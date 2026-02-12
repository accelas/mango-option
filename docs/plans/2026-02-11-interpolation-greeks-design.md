# Interpolation Surface Greeks — Design

## Goal

Add delta, gamma, theta, and rho to all interpolated price surfaces (B-spline, Chebyshev, segmented). Vega already exists. Use the fastest available algorithm per backend, with FD fallback when analytical derivatives are not available.

## Scope

First-order: delta, vega (existing), theta, rho.
Second-order: gamma only.
No cross-Greeks (vanna, volga, charm).

## User-facing API

Each surface type (BSplinePriceTable, ChebyshevPriceTable, segmented variants) gains methods alongside existing `price()` and `vega()`:

```cpp
std::expected<double, GreekError> delta(const PricingParams& params) const;
std::expected<double, GreekError> gamma(const PricingParams& params) const;
std::expected<double, GreekError> theta(const PricingParams& params) const;
std::expected<double, GreekError> rho(const PricingParams& params) const;
```

`price()` and `vega()` stay as `double` return for backward compatibility.

`GreekError` is a lightweight enum: `OutOfDomain`, `NumericalFailure`, `UnsupportedBackend`.

## First-order Greeks via transform weights

Delta, theta, and rho follow the same pattern vega already uses. The coordinate transform maps physical params to internal axes and provides weights encoding the chain rule.

Currently `CoordinateTransform` requires `vega_weights()`. Generalize to:

```cpp
enum class Greek { Delta, Vega, Theta, Rho };

// CoordinateTransform concept requires:
std::array<double, kDim> greek_weights(Greek g, double spot, double strike,
                                        double tau, double sigma, double rate) const;
```

For `StandardTransform4D` (axes: x, tau, sigma, r):
- Delta: `{1/S, 0, 0, 0}`
- Vega: `{0, 0, 1, 0}`
- Theta: `{0, -1, 0, 0}`
- Rho: `{0, 0, 0, 1}`

For `DimensionlessTransform3D` (axes: x, tau' = sigma^2 tau / 2, ln kappa = ln(2r / sigma^2)):
- Delta: `{1/S, 0, 0}`
- Vega: `{0, sigma * tau, -2/sigma}`
- Theta: `{0, sigma^2 / 2, 0}`
- Rho: `{0, 0, 1/r}`

`TransformLeaf` computes any first-order Greek generically:

```cpp
double raw_partial = 0.0;
auto w = xform_.greek_weights(greek, spot, strike, tau, sigma, rate);
for (size_t i = 0; i < kDim; ++i)
    if (w[i] != 0.0)
        raw_partial += w[i] * interp_.partial(i, coords);
```

This replaces the hardcoded `vega_weights()` with one method that handles all four first-order Greeks.

### Scaling

`TransformLeaf` stores normalized prices (`price / K_ref`). All first-order Greeks from the leaf are scaled by `strike / K_ref`. For delta specifically, the chain rule through `x = ln(S/K)` gives `dx/dS = 1/S`, so the weight already encodes that.

## Gamma — second-order derivative

Gamma = d^2 V / dS^2 needs the chain rule through `x = ln(S/K)`. Since x is the moneyness axis (axis 0) in both transforms, the formula is the same regardless of backend:

```
V = f(x, ...) * strike / K_ref

dV/dS = (df/dx)(1/S) * strike / K_ref

d^2V/dS^2 = (d^2f/dx^2 - df/dx) / S^2 * strike / K_ref
```

This needs `partial(0, coords)` and `eval_second_partial(0, coords)` from the interpolant.

Strategy by backend:
- **BSplineND**: Both are analytical (derivative basis functions). Fast, exact.
- **Chebyshev**: `partial()` is already FD. `eval_second_partial()` does not exist. Fall back to FD on `eval()`:
  ```
  d^2f/dx^2 ~ (f(x+h) - 2f(x) + f(x-h)) / h^2
  ```

Compile-time dispatch in `TransformLeaf`:

```cpp
if constexpr (requires { interp_.eval_second_partial(0, coords); }) {
    // Analytical path (B-spline)
} else {
    // FD fallback (Chebyshev, future backends)
}
```

The FD fallback can fail near domain boundaries — this is why Greeks return `std::expected`.

## EEP layer composition

For EEP-decomposed surfaces (standard non-dividend path):

```
American Greek = EEP Greek + European Greek
```

`EuropeanOptionResult` already computes all five Greeks analytically. `AnalyticalEEP` gains:
- `european_delta(params)`
- `european_gamma(params)`
- `european_theta(params)`
- `european_rho(params)`

The `EEPStrategy` concept expands to require these.

Early guard at EEP layer level: when the EEP surface reads `raw <= 0` (deep OTM), skip all partial computation and return European Greek only. Applied once at this level, not repeated in each `TransformLeaf` method.

For segmented surfaces (discrete dividends, no EEP decomposition), `TransformLeaf` computes the full American Greek directly from interpolant partials. `SplitSurface` routes to the correct segment with bracket weighting.

## Files to modify

| File | Change |
|------|--------|
| New: `greek_types.hpp` | `Greek` enum, `GreekError` enum |
| `surface_concepts.hpp` | Replace `vega_weights` with `greek_weights` in `CoordinateTransform` concept. Expand `EEPStrategy` concept with new European Greek methods. |
| `transforms/standard_4d.hpp` | Replace `vega_weights()` with `greek_weights()` |
| `transforms/dimensionless_3d.hpp` | Same |
| `transform_leaf.hpp` | Generic `compute_greek()` using weights. Gamma with compile-time FD fallback. Remove hardcoded `vega()`. |
| `eep/eep_layer.hpp` | Add delta/gamma/theta/rho. Early `raw <= 0` guard. |
| `eep/analytical_eep.hpp` | Add `european_delta`/`gamma`/`theta`/`rho` |
| `price_table.hpp` | Add delta/gamma/theta/rho forwarding |
| `split_surface.hpp` | Add delta/gamma/theta/rho with bracket routing |
| `bspline_surface.hpp` | Type aliases pick up new methods automatically |

No changes to `BSplineND` or `ChebyshevInterpolant` — they already have the derivative methods needed.

## Testing

Greek accuracy tests comparing interpolated Greeks against FDM reference (`AmericanOptionResult`) for delta, gamma, theta, and against Black-Scholes analytical for European-equivalent cases. Test both B-spline and Chebyshev backends, both 4D standard and 3D dimensionless transforms.
