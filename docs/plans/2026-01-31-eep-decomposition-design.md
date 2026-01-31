# EEP Decomposition for Price Tables

## Problem

The price table's B-spline interpolation converges at O(h^2.5) instead of the theoretical O(h^4) for cubic splines. The American early exercise boundary creates a C1 discontinuity that degrades accuracy, especially for short-dated slightly-OTM puts where small vega amplifies price errors into IV errors. Doubling PDE solves from 495 to 812 reduces average error only from 7.5 to 5.1 bps.

## Solution

Decompose the American price into two components:

```
P_American = P_European + EEP
```

The B-spline interpolates only the early exercise premium (EEP). The European component is computed in closed form at query time. EEP is smaller, more localized near the exercise boundary, and lacks the log-normal curvature of the full price surface. The spline approximation should converge faster on this smoother target.

## Architecture

### Decomposition boundary

Store EEP in the builder (Option 1). The generic `PriceTableSurface<N>` remains unchanged — it interpolates whatever tensor it receives. A new `AmericanPriceSurface` wrapper reconstructs the full American price at query time.

```
Build:   PDE → P_Am → subtract P_Eu(BS) → store EEP in tensor → fit B-spline
Query:   EEP_spline(m,τ,σ,r) * (K/K_ref) + P_Eu(S,K,τ,σ,r,q) → American price
```

### Rate model

Flat rate only. The price table collapses yield curves to a zero rate. The European BS formula uses the same flat rate, so EEP is consistent between build and query time.

### Strike scaling

EEP scales linearly with K (degree-1 homogeneity in S, K under constant-rate BS). At query time, scale EEP by K/K_ref and compute P_Eu exactly for the actual strike K. This avoids multiplying a small European price by a large scale factor.

## New Files

### `src/option/european_option.hpp`

European put/call pricing with Greeks. Mirrors the `AmericanOptionSolver` API: accepts `PricingParams`, returns a result object with the same accessor methods. All inline — combinations of `norm_cdf`, `exp`, `sqrt`.

```cpp
namespace mango {

/// European option result with the same interface as AmericanOptionResult
class EuropeanOptionResult {
public:
    double value() const;          // price at current spot
    double value_at(double S) const;  // price at arbitrary spot
    double delta() const;
    double gamma() const;
    double vega() const;
    double theta() const;
    double rho() const;

    // Access underlying params (matches AmericanOptionResult)
    double spot() const;
    double strike() const;
    double maturity() const;
    double volatility() const;
    OptionType option_type() const;
};

/// Solve European option (mirrors AmericanOptionSolver::create / solve)
class EuropeanOptionSolver {
public:
    explicit EuropeanOptionSolver(const PricingParams& params);

    static std::expected<EuropeanOptionSolver, ValidationError>
    create(const PricingParams& params) noexcept;

    EuropeanOptionResult solve() const;

private:
    PricingParams params_;
};

}  // namespace mango
```

The solver/result split matches the American pattern. `EuropeanOptionSolver::solve()` is trivial (closed-form), but the API shape is consistent so callers can swap between American and European solvers.

### `src/option/table/american_price_surface.hpp`

Wraps `PriceTableSurface<4>` and adds European reconstruction.

```cpp
namespace mango {

class AmericanPriceSurface {
public:
    static std::expected<AmericanPriceSurface, ValidationError> create(
        std::shared_ptr<const PriceTableSurface<4>> eep_surface,
        OptionType type);

    /// American price: P_EU(S,K,τ,σ,r,q) + EEP(m,τ,σ,r) * (K/K_ref)
    double price(double spot, double strike, double tau,
                 double sigma, double rate) const;

    double delta(double spot, double strike, double tau,
                 double sigma, double rate) const;

    double vega(double spot, double strike, double tau,
                double sigma, double rate) const;

    /// Access underlying EEP surface for IV solver
    const PriceTableSurface<4>& eep_surface() const;
    const PriceTableMetadata& metadata() const;

private:
    std::shared_ptr<const PriceTableSurface<4>> surface_;
    OptionType type_;
    double K_ref_;
    double dividend_yield_;
};

}  // namespace mango
```

## Modified Files

### `src/option/table/price_table_builder.hpp`

In `extract_tensor()`, subtract the European price from each PDE result before storing:

```cpp
double p_eu = european_option_price(PricingParams{
    m * K_ref, K_ref, tau, sigma, rate, dividend_yield, type}).price;
tensor(m_idx, tau_idx, sigma_idx, r_idx) = p_am - p_eu;
```

Add `bool store_eep = true` to `PriceTableConfig`. Record in `PriceTableMetadata` whether the surface stores EEP or raw prices.

### `src/option/iv_solver_interpolated.hpp`

Accept `AmericanPriceSurface` instead of raw `PriceTableSurface<4>`. The Newton iteration becomes:

```
f(σ) = american_surface.price(S, K, τ, σ, r) - market_price
f'(σ) = american_surface.vega(S, K, τ, σ, r)
```

Overload `create()` to accept either surface type for backward compatibility.

### `src/option/table/price_table_metadata.hpp`

Add a field indicating decomposition mode:

```cpp
enum class SurfaceContent { RawPrice, EarlyExercisePremium };
SurfaceContent content = SurfaceContent::RawPrice;
```

## Tests

### `tests/european_option_test.cc`

- Known values against textbook examples
- Put-call parity: C - P = S*exp(-qT) - K*exp(-rT)
- Edge cases: deep ITM/OTM, τ → 0, σ → 0
- Greeks vs finite differences

### `tests/american_price_surface_test.cc`

- Round-trip: build EEP surface, reconstruct American price, compare against direct PDE
- Greeks accuracy vs FDM Greeks
- Strike scaling consistency

### `tests/price_table_4d_integration_test.cc` (extend)

- EEP mode vs raw mode accuracy comparison on SPY data
- Verify EEP mode reduces max error in short-dated OTM region

## Performance

- **Build time**: One `european_option_price()` call per grid point (~10ns each). Negligible compared to PDE solves.
- **Query time**: One BS evaluation (~10ns) added to B-spline eval (~500ns). Total ~510ns for price, ~30us for IV. No measurable regression.
- **Memory**: Unchanged — same tensor size, same B-spline coefficient count.

## Risks

1. **Discretization mismatch**: The PDE-computed P_Am and closed-form P_Eu use different numerical methods. The EEP tensor absorbs this difference. As long as the same `european_option_price()` is used at build and query time, the decomposition is consistent.
2. **Negative EEP**: EEP should be non-negative (American >= European). PDE discretization error could produce small negatives. Clamp to zero if needed.
3. **Backward compatibility**: Existing users who consume raw price surfaces need the `store_eep` flag defaulting to `true` for new builds. `AmericanPriceSurface::create()` validates the metadata field.
