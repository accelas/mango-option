# Interpolation Framework

How the library composes interpolants, coordinate transforms, EEP decomposition, and dividend segmentation into specialized price surfaces at compile time. Each section explains a design decision and its tradeoffs, building from the bottom of the stack upward.

**Related documents:**
- [Architecture](ARCHITECTURE.md) — Broader library design
- [Mathematical Foundations](MATHEMATICAL_FOUNDATIONS.md) — Chain-rule Greek derivations, EEP formulas
- [API Guide](API_GUIDE.md) — Usage examples

---

## The Problem

A pre-computed price surface answers five-parameter queries: `price(spot, strike, tau, sigma, rate)`. Between the user's query and the interpolant's evaluation, several transformations must happen:

1. **Coordinate mapping.** The interpolant operates in log-moneyness, not spot/strike. A 3D dimensionless surface uses different coordinates entirely.
2. **EEP reconstruction.** The surface stores Early Exercise Premiums, not raw prices. The European component must be added back analytically.
3. **K/K_ref scaling.** The surface is built at a fixed reference strike. Dollar prices must be rescaled for the actual strike.
4. **Dividend segmentation.** With discrete dividends, the maturity axis is split at ex-dividend dates. The correct segment must be selected.
5. **K_ref blending.** Cash dividends break strike homogeneity. Multiple reference strikes are needed, with interpolation between them.

These transformations are orthogonal — any interpolant works with any coordinate system, any of them can be wrapped in EEP decomposition, and segmentation applies identically to B-splines and Chebyshev. The question is how to combine them without a combinatorial explosion of hand-written surface types.

---

## Why Template Composition

The naive approach is a class hierarchy: a `PriceSurface` base class with virtual `price()` and `vega()`, and derived classes for each combination. This has two problems.

First, performance. The IV solver calls `price()` and `vega()` at every Brent iteration — 4-6 times per IV solve, millions of times per batch. Each call evaluates the interpolant (~193ns for B-spline), maps coordinates, reconstructs EEP, and scales by K_ref. Virtual dispatch adds ~5ns per call (indirect branch + pipeline stall), but worse, it prevents inlining. Without inlining, the compiler cannot see through the layer boundaries and must materialize intermediate results, losing ~15-20% on the full evaluation path.

Second, combinatorics. There are 3 interpolant types × 2 coordinate transforms × 2 EEP modes × 3 splitting configurations = 36 potential combinations. Most will never be instantiated, but a class hierarchy would need to account for all of them, and adding a new interpolant would require wiring it through the full hierarchy.

Template composition solves both problems. Each concern is a template that wraps the layer below it. Composition is structural — templates nest directly with no base classes. The compiler sees through all layers and inlines the full evaluation path, recovering the performance of a hand-written monolithic class. New interpolants or transforms only need to satisfy a concept; they compose with existing layers automatically.

The tradeoff is familiar from CRTP: more complex type signatures, longer compile times, and error messages that expose the full nesting. Explicit template instantiation (section 7) bounds the compile-time cost.

---

## The Layer Stack

Four layers, each adding one concern. The stack is built bottom-up; each section explains why that concern is a separate layer rather than folded into an adjacent one.

### Layer 0: Interpolant

The bottom of the stack is a raw N-dimensional interpolation engine. It has no knowledge of options, strikes, or pricing — just coordinates in, value out.

```cpp
template <typename S, size_t N>
concept SurfaceInterpolant = requires(const S& s, std::array<double, N> coords) {
    { s.eval(coords) } -> std::same_as<double>;
    { s.partial(size_t{}, coords) } -> std::same_as<double>;
};
```

The generic adapter `SharedInterp<T, N>` (in `table/shared_interp.hpp`) wraps `shared_ptr<const T>` to satisfy the concept while preserving shared ownership. It forwards `eval()` and `partial()` unconditionally; `eval_second_partial()` is conditionally available via a `requires` clause — present only when `T` provides it.

B-spline surfaces use `SharedBSplineInterp<N>`, a convenience alias for `SharedInterp<BSplineND<double, N>, N>`. The other implementation is `ChebyshevInterpolant<N, Storage>` (barycentric interpolation on Chebyshev-Gauss-Lobatto nodes, with `RawTensor` or `TuckerTensor` storage).

Why a concept instead of a base class? The two interpolants have fundamentally different capabilities. B-splines provide analytical second derivatives (`eval_second_partial`); Chebyshev does not. A base class would either leave the method unimplemented (runtime error) or force a least-common-denominator interface. A concept lets `TransformLeaf` detect the capability at compile time:

```cpp
if constexpr (requires { interp_.eval_second_partial(size_t{0}, coords); }) {
    return interp_.eval_second_partial(0, coords);  // B-spline: O(n) analytical
} else {
    return (f(x+h) - 2*f(x) + f(x-h)) / (h*h);    // Chebyshev: 3 evaluations
}
```

This matters for gamma accuracy: analytical second derivatives are O(h²) from the B-spline order reduction, while FD second derivatives compound two O(h²) first-derivative errors.

### Layer 1: TransformLeaf

`TransformLeaf<Interp, Xform>` sits between the interpolant and the financial layers. It maps the five physical parameters to N-dimensional interpolation coordinates, evaluates the interpolant, and scales the result by `strike / K_ref`.

Why separate coordinate mapping from the interpolant? Because the interpolant is N-dimensional and dimension-agnostic, but the coordinate mapping is problem-specific. `StandardTransform4D` maps to `{ln(S/K), tau, sigma, rate}` — four axes. `DimensionlessTransform3D` collapses sigma and rate into two dimensionless quantities, producing three axes. The same `SharedBSplineInterp` works with either transform, instantiated at `N=4` or `N=3`.

The transform also provides chain-rule weights for Greeks via a `CoordinateTransform` concept:

```cpp
template <typename T>
concept CoordinateTransform = requires(const T& t, ...) {
    { T::kDim } -> std::convertible_to<size_t>;
    { t.to_coords(spot, strike, tau, sigma, rate) }
        -> std::same_as<std::array<double, T::kDim>>;
    { t.greek_weights(Greek{}, spot, strike, tau, sigma, rate) }
        -> std::same_as<std::array<double, T::kDim>>;
};
```

`greek_weights()` returns `d(coord_i)/d(param)` for each axis. For `StandardTransform4D`, delta weights are `{1/S, 0, 0, 0}` — only the log-moneyness axis contributes, so delta needs one partial derivative evaluation. For `DimensionlessTransform3D`, vega weights are `{0, sigma*tau, -2/sigma}` — sigma appears in both the dimensionless time and rate coordinates, so vega needs two partial evaluations. This coupling is inherent in the dimensionless parameterization and is the main accuracy tradeoff for using 3D instead of 4D.

#### The Two Transforms: 4D vs 3D

The `Xform` parameter determines how many dimensions the surface has, representing different points on a build-cost vs accuracy tradeoff.

**StandardTransform4D** maps to `{ln(S/K), tau, sigma, rate}` — four independent axes. Each Greek touches exactly one axis (delta → axis 0, vega → axis 2, etc.), so each Greek needs one interpolant partial evaluation. The downside is that the 4D grid requires more PDE solves (~500 for the High profile).

**DimensionlessTransform3D** collapses sigma and rate into two dimensionless quantities: `tau' = sigma²*tau/2` (dimensionless time) and `ln_kappa = ln(2r/sigma²)` (dimensionless rate). This reduces the grid from 4D to 3D, cutting PDE solves by roughly half.

The tradeoff is in Greek accuracy. Because sigma appears in both tau' and ln_kappa, vega requires two partial evaluations that couple the axes. More fundamentally, the dimensionless parameterization assumes q=0 (dividend yield cannot be absorbed into the coordinates) and r>0 (ln_kappa diverges at r=0). The sigma/rate coupling also means the surface must represent the joint behavior of both parameters in fewer dimensions, which limits accuracy to ~20-50 bps IV error compared to ~2-5 bps for the 4D path.

Use the 3D path when build speed matters more than accuracy, and when q=0 is acceptable (index options, for example).

The leaf also exposes `raw_value()` — the unscaled interpolant output before K_ref scaling and clamping. This seems like a leaky abstraction, but the next layer needs it for an important optimization.

### Layer 2a: EEPLayer

`EEPLayer<Leaf, EEP>` adds the European component back to the leaf's Early Exercise Premium:

```
American price = leaf.price() + european_price()
American delta = leaf.delta() + european_delta()
```

Why separate EEP from the leaf? Because the segmented path (discrete dividends) does not use EEP decomposition. Segments store raw American prices normalized by K_ref, not EEPs. If EEP were baked into `TransformLeaf`, segmented surfaces would need a separate leaf type — duplicating all the coordinate mapping and scaling logic. By making EEP a wrapper, the same `TransformLeaf` serves both paths.

`EEPLayer` has one important optimization. Before computing the leaf's Greek (which requires interpolant partial evaluations — the most expensive step), it checks `leaf.raw_value()`. If the EEP is zero or negative, the point is deep OTM and the option is purely European. The layer returns the analytical European Greek immediately, skipping interpolation derivatives entirely. For a typical option chain, ~30% of queries hit this fast path. This is why `raw_value()` exists on the leaf.

The `EEPStrategy` concept abstracts the European pricing:

```cpp
template <typename E>
concept EEPStrategy = requires(const E& e, double spot, double strike,
                                double tau, double sigma, double rate) {
    { e.european_price(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { e.european_vega(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { e.european_delta(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    // ... gamma, theta, rho
};
```

Currently there is one implementation: `AnalyticalEEP` (Black-Scholes closed-form). The concept exists because a future Heston or local-vol model would need a different European solver while reusing the same layered composition.

### Layer 2b: SplitSurface

`SplitSurface<Inner, Split>` routes queries to one or two surface pieces via a `SplitPolicy`. It handles two independent splitting concerns:

- **TauSegmentSplit**: Discrete dividends split the maturity axis at ex-dividend dates. Each segment is a separate surface. The policy finds the segment covering the query's tau.
- **MultiKRefSplit**: Cash dividends break strike homogeneity. Multiple reference strikes are needed. The policy blends the two K_ref surfaces bracketing the query strike, with linear weights in ln(K_ref) space.

Why is splitting a separate layer? Because it composes with itself. The segmented dividend path needs both tau splitting and K_ref blending:

```
SplitSurface< SplitSurface<TransformLeaf, TauSegmentSplit>, MultiKRefSplit >
```

A query first hits `MultiKRefSplit`, which selects 1-2 K_ref pieces and assigns weights. For each piece, the query hits `TauSegmentSplit`, which finds the correct tau segment. At the bottom, `TransformLeaf` evaluates the segment's interpolant. The weighted results propagate back up through normalize/denormalize at each level.

The `SplitPolicy` concept captures the four operations every split needs:

```cpp
template <typename S>
concept SplitPolicy = requires(const S& s, ...) {
    { s.bracket(...) } -> std::same_as<BracketResult>;   // which piece(s), with weights
    { s.to_local(...) } -> std::same_as<tuple<5 doubles>>; // remap to piece-local coords
    { s.normalize(...) } -> std::same_as<double>;          // per-piece value normalization
    { s.denormalize(...) } -> std::same_as<double>;        // recombine to final value
};
```

`TauSegmentSplit` always returns one piece (weight = 1.0); it remaps tau to the segment's local range and replaces strike with K_ref. `MultiKRefSplit` returns two pieces (the bracketing K_ref values) with complementary weights; it normalizes each piece's value by its K_ref (converting to V/K_ref basis) and denormalizes by multiplying by the actual strike.

### Layer 3: PriceTable

`PriceTable<Inner>` is a thin outermost wrapper. All its methods delegate to `inner_`. It exists to carry runtime metadata — `SurfaceBounds`, `OptionType`, `dividend_yield` — that the IV solver needs but the mathematical layers should not know about.

Why not fold the metadata into the inner layers? Because `TransformLeaf` and `EEPLayer` are pure mathematical transformations. They do not know whether they are being used for puts or calls, or what the domain bounds are. This separation means the same `EEPLayer<TransformLeaf<...>, AnalyticalEEP>` can be constructed with different bounds and option types without any template parameter changes.

The Greek API also lives here: `delta()`, `gamma()`, `theta()`, `rho()`, each returning `std::expected<double, GreekError>`. The `price()` and `vega()` methods take five raw doubles (for IV solver hot-path compatibility); the Greek methods take `PricingParams` (for application use).

---

## Why Gamma Is Special

All first-order Greeks (delta, vega, theta, rho) follow the same pattern: multiply the transform's weight vector by the interpolant's partial derivative vector, scale by K_ref. The code is generic across all four.

Gamma breaks this pattern because it requires a second derivative. For x = ln(S/K):

```
d²V/dS² = (d²V/dx² - dV/dx) / S²
```

The second term (subtracting dV/dx) comes from d²x/dS² = -1/S². This means gamma needs `eval_second_partial` — a method that B-spline interpolants have (the derivative of a cubic B-spline is a quadratic B-spline, computed analytically) but Chebyshev interpolants do not.

Rather than requiring all interpolants to provide second derivatives, `TransformLeaf` uses a compile-time `if constexpr` branch to select the computation method. B-spline gets the analytical path; Chebyshev gets central FD with h = 1e-4 in log-moneyness. The FD fallback costs three interpolant evaluations instead of one, but it is only invoked for Chebyshev surfaces.

This is why `gamma()` is a separate method on every layer rather than being routed through the generic `greek(Greek, params)` path used by delta, vega, theta, and rho.

---

## Type Erasure at the Boundary

Inside the template stack, everything is monomorphic — the compiler knows the exact types at every layer. But the user-facing factory `make_interpolated_iv_solver(config)` must return a single type regardless of which backend and dividend configuration were selected.

The factory uses `AnyInterpIVSolver`, which holds a `std::variant` of all 7 concrete solver types:

```cpp
using SolverVariant = std::variant<
    InterpolatedIVSolver<BSplinePriceTable>,
    InterpolatedIVSolver<BSplineMultiKRefSurface>,
    InterpolatedIVSolver<ChebyshevSurface>,
    InterpolatedIVSolver<ChebyshevRawSurface>,
    InterpolatedIVSolver<ChebyshevMultiKRefSurface>,
    InterpolatedIVSolver<BSpline3DPriceTable>,
    InterpolatedIVSolver<Chebyshev3DPriceTable>
>;
```

Why `std::variant` instead of virtual dispatch? The dispatch happens once per `solve()` call, not once per surface evaluation. Inside `solve()`, the 4-6 Brent iterations call `price()` and `vega()` on the monomorphic surface type — fully inlined. The `std::visit` overhead at the `solve()` boundary is negligible (~2ns) compared to the ~3.5us total solve time.

All 7 template instantiations are explicit in `interpolated_iv_solver.cpp`. Without explicit instantiation, every translation unit that includes the header would re-instantiate the full template stack, adding ~10s per file. Explicit instantiation confines this cost to one compilation unit.

The factory dispatches on two orthogonal config fields:

```
backend variant (BSpline | Chebyshev | Dimensionless)
  × discrete_dividends (none | DiscreteDividendConfig)
```

```
BSplineBackend      + no dividends     → BSplinePriceTable
BSplineBackend      + dividends        → BSplineMultiKRefSurface
ChebyshevBackend    + no dividends     → ChebyshevSurface or ChebyshevRawSurface
ChebyshevBackend    + dividends        → ChebyshevMultiKRefSurface
DimensionlessBackend                   → BSpline3DPriceTable or Chebyshev3DPriceTable
```

The `DimensionlessBackend` path validates that q=0 and rejects discrete dividends at factory time, before any PDE solves.

---

## Concrete Types at a Glance

For reference, here are the full type alias expansions. The naming convention is: interpolant + path → alias.

**Standard path** (EEP decomposition, continuous dividends):

| Alias | Expansion |
|-------|-----------|
| `BSplinePriceTable` | `PriceTable<EEPLayer<TransformLeaf<SharedInterp<BSplineND<double,4>,4>, StandardTransform4D>, AnalyticalEEP>>` |
| `ChebyshevSurface` | Same structure with `ChebyshevInterpolant<4, TuckerTensor<4>>` |
| `ChebyshevRawSurface` | Same with `RawTensor<4>` (no SVD compression) |
| `BSpline3DPriceTable` | `PriceTable<EEPLayer<TransformLeaf<SharedInterp<BSplineND<double,3>,3>, DimensionlessTransform3D>, AnalyticalEEP>>` |
| `Chebyshev3DPriceTable` | Same with `ChebyshevInterpolant<3, TuckerTensor<3>>` |

**Segmented path** (raw prices, discrete dividends):

| Alias | Expansion |
|-------|-----------|
| `BSplineMultiKRefSurface` | `PriceTable<SplitSurface<SplitSurface<TransformLeaf<SharedInterp<BSplineND<double,4>,4>, StandardTransform4D>, TauSegmentSplit>, MultiKRefSplit>>` |
| `ChebyshevMultiKRefSurface` | Same with `ChebyshevInterpolant<4, RawTensor<4>>` |

Segmented leaves do not use `EEPLayer` — they store V/K_ref directly, because the initial condition for each segment comes from the next segment's surface evaluation (not from a closed-form expression), so no clean European decomposition exists.

---

## Key Source Files

| File | Role |
|------|------|
| `table/surface_concepts.hpp` | The four concepts: `SurfaceInterpolant`, `CoordinateTransform`, `EEPStrategy`, `SplitPolicy` |
| `table/transform_leaf.hpp` | Layer 1: coordinate mapping, K_ref scaling, compile-time gamma dispatch |
| `table/eep/eep_layer.hpp` | Layer 2a: EEP reconstruction with early-exit optimization |
| `table/split_surface.hpp` | Layer 2b: composable query routing for segmentation |
| `table/price_table.hpp` | Layer 3: metadata wrapper and public Greek API |
| `table/transforms/standard_4d.hpp` | 4D identity transform with Greek weights |
| `table/transforms/dimensionless_3d.hpp` | 3D dimensionless transform with coupled vega weights |
| `table/splits/tau_segment.hpp` | Tau routing for dividend segments |
| `table/splits/multi_kref.hpp` | K_ref blending for strike homogeneity |
| `table/shared_interp.hpp` | `SharedInterp<T, N>` generic shared-ownership adapter |
| `table/bspline/bspline_surface.hpp` | `SharedBSplineInterp<N>` alias and all B-spline type aliases |
| `table/chebyshev/chebyshev_surface.hpp` | Chebyshev type aliases |
| `interpolated_iv_solver.cpp` | Factory dispatch, explicit instantiations, `AnyInterpIVSolver` variant |
