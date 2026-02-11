# Interpolation Refactor Design

## Goal

Refactor the price surface architecture to support pluggable interpolation
schemes (B-spline, Chebyshev, future), pluggable coordinate transforms
(4D standard, 3D dimensionless), composable surface splits (tau-segmented,
x-piecewise, multi-K_ref), and pluggable EEP decomposition — all behind
compile-time concepts with zero runtime overhead.

## Motivation

PR #386 (dimensionless 3D) and PR #387 (Chebyshev-Tucker experiment) revealed
that the current architecture couples B-spline interpolation into every layer.
The Chebyshev experiment duplicated ~500 lines of adapter code (EEP
reconstruction, segmentation, multi-K_ref blending, vega computation) because
there is no clean interpolation abstraction. Key problems:

1. `PriceTableSurface` hardwires `BSplineND` — no way to swap interpolation
2. `EEPPriceTableInner` bakes EEP reconstruction into B-spline-specific types
3. `SplicedSurface` has 4 template params (Inner, Split, Xform, Combiner) but
   the split logic only handles tau-segmentation and K_ref — no x-piecewise
4. Grid construction (headroom, density heuristics) assumes B-spline convergence
5. Adaptive refinement algorithm is B-spline-specific (iterative axis refinement)

## Architecture

Four concept-based layers, each independently pluggable:

```
InterpolatedIVSolver (unchanged)
  └── AnyIVSolver (type-erased variant at API boundary)
        └── PriceSurface concept (price + vega from 5 params)
              └── SplitSurface<Inner, SplitPolicy>  (composable splits)
                    └── EEPSurfaceAdapter<Interp, Xform, EEP>
                          ├── SurfaceInterpolant concept (eval + partial from N coords)
                          ├── CoordinateTransform concept (5 params → N coords + vega weights)
                          └── EEPStrategy concept (european price/vega + scale)
```

### Layer 1: SurfaceInterpolant concept

The raw interpolation engine. Any scheme that can evaluate a value and a
partial derivative at N-dimensional coordinates.

```cpp
template <typename S, size_t N>
concept SurfaceInterpolant = requires(const S& s, std::array<double, N> coords) {
    { s.eval(coords) } -> std::same_as<double>;
    { s.partial(size_t{}, coords) } -> std::same_as<double>;
};
```

Implementations:
- `BSplineND<N>` — existing, needs thin adapter (rename eval/eval_partial)
- `ChebyshevTuckerND<N>` — already has this interface (future, on experiment branch)

### Layer 2: CoordinateTransform concept

Maps a 5-parameter price query (spot, strike, tau, sigma, rate) to
N-dimensional interpolation coordinates, plus vega weights for chain-rule
differentiation.

```cpp
template <typename T>
concept CoordinateTransform = requires(const T& t, double spot, double strike,
                                        double tau, double sigma, double rate) {
    { T::kDim } -> /* constexpr size_t */;
    { t.to_coords(spot, strike, tau, sigma, rate) }
        -> std::same_as<std::array<double, T::kDim>>;
    { t.vega_weights(spot, strike, tau, sigma, rate) }
        -> std::same_as<std::array<double, T::kDim>>;
};
```

Implementations:

**StandardTransform4D** — identity mapping:
- `to_coords` → `{ln(S/K), tau, sigma, rate}`
- `vega_weights` → `{0, 0, 1, 0}` (direct partial on sigma axis)

**DimensionlessTransform3D** — collapses tau+sigma:
- `to_coords` → `{ln(S/K), σ²τ/2, ln(2r/σ²)}`
- `vega_weights` → `{0, στ, -2/σ}` (chain rule with two terms)

Vega computed generically:
```cpp
double vega = 0;
auto w = transform.vega_weights(spot, strike, tau, sigma, rate);
auto coords = transform.to_coords(spot, strike, tau, sigma, rate);
for (size_t i = 0; i < N; ++i)
    vega += w[i] * interpolant.partial(i, coords);
```

### Layer 3: EEPStrategy concept

Handles decomposition: American = EEP × scale + European. Pluggable to
support future numerical EEP without changing the adapter.

```cpp
template <typename E>
concept EEPStrategy = requires(const E& e, double spot, double strike,
                                double tau, double sigma, double rate,
                                double K_ref) {
    { e.european_price(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { e.european_vega(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { e.scale(strike, K_ref) } -> std::same_as<double>;
};
```

Implementations:
- **AnalyticalEEP** — closed-form BS European. Handles dividend yield and
  escrowed spot for discrete dividends. Current production path.
- **IdentityEEP** — no decomposition (surface stores V/K_ref directly).
  `european_price()` returns 0, `scale()` returns K/K_ref. Used for segmented
  dividend segments.
- **NumericalEEP** — future extension point. Same interface, different European
  computation. Not designed yet.

### Layer 4: EEPSurfaceAdapter

Composes interpolant + transform + EEP into a complete price surface:

```cpp
template <SurfaceInterpolant<N> Interp, CoordinateTransform Xform, EEPStrategy EEP>
class EEPSurfaceAdapter {
    Interp interp_;
    Xform xform_;
    EEP eep_;
    double K_ref_;

public:
    double price(double spot, double strike, double tau, double sigma, double rate) const {
        auto coords = xform_.to_coords(spot, strike, tau, sigma, rate);
        double raw = interp_.eval(coords);
        double eep = std::max(0.0, raw);
        return eep * eep_.scale(strike, K_ref_)
             + eep_.european_price(spot, strike, tau, sigma, rate);
    }

    double vega(double spot, double strike, double tau, double sigma, double rate) const {
        auto coords = xform_.to_coords(spot, strike, tau, sigma, rate);
        double raw = interp_.eval(coords);
        if (raw <= 0.0) {
            // EEP clamped to zero → EEP vega is zero (consistent with price clamp)
            return eep_.european_vega(spot, strike, tau, sigma, rate);
        }
        auto w = xform_.vega_weights(spot, strike, tau, sigma, rate);
        double eep_vega = 0;
        for (size_t i = 0; i < Xform::kDim; ++i)
            eep_vega += w[i] * interp_.partial(i, coords);
        return eep_vega * eep_.scale(strike, K_ref_)
             + eep_.european_vega(spot, strike, tau, sigma, rate);
    }
};
```

Replaces: `EEPPriceTableInner`, `PriceTableInner`, all Chebyshev `*Inner`
adapters from the experiment (~500 lines eliminated).

### Layer 5: SplitSurface

Composable surface splitting along any axis. Replaces `SplicedSurface` (4
template params) with a simpler design (2 params).

The current `SplicedSurface` has a `SliceTransform` that performs per-slice
query remapping (`to_local`) and value normalization (`normalize_value`,
`denormalize`). This is essential for correctness:

- **SegmentedTransform**: remaps global tau → segment-local tau, sets
  strike = K_ref, multiplies result by K_ref
- **KRefTransform**: sets strike = slice's K_ref, divides result by K_ref,
  then denormalizes combined result by multiplying by query strike

In the new design, this per-slice remapping lives in the `SplitPolicy`
itself. Each policy defines both routing (bracket) and per-slice query/value
transforms:

```cpp
struct BracketResult {
    struct Entry { size_t index; double weight; };
    std::array<Entry, 2> entries;
    size_t count;  // 1 = single piece, 2 = overlap blend
};

template <typename S>
concept SplitPolicy = requires(const S& s, double spot, double strike,
                                double tau, double sigma, double rate) {
    { s.bracket(spot, strike, tau, sigma, rate) }
        -> std::same_as<BracketResult>;

    // Per-slice query remapping (e.g., tau → local tau, strike → K_ref)
    { s.to_local(size_t{}, spot, strike, tau, sigma, rate) }
        -> std::same_as<std::tuple<double, double, double, double, double>>;

    // Per-slice value normalization (e.g., multiply by K_ref)
    { s.normalize(size_t{}, strike, double{}) } -> std::same_as<double>;

    // Post-combine denormalization (e.g., multiply by query strike)
    { s.denormalize(double{}, spot, strike, tau, sigma, rate) }
        -> std::same_as<double>;
};

template <typename Inner, SplitPolicy Split>
class SplitSurface {
    std::vector<Inner> pieces_;
    Split split_;

public:
    double price(double spot, double strike, double tau, double sigma, double rate) const {
        auto br = split_.bracket(spot, strike, tau, sigma, rate);
        double result = 0;
        for (size_t i = 0; i < br.count; ++i) {
            auto [ls, lk, lt, lv, lr] = split_.to_local(
                br.entries[i].index, spot, strike, tau, sigma, rate);
            double raw = pieces_[br.entries[i].index].price(ls, lk, lt, lv, lr);
            double norm = split_.normalize(br.entries[i].index, strike, raw);
            result += br.entries[i].weight * norm;
        }
        return split_.denormalize(result, spot, strike, tau, sigma, rate);
    }
    // vega() follows the same pattern
};
```

Split policies:

- **TauSegmentSplit** — routes by tau, remaps to segment-local tau, sets
  strike = K_ref, normalizes by K_ref. Replaces current
  SegmentLookup + SegmentedTransform.
- **MultiKRefSplit** — routes by strike, sets strike = slice K_ref,
  normalizes by dividing by K_ref, denormalizes by multiplying by query
  strike. Replaces current KRefBracket + KRefTransform.
- **IdentitySplit** — pass-through (no remapping, no normalization).
  For vanilla single-surface case and PiecewiseXSplit.
- **PiecewiseXSplit** — routes by x = ln(S/K) to ITM/boundary/OTM elements
  with C∞ bump blending. No query remapping needed. (Future, comes with
  Chebyshev merge.)

Composition examples:

Vanilla B-spline:
```cpp
EEPSurfaceAdapter<BSplineND<4>, StandardTransform4D, AnalyticalEEP>
```

Vanilla 3D dimensionless:
```cpp
EEPSurfaceAdapter<BSplineND<3>, DimensionlessTransform3D, AnalyticalEEP>
```

Discrete dividends + multi-K_ref (showing where remapping lives):
```cpp
SplitSurface<                         // MultiKRefSplit: strike → K_ref, denorm by strike
  SplitSurface<                       // TauSegmentSplit: tau → local tau, strike → K_ref
    EEPSurfaceAdapter<BSplineND<4>, StandardTransform4D, IdentityEEP>,
    TauSegmentSplit>,
  MultiKRefSplit>
```

Note: when TauSegmentSplit and MultiKRefSplit are composed, both remap
strike. MultiKRefSplit (outer) sets strike = slice K_ref first, then
TauSegmentSplit (inner) receives the already-remapped strike. The inner
SegmentedTransform uses the same K_ref as the outer MultiKRef slice, which
is set during surface construction — each inner SplitSurface is built for
one specific K_ref.

### Runtime Performance

The hot path is IV solving (Brent calls `price()` ~10-20 times per query).
All layers are concept-based templates — compiler monomorphizes and inlines
everything. Zero virtual dispatch.

Per `price()` call through the full dividend stack (MultiKRef + TauSegment):
- Each `SplitSurface` level: `bracket()` (1-2 comparisons, ~2ns) returns
  at most 2 entries. Nesting two levels produces up to 2×1 = 2 leaf evals
  (TauSegmentSplit always returns 1 entry). With future PiecewiseXSplit
  nesting: up to 2×2 = 4 leaf evals in overlap zones.
- `to_coords()` computed once at each leaf, not per split level
- Framework overhead: <5ns on top of 200-500ns interpolation

## Builder Pipeline

### Shared: PDE Tensor Builder

Extracts raw N-dimensional tensor of American prices from PDE solves.
Interpolation-agnostic — refactored out of current `PriceTableBuilder`.

Note: `PriceTensor` and `PriceTableAxes` are currently 4D-only aliases
(`PriceTensorND<4>`, `PriceTableAxesND<4>`). The tensor builder and fitter
must be templated on dimension N to support both 3D dimensionless and 4D
standard paths:

```cpp
template <size_t N>
struct TensorBuildResult {
    PriceTensorND<N> tensor;
    PriceTableAxesND<N> axes;
    size_t pde_solves;
    double build_seconds;
};

template <size_t N>
std::expected<TensorBuildResult<N>, PriceTableError>
build_price_tensor(const PriceTensorConfig<N>& config);
```

The existing `PriceTensorND<N>` and `PriceTableAxesND<N>` templates already
exist in the codebase (`price_tensor.hpp:77`, `price_table_axes.hpp:91`) —
they just need to be used directly instead of through the 4D aliases.

Includes the tensor transform callback for EEP decomposition.

### Scheme-Specific: Surface Fitter

```cpp
template <typename F, size_t N>
concept SurfaceFitter = requires(const F& f, const PriceTensorND<N>& tensor,
                                  const PriceTableAxesND<N>& axes) {
    { f.fit(tensor, axes) } -> SurfaceInterpolant<N>;
};
```

- **BSplineFitter<N>** — separable least-squares, clamped cubic knots
- **ChebyshevFitter<N>** — Tucker HOSVD compression (future)

### Scheme-Specific: Adaptive Builder

B-spline and Chebyshev have fundamentally different adaptive algorithms:
- B-spline: iterative axis refinement, add knots, refit, validate
- Chebyshev: nested Clenshaw-Curtis levels, incremental PDE cache reuse

Each adaptive builder owns its full loop. They share utilities (PDE tensor
builder, validation probes, surface assembly) but not the outer loop.

```cpp
template <typename B>
concept AdaptiveBuilder = requires(const B& b, auto config) {
    { b.build(config) } -> /* assembled surface */;
};
```

- **BSplineAdaptiveBuilder** — current `AdaptiveGridBuilder`, refactored
- **ChebyshevAdaptiveBuilder** — CC levels + PDESliceCache (future)

## File Organization

```
src/option/table/
├── surface.hpp                  # SurfaceInterpolant, CoordinateTransform, EEPStrategy concepts
├── eep_surface_adapter.hpp      # EEPSurfaceAdapter template
├── split_surface.hpp            # SplitSurface, SplitPolicy, BracketResult
├── price_tensor.hpp             # (unchanged)
├── price_query.hpp              # (unchanged)
├── tensor_builder.hpp           # PDE batch solve + tensor extraction
│
├── transforms/
│   ├── standard_4d.hpp          # StandardTransform4D
│   └── dimensionless_3d.hpp     # DimensionlessTransform3D
│
├── eep/
│   ├── analytical_eep.hpp       # AnalyticalEEP (BS-based)
│   └── identity_eep.hpp         # IdentityEEP (no decomposition)
│
├── splits/
│   ├── tau_segment.hpp          # TauSegmentSplit
│   └── multi_kref.hpp           # MultiKRefSplit
│
├── bspline/
│   ├── bspline_fitter.hpp       # BSplineFitter<N>
│   └── bspline_adaptive.hpp     # BSplineAdaptiveBuilder
│
└── dimensionless/               # (already exists from PR #386)
    └── ...                      # Refactored to use new concepts
```

No Chebyshev directory — that comes when the experiment merges.
No PiecewiseXSplit — that comes with Chebyshev.

## Public API

Minimal change. One new field on `IVSolverFactoryConfig`:

```cpp
enum class InterpolationScheme : uint8_t {
    BSpline4D,
    Dimensionless3D,
};

struct IVSolverFactoryConfig {
    // ... existing fields unchanged ...
    InterpolationScheme scheme = InterpolationScheme::BSpline4D;
};
```

`make_interpolated_iv_solver` dispatches on scheme. Default is BSpline4D,
so all existing code is unchanged.

**Domain validation for Dimensionless3D:**

`DimensionlessTransform3D` computes `ln(2r/σ²)` and `-2/σ`, which require
`rate > 0` and `sigma > 0`. The factory must reject incompatible configs:

- All rate grid values must be strictly positive (r > 0)
- All vol grid values must be strictly positive (σ > 0)
- Zero or negative rates are not supported by the dimensionless reduction
  because κ = 2r/σ² → 0 or negative, making ln(κ) undefined

When validation fails, the factory returns
`std::unexpected(ValidationError{ValidationErrorCode::InvalidBounds, ...})`
with a clear error. The caller can fall back to `BSpline4D` which has no
such constraint.

The `Dimensionless3D` scheme also does not support the `SegmentedIVPath`
(discrete dividends). The factory rejects this combination — discrete
dividends require per-segment surfaces that the 3D reduction cannot handle
(each segment may need different dividend adjustments). Callers must use
`BSpline4D` for the segmented path.

## What Gets Deleted

- `price_table_surface.hpp` → replaced by SurfaceInterpolant concept + BSplineFitter
- `price_table_inner.hpp` → replaced by EEPSurfaceAdapter with IdentityEEP
- `eep_transform.hpp` → split into eep/analytical_eep.hpp + tensor_builder.hpp
- `spliced_surface.hpp` → replaced by split_surface.hpp (2 template params, not 4)
- `standard_surface.hpp` → type aliases move to factory code
- `spliced_surface_builder.hpp` → assembly logic moves into adaptive builders

## What Stays Unchanged

- `price_tensor.hpp`, `price_query.hpp`, `price_table_axes.hpp`, `price_table_metadata.hpp`
- `adaptive_grid_types.hpp` — config types shared by adaptive builders
- `InterpolatedIVSolver<Surface>` template — still parametrized on PriceSurface concept
- `AnyIVSolver` — stays a variant, grows when new schemes added
- All 116 tests pass throughout migration

## Migration Path

Purely structural refactor — no behavior changes for the B-spline path.

1. Introduce concepts and new adapter types alongside existing code
2. Rewire B-spline path to use new types (EEPSurfaceAdapter, SplitSurface)
3. Rewire dimensionless 3D path to use DimensionlessTransform3D
4. Delete old types (PriceTableSurface, SplicedSurface, etc.)
5. Verify: all tests pass, benchmarks unchanged
6. Later: Chebyshev plugs in by implementing SurfaceInterpolant + ChebyshevAdaptiveBuilder
