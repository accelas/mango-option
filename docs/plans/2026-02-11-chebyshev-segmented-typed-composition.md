# Chebyshev Segmented: Typed SplitSurface Composition

**Date:** 2026-02-11

## Problem

The Chebyshev 4D segmented (discrete dividend) path uses type-erased
`std::function` lambdas that manually reimplement `TauSegmentSplit` and
`MultiKRefSplit` routing. This breaks the typed composition that B-spline
segmented uses, forces `FDVegaLeaf` (FD central differences for vega instead
of analytical partials), and duplicates split logic.

## Goal

Replace the type-erased lambda chain with the same typed `SplitSurface`
composition that B-spline uses. Eliminate `FDVegaLeaf`. Get analytical vega
through `ChebyshevInterpolant::partial()`.

Scope: Chebyshev segmented path only. B-spline path untouched.

## Type Stack

New aliases in `chebyshev_surface.hpp`:

```
ChebyshevSegmentedLeaf    = TransformLeaf<ChebyshevInterpolant<4, RawTensor<4>>, StandardTransform4D>
                            (already exists)

ChebyshevTauSegmented     = SplitSurface<ChebyshevSegmentedLeaf, TauSegmentSplit>       (new)

ChebyshevMultiKRefInner   = SplitSurface<ChebyshevTauSegmented, MultiKRefSplit>         (new)

ChebyshevMultiKRefSurface = PriceTable<ChebyshevMultiKRefInner>                         (new)
```

New solver instantiation:

```cpp
template class InterpolatedIVSolver<ChebyshevMultiKRefSurface>;
```

## Gap Segment Handling

Dividend dates create narrow gap segments (±0.5ms) where no CGL nodes are
placed. The current lambda routes gap queries to the nearest real segment
at query time (20 lines of logic).

Instead, absorb gaps into adjacent real segments at construction time via a
new helper:

```cpp
TauSegmentSplit make_tau_split_from_segments(
    const std::vector<double>& bounds,
    const std::vector<bool>& is_gap,
    double K_ref);
```

Each real segment's range extends to the midpoint of its adjacent gap:

```
Input bounds:  [τ_min, τ_div-ε, τ_div+ε, τ_max]
Input is_gap:  [false,  true,    false]

Output TauSegmentSplit:
  segment 0: tau_start=τ_min,   tau_end=τ_div
  segment 1: tau_start=τ_div,   tau_end=τ_max
```

Queries inside the gap get clamped by `TauSegmentSplit::to_local`'s
`std::clamp`. Same effective behavior, no gap logic in the split policy.

## Build Function Split

`make_segmented_chebyshev_build_fn` currently serves both the adaptive
refinement loop and final per-K_ref builds. Split into two roles:

**During refinement** (unchanged): existing `BuildFn` → `SurfaceHandle`.
Only runs at probe K_ref. Output used for error measurement only.

**Final assembly** (new): returns typed pieces.

```cpp
struct ChebyshevSegmentedPieces {
    std::vector<ChebyshevSegmentedLeaf> leaves;
    TauSegmentSplit tau_split;
};

std::expected<ChebyshevSegmentedPieces, PriceTableError>
build_chebyshev_segmented_pieces(
    double K_ref,
    OptionType option_type,
    double dividend_yield,
    const std::vector<Dividend>& discrete_dividends,
    const std::vector<double>& seg_bounds,
    const std::vector<bool>& seg_is_gap,
    std::span<const double> m_nodes,
    std::span<const double> tau_nodes,
    std::span<const double> sigma_nodes,
    std::span<const double> rate_nodes);
```

Internals extracted from the existing lambda body (segment mapping,
per-segment tensor fill, `ChebyshevInterpolant::build_from_values`).
Each per-K_ref call gets its own `ChebyshevPDECache`.

## Final Assembly

The tail of `build_adaptive_chebyshev_segmented` changes from collecting
`std::function` lambdas to composing typed surfaces:

```cpp
std::vector<ChebyshevTauSegmented> kref_surfaces;
for (double k_ref : K_refs) {
    auto pieces = build_chebyshev_segmented_pieces(
        k_ref, config.option_type, config.dividend_yield,
        config.discrete_dividends, seg_bounds, seg_is_gap,
        grids.moneyness, grids.tau, grids.vol, grids.rate);
    kref_surfaces.emplace_back(
        std::move(pieces->leaves), std::move(pieces->tau_split));
}

ChebyshevMultiKRefInner inner(std::move(kref_surfaces), MultiKRefSplit(K_refs));
ChebyshevMultiKRefSurface surface(std::move(inner), bounds,
                                   config.option_type, config.dividend_yield);
```

Return type changes:

```cpp
struct ChebyshevSegmentedAdaptiveResult {
    ChebyshevMultiKRefSurface surface;    // was: std::function price_fn
    // ... rest unchanged
};
```

## Manual (Non-Adaptive) Path

For perf testing, a manual path that skips the refinement loop:

```cpp
std::expected<ChebyshevMultiKRefSurface, PriceTableError>
build_chebyshev_segmented_manual(
    const SegmentedAdaptiveConfig& config,
    const IVGrid& domain,
    std::array<size_t, 4> cc_levels = {5, 3, 2, 1});
```

Reuses `build_chebyshev_segmented_pieces` and `make_tau_split_from_segments`.
Removes the current "segmented Chebyshev requires adaptive" error.

## Change Set

**Added:**
- `ChebyshevTauSegmented`, `ChebyshevMultiKRefInner`, `ChebyshevMultiKRefSurface` aliases in `chebyshev_surface.hpp`
- `template class InterpolatedIVSolver<ChebyshevMultiKRefSurface>` in `interpolated_iv_solver.cpp`
- `make_tau_split_from_segments()` in `adaptive_refinement.hpp`/`.cpp`
- `build_chebyshev_segmented_pieces()` in `chebyshev_adaptive.hpp`/`.cpp`
- `build_chebyshev_segmented_manual()` in `chebyshev_adaptive.hpp`/`.cpp`

**Modified:**
- `AnyIVSolver::Impl::SolverVariant`: add `InterpolatedIVSolver<ChebyshevMultiKRefSurface>`
- `build_chebyshev_segmented()` factory: construct solver from typed surface
- `build_adaptive_chebyshev_segmented()`: return typed surface

**Deleted (Phase 3 only):**
- `FDVegaLeaf` class
- `using ChebyshevSegmentedSurface = PriceTable<FDVegaLeaf>`
- Its solver instantiation and variant entry

**Unchanged:**
- `TauSegmentSplit`, `MultiKRefSplit`, `SplitSurface`
- `make_segmented_chebyshev_build_fn` (still used during refinement)
- B-spline segmented path
- All other solver instantiations

## Migration Strategy

**Phase 1 (one PR): Add new path alongside old**
- Add all new types, functions, solver instantiation
- Variant temporarily has 8 entries (old + new)
- Old `FDVegaLeaf` path remains functional

**Phase 2 (same PR): Equivalence test**
- Build both paths from same config (K_refs, dividends, CC levels, grids)
- Assert `|price_new - price_old| < ε` across query grid
- Compare vega: analytical vs FD, expect `|vega_new - vega_old| / |vega_new| < 1%`
- Compare IV solve results: `|iv_new - iv_old| < 1e-6`

**Phase 3 (follow-up PR): Delete old path**
- Remove `FDVegaLeaf`, `ChebyshevSegmentedSurface`, old variant entry
- Keep equivalence test as regression (or remove)
