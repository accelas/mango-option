# Adaptive Grid for Segmented (Discrete Dividend) Path

## Problem

`AdaptiveGrid` in `IVSolverFactoryConfig` only works with `StandardIVPath`.
The segmented path (`SegmentedIVPath` for discrete dividends) requires
`ManualGrid` — users must hand-pick grid density.

## Approach: Probe + Max with Segmented PDE

Run adaptive refinement on 2-3 representative K_ref values using the actual
segmented PDE (not a continuous-dividend proxy), take per-axis maximum grid
size across probes, then build all segments with that uniform grid.

### Why segmented PDE for probes (not continuous-dividend proxy)

Discrete dividends change the terminal condition per segment via a jump in S,
creating sharper curvature after dividend dates. The interpolation axes include
tau_segment (not global maturity), so worst-case curvature is segment-local.
Only the segmented PDE exposes it.

### Why uniform grid across segments

All segments must share the same grid so that multi-K_ref interpolation
across segments doesn't hit resolution mismatches at boundaries.

## Design

### 1. Extract refinement loop (callback-based)

Refactor `AdaptiveGridBuilder::build()` into a private helper:

```cpp
// Private, in .cpp
struct SurfaceHandle {
    std::function<double(double spot, double strike, double tau,
                         double sigma, double rate)> price;
};

using BuildFn = std::function<SurfaceHandle(
    const std::vector<double>& moneyness,
    const std::vector<double>& vol,
    const std::vector<double>& rate)>;

struct RefinementContext {
    double spot;
    double dividend_yield;
    OptionType option_type;
    std::vector<double> maturity_grid;  // for standard path
    double maturity;                     // for segmented path
    std::vector<double> moneyness_domain;
    std::vector<double> vol_domain;
    std::vector<double> rate_domain;
};
```

`run_refinement(BuildFn, RefinementContext)` contains the shared loop:
seed grids, iterative build+validate+refine, return final grid sizes
and achieved error.

Existing `build()` becomes a thin wrapper providing a `BuildFn` that
uses `PriceTableBuilder<4>` + `SliceCache`.

### 2. New public method: `build_segmented()`

```cpp
struct SegmentedAdaptiveConfig {
    double spot;
    OptionType option_type;
    double dividend_yield;
    std::vector<Dividend> discrete_dividends;
    double maturity;
    MultiKRefConfig kref_config;
};

[[nodiscard]] std::expected<SegmentedMultiKRefSurface, PriceTableError>
build_segmented(const SegmentedAdaptiveConfig& config,
                const AdaptiveGrid& grid);
```

**Probe K_ref selection:** From the full K_ref list (auto or explicit):
- ATM: closest to spot
- Lowest K_ref: deepest ITM for puts, steepest curvature
- Highest K_ref: deepest OTM

Each probe runs `run_refinement()` with a `BuildFn` that calls
`SegmentedPriceTableBuilder::build()` for that K_ref.

**Grid determination:** Per-axis max grid sizes across probes.
Generate linspace grids across domain bounds with those sizes.

**Final build:** Pass uniform grid to `SegmentedMultiKRefBuilder::build()`
for all K_refs. Return the `SegmentedMultiKRefSurface`.

### 3. Factory integration

In `build_segmented()` (iv_solver_factory.cpp), add `AdaptiveGrid` branch:

```cpp
if (const auto* adaptive = std::get_if<AdaptiveGrid>(&config.grid)) {
    AdaptiveGridBuilder builder(adaptive->params);
    auto surface = builder.build_segmented(seg_config, *adaptive);
    // wrap in InterpolatedIVSolver<SegmentedMultiKRefSurface>
}
```

`ManualGrid` path unchanged.

### 4. SliceCache

Only applies to the standard path (caches by sigma,rate pairs for
`PriceTableBuilder`). The segmented path doesn't use it — each
`SegmentedPriceTableBuilder::build()` is self-contained. The cache
stays in the standard `BuildFn` closure.

## Files

| File | Change |
|------|--------|
| `src/option/table/adaptive_grid_builder.hpp` | Add `build_segmented()`, `SegmentedAdaptiveConfig`, private `run_refinement()` |
| `src/option/table/adaptive_grid_builder.cpp` | Refactor `build()` to use `run_refinement()`, implement `build_segmented()` |
| `src/option/iv_solver_factory.cpp` | Add `AdaptiveGrid` handling in segmented path |
| `tests/adaptive_grid_builder_test.cc` | Test `build_segmented()` probe selection + grid uniformity |
| `tests/iv_solver_factory_test.cc` | Integration test: `AdaptiveGrid` + `SegmentedIVPath` |

## Testing

- Probe selection: verify ATM + lowest + highest K_refs chosen
- Grid uniformity: all segments get same grid sizes
- IV accuracy: segmented adaptive meets target across all K_ref segments
- Regression: existing standard adaptive tests pass unchanged
- Factory integration: `AdaptiveGrid` + `SegmentedIVPath` produces valid solver
