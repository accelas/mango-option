# Adaptive Grid for Segmented (Discrete Dividend) Path

## Problem

`AdaptiveGrid` in `IVSolverFactoryConfig` only works with `StandardIVPath`.
The segmented path (`SegmentedIVPath` for discrete dividends) requires
`ManualGrid` — users must hand-pick grid density.

## Approach: Probe + Max with Segmented PDE

Run adaptive refinement on 2-3 representative K_ref values using actual
segmented PDE surfaces, take per-axis maximum grid size across probes,
then build all segments with that uniform grid.

### Why segmented PDE for probes

Discrete dividends change the terminal condition per segment via a jump
in S, creating sharper curvature after dividend dates. The interpolation
axes include tau_segment (not global maturity), so worst-case curvature
is segment-local. Only the segmented PDE exposes it.

### Why uniform grid across segments

All segments must share the same grid so that multi-K_ref interpolation
doesn't hit resolution mismatches at boundaries.

## Design

### 1. Refactored refinement loop

Extract the ~300-line iterative loop from `AdaptiveGridBuilder::build()`
into a private helper that takes callbacks for surface construction and
validation:

```cpp
// Private types, in .cpp only
struct SurfaceHandle {
    std::function<double(double spot, double strike, double tau,
                         double sigma, double rate)> price;
};

using BuildFn = std::function<SurfaceHandle(
    const std::vector<double>& moneyness,
    const std::vector<double>& vol,
    const std::vector<double>& rate,
    int tau_points)>;  // tau_points used by segmented path

using ValidateFn = std::function<std::expected<double, SolverError>(
    double spot, double strike, double tau,
    double sigma, double rate)>;

struct GridSizes {
    size_t moneyness;
    size_t vol;
    size_t rate;
    int tau_points;
    double achieved_max_error;
};
```

The shared loop `run_refinement(BuildFn, ValidateFn, ctx) -> GridSizes`:
1. Create seed grids (moneyness, vol, rate) from domain bounds
2. For each iteration:
   a. Call `BuildFn` with current grids + tau_points
   b. Latin Hypercube sample validation points
   c. Call `ValidateFn` for fresh reference prices at each sample
   d. Compare interpolated vs reference, compute IV error
   e. Diagnose worst dimension, refine that axis
3. Return final grid sizes and achieved error

**Standard path wrapper:** `BuildFn` calls `PriceTableBuilder<4>`,
`ValidateFn` calls `solve_american_option()` with continuous dividends.
SliceCache stays inside the `BuildFn` closure.

**Segmented path wrapper:** `BuildFn` calls
`SegmentedPriceTableBuilder::build()`, `ValidateFn` calls
`solve_american_option()` with `discrete_dividends` populated.

### 2. Validation with discrete dividends

`solve_american_option(PricingParams)` already handles discrete dividends:
it expands the spatial domain, adds mandatory tau points at dividend dates,
and produces correct reference prices. No new solver needed.

The segmented `ValidateFn` populates `PricingParams.discrete_dividends`
from the config. The existing `compute_error_metric()` works unchanged.

### 3. Tau axis as `tau_points_per_segment`

`SegmentedPriceTableBuilder` generates its tau grid internally from the
scalar `tau_points_per_segment` (default 5, minimum 4 for B-spline).
The adaptive loop cannot pass an explicit tau grid.

Solution: the refinement loop treats the tau axis as the
`tau_points_per_segment` scalar. When the worst dimension is tau, the
loop increments this scalar (e.g., 5 → 7 → 9). The `BuildFn` receives
`tau_points` and passes it through to `SegmentedPriceTableBuilder::Config`.

For the standard path, `tau_points` maps to the maturity grid size as
before (adding midpoints in problematic bins). The callback abstraction
hides this difference.

### 4. Moneyness pre-expansion for uniform grids

`SegmentedPriceTableBuilder` expands moneyness downward by
`total_div / K_ref`. Since this is K_ref-dependent, different K_refs get
different expanded grids, breaking the uniform-grid guarantee.

Solution: **pre-expand the moneyness domain** before probing, using the
worst-case (smallest) K_ref:

```
total_div = sum of all discrete dividend amounts
K_ref_min = smallest K_ref in the full list
expansion = total_div / K_ref_min
expanded_m_min = max(domain_m_min - expansion, 0.01)
```

Pass the pre-expanded moneyness domain to `run_refinement()`. The
adaptive loop operates on this expanded domain for all probes. The
final uniform grid already covers the worst-case expansion.

Then pass a flag or pre-expanded grid to `SegmentedPriceTableBuilder`
so it **skips its own internal expansion** (the grid already covers the
needed range). This can be a boolean `skip_moneyness_expansion` on its
Config, or the builder can detect that the grid already extends below
the expansion threshold.

### 5. Probe K_ref selection

From the full K_ref list (auto-generated or explicit):
- **ATM:** closest to spot
- **Lowest K_ref:** deepest ITM, steepest curvature, largest moneyness
  expansion (`total_div / K_ref` maximized)
- **Highest K_ref:** deepest OTM

If the K_ref list has fewer than 3 entries, probe all of them.

### 6. `build_segmented()` public method

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

Flow:
1. Determine full K_ref list from `kref_config`
2. Select probe K_refs (ATM, lowest, highest)
3. Pre-expand moneyness domain using worst-case K_ref
4. For each probe, run `run_refinement()` with segmented callbacks
5. Take per-axis max grid sizes across probes
6. Generate uniform grids (linspace over expanded domain) with max sizes
7. Build all K_ref segments via `SegmentedMultiKRefBuilder::build()`
   with the uniform grid and `skip_moneyness_expansion = true`
8. Return `SegmentedMultiKRefSurface`

### 7. Factory integration

In `build_segmented()` (iv_solver_factory.cpp), add `AdaptiveGrid` branch:

```cpp
if (const auto* adaptive = std::get_if<AdaptiveGrid>(&config.grid)) {
    AdaptiveGridBuilder builder(adaptive->params);
    auto surface = builder.build_segmented(
        SegmentedAdaptiveConfig{...}, *adaptive);
    if (!surface)
        return std::unexpected(ValidationError{
            ValidationError::InvalidGridSize, surface.error().message});
    // wrap in InterpolatedIVSolver<SegmentedMultiKRefSurface>
}
```

`ManualGrid` path unchanged. Error mapping: `PriceTableError.message`
converts to `ValidationError` with `InvalidGridSize` code.

### 8. SliceCache

Only applies to the standard path. The segmented path doesn't use it —
each `SegmentedPriceTableBuilder::build()` is self-contained. The cache
stays inside the standard `BuildFn` closure.

## Files

| File | Change |
|------|--------|
| `src/option/table/adaptive_grid_builder.hpp` | Add `build_segmented()`, `SegmentedAdaptiveConfig` |
| `src/option/table/adaptive_grid_builder.cpp` | Extract `run_refinement()`, implement `build_segmented()` |
| `src/option/table/segmented_price_table_builder.hpp` | Add `skip_moneyness_expansion` flag to Config |
| `src/option/table/segmented_price_table_builder.cpp` | Honor `skip_moneyness_expansion` flag |
| `src/option/iv_solver_factory.cpp` | Add `AdaptiveGrid` handling in segmented path |
| `tests/adaptive_grid_builder_test.cc` | Test `build_segmented()` |
| `tests/iv_solver_factory_test.cc` | Integration test: `AdaptiveGrid` + `SegmentedIVPath` |

## Testing

- **Probe selection:** verify ATM + lowest + highest K_refs chosen;
  verify all probed when list < 3
- **Grid uniformity:** all segments get same grid sizes after adaptive
- **Pre-expansion:** moneyness domain covers worst-case `total_div / K_ref`
- **IV accuracy:** segmented adaptive meets target across all K_ref segments
- **Tau refinement:** verify `tau_points_per_segment` increases when tau
  is the worst dimension
- **Large dividends:** stress test with `total_div / K_ref > 0.3`
- **No dividends:** single segment degenerates to standard path behavior
- **Regression:** existing standard adaptive tests pass unchanged
- **Factory integration:** `AdaptiveGrid` + `SegmentedIVPath` produces
  valid solver with correct IV
