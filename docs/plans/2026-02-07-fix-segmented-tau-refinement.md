# Fix segmented tau refinement in adaptive grid builder

## Problem

The adaptive builder's tau refinement is disconnected from the segmented
builder. It grows a global `maturity_grid`, but only passes
`maturity_grid.size()` as a scalar `tau_points_per_segment`. The actual
grid positions are discarded. Each segment creates its own local tau grid
independently, and `SegmentLookup` does hard switching (no blending).

This causes tau to dominate every refinement iteration (700-4900 bps
errors at segment boundaries), consuming all refinement budget and
preventing moneyness/vol/rate improvement. The error oscillates rather
than decreasing because changing the count changes each segment's
B-spline fit unpredictably.

## Fix: remove tau from adaptive refinement for segmented surfaces

1. Add `tau_target_dt` (default 0.05) to `AdaptiveGridParams`
2. Add `skip_tau_refinement` flag to `RefinementContext`
3. In `probe_and_build()`: set `skip_tau_refinement = true`, pass
   `tau_target_dt` through `make_seg_config()`
4. In `run_refinement()`: skip dimension 1 (tau) when selecting worst
   dimension if `skip_tau_refinement` is set
5. Tau grid for segmented surfaces is set deterministically by the
   existing `tau_target_dt` logic in `make_segment_tau_grid()`

## Files to modify

- `src/option/table/adaptive_grid_builder.cpp` — RefinementContext,
  run_refinement worst-dim selection, probe_and_build, make_seg_config
- `src/option/table/adaptive_grid_types.hpp` — add tau_target_dt to
  AdaptiveGridParams

## Verification

- `bazel test //...` — all tests pass
- `bazel build //benchmarks/...` — benchmarks compile
- Run `interp_iv_safety` — dividend IV errors should improve (refinement
  budget now goes to moneyness/vol/rate instead of tau)
