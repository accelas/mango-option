# Segmented IV Accuracy Improvements

## Problem

The `interp_iv_safety` benchmark reveals two accuracy gaps in the
discrete dividend interpolated IV path:

1. **Adaptive coverage gap.** `select_probes()` picks only 2-3
   K_refs/strikes (front, back, ATM-nearest). Deep OTM/ITM error
   regions are missed. The segmented path also uses BS vega instead of
   FD American vega for its error metric, under-ranking OTM errors.

2. **Long-tenor segment chaining degradation.** Each segment chains
   its initial condition from the prior segment's B-spline surface.
   Error compounds across segments. `tau_points_per_segment` is
   constant regardless of segment width, so wide segments are
   under-resolved.

Observed: sigma=0.30, T=2y, K=80 hits 228 bps IV error. Target 2 bps
is not met anywhere in the deep OTM long-tenor region.

## Target

- < 20 bps everywhere in the moneyness x maturity domain
- < 5 bps near ATM (K=95-105)
- Reasonable build time (< 2x current)

Strict 2 bps everywhere is deferred. Deep OTM + long tenor + high vol
is inherently harder; 20 bps is acceptable for production use.

## Design

### 1. Adaptive Grid Builder (`adaptive_grid_builder.cpp`)

**1a. Probe all strikes for small N.**
`select_probes`: if N <= 15, return all items. Above 15, use
percentile sampling {min, p25, ATM, p75, max}. Currently always
picks <= 3.

**1b. Fix probe validation strike in `run_refinement`.**
Add `std::optional<double> fixed_strike` to `RefinementContext`. When
set, validation queries use that strike instead of `spot/m` from LHS
samples. `probe_and_build` sets this for each probe. Currently, the
arbitrary `spot/m` strike is meaningless for per-strike segmented
surfaces built around a specific K_ref.

**1c. FD vega in final validation only.**
Keep BS vega during `run_refinement` inner loop (cheap, sufficient for
refinement direction). Use FD American vega (central difference at
sigma +/- eps, 3 FD solves per sample) only in the new validation
pass. Limits cost increase to the final check.

**1d. Preserve anchor moneyness knots.**
After `probe_and_build` computes `final_m = linspace(min, max, N)`,
merge in forced anchors: m=1.0 (ATM) and original user-provided
moneyness knots from `domain.moneyness`. Sorted insertion with
dedup (tolerance 1e-6). Ensures interpolation grid includes
financially meaningful locations.

### 2. Segmented Price Table Builder (`segmented_price_table_builder`)

**2a. Width-proportional tau points.**
New config fields:

    double tau_target_dt = 0.0;  // 0 = legacy constant mode
    int tau_points_min = 4;      // B-spline minimum
    int tau_points_max = 30;     // cap for very wide segments

When `tau_target_dt > 0`, `make_segment_tau_grid` computes:

    n = clamp(ceil(seg_width / tau_target_dt) + 1,
              tau_points_min, tau_points_max)

When `tau_target_dt == 0`, falls back to existing constant
`tau_points_per_segment`. Backward compatible.

The adaptive builder sets `tau_target_dt` from the shortest segment
width divided by the refined tau_points, so wider segments
automatically get proportionally more points.

### 3. Validation Pass for `build_segmented_strike`

**3a. Final validation + retry.**
`build_segmented_strike` currently has no post-build validation
(unlike `build_segmented` which does LHS validation + retry).

Add a validation pass:
- Sample (tau, sigma, rate) via `latin_hypercube_3d`
- Evaluate at each strike in the strike list
- Compute error using FD American vega (the only place FD vega is
  used, per cost control decision)
- Track per-strike max error and overall 95th-percentile error

**3b. Retry logic.**
If max_error > 20 bps or p95_error > 5 bps:
- Bump moneyness grid +2 points
- Bump vol/rate grids +1 point each
- Reduce tau_target_dt by 20%
- Rebuild and re-validate
- Max 2 retries; return best result with `target_met = false` if
  still failing

**3c. Reporting.**
Add `max_iv_error` and `p95_iv_error` to `StrikeAdaptiveResult` so
callers can inspect how close to target the build landed.

## Deferred (Phase 2)

- Sqrt-spaced tau grid (cluster near segment start) — diminishing
  returns once width-proportional scaling is in place
- Boundary tensor snapshot IC — replace full spline eval at segment
  boundary with extracted tensor slice. Good architecture but lower
  ROI than probing/validation fixes
- Per-strike adaptive grids — each strike gets its own grid density.
  Eliminates coverage gap entirely but higher complexity
- Periodic FD re-anchoring — hard-caps cumulative chaining drift for
  very long maturities

## Testing

**Unit tests:**
- `select_probes`: N <= 15 returns all, N > 15 returns percentiles
- `make_segment_tau_grid` with tau_target_dt: verify scaling, clamping
- Anchor knot merging: m=1.0 and user knots survive into final grid

**Integration regression:**
New test in `adaptive_grid_builder_test.cc`: build per-strike
segmented surface with sigma=0.30, T=2.0, quarterly $0.50, K=80.
Assert IV error < 20 bps. This is the exact case hitting 228 bps
today.

**Benchmark verification:**
Re-run `interp_iv_safety`. The sigma=0.30 dividend heatmap should
show no cell > 20 bps and ATM cells < 5 bps.

## Files Changed

| File | Changes |
|------|---------|
| `src/option/table/adaptive_grid_builder.hpp` | `RefinementContext::fixed_strike`, `StrikeAdaptiveResult` error fields |
| `src/option/table/adaptive_grid_builder.cpp` | `select_probes`, `probe_and_build` anchors, `build_segmented_strike` validation |
| `src/option/table/segmented_price_table_builder.hpp` | `Config::tau_target_dt/min/max` |
| `src/option/table/segmented_price_table_builder.cpp` | `make_segment_tau_grid` width scaling |
| `tests/adaptive_grid_builder_test.cc` | Unit + regression tests |
