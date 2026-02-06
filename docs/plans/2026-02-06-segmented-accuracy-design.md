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

### 0. Prerequisites (already done)

**0a. API Return Type.**
`build_segmented()` and `build_segmented_strike()` now return
`SegmentedAdaptiveResult` / `StrikeAdaptiveResult` instead of raw
surfaces. These structs carry `.surface`, `.grid` (the chosen
`ManualGrid`), and `.tau_points_per_segment`. This is committed on
the `cleanup/remove-segmented-legacy` branch.

**0b. Templatize Latin Hypercube by dimension.**
`latin_hypercube_4d` → `latin_hypercube<N>` template (same pattern as
`PriceTableBuilder<N>`, `BSplineND<N>`). `latin_hypercube_4d` remains
as a thin backward-compatible wrapper. This lets the validation pass
use `latin_hypercube<3>` natively instead of discarding a dimension
from a 4D sample. Also on `cleanup/remove-segmented-legacy`.

### 1. Adaptive Grid Builder (`adaptive_grid_builder.cpp`)

**1a. Probe all strikes for small N.**
`select_probes` logic (inline in `build_segmented` and
`build_segmented_strike`): if N <= 15, return all items. Above 15,
use percentile sampling {min, p25, ATM, p75, max}. Currently always
picks <= 3.

**1b. Fix probe validation strike in `run_refinement`.**
Add `std::optional<double> fixed_strike` to `RefinementContext`. When
set, validation queries use that strike instead of `spot/m` from LHS
samples. `probe_and_build` sets this for each probe. Currently, the
arbitrary `spot/m` strike is meaningless for per-strike segmented
surfaces built around a specific K_ref.

Note: when `fixed_strike` is set, the refinement loop samples 3
dimensions (tau, sigma, rate) via `latin_hypercube<3>`. The LHS
template is parameterized by dimension (matching the B-spline
pattern), so `latin_hypercube<3>` returns
`std::vector<std::array<double, 3>>` natively.

**1c. FD vega in final validation only.**
Keep BS vega during `run_refinement` inner loop (cheap, sufficient for
refinement direction). Use FD American vega (central difference at
sigma ± 0.5%, 2 additional FD solves per sample) only in the
`build_segmented_strike` validation pass (Section 3). This limits
cost increase to the final check.

**1d. Preserve anchor moneyness knots.**
After `probe_and_build` computes `final_m = linspace(min, max, N)`,
merge in forced anchors: m=1.0 (ATM) and original user-provided
moneyness knots from `domain.moneyness`. Sorted insertion with
dedup (tolerance 1e-6). Ensures interpolation grid includes
financially meaningful locations.

**1e. Tolerance consistency.**
The refinement inner loop (`run_refinement`) keeps its existing
`target_iv_error` (default 2e-5 = 2 bps) with BS vega. This is a
refinement *direction* signal, not the acceptance criterion. The
actual acceptance decision happens in Section 3's validation pass,
which uses FD American vega and the relaxed thresholds (20 bps max,
5 bps p95). No change to `run_refinement`'s convergence check.

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

Add a validation pass after the surface is built:
- Sample (tau, sigma, rate) via `latin_hypercube<3>` with N=64
- For each sample, evaluate at every strike in the strike list
- Compute IV error: `price_error / american_vega`, where
  `american_vega` is computed via central difference
  `(P(σ+h) - P(σ-h)) / 2h` with `h = 0.005` (2 extra FD solves
  per validation point)
- Floor `american_vega` at `params.vega_floor` to avoid division
  blowup in deep OTM regions
- Track per-strike max error, overall max error, and p95 error

**3b. Acceptance and retry logic.**
Acceptance thresholds (separate from refinement's `target_iv_error`):
- `max_acceptable_iv_error = 20e-4` (20 bps)
- `p95_acceptable_iv_error = 5e-4` (5 bps)

If either threshold is breached:
- Bump moneyness grid +2 points
- Bump vol/rate grids +1 point each
- Reduce `tau_target_dt` by 20%
- Rebuild and re-validate
- Max 2 retries; return best result with `target_met = false` if
  still failing

Note: ATM and OTM regions are not tracked separately. The p95
threshold naturally enforces that most of the domain (including ATM)
is below 5 bps. Deep OTM points with low vega are allowed up to the
20 bps max. This avoids the complexity of separate ATM/OTM targets.

**3c. Reporting.**
Add to `StrikeAdaptiveResult`:
- `double max_iv_error` — worst IV error across all validation points
- `double p95_iv_error` — 95th percentile IV error
- `bool target_met` — whether both thresholds were satisfied

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
| `src/option/table/adaptive_grid_types.hpp` | `StrikeAdaptiveResult`: add `max_iv_error`, `p95_iv_error`, `target_met` fields |
| `src/option/table/adaptive_grid_builder.cpp` | `select_probes` (N<=15 → all), `fixed_strike` in probes, anchor knot merging, `build_segmented_strike` validation pass, FD American vega in validation, tau_target_dt wiring |
| `src/option/table/segmented_price_table_builder.hpp` | `Config::tau_target_dt`, `Config::tau_points_min`, `Config::tau_points_max` |
| `src/option/table/segmented_price_table_builder.cpp` | `make_segment_tau_grid` width-proportional scaling |
| `tests/adaptive_grid_builder_test.cc` | Unit + regression tests |

Note: `adaptive_grid_builder.hpp` and `iv_solver_factory.cpp` changes
(API return type) are a prerequisite already done on
`cleanup/remove-segmented-legacy`.
