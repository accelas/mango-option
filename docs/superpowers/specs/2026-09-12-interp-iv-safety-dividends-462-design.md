# Retune `interp_iv_safety` dividends path and measure sparse-K_ref accuracy (#462)

Date: 2026-09-12. Branch `fix/462-dividends-retune`. Issue #462 (follow-up 6 of 7
from PR #454).

## 1. Problem

`benchmarks/interp_iv_safety.cc --path=dividends` is dead on `main`. Both of its
builds fail, the benchmark swallows the error code, and its TV/K comparison
table prints `0.0` for the failed backends as if the error were nil.

Measured on `main` (f510c1ab) with a throwaway driver, PUT, spot 100, rate
5%, yield 2%, three $0.50 dividends at 0.25/0.50/0.75 of maturity, T = 1:

| Backend | Config | Result |
|---|---|---|
| B-spline | S/K 0.70–1.30, K_refs 80–120 step 5, 2 bps | `NoViableSurface` at every commit since #454 |
| B-spline | S/K 0.83–1.25 (strikes 80–120), K_refs 80–120 step 5, 5 bps | viable, max 1210 bps, avg 50 bps |
| B-spline | S/K 0.92–1.08, K_refs 90–110 step 2.5, 10 bps (documented) | viable, max 74 bps |
| B-spline | strikes 80–120, K_refs 80–120 step 2.5 | `NoViableSurface` |
| Chebyshev | strikes 80–120, K_refs 80–120 step 5, 5 bps | `NoViableSurface` (viable at 231.5 bps before 14c416ff; regression filed as #500) |
| Chebyshev | strikes 80–120, K_refs 80–120 step 2.5, 5 bps | viable, max 1282 bps, avg 37 bps, sizing loop 66–308 bps |
| Chebyshev | documented config | viable, max 74.4 bps |

Two facts drive the design:

1. The B-spline config is incoherent. Its moneyness range implies strikes
   76.9–142.9 but its K_refs span only 80–120. The API guide already names
   this pairing as unsupported. That is config rot, not a defect.
2. On the wide 80–120 band the K_ref blend dominates the error: the Chebyshev
   sizing loop reaches 66–308 bps on a single K_ref surface, but the assembled
   blend measures 1282 bps at 2.5% spacing and refuses at 5%. That is the
   sparse-K_ref finding parked in PR #449, now with numbers. Its fix is the
   spot-scaling work in #460, not a config change.

## 2. Goals

- G1. `--path=dividends` builds on both backends and reports real numbers for
  the configuration users are told to copy.
- G2. A failed build is visible: error code printed, `n/a` in tables, non-zero
  exit.
- G3. A measured answer to the sparse-K_ref question: blend error as a
  function of K_ref spacing, maturity and strike position, separated from the
  per-K_ref interpolation error, recorded in the API guide and on #460.

## 3. Non-goals

- No library code changes. No new validation in `create()` (that is #460).
- No fix or diagnosis of the Chebyshev wide-band regression (#500).
- No new tests. The retuned config is the documented config family, which the
  nightly `slow` suite already pins (`DocumentedAdaptiveDiscreteDividendConfig`,
  `DocumentedBSplineConfigReportsAccuracyAndSolves`).
- No change to the vanilla, q0 or Chebyshev-4D paths of the benchmark.

## 4. Design

### D1. Dividends path uses the documented config family

Both `build_div_solvers()` (B-spline, one solver per maturity) and
`run_chebyshev_dividends()` (Chebyshev, one surface at T = 1) use:

| Field | Value | Source |
|---|---|---|
| S/K moneyness | {0.92, 0.95, 1.0, 1.05, 1.08} | documented config |
| vol | {0.10, 0.15, 0.20, 0.30} | documented config |
| rate | {0.02, 0.03, 0.05, 0.07} | documented config |
| K_refs | {90, 92.5, 95, 97.5, 100, 102.5, 105, 107.5, 110} | documented config |
| `AdaptiveGridParams` | `{.target_iv_error = 1e-3}`, defaults otherwise | documented config |
| dividend yield, schedule | `kDivYield`, `make_div_schedule(T)` | benchmark constants |

The grid, K_refs and adaptive parameters are verbatim from
`documented_adaptive_dividend_config()` in `tests/iv_solver_factory_slow_test.cc`.
The schedule and yield stay the benchmark's own because the FDM reference
prices and the vanilla path use them; the documented schedule (2 × $1.50)
is a different, harsher stress that the pin covers.

The B-spline per-maturity loop keeps all eight maturities; each build takes
about a second at the documented config. The Chebyshev build stays a single
T = 1 surface; it takes minutes, as it did before.

### D2. Dividends path has its own strike set

The vanilla path keeps `kStrikes` (80–120 step 5). The dividends path
measures `kDivStrikes = {93, 95, 97.5, 100, 102.5, 105, 107}`: every strike
lies inside the band S/K ∈ [0.92, 1.08] (strikes 92.6–108.7), and the set
mixes K_ref anchors (95, 97.5, 100, 102.5, 105) with off-anchor strikes near
the band edges (93, 107).

To carry two strike sets the benchmark's per-path containers become generic
over the strike count: `PriceGrid`, `ErrorTable`, `TVKMask`,
`generate_prices`, `compute_errors_*`, `print_heatmap`, `compute_tvk_mask`
and `print_tvk_comparison` take the strike array (or its size) as a template
parameter or a `std::span`, and the maturity-label and strike-header printing
read from the array they are given. This is a mechanical change; no output
format changes for the vanilla and q0 paths.

### D3. Honest failure reporting

- A failed adaptive build prints the `PriceTableErrorCode` name and the
  error's `axis_index` and `count`, via a small `code_name()` switch in the
  benchmark (the library has no string mapping for this enum; the switch is
  exhaustive so a new enumerator is a compile warning).
- `print_tvk_comparison` takes each algorithm's error table as optional
  (`const ErrorTable*` may be null). A null table prints `n/a` instead of
  `0.0 (0)`.
- `main` tracks whether any requested build failed and returns 1 in that case,
  after printing everything it could.

### D4. `--path=kref`: the K_ref spacing sweep

A new path that isolates the blend error. It uses the manual, non-adaptive
segmented builder so that grids are fixed and the only variable is K_ref
spacing.

**Surfaces.** For each spacing Δ ∈ {10, 5, 2.5, 1.25} dollars, K_refs are
`80, 80+Δ, …, 120` (5, 9, 17, 33 K_refs). Each K_ref surface is
`SegmentedPriceTableBuilder::build(Config{...})` with a fixed log-moneyness
grid of 41 uniform points on [−0.30, 0.30] (covers ln(100/120) − dividend
shift through ln(100/80)), vol grid {0.10, 0.15, 0.20, 0.30, 0.50}, rate grid
{0.03, 0.05, 0.07}, `tau_points_per_segment = 5`, the benchmark's yield and
`make_div_schedule(T)`. The surfaces are assembled with
`build_multi_kref_surface` and wrapped as `BSplineMultiKRefSurface` with
bounds m ∈ [−0.30, 0.30], tau ∈ [0, T], and the vol and rate grid ranges.
The IV solver is `InterpolatedIVSolver<BSplineMultiKRefSurface>::create`.

**Maturities.** T ∈ {30/365, 180/365, 1.0}, one surface set per maturity,
schedule scaled by `make_div_schedule(T)` as elsewhere in the benchmark.

**Query strikes.** Inside the window [85, 115] (to avoid the outer bands,
where the blend clamps to a single K_ref):

- anchors: the spacing's own K_refs in the window;
- mid-anchors: midpoints between adjacent K_refs in the window.

At Δ = 1.25 the anchors of every coarser spacing are also anchors, so the
anchor column at fine spacing is the per-surface floor for the same strikes.

**Error.** For each query (T, K, σ ∈ {0.15, 0.30}, r = 0.05): the FDM
reference price is `solve_american_option` with the same contract; the FDM IV
is `solve_fdm_iv_div` (the benchmark's Brent inversion); the interpolated IV
is the solver's `solve` on that price. Error is |Δ IV| in bps, the same
metric as the rest of the benchmark. Inversion failures count as failed
points and print as `---`.

**Output.** One table per σ:

```
K_ref spacing sweep — σ=15%  (IV error, bps; window K∈[85,115])
             anchors              mid-anchors
  Δ     T   max    rms    n     max    rms    n
  10   30d  ...
```

followed by a one-line summary per spacing: the ratio of mid-anchor RMS to
anchor RMS, which is the blend's contribution.

**Cost.** 4 spacings × 3 maturities × (5+9+17+33) K_ref surfaces × 15 (σ, r)
PDE solves ≈ 2900 short fixed-expiry solves, plus about 300 FDM references
and Brent inversions. Under a minute.

### D5. Documentation

- `docs/API_GUIDE.md`, section "Reference-strike configuration" (the K_ref
  paragraphs near line 761): add a short table of the sweep's mid-anchor
  errors by spacing and maturity, the date and command that produced it, and
  a one-sentence spacing rule derived from the numbers (the largest Δ whose
  mid-anchor RMS stays within 2× the anchor RMS at every measured maturity;
  the exact figure is filled in from the run during execution).
- Comment on #460 with the same table and rule as its measured baseline.
- `CLAUDE.md` Pattern 3/4 text is unchanged; it already states the
  span-and-resolve rule.

### D6. Cross-references

- In `build_div_solvers()` and `run_chebyshev_dividends()`: a comment naming
  `documented_adaptive_dividend_config()` in `tests/iv_solver_factory_slow_test.cc`
  as the source of the grid/K_ref/adaptive values.
- Above `documented_adaptive_dividend_config()`: a comment naming
  `benchmarks/interp_iv_safety.cc` as a copy that must move with it.

## 5. Acceptance criteria

- AC1. `bazel build //benchmarks/...` succeeds with no new warnings.
- AC2. `interp_iv_safety --path=dividends` builds all eight B-spline
  per-maturity solvers and the Chebyshev surface on this branch,
  prints their diagnostics, and exits 0. Its heatmaps use the seven
  `kDivStrikes` columns.
- AC3. With a deliberately incoherent config (verified once by hand, not
  committed) the benchmark prints the error code name and `n/a` rows and
  exits 1.
- AC4. `interp_iv_safety --path=kref` completes and prints the four-spacing
  table for each σ, with no failed points at Δ ≤ 5 in the window.
- AC5. `--path=all`, `--path=bspline`, `--path=chebyshev`, `--path=q0` produce
  the same tables as before, byte-for-byte apart from timing lines.
- AC6. `bazel test //...` stays at 156/156.
- AC7. The API guide table and the #460 comment contain the numbers from the
  committed sweep run.

## 6. Risks and assumptions

- The B-spline per-maturity solvers at short maturities (7d, 14d) with the
  scaled schedule have not been run at the documented config yet; the doc
  config was measured at T = 1 only. If a short maturity refuses, AC2 fails
  and the cause is reported rather than tuned away; the likely outcome is a
  `[skip]` line with the code name, which D3 makes visible.
- The kref sweep's manual B-spline surfaces inherit any per-K_ref fit
  instability (#458). Reporting anchor error next to mid-anchor error keeps
  the blend contribution readable even if the floor is noisy.
- The Chebyshev dividends build takes minutes; the benchmark is manual-tagged
  and not in CI, so this is acceptable and unchanged from today.

## 7. Decisions (brainstorm record)

Q1. **Scope of half (b), the sparse-K_ref finding.** Options: (i) measure and
document only, enforcement stays in #460; (ii) measure, document, and add a
`create()`-time spacing check; (iii) defer (b) to #460 entirely.
**Chosen: (i).** Enforcement would compete with #460's span validation, and
#460 needs the measured baseline first.

Q2. **Strike coverage of the retuned dividends path.** Options: (i) documented
±8% band with its own in-band strike set; (ii) keep 80–120 with 2.5% K_refs
(17 K_refs, ~1280 bps, ~13 min Chebyshev); (iii) keep the 80–120 strike table
and print out-of-band strikes as `n/a`. **Chosen: (i).** It measures the
config users copy and runs fast; the wide-band behaviour is what the sweep
reports.

Q3. **Instrument for the measurement.** Options: (i) new `--path=kref` in
`interp_iv_safety`, manual segmented B-spline; (ii) a standalone benchmark
binary; (iii) sweep on both backends. **Chosen: (i).** The blend lives in
`MultiKRefSplit` and is backend-independent; one fast backend suffices, and
the sweep belongs next to the diagnostic it explains.

Q4. **Regression protection for the retuned config.** Options: (i) rely on
the existing nightly pins plus cross-reference comments; (ii) a new slow
test on the benchmark's exact config; (iii) a shared config helper linked by
both. **Chosen: (i).** A second near-identical fixture adds drift of its own;
a shared library target for one fixture is not worth a BUILD target.

Q5. **The Chebyshev wide-band regression found during triage.** Options:
(i) file a new issue and keep #462 as designed; (ii) fold the fix into #462;
(iii) pause #462 until it is fixed. **Chosen: (i)**, filed as #500. The
documented band does not touch the regressed region.

Approved by the user on 2026-09-12 for spec writing and design review.
