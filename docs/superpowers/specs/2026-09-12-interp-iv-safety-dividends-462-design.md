# Retune `interp_iv_safety` dividends path and measure sparse-K_ref accuracy (#462)

Date: 2026-09-12. Branch `fix/462-dividends-retune`. Issue #462 (follow-up 6 of 7
from PR #454). Revision 2 (after Codex design review round 1).

## 1. Problem

`benchmarks/interp_iv_safety.cc --path=dividends` is dead on `main`. Both of its
builds fail, the benchmark swallows the error code, and its TV/K comparison
table prints `0.0` for the failed backends as if the error were nil.

Measured on `main` (f510c1ab) with a throwaway driver. All figures below are
the builder's own holdout diagnostic (filtered price error over FD vega),
not IV-inversion error; the two are compared in D4. PUT, spot 100, rate 5%,
yield 2%, three $0.50 dividends at 0.25/0.50/0.75 of maturity, T = 1:

| Backend | Config | Result |
|---|---|---|
| B-spline | S/K 0.70–1.30, K_refs 80–120 step 5, 2 bps | `NoViableSurface` at every commit since #454 |
| B-spline | S/K 0.83–1.25 (strikes 80–120), K_refs 80–120 step 5, 5 bps | viable, max 1210 bps, avg 50 bps |
| B-spline | S/K 0.92–1.08, K_refs 90–110 step 2.5, 10 bps (documented grid) | viable, max 74 bps |
| B-spline | strikes 80–120, K_refs 80–120 step 2.5 | `NoViableSurface` |
| Chebyshev | strikes 80–120, K_refs 80–120 step 5, 5 bps | `NoViableSurface` (viable at 231.5 bps before 14c416ff; regression filed as #500) |
| Chebyshev | strikes 80–120, K_refs 80–120 step 2.5, 5 bps | viable, max 1282 bps, avg 37 bps; sizing loop 66–308 bps |
| Chebyshev | documented grid | viable, max 74.4 bps |

Three facts drive the design:

1. The B-spline config is incoherent. Its moneyness range implies strikes
   76.9–142.9 but its K_refs span only 80–120. The API guide already names
   this pairing as unsupported. That is config rot, not a defect.
2. On the wide 80–120 band the assembled K_ref blend measures far worse than
   the single-K_ref surface the sizing loop refined (1282 bps against
   66–308 bps at 2.5% spacing). That gap is the motivation for the controlled
   sweep in D4; the sweep is what establishes how much of it the blend policy
   owns. The library-side fix, if the blend owns it, is the spot-scaling work
   in #460, not a config change.
3. The benchmark's dividend schedule scales with maturity (three dividends at
   0.25 T, 0.5 T, 0.75 T), so a 7-day option carries three dividends 1.75
   days apart. With the documented grid the B-spline segmented builder refuses
   that at every maturity ≤ 180d. Viability depends erratically on dividend
   placement (matrix in #501); the only tested schedule that builds at all
   eight maturities is a fixed quarterly calendar starting at 0.25 y.

## 2. Goals

- G1. `--path=dividends` builds on both backends at every maturity it reports
  and prints real numbers for the configuration users are told to copy.
- G2. A failed build is visible: error code printed, `n/a` in tables, non-zero
  exit.
- G3. A measured answer to the sparse-K_ref question: the blend policy's
  error as a function of K_ref spacing, maturity and strike position, separated
  from the per-K_ref surface error by a same-query control, recorded in the
  API guide and on #460.

## 3. Non-goals

- No library code changes. No new validation in `create()` (that is #460).
- No fix or diagnosis of the Chebyshev wide-band regression (#500) or the
  B-spline placement fragility (#501).
- No change to the vanilla, q0 or Chebyshev-4D paths of the benchmark, nor to
  `make_div_schedule` in `iv_benchmark_common.hpp` (`iv_fdm_sweep` uses it).
- No general guarantee about K_ref spacing. The sweep publishes a conditional
  baseline for one spot, one option type, one rate and one schedule family.

## 4. Design

### D1. Dividends path: documented grid, fixed quarterly calendar

Both `build_div_solvers()` (B-spline, one solver per maturity) and
`run_chebyshev_dividends()` (Chebyshev, one surface at T = 1) use:

| Field | Value | Source |
|---|---|---|
| S/K moneyness | {0.92, 0.95, 1.0, 1.05, 1.08} | documented config |
| vol | {0.10, 0.15, 0.20, 0.30} | documented config |
| rate | {0.02, 0.03, 0.05, 0.07} | documented config |
| K_refs | {90, 92.5, 95, 97.5, 100, 102.5, 105, 107.5, 110} | documented config |
| `AdaptiveGridParams` | `{.target_iv_error = 1e-3}`, defaults otherwise | documented config |
| dividend yield | `kDivYield` (2%) | benchmark constant |
| schedule | `quarterly_div_schedule(T)`: $0.50 at calendar 0.25, 0.50, 0.75, … < T | new, this path only |

The grid, K_refs and adaptive parameters are verbatim from
`documented_adaptive_dividend_config()` in `tests/iv_solver_factory_slow_test.cc`.
The schedule is new: a fixed quarterly calendar, filtered to the option's
life, so a 1-year option carries the same three dividends the benchmark used
before and a 7-day option carries none. Maturities below 0.25 y therefore run
the segmented path with a single segment and no dividend; their rows are
labelled with the dividend count so a reader does not mistake them for
dividend-bearing rows.

Measured on `main` with this schedule and the documented grid (holdout
diagnostic, bps):

| 7d | 14d | 30d | 60d | 90d | 180d | 1y | 2y |
|---|---|---|---|---|---|---|---|
| 1591 | 962 | 329 | 208 | 965 | 82 | 46 | 36 |

All eight build. Chebyshev at T = 1 with the same schedule: viable, 69.6 bps
max, 9.9 bps avg, 11 minutes at 3 threads. The short-maturity figures are
poor and the benchmark reports them as they are; they are the documented
config applied to maturities it was not tuned for, and part of what #501
records.

**Wrapping.** The B-spline per-maturity solver publishes
`result->sample_bounds` (spec D2 of #454), not bounds assembled from the
input arrays, and passes the build schedule to
`InterpolatedIVSolver::create(surface, {}, schedule)` so query schedules are
validated. The Chebyshev solver does the same with `result->sample_bounds`
and the T = 1 schedule.

### D2. Reference prices and strike set for the dividends path

**Strike set.** The vanilla path keeps `kStrikes` (80–120 step 5). The
dividends path measures `kDivStrikes = {93, 95, 97.5, 100, 102.5, 105, 107}`:
every strike lies inside the band S/K ∈ [0.92, 1.08] (strikes 92.6–108.7), and
the set mixes K_ref anchors (95, 97.5, 100, 102.5, 105) with off-anchor
strikes near the band edges (93, 107). Strike headers print with one decimal.

**Two reference grids.** The two backends answer different contracts at a
maturity below T = 1, so each gets reference prices for the contract it
actually represents:

- B-spline per-maturity: contract maturity T_i with `quarterly_div_schedule(T_i)`.
  The solver for T_i was built with that schedule and is queried at
  tau = T_i, so the reference and the surface agree by construction.
- Chebyshev fixed-expiry: the T = 1 surface queried at tau = T_i represents
  the 1-year contract observed 1 − T_i later. Its reference at T_i is the
  contract with maturity T_i and schedule
  `rolled_dividends(quarterly_div_schedule(1.0), 1.0, T_i)` from
  `mango/option/dividend_utils.hpp`. Maturities above 1 y have no Chebyshev
  row (as today).

Both grids are produced by `generate_prices` given a schedule function; the
existing scaled-schedule call site for the vanilla and q0 paths is unchanged.
Each backend's heatmap and TV/K mask use its own reference grid, and the
TV/K comparison prints the two backends in separate blocks headed by their
schedule, since their masks differ. The recovered FDM IV then equals the
displayed 15% or 30% on both backends, which keeps every query inside the
documented vol range [0.10, 0.30].

**Containers.** To carry two strike sets the per-path containers become
generic over the strike count: `PriceGrid`, `ErrorTable`, `TVKMask` and the
functions that take them (`generate_prices`, `compute_errors_*`,
`print_heatmap`, `compute_tvk_mask`, `print_tvk_comparison`) take
`template <size_t NS>` with the strike array supplied alongside. Maturity and
vol dimensions stay fixed. The Chebyshev T = 1 point diagnostic loop and the
opening banner read from the strike array they are given. No output format
changes for the vanilla and q0 paths.

### D3. Honest failure reporting

- A failed adaptive build prints the `PriceTableErrorCode` name and the
  error's `axis_index` and `count`, via a `code_name()` switch in the
  benchmark; the switch is exhaustive so a new enumerator is a compile
  warning. A failed `InterpolatedIVSolver::create` prints its
  `ValidationErrorCode` value.
- Every aggregate (heatmap RMS, TV/K cell, sweep cell) prints `n/a` when it
  has zero contributing points; the current printers print `0.0` for `n == 0`
  and that is what hid the failure.
- `print_tvk_comparison` takes each algorithm's error table as a nullable
  pointer; a null table (no surface built) prints `n/a` in every cell.
- Every table reports counts: attempted, succeeded, failed (reference or
  inversion), filtered (TV/K). The B-spline per-maturity block lists which
  maturities built and which did not.
- Exit status: `main` returns 1 if any *build* the requested path needed
  failed (adaptive build, manual build, or solver wrap). Per-point reference
  or inversion failures never change the exit status; they are data and are
  counted in the tables.

### D4. `--path=kref`: the K_ref spacing sweep

A new path that measures the K_ref blend policy with a same-query control.
It uses the manual, non-adaptive segmented builder so that input knots are
fixed and the only variable across spacings is the K_ref set.

**Surfaces.** For each spacing Δ ∈ {10, 5, 2.5, 1.25} dollars (Δ/spot =
10%, 5%, 2.5%, 1.25%), K_refs are `80, 80+Δ, …, 120` (5, 9, 17, 33 K_refs).
Each K_ref surface is `SegmentedPriceTableBuilder::build(Config{...})` with:

- log-moneyness input knots: 41 uniform points on [−0.30, 0.30]; the builder
  itself expands the grid downward by total dividend / K_ref and adds upper
  headroom, so the *input* knots are fixed while the final grid differs
  slightly per K_ref;
- vol knots {0.10, 0.15, 0.20, 0.30, 0.50}; rate knots {0.02, 0.03, 0.05, 0.07}
  (the builder requires at least four knots per axis);
- `tau_points_per_segment = 5`, `kDivYield`, `quarterly_div_schedule(T)`.

The surfaces are assembled with `build_multi_kref_surface` and wrapped as
`BSplineMultiKRefSurface` with bounds m ∈ [−0.30, 0.30], tau ∈ [0, T], and
the vol and rate knot ranges, matching `manual_segmented_bounds`. The IV
solver is `InterpolatedIVSolver<BSplineMultiKRefSurface>::create(surface, {},
schedule)`.

**Maturities.** T ∈ {90/365, 180/365, 1.0}: one, two and three dividends
under the quarterly calendar. Rate 5%, σ ∈ {0.15, 0.30}.

**Query strikes.** Inside the window [85, 115], inclusive, so every query is
at least one coarse spacing away from the clamped outer bands:

- anchors: the spacing's K_refs in the window;
- mid-anchors: all midpoints between adjacent K_refs that lie in the window.

**Per-query quantities.** For a query (T, K, σ) with bracketing K_refs L ≤ K ≤ H
and w = (K − L)/(H − L):

- P_FDM(K): `solve_american_option` at (spot, K, T, σ, r) with the schedule;
  vega_FDM(K) by the same central σ-bump `make_fd_vega_refs_fn` in
  `src/option/table/adaptive_metrics.cpp` uses (two more solves). At anchors P_FDM(L) and P_FDM(H) are the
  anchor references already solved.
- B_FDM(K) = K · [(1 − w) · P_FDM(L)/L + w · P_FDM(H)/H]: the blend policy
  applied to exact prices. This is the same-query control: it is what the
  assembled surface would return if every K_ref surface were exact.
- P̂(K): the assembled surface price.
- Signed price decomposition: P̂(K) − P_FDM(K) = [P̂(K) − B_FDM(K)] +
  [B_FDM(K) − P_FDM(K)]. The first term is surface approximation (fit, PDE,
  extraction), the second is the blend policy's own error. At anchors the
  second term is zero by construction.
- IV-equivalent estimates: each price term divided by vega_FDM(K), labelled
  as estimates.
- End-to-end IV inversion error: |IV_interp − IV_FDM| in bps, where IV_FDM is
  `solve_fdm_iv_div` on P_FDM(K) and IV_interp is the solver's `solve` on the
  same price. This is the benchmark's usual metric, reported next to the
  decomposition, never mixed with it.

**Output.** One block per σ, one row per (Δ, T):

```
K_ref spacing sweep — σ=15%  (window K∈[85,115]; bps = price / FD vega unless "inv")
                     mid-anchors                                   anchors
  Δ     T    n  blend max  blend rms  surf max  surf rms  inv max  fail |  n  surf max  surf rms  inv max  fail
  10   90d  ...
```

`blend` is [B_FDM − P_FDM]/vega, `surf` is [P̂ − B_FDM]/vega, `inv` is the
inversion error, `fail` counts queries whose reference solve or inversion
failed. A cell with n = 0 prints `n/a`. A row whose `fail` exceeds 10% of its
queries is marked `incomplete` and excluded from D5.

**Resolution check.** After the sweep, the Δ = 2.5 row at T = 1 is rebuilt
with 81 moneyness knots and `tau_points_per_segment = 9` and printed as one
extra row labelled `fine`. If `surf` moves by more than a factor of two the
per-surface floor is not converged and the sweep says so in its footer; the
blend column is independent of the floor by construction, so the conclusion
about the blend policy stands either way.

**Cost.** Surfaces: 3 maturities × (5 + 9 + 17 + 33) K_refs × 20 (σ, r) knot
pairs = 3840 short fixed-expiry solves, plus the fine row. Queries: about
550 (T, K, σ) points, each three FD solves for price and vega plus a Brent
inversion. Runtime is measured during execution and printed; the target is a
few minutes, not a promise.

### D5. Documentation

- `docs/API_GUIDE.md`, the K_ref paragraphs under "Discrete Dividends with
  Adaptive Grid" (near line 761): add a short table of the sweep's mid-anchor
  `blend max` and `blend rms` per (Δ, T) for both σ, headed by the command,
  date, commit and build flags that produced it. The accompanying text is a
  conditional baseline, not a rule: it names the spacings whose `blend max`
  stays at or below 10 bps IV-equivalent at every measured (T, σ) with no
  `incomplete` row, states explicitly that none may qualify, and states the
  conditions (spot 100, PUT, r = 5%, quarterly $0.50 calendar, manual B-spline
  segmented surfaces) outside which the numbers do not transfer.
- Comment on #460 with the same table and text as its measured baseline.
- Terms used in the sweep output and guide, defined where first used:
  *anchor* (a strike equal to a K_ref), *mid-anchor* (the midpoint between two
  adjacent K_refs), *blend policy* (`MultiKRefSplit`: query each bracketing
  K_ref surface at (spot, K_ref), normalize by K_ref, interpolate linearly in
  strike, multiply by strike). S/K and log(S/K) are named as such.
- `CLAUDE.md` Pattern 3/4 text is unchanged; it already states the
  span-and-resolve rule.

### D6. Cross-references and regression coverage

- In `build_div_solvers()` and `run_chebyshev_dividends()`: a comment naming
  `documented_adaptive_dividend_config()` in `tests/iv_solver_factory_slow_test.cc`
  as the source of the grid/K_ref/adaptive values.
- Above `documented_adaptive_dividend_config()`: a comment naming
  `benchmarks/interp_iv_safety.cc` as a copy that must move with it.
- **Open for the go/no-go (Codex round 1 disagreed with decision Q4):** the
  existing nightly pins use a different yield, schedule and a single build
  maturity, so they do not protect the benchmark's eight-maturity
  configuration. The proposed addition, if the user accepts it, is one
  `slow`-tagged case in the existing `iv_solver_factory_slow_test` target that
  builds the B-spline per-maturity configuration of D1 at all eight
  maturities and asserts viability with `holdout_points_measured > 0`; no
  accuracy number is pinned, and no Chebyshev case is added (minutes per
  build). Until the user decides, the spec's default remains the user's Q4
  choice: no new test.

## 5. Acceptance criteria

- AC1. `bazel build //benchmarks:interp_iv_safety` succeeds with no new
  warnings (the target is `manual`, so the wildcard does not cover it).
- AC2. `interp_iv_safety --path=dividends` on this branch builds all eight
  B-spline per-maturity solvers and the Chebyshev surface, prints their
  diagnostics and dividend counts, uses the seven `kDivStrikes` columns, and
  exits 0.
- AC3. With a deliberately incoherent config (verified once by hand, not
  committed) the benchmark prints the error code name and `n/a` cells and
  exits 1.
- AC4. `interp_iv_safety --path=kref` completes, prints the four-spacing
  blocks for both σ with the `fine` row and counts, and has no `incomplete`
  row at Δ ≤ 5.
- AC5. `--path=bspline`, `--path=chebyshev` and `--path=q0` produce the same
  tables as before apart from timing lines and the usage/banner lines.
  `--path=all` includes `kref` and its dividends sections change as designed.
- AC6. The fast suite (`bazel test //tests/... --test_tag_filters=-manual,-slow`)
  passes with the same count as on `main`; `bazel test //tests:iv_solver_factory_slow_test`
  passes.
- AC7. The API guide table and the #460 comment contain the numbers from the
  committed sweep run, with its provenance line.

## 6. Risks and assumptions

- The D1 schedule was chosen because it builds today; #501 shows the
  B-spline segmented builder refuses other realistic placements. A library
  change that fixes #501 may shift the short-maturity numbers, and the
  benchmark's non-zero exit and code-name output make any new refusal visible.
- The sweep's manual B-spline surfaces inherit any per-K_ref fit instability
  (#458). The same-query control keeps the blend column independent of that
  floor; the `surf` column and the `fine` row show how large the floor is.
- The Chebyshev dividends build takes about ten minutes; the benchmark is
  manual-tagged and not in CI, so this is acceptable and unchanged from today.
- The sweep runs at one spot, one rate and one schedule family. D5 states the
  conditions; the numbers are a baseline for #460, not a guarantee.

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
the sweep belongs next to the diagnostic it explains. Round 1 of design review
replaced the anchor-vs-mid-anchor comparison with the same-query control in D4.

Q4. **Regression protection for the retuned config.** Options: (i) rely on
the existing nightly pins plus cross-reference comments; (ii) a new slow
test on the benchmark's exact config; (iii) a shared config helper linked by
both. **Chosen: (i).** A second near-identical fixture adds drift of its own;
a shared library target for one fixture is not worth a BUILD target.
Design review round 1 disagreed: the pins cover a different schedule and one
maturity. Carried to the go/no-go as D6's open item.

Q5. **The Chebyshev wide-band regression found during triage.** Options:
(i) file a new issue and keep #462 as designed; (ii) fold the fix into #462;
(iii) pause #462 until it is fixed. **Chosen: (i)**, filed as #500. The
documented band does not touch the regressed region.

Q6 (added after round 1, decided from measurement, not a user question).
**Dividend schedule for the dividends path.** Options: (i) keep the scaled
`make_div_schedule(T)`; (ii) fixed quarterly calendar from 0.25 y; (iii) fixed
quarterly calendar from an earlier first ex-dividend so short maturities
carry a dividend. **Chosen: (ii).** (i) refuses at every maturity ≤ 180d on
B-spline; (iii) refuses at some maturity for every first-date tried (#501);
(ii) is the only placement that builds at all eight.

Approved by the user on 2026-09-12 for spec writing and design review.
