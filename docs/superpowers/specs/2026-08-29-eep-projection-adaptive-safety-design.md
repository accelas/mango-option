# Exact EEP Projection + Adaptive Refinement Safety Design

**Issue:** #434 (expanded scope per issue comment of 2026-07-23)

**Status:** Draft

**Date:** 2026-08-29

**Supersedes:** the *scope* of `2026-07-23-eep-floor-correction-design.md`. That
document's decision (exact projection `max(0, x)`) and its projection test
design remain authoritative and are incorporated here unchanged. This document
adds the adaptive-refinement and IV-inversion work required before the
projection can merge.

## Context

The July design replaced the debiased-softplus EEP floor with the exact
projection `eep_floor(x) = max(0, x)`. A test-first implementation
(`fix/issue-434-eep-projection`, cherry-picked into this branch) passed all
direct contract tests but deterministically regressed the adaptive 4D B-spline
diagnostic (`interp_iv_safety --path=q0`): 7.3 → 289.3 bps RMS at TV/K ≥ 1e-4,
σ = 30%. The branch was paused pending an expanded safety contract
(issue comment, 7-item checklist).

A design-exploration prototype (2026-08-28/29, 8 experiment rounds on branch
`proto/434-adaptive-safety`, ~20 instrumented adaptive builds) identified the
actual root causes. They are pre-existing defects in the adaptive refinement
loop; the exact projection merely perturbed which of them fired.

### Prototype findings (evidence base)

**P1 — The refinement loop measures the wrong domain.** Validation samples are
drawn from the *headroom-expanded* fit domain, and the headroom is ~10×
oversized: `extract_chain_domain` passes the **user knot count** (7 for the
benchmark configs) to `spline_support_headroom` instead of the actual
moneyness grid size (≥ 60), producing ±0.31 log-moneyness of headroom instead
of ±0.03. The least-squares fit oscillates wildly in that band (observed
∂Price/∂σ ≈ −1.2×10⁴ $/vol against prices ≤ $130, both under softplus and
exact projection). Consequences, all observed:

- reported `max_error` at the seed grid is ~8,000 bps while user-visible
  quality is single-digit bps — the 2 bps convergence target has effectively
  never been reachable, so every build exhausts `max_iter`;
- error-bin attribution is dominated by corner noise (bin concentrations
  0.25–0.31 against a uniform baseline of 0.20 — no usable signal), so axis
  selection is effectively arbitrary;
- clamping validation sampling to the user domain drops the seed reading to
  123 bps and makes attribution meaningful.

**P2 — Refinement degrades the fit, and the loop keeps the damage.** Under
honest (user-domain) measurement, midpoint-insertion refinement made the
holdout error strictly worse in both benchmark configs (q0: 102 → 1,099 bps;
vanilla: 931 → 27,683 bps under the old selection), and at 160 moneyness
points the B-spline build **fails outright** (clustered knots), which today
propagates as failure of the entire solver build even though good candidates
were already in hand. The current loop returns the **last** iteration
unconditionally; under softplus, the production vanilla path returns a surface
whose full-domain holdout error is ~1,800 bps although its own iteration-0
candidate measured ~35 bps. This is a live production defect independent of
the projection.

**P3 — Retention against a fixed user-domain holdout fixes the regression.**
Best-candidate retention alone restored the q0 diagnostic to 4.8–5.1 bps
(TV/K ≥ 1e-4, σ = 30%; baseline before regression: 7.3–8.7) and 1.2–1.3 bps at
σ = 15%, TV/K ≥ 1e-3. The control run proved the holdout must be sampled from
the user domain: with a corner-polluted holdout, retention *misranks*
candidates (vanilla test slice 50 → 403 bps).

**P4 — Concentration-threshold multi-axis selection and standalone
trial-build tie-breaks are dead mechanisms.** With diffuse attribution
(P1), no dimension ever clears a concentration threshold, and the tie-break
never triggers. What worked: greedy coordinate descent with **measured
acceptance** — refine the top-scoring untried axis starting from the best
candidate's grids, accept only on ≥ 2% relative holdout improvement,
otherwise mark the axis tried and take the next. The 2% threshold is
required: a hairline 931.5 → 931.2 "improvement" otherwise reopens the axis
set and burns the entire budget re-trying useless axes. On the vanilla config
this walk reached the rate axis, collapsing the holdout from
931 → 80.9 bps max (21.3 → 5.3 bps avg).

**P5 — Zero-violation monotonicity gating is unusable.** Every surface the
pipeline has ever produced — including production softplus surfaces that pass
all benchmarks — has σ-monotonicity violations somewhere in its domain (22–57
at the seed). Violation counts barely discriminate good from catastrophic
candidates (1 vs 3 in the user domain); holdout error separates them by four
orders of magnitude. In-domain violations are real but small
(−2 … −14 $/vol): at low time value, vega → 0 and *no* least-squares fit can
be strictly monotone there — small price wiggles are unavoidable in principle.
The right defenses are (a) candidate ranking by holdout error at build time
and (b) multiple-root detection at query time. Build-time monotonicity
statistics are reported as diagnostics, not used as a gate.

## Goals

1. Land the exact EEP projection (July design, unchanged).
2. Make the adaptive refinement loop measure user-visible quality: validation
   and holdout sampling over the user domain.
3. Fix the oversized-headroom defect.
4. Never return a worse surface than the best one built: fixed-holdout
   candidate retention, including on mid-loop build failure.
5. Replace noise-driven axis selection with measured acceptance
   (greedy backtracking).
6. Surface honest build diagnostics (`target_met`, achieved errors, picked
   iteration, monotonicity statistics) to the API user.
7. Defend IV inversion against multi-root brackets and negative vega
   (configurable, default on).
8. Regression coverage for the deterministic q0 bifurcation and the
   wrong-root failure; before/after `interp_iv_safety` evidence.

## Non-goals

- Root-causing the fit degradation under knot insertion or the deterministic
  160-point build failure (follow-up issue; this design only makes them
  non-fatal).
- Monotonicity certificates or shape-constrained (monotone) fitting
  (follow-up issue).
- Changing the segmented multi-K_ref probe aggregation (it discards refined
  knot positions and keeps only sizes — follow-up issue). The segmented path
  still inherits every shared-loop improvement below.
- Denoising positive PDE residuals; changing interpolation algorithms or
  coordinate transforms; persisted-table migration (all per the July design).
- Any European-price shortcut in the American solver (#439 constraint).

## Design

### D1. Exact EEP projection

As specified in `2026-07-23-eep-floor-correction-design.md`:
`eep_floor(x) = max(0, x)` in `src/option/table/eep/eep_decomposer.hpp`, with
that document's direct-contract, shared-decomposition, and integration tests
(already present on this branch via cherry-pick). Documentation updates
(`docs/ARCHITECTURE.md` §4, code comments) describe the exact projection.

### D2. Sampling domain separation (fit domain vs. measurement domain)

`RefinementContext` gains a second bounds member:

```cpp
struct RefinementContext {
    double spot;
    double dividend_yield;
    OptionType option_type;
    SurfaceBounds bounds;         // fit domain (with headroom) — unchanged
    SurfaceBounds sample_bounds;  // user-facing measurement domain
};
```

- `sample_bounds` is the domain the user actually asked for: log-moneyness
  span of the user's strikes/moneyness grid, the user's τ, σ, r ranges
  (after the existing `expand_domain_bounds` minimum-spread widening, which
  is a usability floor, not headroom).
- `extract_chain_domain` (and the segmented equivalent
  `expand_segmented_domain` call sites) populate both members; headroom is
  applied only to `bounds`.
- `run_refinement` draws **all** validation samples — the per-iteration fresh
  samples and the fixed holdout (D4) — from `sample_bounds`. Error-bin
  normalization also uses `sample_bounds` so bins describe the measured
  region.
- The fit still covers `bounds`; the surface's queryable range is unchanged
  by this item (see D3 for the headroom size fix).

### D3. Headroom size fix

`spline_support_headroom(domain_width, n_knots)` must receive the actual
moneyness grid size, not the user strike count. `extract_chain_domain` uses
`max(chain.strikes.size(), AdaptiveGridParams::min_moneyness_points)`; to
avoid plumbing params into a context helper, the function signature gains the
expected knot count:

```cpp
std::expected<RefinementContext, PriceTableError>
extract_chain_domain(const OptionGrid& chain, size_t expected_m_knots);
```

For the benchmark configs this shrinks moneyness headroom from ±0.31 to
±0.03 log-moneyness. Behavior change: queries in the removed band — which
previously returned oscillating garbage with healthy-looking vega — are now
rejected by the existing bounds check. This is deliberate and documented.

### D4. Fixed holdout with cached references

Before iteration 0, `run_refinement` generates one holdout set of
`params.validation_samples` points via `latin_hypercube_4d` with seed
`params.lhs_seed ^ 0x484F4C44` ("HOLD"), scaled to `sample_bounds`, and
computes per point, **once**:

- the FD American reference price (`validate_fn`);
- the two σ-bumped FD prices used by the FD-vega error metric.

`make_fd_vega_error_fn` is refactored so the vega bump solves can be supplied
from a cache: a new overload/struct `CachedErrorRefs { double ref_price;
double vega; }` per holdout point, so per-iteration holdout evaluation costs
only `validation_samples` surface interpolations plus arithmetic (no FD
solves after setup). Points whose reference solve fails are excluded from the
holdout (recorded in diagnostics).

Holdout score per candidate: `holdout_max` and `holdout_avg` over the same
TV/K-filtered IV-error metric used by fresh validation.

Convergence now requires both sets under target:
`fresh_max <= target_iv_error && holdout_max <= target_iv_error`.

### D5. Candidate retention

Each iteration records a candidate `{grids, holdout_max, holdout_avg,
monotonicity stats, iteration index}`. At loop exit (converged, budget
exhausted, all axes exhausted, or build failure):

- pick the candidate with the lowest `holdout_max` (ties: lowest
  `holdout_avg`, then earliest iteration);
- if the picked candidate is not the surface most recently built, rebuild it
  once via `build_fn` (deterministic: same grids → same surface) so the
  caller's captured surface state matches the returned grids;
- `achieved_max_error`/`achieved_avg_error` report the picked candidate's
  holdout numbers; `target_met = (holdout_max <= target && fresh
  convergence was observed for that candidate)`.

**Mid-loop build failure:** if `build_fn` fails and at least one candidate
exists, the loop stops and falls back to the retained best (recording the
failure in diagnostics). Only a failure with zero candidates (i.e., the seed
build) propagates as an error. This replaces today's behavior where any
mid-loop failure kills the entire solver build.

Best-effort semantics (user decision): a safe-but-unmet-target result is
returned successfully with `target_met = false` and achieved errors visible
via D7. There is no rejection gate on monotonicity statistics (P5).

### D6. Refinement selection: greedy backtracking with measured acceptance

Replaces single-axis `worst_dimension()` selection. State: `tried[4]`
(axes attempted since the last accepted improvement).

Per refinement step:

1. Score each axis d: `score[d] = concentration[d] * dim_error_mass[d]`
   (the existing `ErrorBins` quantities, now computed over `sample_bounds`).
2. Pick the highest-scoring axis with `tried[d] == false`. If none remain,
   stop (all axes exhausted; retention picks the final result).
3. Reset the working grids to the **best candidate's** grids, then apply the
   existing `refine_fn` for the picked axis.
4. After the next iteration's build + evaluation: if
   `holdout_max < best_holdout_max * (1 - kMinRelImprovement)` with
   `kMinRelImprovement = 0.02`, the step is **accepted**: clear `tried`.
   Otherwise mark `tried[d] = true` (the candidate is still recorded and
   remains eligible for retention).

Budget: `max_iter` continues to bound total iterations (builds). Its default
rises from 5 to 8 so a full 4-axis backtracking walk plus accepted
improvements fits (prototype: the rate axis was reached at iteration 4 of 5,
leaving no budget to continue from the improvement). Cost note: iterations
beyond the old default occur only when earlier iterations failed to converge,
and each costs one table build plus `validation_samples` FD solves
(vega bumps for fresh samples included) — the holdout itself adds no
per-iteration FD cost (D4).

The concentration-threshold multi-axis mechanism and the standalone
trial-build tie-break from the earlier draft are **not** implemented (P4).

### D7. Build diagnostics surfaced to the user

New struct in `adaptive_grid_types.hpp`:

```cpp
struct BuildDiagnostics {
    bool target_met = false;
    double achieved_max_error = 0.0;   // holdout, picked candidate
    double achieved_avg_error = 0.0;
    size_t picked_iteration = 0;
    size_t total_iterations = 0;
    bool build_failure_fallback = false; // a mid-loop build failed
    size_t holdout_points = 0;           // valid holdout references
    size_t monotonicity_violations = 0;  // picked candidate, sample_bounds scan
    double worst_vega_slope = 0.0;       // most negative dPrice/dσ observed
    std::vector<IterationStats> iterations;
};
```

- `run_refinement` fills it (monotonicity statistics from a σ-scan of the
  picked candidate at the holdout (m, τ, r) points — 7 σ values across the
  user σ-range; diagnostics only, never a gate).
- The adaptive builders (`BSplineAdaptiveResult`,
  `BSplineSegmentedAdaptiveResult`, Chebyshev equivalents) carry it.
- `AnyPriceTable` and `AnyInterpIVSolver` store it when built through an
  adaptive path and expose `std::optional<BuildDiagnostics>
  build_diagnostics() const` (nullopt for non-adaptive/manual builds and
  Parquet-loaded tables).

### D8. IV inversion defense (query time)

In `InterpolatedIVSolver::solve` (src/option/interpolated_iv_solver.hpp):

1. **Signed vega pre-check.** The existing pre-check takes `std::abs(vega)`
   at three probe vols; a strongly negative vega therefore counts as healthy.
   Change: use the signed maximum. `max_vega = max(vega_probes)` (signed);
   reject with `VegaTooSmall` only when `max_vega < vega_threshold`.
   A negative probe no longer masquerades as sensitivity; surfaces negative
   at all probes fail the same check naturally.
2. **Bracket scan** (new, config `detect_multiple_roots`, default `true`):
   before Brent, evaluate the objective at 9 equally spaced σ in
   `[sigma_min, sigma_max]` (endpoints included).
   - More than one sign change → return new `IVErrorCode::MultipleRoots`
     (`final_error` = number of sign changes, `last_vol` = lowest crossing).
   - Exactly one sign change → run Brent on the bracketing subinterval
     (narrowed bracket; partially offsets the scan's ~9 × 250 ns cost).
   - No sign change → fall through to Brent on the full bracket, which
     reports `BracketingFailed` exactly as today.
   - Scan cost is ~2 μs against a ~3.5 μs solve; `detect_multiple_roots =
     false` restores today's path unchanged.
3. **Error-code wiring** (the #449 drill): `MultipleRoots` appended to
   `IVErrorCode` (src/support/error_types.hpp), message/helper switch arms,
   `iv_result.hpp` mapping, Python bindings enum + message
   (src/python/mango_bindings.cpp), C API `map_iv_error`
   (src/ffi/mango_c_api.cpp → `MANGO_ERR_SOLVER` class alongside
   `BracketingFailed`'s mapping; follow its existing category).

### D9. Callers and shared-path coverage

`run_refinement` is the single shared loop; D2/D4/D5/D6 changes apply
automatically to:

- `build_adaptive_bspline` (4D single-K_ref; also gains the D3 headroom fix),
- `build_adaptive_bspline_segmented` (per-probe; aggregation unchanged),
- `build_adaptive_chebyshev` / `build_adaptive_chebyshev_segmented`.

The Chebyshev exact-grids mode (`InitialGrids::exact`) keeps CGL node
placement: the backtracking reset (D6 step 3) restores the best candidate's
node set verbatim, and `refine_fn` remains the backend's own.

## Decisions

Brainstorm Q&A (2026-08-28, user choices) and prototype-driven revisions:

1. **Scope: full expanded scope** (all checklist items in one branch) —
   chosen over "projection + safety net" and smaller splits. Prototype
   evidence subsequently *narrowed the mechanism list* (multi-axis and
   standalone tie-break dropped as valueless — P4) while adding root-cause
   fixes (D2, D3) not in the original checklist; the checklist's intent
   (comparable validation, retention, honest diagnostics, measured
   refinement, IV defense, regression coverage, benchmark evidence) is fully
   covered.
2. **Refinement selection: "both multi-axis + measured tie-break"** — the
   prototype showed the multi-axis trigger never fires and the tie-break is
   subsumed by measured acceptance; the user approved the revised greedy
   backtracking design (2026-08-29) as the implementation of "measured
   improvement".
3. **Unmet-but-safe target: best-effort + surfaced diagnostics** — no hard
   error, no strict-mode flag (D5, D7).
4. **IV defense: bracket scan, configurable, on by default** — plus signed
   vega pre-check (D8). Post-hoc-only checking rejected (cannot detect a
   wrong root on a locally positive slope — exactly the observed q0
   failure).
5. **Rigor level: prototype first** — executed; findings above. Certificate /
   shape-constrained fitting deferred to a follow-up issue by the evidence
   (build-time strict monotonicity is unattainable where vega → 0; the
   query-time scan is the sound defense there).
6. **Retention criterion: holdout error, not monotonicity violations** (P5).
7. **Improvement threshold 2% relative** — from P4's budget-burn observation.
8. **`max_iter` default 5 → 8** — required for the backtracking walk to
   both reach and exploit late-axis improvements (D6).

## Test design

### Unit: run_refinement (synthetic callbacks)

The loop is callback-driven; tests use synthetic `BuildFn`/`ValidateFn`/
`ComputeErrorFn` that require no PDE solves:

- **Retention picks best:** builds whose holdout error degrades after
  iteration k → returned grids are iteration k's; `picked_iteration == k`.
- **Build-failure fallback:** `build_fn` fails at iteration j > 0 → success,
  `build_failure_fallback == true`, best prior candidate returned; failure at
  iteration 0 → error propagates.
- **Backtracking walk:** an error field reducible only along axis 2 → axes
  0/1 (higher scores by construction) are tried, rejected, and axis 2's
  improvement is accepted; `tried` resets after acceptance.
- **ε-threshold:** a 0.5% improvement does not reset `tried`; a 5%
  improvement does.
- **All-axes-exhausted stop:** no axis improves → loop stops before
  `max_iter`, retention returns the seed.
- **Convergence requires holdout:** fresh samples under target but holdout
  above → not converged.
- **Holdout caching:** count `validate_fn` invocations — exactly
  holdout-setup + fresh-validation counts (no per-iteration holdout solves).
- **Rebuild consistency:** picked ≠ last ⇒ one extra `build_fn` call with the
  picked grids.
- **Sampling domain:** all sample coordinates within `sample_bounds`.

### Unit: headroom and domain

- `extract_chain_domain(chain, 60)` headroom equals
  `3 * width / 59` (not `3 * width / (n_strikes - 1)`).
- `sample_bounds` equals user ranges (with minimum-spread widening),
  `bounds` equals `sample_bounds` plus headroom on moneyness.

### Unit: IV defense

- Synthetic price table whose reconstructed price oscillates in σ (three
  crossings for a chosen market price) → `MultipleRoots`; the same query with
  `detect_multiple_roots = false` reproduces today's (wrong-root) behavior —
  documenting the defended failure.
- Monotone surface → identical root with and without the scan (narrowed
  bracket converges to the same σ within tolerance).
- Signed pre-check: probe vegas {−5, −3, −1} → `VegaTooSmall` (max signed
  < threshold); {−5, +2, +1} with threshold 0.5 → passes to the scan.
- Error-code wiring: `MultipleRoots` mapped in `iv_result`, Python enum, and
  C API (mapping test per #449 pattern).

### Integration / regression

- **q0 bifurcation regression:** adaptive B-spline build on a scaled-down
  q = 0 PUT config (reduced grids/samples for test budget) asserts: build
  succeeds, `build_diagnostics()` present, `achieved_max_error` below a
  generous bound (e.g. 100 bps) — this fails against the pre-fix loop, which
  returns the catastrophic final iteration.
- **Projection tests** from the July design (already on branch).
- **Diagnostics plumbing:** factory-built adaptive solver exposes
  diagnostics; manual build returns `nullopt`.

### Benchmark evidence (acceptance, not CI)

Before/after `interp_iv_safety` for `--path=q0`, `--path=bspline`,
`--path=dividends`:

- q0, σ = 30%, TV/K ≥ 1e-4 must improve from 289.3 bps to ≤ 10 bps
  (prototype: 5.1);
- no path deteriorates beyond the benchmark's 0.1 bps reported precision on
  TV/K ≥ 1e-3 metrics without a written explanation in the PR;
- the vanilla σ = 30% slice is the known watch item (vol-knot phase, P4/D6
  budget change is expected to close it); record and explain the final
  numbers either way.

## Acceptance criteria

1. July design's projection criteria (all — see that document).
2. Validation and holdout sampling occur strictly within the user domain;
   headroom no longer scales with user knot count.
3. The adaptive loop never returns a surface with a worse holdout score than
   the best candidate it built; mid-loop build failure after iteration 0 is
   non-fatal.
4. Refinement steps are accepted only on ≥ 2% relative holdout improvement;
   axis search backtracks; loop stops when all axes are exhausted.
5. `build_diagnostics()` available on adaptive-built tables/solvers with
   honest `target_met` and achieved errors.
6. `MultipleRoots` returned (never a spurious root) for multi-crossing
   brackets when `detect_multiple_roots` is on; scan removable by config;
   signed vega pre-check.
7. All error-code surfaces (C++, Python, C API) handle the new code.
8. Full test suite, benchmarks build, Python bindings build; benchmark
   evidence recorded per above.

## Follow-up issues (to file at merge)

1. Fit degradation under knot insertion: clustered midpoint insertion
   degrades and eventually breaks the B-spline fit (deterministic failure at
   160 points). Root-cause and fix (knot spacing constraints, regularization,
   or kink-aware placement near the exercise boundary).
2. Monotonicity certificate (B-spline derivative-coefficient bound coupled
   with per-interval European vega lower bounds) and/or shape-constrained
   σ-axis fitting; certified surfaces may skip the query-time scan.
3. Segmented multi-K_ref probe path discards refined knot positions
   (aggregates sizes only, rebuilds uniform) — evaluate carrying positions.
4. Sparse explicit K_ref grids accuracy (parked from PR #449 work).
