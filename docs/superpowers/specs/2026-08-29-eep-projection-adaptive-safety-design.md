# Exact EEP Projection + Adaptive Refinement Safety Design

**Issue:** #434 (expanded scope per issue comment of 2026-07-23)

**Status:** Approved — design review converged after round 3 (rounds 1–2
fixed structural blockers; round 3's findings were descendant refinements of
the established principles — *exploration must be able to continue toward
viability*, *viability must cover every evaluation made of a candidate*, and
*D8 semantics must never overclaim* — folded below without design change)

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
candidate's grids, restart the axis walk only on ≥ 2% relative holdout
improvement, otherwise mark the axis tried and take the next. The 2%
threshold is required: a hairline 931.5 → 931.2 "improvement" otherwise
reopens the axis set and burns the entire budget re-trying useless axes. On
the vanilla config this walk reached the rate axis, collapsing the holdout
from 931 → 80.9 bps max (21.3 → 5.3 bps avg).

**P5 — Zero-violation monotonicity gating is unusable.** Every surface the
pipeline has ever produced — including production softplus surfaces that pass
all benchmarks — has σ-monotonicity violations somewhere in its domain (22–57
at the seed). Violation counts barely discriminate good from catastrophic
candidates (1 vs 3 in the user domain); holdout error separates them by four
orders of magnitude. In-domain violations are real but small
(−2 … −14 $/vol): where time value is small, vega → 0 and demanding strict
monotonicity from an *unconstrained* least-squares fit is inappropriate —
small price wiggles are expected there. (A shape-constrained fit could
enforce a nondecreasing σ-profile; that is the deferred follow-up, not this
branch.) The defenses are therefore (a) a *viability gate* on holdout error
at build time (D5), (b) monotonicity statistics as diagnostics (D7), and (c)
multiple-root screening at query time (D8). The result is **screened, not
proven unique-root safe** — the spec never claims otherwise.

## Goals

1. Land the exact EEP projection (July design, unchanged).
2. Make the adaptive refinement loop measure user-visible quality: validation
   and holdout sampling over the user domain; published query bounds match
   the measured domain.
3. Fix the oversized-headroom defect.
4. Never return a surface with a worse holdout score than the best viable one
   built: fixed-holdout candidate retention, including on mid-loop build
   failure; reject the build when no candidate is viable.
5. Replace noise-driven axis selection with measured-acceptance greedy
   coordinate descent, with well-defined exploration/restart/retention
   states and backend state rollback.
6. Surface honest build diagnostics (`target_met`, achieved errors, picked
   iteration, monotonicity statistics) to C++ and Python users.
7. Screen IV inversion against multi-root brackets and negative vega with a
   uniform 17-point bracket scan (configurable, default on) with an honest,
   explicitly bounded detection contract.
8. Regression coverage for the deterministic q0 bifurcation and the
   wrong-root failure; before/after `interp_iv_safety` evidence.

## Non-goals

- Root-causing the fit degradation under knot insertion or the deterministic
  160-point build failure (follow-up issue; this design only makes them
  non-fatal).
- Monotonicity certificates or shape-constrained (monotone) fitting
  (follow-up issue). Certified-monotone surfaces are the *complete* answer to
  root uniqueness; D8 is an explicit interim screen.
- Carrying refined knot *positions* through the segmented B-spline probe
  aggregation (follow-up issue). This design does fix the segmented retry
  path's unvalidated-return defect (D9).
- Adding a retry stage to the segmented Chebyshev builder (it has none
  today; it gains final validation + the viability gate only).
- Exposing build diagnostics or the `detect_multiple_roots` toggle through
  the C API (ABI stability; revisit on the next planned ABI rev — the C API
  keeps the default-on screen).
- Denoising positive PDE residuals; changing interpolation algorithms or
  coordinate transforms; persisted-table migration (all per the July design).
  `BuildDiagnostics` never enters `PriceTableData`/Parquet serialization.
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
    SurfaceBounds bounds;         // fit domain (support incl. headroom)
    SurfaceBounds sample_bounds;  // user-facing measurement domain
};
```

- `sample_bounds` is the domain the user actually asked for: log-moneyness
  span of the user's strikes/moneyness grid, the user's τ, σ, r ranges
  (after the existing `expand_domain_bounds` minimum-spread widening, which
  is a usability floor, not headroom).
- **Fit-domain construction is backend-specific** and always derives from
  `sample_bounds` plus support extension:
  - B-spline: moneyness support headroom per D3 (expected seeded density);
  - Chebyshev: the existing CC-level-based extension, applied to
    `sample_bounds`. `bounds` reflects the nodes actually supplied to the
    builder.
- `run_refinement` draws **all** validation samples — the per-iteration fresh
  samples and the fixed holdout (D4) — from `sample_bounds`. Error-bin
  normalization also uses `sample_bounds`.
- **Published query bounds = `sample_bounds`.** The surface's advertised
  bounds (what `is_in_bounds` / the wrappers enforce) are the sample domain;
  the fit headroom remains interpolation *support* only and is not
  queryable. This closes the gap where unmeasured headroom was reachable by
  user queries. (Behavior change, documented: queries in the former
  headroom band — which returned oscillating garbage with healthy-looking
  vega — are now rejected by the bounds check.)
- **Bin → interval conversion:** bins are normalized over `sample_bounds`;
  `run_refinement` converts the selected problematic bins into *physical*
  intervals and passes them to `RefineFn` (D6 signature). `RefineFn` no
  longer derives intervals from `grid.front()/back()`.

### D3. Headroom size fix

`spline_support_headroom(domain_width, n_knots)` must receive the **expected
moneyness grid density**, not the user strike count. The expected density is
`max(user_moneyness_knots, params.min_moneyness_points)`; this approximates
the seeded grid size (`seed_grid` may add up to two domain endpoints), which
is acceptable — the quantity controls headroom scale, not an exact support
width. `extract_chain_domain` gains the parameter:

```cpp
std::expected<RefinementContext, PriceTableError>
extract_chain_domain(const OptionGrid& chain, size_t expected_m_knots);
```

The segmented B-spline builder computes headroom with the same rule at
`build_adaptive()` time (when `AdaptiveGridParams` is available), keeping
separate `sample_domain_` / `fit_domain_` members; `create()` no longer
bakes headroom into a single domain.

For the benchmark configs this shrinks moneyness headroom from ±0.31 to
±0.03 log-moneyness.

**Parameter validation** (applies to every adaptive entry point):
`target_iv_error > 0` and finite; `vega_floor > 0` and finite;
`refinement_factor > 1` and finite; `max_iter >= 1`;
`validation_samples >= 8`; violations ⇒
`PriceTableErrorCode::InvalidConfig`.

**Fit-bounds construction is per-backend** (no double headroom):
`extract_chain_domain(chain, expected_m_knots)` builds B-spline fit bounds;
the Chebyshev builders construct their own fit bounds from `sample_bounds`
via their CC-level extension and must not additionally apply the B-spline
headroom.

### D4. Fixed holdout with cached references

**Callback contract.** The current `ComputeErrorFn` hides two FD solves
(vega bumps) inside `make_fd_vega_error_fn`, so references cannot be cached
through the existing signature. The error metric is split:

```cpp
/// Per-point reference data, computed once per holdout point.
struct ErrorRefs {
    double ref_price = 0.0;  // FD American price
    double vega = 0.0;       // FD central-difference American vega
};

/// Produce refs for one point (base solve + two sigma-bump solves).
/// Any failed or non-finite solve => unexpected.
using PrepareRefsFn = std::function<std::expected<ErrorRefs, SolverError>(
    double spot, double strike, double tau, double sigma, double rate)>;

/// Score one point from interpolated price + cached refs. Pure arithmetic.
/// Contract: returns a finite, nonnegative error (loop treats anything else
/// as a non-viable evaluation).
using ScoreErrorFn = std::function<double(
    double interp, const ErrorRefs& refs,
    double spot, double strike, double tau,
    double sigma, double rate)>;
```

`make_fd_vega_error_fn` is replaced by `make_fd_vega_refs_fn`
(a `PrepareRefsFn`) plus `make_iv_score_fn` (a `ScoreErrorFn` carrying the
TV/K filter, vega floor, and the *target-level noise clamp* — the same
arithmetic as today's `compute_iv_error`; note it is a clamp applied only
when the price error is within the noise floor, **not** a global cap, so
catastrophic scores are never capped below the viability bound). `run_refinement`'s signature takes `PrepareRefsFn` and
`ScoreErrorFn` in place of `ValidateFn` + `ComputeErrorFn`; fresh
per-iteration validation uses the same pair (its refs are simply not
reused). `ValidateFn` remains for reference-price generation inside
`make_fd_vega_refs_fn` and for callers that need plain solves.

**Holdout.** Before iteration 0, `run_refinement` generates one holdout set
of `params.validation_samples` points via `latin_hypercube_4d` with seed
`params.lhs_seed ^ 0x484F4C44` ("HOLD"), scaled to `sample_bounds`, and
computes `ErrorRefs` per point once.

- A holdout point is **valid** only if `PrepareRefsFn` succeeded and all
  values are finite. Invalid points are excluded and counted in diagnostics.
- Fewer than `max(4, validation_samples / 4)` valid holdout points ⇒ the
  build fails with `PriceTableErrorCode::ValidationFailed` (a holdout that
  cannot measure cannot certify retention).
- A non-finite interpolated price or score at any valid holdout point makes
  the candidate **non-viable** (D5); it is recorded but never returned.

Holdout score per candidate: `holdout_max` and `holdout_avg` over the valid
points. Per-iteration holdout cost after setup: `validation_samples` surface
interpolations plus arithmetic — no FD solves.

Convergence requires both sets under target:
`fresh_max <= target_iv_error && holdout_max <= target_iv_error`.

### D5. Candidate retention and the viability gate

Each iteration records a candidate `{grids, backend state snapshot (D6),
holdout_max, holdout_avg, error bins, iteration index, viable flag}`.

**Viability (the "reject when every candidate is unsafe" contract, restated
in measurable terms):** a candidate is *viable* iff

- **every evaluation made of it** — every valid holdout point *and* every
  valid fresh-sample point evaluated for that candidate — produced a finite
  interpolated price and a finite, nonnegative score (a NaN on a fresh
  in-domain point disqualifies the candidate even if the fixed holdout
  missed that location; ranking still uses holdout error alone), and
- `holdout_max <= kViabilityBound` with **`kViabilityBound = 0.20`
  (2,000 bps of IV, absolute)**. This is an operational garbage detector,
  not a quality bar: it is independent of `target_iv_error` by design (a
  strict 0.7 bps target must not brand a 50 bps surface "non-viable", and a
  loose target must not admit a 30-percentage-point surface). Calibration
  from the prototype's user-domain measurements: healthy candidates measured
  35–931 bps maxima (all ≤ 0.094); catastrophic candidates measured
  27,683 bps – 2.3×10⁹ bps (all ≥ 2.7). Quality relative to the *target* is
  reported honestly through `target_met` and the achieved errors (D7), and
  ranking picks the best viable candidate regardless of the bound.

**Three candidate roles (distinct by design):**

- **Exploration base:** the best *evaluated* candidate (lowest
  `holdout_max` among candidates with finite scores, viability not
  required; if every candidate is non-finite, the seed grids). Backtracking
  resets (D6) go here, so a non-viable seed can still be refined toward
  viability, and a sub-2% improvement naturally becomes the next base.
- **Walk-restart reference:** the previous best `holdout_max`; only a
  ≥ 2% relative improvement over it clears `tried` (D6).
- **Returned candidate:** the best *viable* candidate (lowest
  `holdout_max`; ties: lowest `holdout_avg`, then earliest iteration).

At loop exit (converged, budget exhausted, all axes exhausted, or build
failure):

- if no candidate is viable, the build fails with new
  `PriceTableErrorCode::NoViableSurface`;
- if the returned candidate is not the surface most recently built, rebuild
  it once via `build_fn` with its grids (deterministic: same grids → same
  surface) so the caller's captured surface state matches the returned
  grids. The rebuild is recorded in diagnostics (`final_rebuild = true`,
  with its own `IterationStats` entry marked `refined_dim = -2` to
  distinguish it) and does not consume `max_iter` budget. If this final
  rebuild fails, the build fails with the rebuild's error — the loop must
  never return grids that do not describe the caller's captured surface;
- `achieved_max_error`/`achieved_avg_error` report the returned candidate's
  holdout numbers; `target_met = (returned holdout_max <= target_iv_error
  && that iteration also satisfied the fresh-sample convergence check)`.

**Mid-loop build failure — exploration continues:** a failed refinement
trial build must not strand exploration (a non-viable seed might only be
recoverable through a *different* axis). Rules:

- **Seed build failure** (iteration 0): propagates as an error.
- **Failed refinement trial** (`build_fn` fails for a refined-grid build):
  mark the attempted axis `tried`, restore the exploration base and its
  backend state, record the attempt in `IterationStats` with
  `build_failed = true` (new field) and `refined_dim` = the attempted axis,
  consume one unit of `max_iter` budget, set
  `build_failure_fallback = true`, and continue at the axis-selection step
  while untried axes and budget remain.
- **Exploration exhausted** (all axes tried or budget consumed): retention
  runs as usual — best viable candidate returned; no viable candidate ⇒
  `NoViableSurface` (the terminal error is `NoViableSurface`, not the
  trial-build error; the failures are visible in `IterationStats`).
- **Final-rebuild failure:** propagates as an error (D5 above).

**Budget edge contracts:** `max_iter == 1` ⇒ the seed is built, evaluated,
recorded, and returned via the same retention path (no refinement).
Convergence at iteration 0 ⇒ `picked_iteration == 0`,
`total_iterations == 1`.

**Factory error surfacing:** `to_validation_error` currently collapses
unrecognized `PriceTableErrorCode`s to `InvalidGridSize`. New mappings:
`PriceTableErrorCode::NoViableSurface` → new
`ValidationErrorCode::NoViableSurface`; `ValidationFailed` → new
`ValidationErrorCode::AdaptiveValidationFailed`. Both wired through the
exhaustive switches (error_types helpers, Python bindings enum + message,
C API `map_validation_error` → `MANGO_ERR_VALIDATION`), appended at the end
of their enums (stable existing values).

Best-effort semantics (user decision): a viable-but-unmet-target result is
returned successfully with `target_met = false` and achieved errors visible
via D7.

### D6. Refinement selection: greedy coordinate descent with a measured
walk-restart

Replaces single-axis `worst_dimension()` selection. State: `tried[4]`
(axes attempted since the last walk restart).

**RefineFn signature** (new; replaces the ErrorBins-coupled form):

```cpp
struct RefineOutcome {
    bool changed = false;  // grids actually changed
    int changed_dim = -1;  // the axis that actually changed (may differ
                           // from the requested axis only if the backend
                           // documents redirection; see below)
};

/// Focus intervals are physical coordinates within sample_bounds (D2).
using RefineFn = std::function<RefineOutcome(
    size_t requested_dim,
    std::span<const std::pair<double, double>> focus_intervals,
    std::vector<double>& moneyness,
    std::vector<double>& tau,
    std::vector<double>& vol,
    std::vector<double>& rate)>;
```

- The B-spline refiner must return `changed = false` when the requested
  axis is at `max_points_per_dim` or no midpoint is insertable (today it
  returns success unconditionally — required fix), and never redirects.
- The Chebyshev refiners currently redirect the requested dimension to a
  lowest-level dimension under a balance rule; under coordinate descent
  that fallback is **removed** — the refiner advances exactly the requested
  axis or returns `changed = false` (axis at its level cap).
  `changed_dim`, not the requested axis, is what the loop's bookkeeping
  records (defense in depth; with redirection removed they always match).

**Backend state rollback.** Restoring grid vectors is not sufficient for
Chebyshev (its refiners advance per-axis level counters held outside the
grids). `run_refinement` gains optional state hooks:

```cpp
struct RefineStateHooks {
    std::function<std::shared_ptr<const void>()> snapshot;  // opaque
    std::function<void(const std::shared_ptr<const void>&)> restore;
};
```

Backends with external refinement state (Chebyshev level counters) provide
both hooks; the B-spline paths pass none (grids are the whole state). A
candidate's record includes the snapshot taken at its evaluation; the
backtracking reset (step 3) restores it together with the grids. Restoring
then re-refining a previously rejected axis after a walk restart must yield
the correct next level (test-pinned).

Per refinement step:

1. Score each axis d over the **exploration base's** recorded error bins:
   `score[d] = concentration[d]` — the max-bin fraction, defined as `0`
   when the bin total is zero (possible when fresh samples met target but
   the holdout did not). Ties, including the all-zero case, break by
   dimension order (moneyness, τ, σ, r); with empty bins the focus
   intervals passed to `refine_fn` are empty, which the refiner treats as
   uniform refinement. (`dim_error_mass` is identical across dimensions by
   construction and is dropped from the score; concentration is documented
   as a *proposal-ordering heuristic only*. The measured walk-restart, not
   the score, decides what is kept.)
2. Pick the highest-scoring axis with `tried[d] == false`. If none remain,
   stop (all axes exhausted; retention picks the final result).
3. Reset the working grids **and backend state** to the exploration base's
   (D5 roles), then call `refine_fn(picked_dim, focus_intervals, ...)` with
   the physical focus intervals from the base's bins (D2).
4. If `refine_fn` returns `changed = false`, mark `tried[d] = true`
   immediately **without consuming a build** and continue at step 2.
5. After the next iteration's build + evaluation: if
   `holdout_max < prev_best_holdout_max * (1 - kMinRelImprovement)` with
   `kMinRelImprovement = 0.02`, the walk **restarts**: clear `tried`.
   Otherwise mark `tried[changed_dim] = true`. Either way the candidate is
   recorded; if it improved the best evaluated score at all (even < 2%), it
   becomes the next exploration base (D5 roles) — the 2% threshold governs
   only the axis-walk restart, not which candidate exploration builds on.

Budget: `max_iter` continues to bound total built iterations. Its default
rises from 5 to 8 (comment in `adaptive_grid_types.hpp` updated) so a full
4-axis backtracking walk plus accepted improvements fits (prototype: the
rate axis was reached at iteration 4 of 5, leaving no budget to exploit the
improvement). Cost note: iterations beyond the old default occur only when
earlier iterations failed to converge, and each costs one table build plus
fresh-validation FD solves — the holdout adds no per-iteration FD cost (D4).

The concentration-threshold multi-axis mechanism and the standalone
trial-build tie-break from the earlier draft are **not** implemented (P4).

### D7. Build diagnostics surfaced to the user

New struct in `adaptive_grid_types.hpp`:

```cpp
struct BuildDiagnostics {
    bool target_met = false;
    double achieved_max_error = 0.0;   // holdout, returned candidate
    double achieved_avg_error = 0.0;
    size_t picked_iteration = 0;
    size_t total_iterations = 0;       // built iterations, excl. final rebuild
    bool final_rebuild = false;        // picked != last, rebuilt once
    bool build_failure_fallback = false;
    size_t holdout_points = 0;         // valid holdout references
    size_t holdout_points_invalid = 0;
    size_t monotonicity_violations = 0;   // returned candidate, see below
    size_t monotonicity_points_invalid = 0;  // non-finite scan prices
    double worst_vega_slope = 0.0;        // most negative dPrice/dσ observed
    std::vector<IterationStats> iterations;  // final rebuild: refined_dim = -2
};
```

`IterationStats` gains `bool build_failed = false` (failed refinement
trials, D5); the final rebuild's entry uses `refined_dim = -2`.

**Monotonicity statistics** (diagnostics only, never a gate): on the
returned candidate, at each valid holdout `(m, τ, r)`, scan 7 equally spaced
σ across the user σ-range (`sample_bounds`). Count a violation when
`price(σ_{k+1}) < price(σ_k) − max(1e-8 * spot, price_tol)` where
`price_tol = target_iv_error * vega_floor` (the same noise floor the error
metric uses); the reported slope is `Δprice / Δσ` for the worst violating
step. Non-finite prices increment `monotonicity_points_invalid`, not
violations. Degenerate σ-range (σ_min == σ_max) ⇒ scan skipped, zero
violations.

**Ownership path.** Adaptive builder results (`BSplineAdaptiveResult`,
`BSplineSegmentedAdaptiveResult`, Chebyshev equivalents) carry a
`BuildDiagnostics` **and the `sample_bounds`**; the factory's adaptive
wrapping supplies the published surface bounds from the carrier's
`sample_bounds` instead of deriving them from the spline/leaf knots (which
span the fit domain) — this is how D2's published-bounds rule reaches the
constructed surfaces on every adaptive path. Internal factory helpers
return `{table, diagnostics}` carriers; `AnyPriceTable::Impl` and `AnyInterpIVSolver::Impl` each gain a
separate `std::optional<BuildDiagnostics>` member (`nullopt` for manual
builds and Parquet-loaded tables); `AnyPriceTable::make_iv_solver` copies it
into the solver wrapper. `to_data()` and all serialization visit only the
table — diagnostics never enter `PriceTableData`/Parquet.

**Language surface:** C++ accessor `build_diagnostics()` on both wrappers;
Python property `build_diagnostics` (dict, via pybind11) on `PriceTable` and
`InterpolatedIVSolver`; the Python solver config also gains
`detect_multiple_roots` (D8). C API: deferred (Non-goals).

**Existing factory gap, made explicit:** the continuous (non-segmented)
Chebyshev factory path currently ignores `config.adaptive` and always uses
the fixed builder; that behavior is unchanged in this branch and its
diagnostics are `nullopt` (documented).

**Segmented builders:** the diagnostics exposed by a segmented table
describe the **final assembled surface** (D9); the per-probe loop
diagnostics are appended to `iterations` for forensics.

### D8. IV inversion screen (query time)

In `InterpolatedIVSolver::solve` (src/option/interpolated_iv_solver.hpp):

1. **Signed vega pre-check.** The existing pre-check takes `std::abs(vega)`
   at three fixed probe vols, so a strongly negative vega counts as healthy
   and probes may fall outside the bracket. Changes: probe vols become the
   quartile points of the *actual* bracket `[sigma_min, sigma_max]`
   (25% / 50% / 75%); the check uses the **signed** maximum, initialized to
   `-infinity`; non-finite probe vega ⇒ `NumericalInstability`. Reject with
   `VegaTooSmall` when `max_signed_vega < vega_threshold`.
2. **Bracket screen** (new, config `detect_multiple_roots`, default
   `true`): before Brent, evaluate the objective at **17 equally spaced σ**
   in `[sigma_min, sigma_max]` (endpoints included; uniform bracket/16
   spacing — no selective subdivision, whose detection guarantee round 2
   showed to be unsound).
   - **Zero tolerance:** the objective is a *price* difference; samples
     with `|objective| <= zero_tol`, `zero_tol = 1e-9 * spot`, are
     *zeros* (a dollar tolerance, deliberately distinct from
     `config_.tolerance`, which the root finder uses for both its interval
     and objective convergence criteria). Consecutive zeros collapse into
     one zero run. A zero run spanning **all 17 samples** is an unresolved
     continuum of roots ⇒ `MultipleRoots` (`final_error = 0`,
     `last_vol = sigma_min`).
   - **Transition counting:** transitions are counted between nonzero
     samples of opposite sign; a zero run *between* opposite signs counts
     as exactly one transition. A zero run flanked by the *same* sign on
     both sides is a tangency contact ⇒ `MultipleRoots` (an
     even-multiplicity contact means root selection is ambiguous). A zero
     run at a bracket *endpoint* flanked inward by a single nonzero sign is
     a boundary root: with no other transition, return that endpoint σ
     directly **only if the endpoint also satisfies the solver's configured
     convergence tolerance** (`|objective| <= config_.tolerance` — a user's
     tighter tolerance is never silently loosened by `zero_tol`); otherwise
     report `BracketingFailed` as the un-screened path would (the scan
     found no true bracket). With any other transition ⇒ `MultipleRoots`.
   - More than one transition ⇒ `IVErrorCode::MultipleRoots`
     (`final_error` = transition count, `last_vol` = the low-σ endpoint of
     the lowest transition interval — an interval bound, not a root).
   - Exactly one transition ⇒ run Brent on that subinterval (narrowed
     bracket; offsets part of the scan cost). After convergence, a
     **post-hoc slope check**: the objective FD slope across the narrowed
     interval's scan endpoints must be positive; otherwise
     `MultipleRoots`.
   - No transition ⇒ fall through to Brent on the full bracket, which
     reports `BracketingFailed` exactly as today.
   - Non-finite objective at any scan point ⇒ `NumericalInstability`.
   - **Honest contract (documented in the header):** this is a *screen*,
     not a proof of uniqueness. Guarantee: any objective sign excursion
     spanning at least one bracket/16 cell is detected, as is tangency
     within `zero_tol` at a scan point. Folds narrower than one cell that
     also evade the post-hoc slope check can pass. Certified-monotone
     surfaces (follow-up issue) are the complete solution; the acceptance
     criterion for this branch is the screen contract, not absolute
     uniqueness.
   - Cost: 17 surface evals (~4 μs) against a ~3.5 μs solve;
     `detect_multiple_roots = false` restores today's path unchanged.
3. **Error-code wiring:** `MultipleRoots` appended at the end of
   `IVErrorCode` (src/support/error_types.hpp) with message/helper switch
   arms; Python bindings enum + message (src/python/mango_bindings.cpp);
   C API `map_iv_error` (src/ffi/mango_c_api.cpp) maps it to
   **`MANGO_ERR_BRACKETING`** (the `BracketingFailed` category — verified
   against the existing mapping). `iv_result.hpp` needs no change (it maps
   validation errors only). The C config struct is **not** extended (ABI;
   Non-goals) — C API callers get the default-on screen.

### D9. Callers and shared-path coverage

`run_refinement` is the single shared loop; D2/D4/D5/D6 changes apply to all
adaptive builders. Per-caller contracts (they differ — the two segmented
implementations do **not** share an architecture):

- **`build_adaptive_bspline` (4D single-K_ref):** captures `last_spline` by
  reference in `build_fn`; the D5 final-rebuild rule keeps `last_spline`
  consistent with the returned grids. Gains the D3 headroom fix; passes no
  state hooks. Published bounds = `sample_bounds` (D2).
- **`build_adaptive_chebyshev` (4D, continuous):** passes state hooks for
  its level counters (D6); its refiner's balance-rule redirection is
  removed (D6). The caller's existing unconditional extra `build_fn` call
  after `run_refinement` is removed — it consumes the loop's captured
  surface directly (the loop now guarantees last-built == returned).
- **`build_adaptive_bspline_segmented` (multi-K_ref probes):** per-probe
  loops gain retention/backtracking automatically (probe *size* selection
  improves; probe-loop build failures with viable candidates are
  non-fatal). Aggregation to uniform final grids is unchanged (Non-goals).
  **Probe measurement is band-scoped** (execution amendment, 2026-08-29):
  each probe's references are solved at the probe's own scaled coordinates
  (`scale = strike / K_ref`, spot and price/vega scaled together, same
  dividend schedule), and each probe is measured only over the strike band
  it dominates in the assembly — geometric midpoints to its neighbouring
  K_refs (log-spaced), outer bands extending to the user strike range, a
  single K_ref serving the whole range. A probe whose band is empty is
  skipped (`IterationStats.refined_dim = -3`); a degenerate band is
  minimally widened. Rationale: the assembled surface blends bracketing
  K_refs by strike, so probe error at strikes a probe never dominates is
  not user-observable; measuring it gated builds on noise. The assembled
  surface's own final validation below remains the safety authority.
  The **final assembled surface** contract:
  1. final validation (LHS over the **sample domain**, D2) computes
     `ErrorRefs` per point once (`PrepareRefsFn`); points follow D4's
     validity rules; fewer than `max(4, validation_samples / 4)` valid
     points ⇒ `ValidationFailed`;
  2. the original assembled surface is scored on those cached refs; if its
     final max error exceeds the target **or it fails D5 viability**, the
     existing bumped-grid retry is built and scored **on the same cached
     refs** (identical coordinates — no second reference generation);
  3. D5 viability (finite scores + `kViabilityBound`) applies to both; the
     builder returns the lowest-error **viable** one (`used_retry` set
     accordingly); both non-viable ⇒ `NoViableSurface`. Today's behavior —
     returning the retry surface unvalidated with the pre-retry error
     numbers — is a defect under Goal 4 and is fixed;
  4. the surface's `BuildDiagnostics` (D7) describes the returned final
     surface (achieved errors from this final validation, plus a
     monotonicity scan of it); `target_met` reflects the returned
     surface's final max error.
- **`build_adaptive_chebyshev_segmented`:** a single refinement loop at
  `K_ref = spot` (no probes, no retry — none is added). Gains: sizing-loop
  retention via the shared loop, then the assembled all-K_ref surface gets
  **mandatory final validation + the viability gate** exactly as steps 1/3/4
  above (minus the retry). Previously the assembled surface was returned
  without final validation.

The Chebyshev exact-grids mode (`InitialGrids::exact`) keeps CGL node
placement through the state hooks: restore returns the node set and level
counters verbatim.

## Decisions

Brainstorm Q&A (2026-08-28/29, user choices) and prototype/review-driven
revisions:

1. **Scope: full expanded scope** (all checklist items in one branch) —
   chosen over "projection + safety net" and smaller splits. Prototype
   evidence subsequently *narrowed the mechanism list* (multi-axis and
   standalone tie-break dropped as valueless — P4) while adding root-cause
   fixes (D2, D3) not in the original checklist.
2. **Refinement selection: "both multi-axis + measured tie-break"** — the
   prototype showed the multi-axis trigger never fires and the tie-break is
   subsumed by measured acceptance; the user approved the revised greedy
   backtracking design (2026-08-29) as the implementation of "measured
   improvement". Review round 2 sharpened it into the three-role model
   (exploration base = best evaluated; 2% = walk-restart threshold only;
   returned = best viable), which also lets a non-viable seed be refined
   toward viability.
3. **Unmet-but-safe target: best-effort + surfaced diagnostics** — no hard
   error, no strict-mode flag (D5, D7). The issue's "reject when every
   candidate is unsafe" item is honored through the *viability gate*:
   finite scores + an **absolute** `kViabilityBound = 0.20` IV-error bound
   (an operational garbage detector, deliberately independent of the
   accuracy target) + `NoViableSurface`. Violation-count gating measurably
   fails to discriminate (P5); holdout error separates healthy from
   catastrophic by ≥ 30× in every prototype measurement.
4. **IV defense: bracket scan, configurable, on by default** — revised in
   review rounds 1–2 to a **uniform 17-point screen** (selective
   subdivision's guarantee was unsound) with defined zero/tangency/endpoint
   semantics, a post-hoc slope check, and an explicitly documented
   screen-not-proof contract. Post-hoc-only checking remains rejected
   (cannot detect a wrong root on a locally positive slope). Absolute
   uniqueness is delivered by the certified-monotone follow-up, not this
   branch.
5. **Rigor level: prototype first** — executed; findings above.
   Certificate / shape-constrained fitting deferred to a follow-up issue;
   the correct statement is that *unconstrained* least-squares cannot be
   expected to be monotone where vega → 0, so gating on it is inappropriate
   (constrained fitting remains possible and is the follow-up).
6. **Retention criterion: holdout error, not monotonicity violations** (P5),
   with the D5 viability gate as the rejection mechanism.
7. **Walk-restart threshold 2% relative** — from P4's budget-burn
   observation; sub-threshold improvements still advance the exploration
   base (round-2 fix), they just don't reopen tried axes.
8. **`max_iter` default 5 → 8** — required for the backtracking walk to
   both reach and exploit late-axis improvements (D6).
9. **Diagnostics surface: C++ + Python; C API deferred** — the C structs
   are frozen this branch for ABI stability. Python gains the
   `detect_multiple_roots` config field and the diagnostics property.
10. **Published bounds = sample domain** (round-2 fix): the headroom band
    is interpolation support, not queryable surface.

## Test design

### Unit: run_refinement (synthetic callbacks)

The loop is callback-driven; tests use synthetic `BuildFn` / `PrepareRefsFn`
/ `ScoreErrorFn` / `RefineFn` (D4/D6 signatures) that require no PDE solves:

- **Retention picks best viable:** builds whose holdout error degrades after
  iteration k → returned grids are iteration k's; `picked_iteration == k`;
  `final_rebuild == true` and one extra `build_fn` call with the picked
  grids; the rebuild's `IterationStats` entry has `refined_dim == -2` and
  `total_iterations` excludes it.
- **Viability gate:** all candidates above `kViabilityBound` →
  `NoViableSurface`; one candidate non-finite at a holdout point → never
  returned even with the lowest score.
- **Non-viable seed exploration:** seed above `kViabilityBound`, refinement
  brings a later candidate under it → success returning the later
  candidate (regression for round-2 critical #1).
- **Sub-threshold base advance:** a 0.5% improvement does not clear `tried`
  but does become the next exploration base (verified via the grids the
  next `refine_fn` call receives); a 5% improvement clears `tried`.
- **Build-failure exploration:** `build_fn` fails for axis 0's trial while
  axis 2 can still produce the first viable candidate → the loop marks
  axis 0 tried, restores the base, continues, and succeeds via axis 2
  (`build_failure_fallback == true`, the failed attempt in
  `IterationStats` with `build_failed == true`); failure at iteration 0 →
  error propagates; all axes failed/no-op with no viable candidate →
  `NoViableSurface`; final-rebuild failure → error propagates.
- **Fresh-sample viability:** a candidate returning NaN on one fresh sample
  but finite on the whole holdout is non-viable and never returned.
- **Backtracking walk:** an error field reducible only along axis 2 → axes
  with higher concentration scores are tried, rejected, and axis 2's
  improvement restarts the walk.
- **No-op refinement:** `refine_fn` returning `changed = false` marks the
  axis tried without consuming a build; all axes no-op → loop stops before
  `max_iter` with the seed retained.
- **State hooks:** snapshot/restore invoked around each backtracking reset;
  a Chebyshev-style counter restored after a rejected trial yields the
  correct next level when the axis is retried post-restart.
- **Convergence requires holdout:** fresh samples under target but holdout
  above → not converged.
- **Holdout caching:** count `PrepareRefsFn` invocations — exactly
  holdout-setup + fresh-validation counts (no per-iteration holdout
  re-preparation).
- **Holdout validity:** failed refs excluded and counted; below the
  minimum-valid threshold → `ValidationFailed`; `validation_samples < 8`,
  `max_iter == 0`, `target_iv_error <= 0`, `refinement_factor <= 1` →
  `InvalidConfig`; `max_iter == 1` → seed retained.
- **Sampling domain:** all sample coordinates within `sample_bounds`; bins
  handed to `refine_fn` as physical intervals inside `sample_bounds`.

### Unit: headroom, domain, and bounds

- `extract_chain_domain(chain, 60)` headroom equals `3 * width / 59` (not
  `3 * width / (n_strikes - 1)`).
- `sample_bounds` equals user ranges (with minimum-spread widening);
  `bounds` equals `sample_bounds` plus headroom on moneyness.
- Published surface bounds equal `sample_bounds`: a query at
  `sample_bounds.m_min - ε` (inside the old headroom band) is rejected by
  `is_in_bounds`.
- Segmented B-spline: headroom applied at `build_adaptive()` with the D3
  rule; final validation samples drawn from the sample domain.

### Unit: IV screen

- Synthetic surface with three crossings inside the bracket →
  `MultipleRoots`; the same query with `detect_multiple_roots = false`
  reproduces today's (wrong-root) behavior — documenting the defended
  failure.
- **Fold spanning ≥ one bracket/16 cell** inside one former 9-point
  interval → detected (`MultipleRoots`) by the 17-point scan.
- **Tangency:** objective touching zero (within `zero_tol`) at a scan
  point, same sign both sides → `MultipleRoots`.
- **Endpoint root:** monotone objective with its only zero run at
  `sigma_min` and `|objective| <= config tolerance` → returns `sigma_min`;
  same shape with a *tighter* configured tolerance than the residual →
  `BracketingFailed`, never a silently loose success; endpoint zero plus an
  interior transition → `MultipleRoots`.
- **All-zero scan:** objective within `zero_tol` at all 17 samples →
  `MultipleRoots` with `final_error == 0`.
- Monotone surface → identical root with and without the screen (narrowed
  bracket converges to the same σ within tolerance); scan cost exactly 17
  evals.
- Post-hoc slope check: converged root whose narrowed-interval scan slope
  is negative → `MultipleRoots`.
- Signed pre-check: probe vegas {−5, −3, −1}, threshold 0.5 →
  `VegaTooSmall`; {−5, +2, +1} → passes to the screen; probes evaluated at
  bracket quartiles (regression: bracket narrower than [0.10, 0.50]);
  non-finite probe → `NumericalInstability`.
- Error-code wiring: `MultipleRoots`, `NoViableSurface`,
  `AdaptiveValidationFailed` present in Python enums with messages; C API
  maps `MultipleRoots` → `MANGO_ERR_BRACKETING` and the new validation
  codes → `MANGO_ERR_VALIDATION` (mapping tests per the #449 pattern).

### Integration / regression

- **q0 bifurcation regression:** adaptive B-spline build on a scaled-down
  q = 0 PUT config (reduced grids/samples for test budget) asserts: build
  succeeds; `build_diagnostics()` present with sane `achieved_max_error`
  (≤ 100 bps); and **IV round-trips through the returned solver** at the
  known wrong-root region (σ = 30%, T = 30d, K = 80 analog) recover the
  true vol within tolerance or return `MultipleRoots` — never a spurious
  low root. This fails against the pre-fix loop.
- **Segmented retry regression:** a config that forces the retry path →
  the returned surface's diagnostics describe the returned surface; a retry
  that scores worse than the original on the shared refs is not returned;
  an original that meets a loose target but fails viability still triggers
  the retry.
- **Segmented Chebyshev final validation:** assembled surface exceeding
  `kViabilityBound` on final validation → `NoViableSurface` (previously
  returned silently).
- **Projection tests** from the July design (already on branch).
- **Diagnostics plumbing:** factory-built adaptive solver exposes
  diagnostics (C++ and Python); manual build, continuous-Chebyshev factory
  path, and Parquet round-trip return `nullopt`; serialized bytes contain
  no diagnostics fields.

### Benchmark evidence (acceptance, not CI)

Before/after `interp_iv_safety` for `--path=q0`, `--path=bspline`,
`--path=dividends`:

- q0, σ = 30%, TV/K ≥ 1e-4 must improve from 289.3 bps to ≤ 10 bps
  (prototype: 5.1);
- no path deteriorates beyond the benchmark's 0.1 bps reported precision on
  TV/K ≥ 1e-3 metrics without a written explanation in the PR;
- the vanilla σ = 30% slice is the known watch item (vol-knot phase; the D6
  budget change is expected to close it); record and explain the final
  numbers either way.

## Acceptance criteria

1. July design's projection criteria (all — see that document).
2. Validation and holdout sampling occur strictly within the user domain;
   headroom no longer scales with user knot count; published query bounds
   equal the sample domain.
3. The adaptive loop never returns a surface with a worse holdout score than
   the best viable candidate it built; a non-viable seed can still be
   refined toward viability; mid-loop build failure after a viable candidate
   exists is non-fatal; no viable candidate ⇒ `NoViableSurface`, surfaced
   intact through the factory error mapping.
4. The axis walk restarts only on ≥ 2% relative holdout improvement;
   sub-threshold improvements advance the exploration base without
   restarting the walk; backend state is restored on backtracking; no-op
   refinements consume no build; the loop stops when all axes are
   exhausted.
5. `build_diagnostics()` available on adaptive-built tables/solvers (C++
   and Python) with honest `target_met` and achieved errors; absent from
   serialization; segmented diagnostics describe the returned final
   surface; the segmented B-spline retry is returned only when it scores
   better on shared references; segmented Chebyshev gains final validation
   + the viability gate.
6. The D8 screen contract holds: ≥ one-cell sign excursions, tangency at
   scan points, and multi-transition brackets return `MultipleRoots`;
   boundary roots return the endpoint; single-transition brackets Brent on
   the narrowed subinterval with a positive post-hoc slope; the screen is
   removable by config (C++/Python); signed vega pre-check on in-bracket
   quartile probes. The documented contract is screen-not-proof.
7. All error-code surfaces (C++, Python, C API mapping) handle
   `MultipleRoots`, `NoViableSurface`, and `AdaptiveValidationFailed`;
   C structs unchanged; enum values appended, existing values stable.
8. Full test suite, benchmarks build, Python bindings build; benchmark
   evidence recorded per above.

## Follow-up issues (to file at merge)

1. Fit degradation under knot insertion: clustered midpoint insertion
   degrades and eventually breaks the B-spline fit (deterministic failure at
   160 points). Root-cause and fix (knot spacing constraints, regularization,
   or kink-aware placement near the exercise boundary).
2. Monotonicity certificate (B-spline derivative-coefficient bound coupled
   with per-interval European vega lower bounds) and/or shape-constrained
   σ-axis fitting; certified surfaces may skip the query-time screen — this
   is the complete answer to root uniqueness that D8 explicitly is not.
3. Segmented B-spline probe path discards refined knot positions
   (aggregates sizes only, rebuilds uniform) — evaluate carrying positions.
4. Sparse explicit K_ref grids accuracy (parked from PR #449 work).
5. C ABI rev: expose `BuildDiagnostics` and `detect_multiple_roots` through
   the C API.
6. Continuous Chebyshev factory path ignores `config.adaptive` (documented
   as unchanged in this branch) — wire it or remove the config surface.
