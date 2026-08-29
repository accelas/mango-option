# Exact EEP Projection + Adaptive Refinement Safety Design

**Issue:** #434 (expanded scope per issue comment of 2026-07-23)

**Status:** Draft (design review round 2)

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
The defenses are therefore (a) a *catastrophe gate* on holdout error at build
time (D5), (b) monotonicity statistics as diagnostics (D7), and (c)
multiple-root detection at query time (D8).

## Goals

1. Land the exact EEP projection (July design, unchanged).
2. Make the adaptive refinement loop measure user-visible quality: validation
   and holdout sampling over the user domain.
3. Fix the oversized-headroom defect.
4. Never return a surface with a worse holdout score than the best viable one
   built: fixed-holdout candidate retention, including on mid-loop build
   failure; reject the build when no candidate is viable.
5. Replace noise-driven axis selection with measured acceptance
   (greedy backtracking) with well-defined backend state rollback.
6. Surface honest build diagnostics (`target_met`, achieved errors, picked
   iteration, monotonicity statistics) to C++ and Python users.
7. Defend IV inversion against multi-root brackets and negative vega with a
   bounded, vega-aware bracket screen (configurable, default on) and honest
   documentation of what it can and cannot detect.
8. Regression coverage for the deterministic q0 bifurcation and the
   wrong-root failure; before/after `interp_iv_safety` evidence.

## Non-goals

- Root-causing the fit degradation under knot insertion or the deterministic
  160-point build failure (follow-up issue; this design only makes them
  non-fatal).
- Monotonicity certificates or shape-constrained (monotone) fitting
  (follow-up issue). Certified-monotone surfaces are the *complete* answer to
  root uniqueness; D8 is an explicit interim screen.
- Carrying refined knot *positions* through the segmented multi-K_ref probe
  aggregation (follow-up issue). This design does fix the segmented retry
  path's unvalidated-return defect (D9) because it directly contradicts
  Goal 4.
- Exposing build diagnostics or the `detect_multiple_roots` toggle through
  the C API (ABI stability; revisit on the next planned ABI rev — the C API
  keeps the default-on scan).
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
    SurfaceBounds bounds;         // fit domain (with headroom) — unchanged
    SurfaceBounds sample_bounds;  // user-facing measurement domain
};
```

- `sample_bounds` is the domain the user actually asked for: log-moneyness
  span of the user's strikes/moneyness grid, the user's τ, σ, r ranges
  (after the existing `expand_domain_bounds` minimum-spread widening, which
  is a usability floor, not headroom).
- `extract_chain_domain` and the segmented domain construction populate both
  members; headroom is applied only to `bounds`. For the segmented builders
  this requires keeping separate `sample_domain_` and `fit_domain_` members
  and computing the moneyness headroom at `build_adaptive()` time (when
  `AdaptiveGridParams` is available), not in `create()` (D3).
- `run_refinement` draws **all** validation samples — the per-iteration fresh
  samples and the fixed holdout (D4) — from `sample_bounds`. Error-bin
  normalization also uses `sample_bounds`.
- **Bin → interval conversion:** because bins are normalized over
  `sample_bounds` but grids span the fit domain, `run_refinement` converts
  the selected problematic bins into *physical* intervals (in
  `sample_bounds` coordinates) and passes those intervals to `RefineFn`.
  `RefineFn` no longer derives intervals from `grid.front()/back()`.
- The fit still covers `bounds`; the surface's queryable range is unchanged
  by this item (see D3 for the headroom size fix). Knots outside
  `sample_bounds` are intentionally unmeasured: the holdout ranks
  user-domain accuracy only, which is the objective.

### D3. Headroom size fix

`spline_support_headroom(domain_width, n_knots)` must receive the **expected
moneyness grid density**, not the user strike count. The expected density is
`max(user_moneyness_knots, params.min_moneyness_points)`; this is an
approximation of the seeded grid size (`seed_grid` may add up to two domain
endpoints), which is acceptable — the quantity controls headroom scale, not
an exact support width. `extract_chain_domain` gains the parameter:

```cpp
std::expected<RefinementContext, PriceTableError>
extract_chain_domain(const OptionGrid& chain, size_t expected_m_knots);
```

The segmented builders compute headroom with the same rule at
`build_adaptive()` time (D2).

For the benchmark configs this shrinks moneyness headroom from ±0.31 to
±0.03 log-moneyness. Behavior change: queries in the removed band — which
previously returned oscillating garbage with healthy-looking vega — are now
rejected by the existing bounds check. This is deliberate and documented.

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
using ScoreErrorFn = std::function<double(
    double interp, const ErrorRefs& refs,
    double spot, double strike, double tau,
    double sigma, double rate)>;
```

`make_fd_vega_error_fn` is replaced by `make_fd_vega_refs_fn`
(a `PrepareRefsFn`) plus `make_iv_score_fn` (a `ScoreErrorFn` carrying the
TV/K filter, vega floor, and cap — the same arithmetic as today's
`compute_iv_error`). `run_refinement`'s signature takes `PrepareRefsFn` and
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
- `params.validation_samples < 8` with adaptive refinement ⇒
  `InvalidConfig` (a 1-point holdout is not a safety signal).
- A non-finite interpolated price at any valid holdout point makes the
  candidate **non-viable** (D5); it is recorded but never picked.

Holdout score per candidate: `holdout_max` and `holdout_avg` over the valid
points. Per-iteration holdout cost after setup: `validation_samples` surface
interpolations plus arithmetic — no FD solves.

Convergence now requires both sets under target:
`fresh_max <= target_iv_error && holdout_max <= target_iv_error`.

### D5. Candidate retention and the viability gate

Each iteration records a candidate `{grids, backend state snapshot (D6),
holdout_max, holdout_avg, error bins, monotonicity stats, iteration index,
viable flag}`.

**Viability (the "reject when every candidate is unsafe" contract, restated
in measurable terms):** a candidate is *viable* iff

- every valid holdout point produced a finite interpolated price and finite
  score, and
- `holdout_max <= kCatastrophicMultiple * target_iv_error`, with
  `kCatastrophicMultiple = 50` (2 bps target ⇒ 100 bps bound; prototype
  calibration: healthy seeds measured 35–102 bps *maximums* under a 2 bps
  target only in the q0 config's hardest corner — the bound is deliberately
  generous; catastrophic candidates measured 10³–10⁹ bps).

At loop exit (converged, budget exhausted, all axes exhausted, or build
failure):

- pick the **viable** candidate with the lowest `holdout_max` (ties: lowest
  `holdout_avg`, then earliest iteration);
- if no candidate is viable, the build fails with new
  `PriceTableErrorCode::NoViableSurface`;
- if the picked candidate is not the surface most recently built, rebuild it
  once via `build_fn` with the picked grids (deterministic: same grids →
  same surface) so the caller's captured surface state matches the returned
  grids. The rebuild is recorded in `IterationStats` (it is a real build)
  but does not consume `max_iter` budget. If this final rebuild fails, the
  build fails with the rebuild's error — the loop must never return grids
  that do not describe the caller's captured surface;
- `achieved_max_error`/`achieved_avg_error` report the picked candidate's
  holdout numbers; `target_met = (picked holdout_max <= target_iv_error
  && the picked iteration also satisfied the fresh-sample convergence
  check)`.

**Mid-loop build failure:** if `build_fn` fails and at least one viable
candidate exists, the loop stops and falls back to the retained best
(recording `build_failure_fallback = true`). A failure with no viable
candidates (including the seed build) propagates as an error.

**Budget edge contracts:** `max_iter == 0` ⇒ `InvalidConfig`.
`max_iter == 1` ⇒ the seed is built, evaluated, recorded, and returned via
the same retention path (no refinement). Convergence at iteration 0 ⇒
`picked_iteration == 0`, `total_iterations == 1`.

Best-effort semantics (user decision): a viable-but-unmet-target result is
returned successfully with `target_met = false` and achieved errors visible
via D7.

### D6. Refinement selection: greedy backtracking with measured acceptance

Replaces single-axis `worst_dimension()` selection. State: `tried[4]`
(axes attempted since the last accepted improvement).

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
then re-refining a previously rejected axis after an acceptance must yield
the correct next level (test-pinned).

Per refinement step:

1. Score each axis d over the **best viable candidate's** recorded error
   bins: `score[d] = concentration[d]` — the max-bin fraction. (The current
   `dim_error_mass` is identical across dimensions by construction, so it
   adds no cross-axis information; concentration is documented as a
   *proposal-ordering heuristic only*. Measured acceptance, not the score,
   decides what is kept.)
2. Pick the highest-scoring axis with `tried[d] == false`. If none remain,
   stop (all axes exhausted; retention picks the final result).
3. Reset the working grids **and backend state** to the best viable
   candidate's, then apply `refine_fn` for the picked axis with the
   physical focus intervals from that candidate's bins (D2).
4. `refine_fn` must return `true` only if the grids actually changed. A
   no-op (axis at `max_points_per_dim`, or no insertable midpoint) returns
   `false`; the loop marks `tried[d] = true` immediately **without
   consuming a build**, and continues at step 2. (The current B-spline
   `refine_fn` returns `true` unconditionally — this is a required fix.)
5. After the next iteration's build + evaluation: if
   `holdout_max < best_holdout_max * (1 - kMinRelImprovement)` with
   `kMinRelImprovement = 0.02`, the step is **accepted**: clear `tried`.
   Otherwise mark `tried[d] = true`. The candidate is recorded and remains
   retention-eligible either way.

Budget: `max_iter` continues to bound total built iterations. Its default
rises from 5 to 8 so a full 4-axis backtracking walk plus accepted
improvements fits (prototype: the rate axis was reached at iteration 4 of 5,
leaving no budget to exploit the improvement). Cost note: iterations beyond
the old default occur only when earlier iterations failed to converge, and
each costs one table build plus fresh-validation FD solves — the holdout
adds no per-iteration FD cost (D4).

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
    size_t total_iterations = 0;       // built iterations, excl. final rebuild
    bool final_rebuild = false;        // picked != last, rebuilt once
    bool build_failure_fallback = false;
    size_t holdout_points = 0;         // valid holdout references
    size_t holdout_points_invalid = 0;
    size_t monotonicity_violations = 0;  // picked candidate, see below
    double worst_vega_slope = 0.0;       // most negative dPrice/dσ observed
    std::vector<IterationStats> iterations;
};
```

**Monotonicity statistics** (diagnostics only, never a gate): on the picked
candidate, at each valid holdout `(m, τ, r)`, scan 7 equally spaced σ across
the user σ-range (`sample_bounds`). Count a violation when
`price(σ_{k+1}) < price(σ_k) − max(1e-8 * spot, price_tol)` where
`price_tol = target_iv_error * vega_floor` (the same noise floor the error
metric uses); the reported slope is `Δprice / Δσ` for the worst violating
step. Non-finite prices in the scan increment `holdout_points_invalid`-style
accounting (a dedicated counter inside the stats) rather than violations.
Degenerate σ-range (σ_min == σ_max) ⇒ scan skipped, zero violations.

**Ownership path.** Adaptive builder results (`BSplineAdaptiveResult`,
`BSplineSegmentedAdaptiveResult`, Chebyshev equivalents) carry a
`BuildDiagnostics`. Factory helpers return it alongside the table;
`AnyPriceTable::Impl` and `AnyInterpIVSolver::Impl` each gain a separate
`std::optional<BuildDiagnostics>` member (`nullopt` for manual builds and
Parquet-loaded tables); `AnyPriceTable::make_iv_solver` copies it into the
solver wrapper. `to_data()` and all serialization visit only the table —
diagnostics never enter `PriceTableData`/Parquet.

**Language surface:** C++ accessor `build_diagnostics()` on both wrappers;
Python property `build_diagnostics` (dict-like, via pybind11) on
`PriceTable` and `InterpolatedIVSolver`. C API: deferred (Non-goals).

**Segmented builders:** the diagnostics exposed by a segmented table
describe the **final assembled surface** (D9), not a discarded probe. The
per-probe loop diagnostics are appended to `iterations` for forensics.

### D8. IV inversion defense (query time)

In `InterpolatedIVSolver::solve` (src/option/interpolated_iv_solver.hpp):

1. **Signed vega pre-check.** The existing pre-check takes `std::abs(vega)`
   at three probe vols, so a strongly negative vega counts as healthy.
   Changes: probe vols become the quartile points of the *actual* bracket
   `[sigma_min, sigma_max]` (25% / 50% / 75%), not fixed constants that may
   fall outside it; the check uses the **signed** maximum, initialized to
   `-inf`; non-finite probe vega ⇒ `NumericalInstability`. Reject with
   `VegaTooSmall` when `max_signed_vega < vega_threshold`.
2. **Bracket screen** (new, config `detect_multiple_roots`, default
   `true`): before Brent, evaluate the objective at 9 equally spaced σ in
   `[sigma_min, sigma_max]` (endpoints included).
   - **Near-zero handling:** samples with `|objective| <= zero_tol`
     (`zero_tol = config_.tolerance`) are *zeros*; consecutive zeros
     collapse into one zero run. Sign transitions are counted between
     nonzero samples of opposite sign, with a zero run between opposite
     signs counting as exactly one transition. A zero run flanked by the
     same sign counts as a tangent contact: it is reported as
     `MultipleRoots` (an even-multiplicity contact means the price fold
     touches the market price — root selection is ambiguous).
   - **Fold subdivision (vega-aware):** for each of the 8 intervals whose
     endpoint objective *slopes* (finite differences of the 9 samples)
     indicate a fold (slope sign change), evaluate the interval midpoint
     once more (bounded: ≤ 8 extra evals, typically 0–2) and recount
     transitions including those points.
   - More than one transition → `IVErrorCode::MultipleRoots`
     (`final_error` = transition count, `last_vol` = the low σ endpoint of
     the lowest transition interval).
   - Exactly one transition → run Brent on that subinterval (narrowed
     bracket; offsets part of the scan cost). After convergence, a
     **post-hoc slope check**: FD vega at the root (reusing the bracket's
     nearest scan samples) must be `> 0`; otherwise `MultipleRoots`.
   - No transition → fall through to Brent on the full bracket, which
     reports `BracketingFailed` exactly as today.
   - Non-finite objective at any scan point ⇒ `NumericalInstability`.
   - **Honest contract (documented in the header):** this is a *screen*,
     not a proof of uniqueness. It detects any fold the 9+8-point
     resolution straddles (bracket/16 ≈ 0.03 vol resolution for the default
     bracket) plus tangency at scan points; folds narrower than the
     resolution that also evade the post-hoc slope check can pass.
     Certified-monotone surfaces (follow-up issue) are the complete
     solution; the acceptance criterion for this branch is the screen
     contract, not absolute uniqueness.
   - Cost: 9–17 surface evals (~2–4 μs) against a ~3.5 μs solve;
     `detect_multiple_roots = false` restores today's path unchanged.
3. **Error-code wiring:** `MultipleRoots` appended to `IVErrorCode`
   (src/support/error_types.hpp) with message/helper switch arms; Python
   bindings enum + message (src/python/mango_bindings.cpp); C API
   `map_iv_error` (src/ffi/mango_c_api.cpp) maps it to
   **`MANGO_ERR_BRACKETING`** (the same category as `BracketingFailed` —
   verified against the existing mapping). `iv_result.hpp` needs no change
   (it maps validation errors only). The C config struct is **not**
   extended (ABI; Non-goals) — C API callers get the default-on screen.

### D9. Callers and shared-path coverage

`run_refinement` is the single shared loop; D2/D4/D5/D6 changes apply to all
four adaptive builders. Per-caller contracts:

- **`build_adaptive_bspline` (4D single-K_ref):** captures `last_spline` by
  reference in `build_fn`; the D5 final-rebuild rule keeps `last_spline`
  consistent with the returned grids. Gains the D3 headroom fix and passes
  no state hooks.
- **`build_adaptive_chebyshev` (4D):** passes state hooks for its level
  counters (D6). The caller's existing unconditional extra `build_fn` call
  after `run_refinement` is removed — it consumes the loop's captured
  surface directly (the loop now guarantees last-built == picked).
- **Segmented builders (B-spline and Chebyshev, multi-K_ref):** the
  per-probe loops gain retention/backtracking automatically, improving the
  probe *size* selection and making probe-loop build failures non-fatal.
  Aggregation to uniform final grids is unchanged (Non-goals). The
  guarantees of D5 apply *within each probe loop*; the **final assembled
  surface** gets its own contract:
  - the existing final multi-K_ref validation (LHS over the domain) is
    performed on the **sample domain** (D2), not the headroom domain;
  - **retry fix:** when the bumped-grid retry surface is built, it is
    validated with the same final validation; the builder returns whichever
    of (original, retry) has the lower final max error, with diagnostics
    (`achieved_*`, `target_met`, `used_retry`) describing the returned
    surface. Today's behavior — returning the retry surface unvalidated
    with the pre-retry error numbers — is a defect under Goal 4 and is
    fixed here;
  - the final surface's `BuildDiagnostics` (D7) is computed from this final
    validation plus a monotonicity scan of the final surface.

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
   improvement".
3. **Unmet-but-safe target: best-effort + surfaced diagnostics** — no hard
   error, no strict-mode flag (D5, D7). The issue's "reject when every
   candidate is unsafe" item is honored through the *viability gate*
   (finite scores + `50 × target` catastrophe bound + `NoViableSurface`),
   which measurably separates good from catastrophic candidates where
   violation counts do not (P5; design-review round 1).
4. **IV defense: bracket scan, configurable, on by default** — strengthened
   in review round 1 to a vega-aware screen with fold subdivision, tangency
   handling, a post-hoc slope check, and an explicitly documented
   screen-not-proof contract. Post-hoc-only checking remains rejected
   (cannot detect a wrong root on a locally positive slope). Absolute
   uniqueness is delivered by the certified-monotone follow-up, not this
   branch.
5. **Rigor level: prototype first** — executed; findings above. Certificate /
   shape-constrained fitting deferred to a follow-up issue by the evidence
   (build-time strict monotonicity is unattainable where vega → 0).
6. **Retention criterion: holdout error, not monotonicity violations** (P5),
   with the D5 viability gate as the rejection mechanism.
7. **Improvement threshold 2% relative** — from P4's budget-burn
   observation.
8. **`max_iter` default 5 → 8** — required for the backtracking walk to
   both reach and exploit late-axis improvements (D6).
9. **Diagnostics surface: C++ + Python; C API deferred** — the C config and
   result structs are frozen this branch for ABI stability (review round 1).

## Test design

### Unit: run_refinement (synthetic callbacks)

The loop is callback-driven; tests use synthetic `BuildFn` / `PrepareRefsFn`
/ `ScoreErrorFn` (D4 signatures) that require no PDE solves:

- **Retention picks best:** builds whose holdout error degrades after
  iteration k → returned grids are iteration k's; `picked_iteration == k`;
  `final_rebuild == true` and one extra `build_fn` call with the picked
  grids.
- **Viability gate:** all candidates above `50 × target` →
  `NoViableSurface`; one candidate non-finite at a holdout point → never
  picked even with the lowest score.
- **Build-failure fallback:** `build_fn` fails at iteration j > 0 with a
  viable candidate → success, `build_failure_fallback == true`; failure at
  iteration 0 → error propagates; final-rebuild failure → error propagates.
- **Backtracking walk:** an error field reducible only along axis 2 → axes
  with higher concentration scores are tried, rejected, and axis 2's
  improvement is accepted; `tried` resets after acceptance.
- **ε-threshold:** a 0.5% improvement does not reset `tried`; a 5%
  improvement does.
- **No-op refinement:** a capped axis (refine_fn returns false) is marked
  tried without consuming a build; all axes capped → loop stops before
  `max_iter` with the seed retained.
- **State hooks:** snapshot/restore invoked around each backtracking reset;
  a Chebyshev-style counter restored after a rejected trial yields the
  correct next level when the axis is retried post-acceptance.
- **Convergence requires holdout:** fresh samples under target but holdout
  above → not converged.
- **Holdout caching:** count `PrepareRefsFn` invocations — exactly
  holdout-setup + fresh-validation counts (no per-iteration holdout
  re-preparation).
- **Holdout validity:** failed refs excluded and counted; below the
  minimum-valid threshold → `ValidationFailed`; `validation_samples < 8` →
  `InvalidConfig`; `max_iter == 0` → `InvalidConfig`; `max_iter == 1` →
  seed retained.
- **Sampling domain:** all sample coordinates within `sample_bounds`; bins
  handed to `refine_fn` as physical intervals inside `sample_bounds`.

### Unit: headroom and domain

- `extract_chain_domain(chain, 60)` headroom equals `3 * width / 59` (not
  `3 * width / (n_strikes - 1)`).
- `sample_bounds` equals user ranges (with minimum-spread widening);
  `bounds` equals `sample_bounds` plus headroom on moneyness.
- Segmented builder: headroom applied at `build_adaptive()` with the D3
  rule; final validation samples drawn from the sample domain.

### Unit: IV defense

- Synthetic surface with three crossings inside the bracket →
  `MultipleRoots`; the same query with `detect_multiple_roots = false`
  reproduces today's (wrong-root) behavior — documenting the defended
  failure.
- **Two roots inside one initial scan interval** (fold width < bracket/8,
  ≥ bracket/16) → caught by fold subdivision → `MultipleRoots`.
- **Tangent contact** (objective touches zero without crossing at a scan
  point) → `MultipleRoots`.
- Monotone surface → identical root with and without the screen (narrowed
  bracket converges to the same σ within tolerance); scan cost bounded at
  17 evals.
- Post-hoc slope check: converged root with locally negative FD vega →
  `MultipleRoots`.
- Signed pre-check: probe vegas {−5, −3, −1}, threshold 0.5 →
  `VegaTooSmall`; {−5, +2, +1} → passes to the screen; probes evaluated at
  bracket quartiles (regression: bracket narrower than [0.10, 0.50]);
  non-finite probe → `NumericalInstability`.
- Error-code wiring: `MultipleRoots` in Python enum with message; C API
  maps to `MANGO_ERR_BRACKETING` (mapping test per the #449 pattern).

### Integration / regression

- **q0 bifurcation regression:** adaptive B-spline build on a scaled-down
  q = 0 PUT config (reduced grids/samples for test budget) asserts: build
  succeeds; `build_diagnostics()` present with sane `achieved_max_error`
  (≤ 100 bps); and **IV round-trips through the returned solver** at the
  known wrong-root region (σ = 30%, T = 30d, K = 80 analog) recover the
  true vol within tolerance or return `MultipleRoots` — never a spurious
  low root. This fails against the pre-fix loop.
- **Segmented retry regression:** a config that forces the retry path →
  the returned surface's diagnostics describe the returned surface, and a
  retry that validates worse than the original is not returned.
- **Projection tests** from the July design (already on branch).
- **Diagnostics plumbing:** factory-built adaptive solver exposes
  diagnostics (C++ and Python); manual build and Parquet round-trip return
  `nullopt`; serialized bytes contain no diagnostics fields.

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
   headroom no longer scales with user knot count.
3. The adaptive loop never returns a surface with a worse holdout score than
   the best viable candidate it built; mid-loop build failure after a viable
   candidate exists is non-fatal; no viable candidate ⇒ `NoViableSurface`.
4. Refinement steps are accepted only on ≥ 2% relative holdout improvement;
   axis search backtracks with correct backend state restoration; no-op
   refinements consume no build; loop stops when all axes are exhausted.
5. `build_diagnostics()` available on adaptive-built tables/solvers (C++
   and Python) with honest `target_met` and achieved errors; absent from
   serialization; segmented diagnostics describe the returned final
   surface, and the segmented retry is returned only when it validates
   better.
6. The D8 screen contract holds: multi-transition and tangency brackets
   return `MultipleRoots`; single-transition brackets Brent on the narrowed
   subinterval with a positive post-hoc slope; the screen is removable by
   config; signed vega pre-check on in-bracket probes. The documented
   contract is screen-not-proof.
7. All error-code surfaces (C++, Python, C API mapping) handle
   `MultipleRoots`; C structs unchanged.
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
3. Segmented multi-K_ref probe path discards refined knot positions
   (aggregates sizes only, rebuilds uniform) — evaluate carrying positions.
4. Sparse explicit K_ref grids accuracy (parked from PR #449 work).
5. C ABI rev: expose `BuildDiagnostics` and `detect_multiple_roots` through
   the C API.
