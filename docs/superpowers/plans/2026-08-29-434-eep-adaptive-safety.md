# EEP Projection + Adaptive Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the exact EEP projection together with the adaptive-refinement
safety redesign (user-domain measurement, holdout retention, viability gate,
measured backtracking, build diagnostics) and the query-time IV multi-root
screen, per the approved spec.

**Architecture:** All controller changes concentrate in the shared
`run_refinement` loop (`src/option/table/adaptive_refinement.{hpp,cpp}`) and
flow to the four adaptive builders; the IV screen is local to
`InterpolatedIVSolver::solve`; diagnostics ride a sidecar (never serialized).

**Tech Stack:** C++23, Bazel, GoogleTest, pybind11.

**Spec:** `docs/superpowers/specs/2026-08-29-eep-projection-adaptive-safety-design.md`
(committed on this branch; READ IT FIRST — it pins every semantic referenced
here). The July projection spec
`docs/superpowers/specs/2026-07-23-eep-floor-correction-design.md` governs D1.

## Global Constraints

- Exact values (verbatim from spec): holdout seed `params.lhs_seed ^
  0x484F4C44`; minimum valid holdout points `max(4, validation_samples/4)`;
  `validation_samples >= 8`; `kViabilityBound = 0.20` (absolute IV error);
  walk-restart threshold `kMinRelImprovement = 0.02`; `max_iter` default
  `8`; IV screen: 17 equally spaced σ samples, `zero_tol = 1e-9 * spot`;
  monotonicity scan: 7 σ points, violation tolerance
  `max(1e-8 * spot, target_iv_error * vega_floor)`; final-rebuild
  `IterationStats.refined_dim = -2`.
- Candidate roles (spec D5): exploration base = best *evaluated*; 2% governs
  only the axis-walk restart; returned = best *viable*. Viability covers
  holdout AND fresh evaluations (finite, nonnegative) plus the 0.20 bound.
- `BuildDiagnostics` must NEVER enter `PriceTableData`/Parquet.
- C API structs unchanged (no new fields); new enum values appended at the
  END of their enums.
- Library code: no printf/fprintf (USDT only); SPDX header on every new
  file; `-Werror` clean.
- Every bug found gets a regression test (CLAUDE.md).
- Never bypass the failing-test step: run, observe failure, then implement.
- Merge-conflict note: PR #449 (open) appends `DiscreteDividendMismatch` to
  `ValidationErrorCode`/`IVErrorCode`. If it merges before this branch,
  rebase and append this plan's new values AFTER those.

---

### Task 1: Error-code plumbing

**Files:**
- Modify: `src/support/error_types.hpp` (enums + message helpers)
- Modify: `src/python/mango_bindings.cpp` (enum values + message strings)
- Modify: `src/ffi/mango_c_api.cpp` (`map_iv_error`, validation mapping)
- Test: `tests/error_types_test.cc` (extend existing if present, else the
  existing test that covers error messages; `tests/ffi_c_api_test.cc`-style
  mapping test alongside the #449 pattern — find with
  `grep -rn "map_iv_error\|MANGO_ERR_BRACKETING" tests/`)

**Interfaces:**
- Produces: `IVErrorCode::MultipleRoots` (appended after `PDESolveFailed`);
  `PriceTableErrorCode::ValidationFailed` and
  `PriceTableErrorCode::NoViableSurface` (appended at enum end);
  `ValidationErrorCode::NoViableSurface` and
  `ValidationErrorCode::AdaptiveValidationFailed` (appended at enum end).
- C API: `MultipleRoots` → `MANGO_ERR_BRACKETING`; both new
  ValidationErrorCodes → `MANGO_ERR_VALIDATION`.

- [ ] **Step 1: Write failing tests** — message-helper coverage for each new
  code (exhaustive switches will fail to compile until arms are added — that
  compile failure IS the failing test) plus mapping assertions:

```cpp
TEST(ErrorCodeTest, MultipleRootsHasMessage) {
    EXPECT_NE(mango::iv_error_message(mango::IVErrorCode::MultipleRoots),
              nullptr);
}
// C API mapping test (same pattern as the DiscreteDividendMismatch test
// added by the #449 branch if visible, else follow existing mapping tests):
// IVError{MultipleRoots} through the C boundary yields MANGO_ERR_BRACKETING.
```

- [ ] **Step 2: Run** `bazel test //tests:error_types_test` (or the target
  found in Step 1) — expect FAIL/compile error.
- [ ] **Step 3: Implement** — append enum values with doc comments, add all
  switch arms (helpers in error_types.hpp, `iv_result.hpp` needs NO change —
  verified in review), Python `py::enum_` value + message, C API cases.
- [ ] **Step 4:** `bazel test //tests:error_types_test` + the mapping test —
  PASS. `bazel build //src/python:mango_option` compiles.
- [ ] **Step 5: Commit** `feat: add MultipleRoots and adaptive-build error codes`

### Task 2: Params, IterationStats, BuildDiagnostics types

**Files:**
- Modify: `src/option/table/adaptive_grid_types.hpp`
- Test: `tests/adaptive_grid_builder_test.cc` (param-validation cases) — or a
  new `tests/adaptive_refinement_unit_test.cc` created here and grown in
  Tasks 3–6 (`cc_test` in `tests/BUILD.bazel`, deps on
  `//src/option/table:adaptive_refinement` — check actual target name with
  `grep -n adaptive tests/BUILD.bazel src/option/table/BUILD.bazel`).

**Interfaces (produces, verbatim):**

```cpp
// AdaptiveGridParams: max_iter default 5 -> 8 (update comment).
// IterationStats: add
    bool build_failed = false;  ///< Refinement trial build failed (D5)
// New struct (same header):
struct BuildDiagnostics {
    bool target_met = false;
    double achieved_max_error = 0.0;   // holdout, returned candidate
    double achieved_avg_error = 0.0;
    size_t picked_iteration = 0;
    size_t total_iterations = 0;       // built iterations, excl. final rebuild
    bool final_rebuild = false;
    bool build_failure_fallback = false;
    size_t holdout_points = 0;
    size_t holdout_points_invalid = 0;
    size_t monotonicity_violations = 0;
    size_t monotonicity_points_invalid = 0;
    double worst_vega_slope = 0.0;
    std::vector<IterationStats> iterations;  // final rebuild: refined_dim = -2
};
```

- [ ] **Step 1: Failing test** — param validation (loop entry is Task 6, so
  here just pin the struct compiles and defaults):

```cpp
TEST(AdaptiveGridParamsTest, DefaultMaxIterIsEight) {
    EXPECT_EQ(mango::AdaptiveGridParams{}.max_iter, 8u);
}
TEST(BuildDiagnosticsTest, DefaultsAreEmpty) {
    mango::BuildDiagnostics d;
    EXPECT_FALSE(d.target_met);
    EXPECT_EQ(d.holdout_points, 0u);
}
```

- [ ] **Step 2:** run — FAIL (max_iter is 5 / struct absent).
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run — PASS.
- [ ] **Step 5: Commit** `feat: add BuildDiagnostics and raise adaptive max_iter to 8`

### Task 3: Split the error metric into PrepareRefsFn + ScoreErrorFn

**Files:**
- Modify: `src/option/table/adaptive_refinement.hpp` (typedefs, factories)
- Modify: `src/option/table/adaptive_refinement.cpp`
- Modify (compile-fix only): `src/option/table/bspline/bspline_adaptive.cpp`,
  `src/option/table/chebyshev/chebyshev_adaptive.cpp` — swap
  `make_fd_vega_error_fn` for the new pair; loop behavior unchanged in this
  task.
- Test: `tests/adaptive_refinement_unit_test.cc`

**Interfaces (produces, verbatim from spec D4):**

```cpp
struct ErrorRefs { double ref_price = 0.0; double vega = 0.0; };
using PrepareRefsFn = std::function<std::expected<ErrorRefs, SolverError>(
    double spot, double strike, double tau, double sigma, double rate)>;
using ScoreErrorFn = std::function<double(
    double interp, const ErrorRefs& refs,
    double spot, double strike, double tau,
    double sigma, double rate)>;
PrepareRefsFn make_fd_vega_refs_fn(const AdaptiveGridParams& params,
                                   const ValidateFn& validate_fn);
ScoreErrorFn make_iv_score_fn(const AdaptiveGridParams& params,
                              OptionType option_type);
```

`make_fd_vega_refs_fn`: base solve + the two σ-bump solves exactly as in
today's `make_fd_vega_error_fn` (`eps = max(1e-4, 0.01 * sigma)`, dn clamp
`1e-4`); any failed or non-finite solve → `std::unexpected`. `make_iv_score_fn`:
today's TV/K 1e-4 filter + `compute_iv_error` arithmetic (vega floor +
target-level noise clamp — NOT a global cap), reading vega from `refs`.
`run_refinement` signature replaces `ValidateFn validate_fn, ComputeErrorFn
compute_error` with `PrepareRefsFn prepare_refs, ScoreErrorFn score`;
`evaluate_samples` calls `prepare_refs` per fresh sample then `score`.
Delete `ComputeErrorFn` and `make_fd_vega_error_fn`.

- [ ] **Step 1: Failing tests** —

```cpp
// Score equivalence with the old arithmetic:
TEST(ScoreFnTest, MatchesComputeIvError) {
    mango::AdaptiveGridParams p;  // target 2e-5, floor 1e-4
    auto score = mango::make_iv_score_fn(p, mango::OptionType::PUT);
    mango::ErrorRefs refs{.ref_price = 5.0, .vega = 20.0};
    // price_error 0.01 / vega 20 = 5e-4
    EXPECT_NEAR(score(5.01, refs, 100.0, 100.0, 1.0, 0.2, 0.05), 5e-4, 1e-12);
}
TEST(ScoreFnTest, TvkFilterZeroesDeepItm) {
    mango::AdaptiveGridParams p;
    auto score = mango::make_iv_score_fn(p, mango::OptionType::PUT);
    // K=100, S=100 put ref 0.005 -> TV/K = 5e-5 < 1e-4 -> filtered
    mango::ErrorRefs refs{.ref_price = 0.005, .vega = 1.0};
    EXPECT_EQ(score(1.0, refs, 100.0, 100.0, 0.01, 0.2, 0.05), 0.0);
}
TEST(PrepareRefsTest, PropagatesSolveFailure) {
    mango::ValidateFn failing = [](double, double, double, double, double)
        -> std::expected<double, mango::SolverError> {
        return std::unexpected(mango::SolverError{});
    };
    auto prep = mango::make_fd_vega_refs_fn(mango::AdaptiveGridParams{}, failing);
    EXPECT_FALSE(prep(100, 100, 1.0, 0.2, 0.05).has_value());
}
```

  (Adjust `SolverError{}` construction to the real type — read
  `src/support/error_types.hpp` first.)
- [ ] **Step 2:** run — FAIL (functions absent).
- [ ] **Step 3:** implement; mechanical caller swaps; keep loop semantics.
- [ ] **Step 4:** `bazel test //tests:adaptive_refinement_unit_test
  //tests:adaptive_grid_builder_test` — PASS; `bazel build //...` clean.
- [ ] **Step 5: Commit** `refactor: split adaptive error metric into prepare/score`

### Task 4: RefineFn/RefineOutcome + B-spline refiner rewrite

**Files:**
- Modify: `src/option/table/adaptive_refinement.hpp` (RefineFn, RefineOutcome)
- Modify: `src/option/table/bspline/bspline_adaptive.cpp`
  (`make_bspline_refine_fn`)
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.cpp` (signature
  only in this task; redirection removal is Task 7)
- Modify: `src/option/table/adaptive_refinement.cpp` (loop calls; still
  single-axis `worst_dimension()` selection in this task — the walk arrives
  in Task 6; loop computes physical focus intervals from bins here)
- Test: `tests/adaptive_refinement_unit_test.cc`

**Interfaces (produces, spec D6 verbatim):**

```cpp
struct RefineOutcome { bool changed = false; int changed_dim = -1; };
using RefineFn = std::function<RefineOutcome(
    size_t requested_dim,
    std::span<const std::pair<double, double>> focus_intervals,
    std::vector<double>& moneyness,
    std::vector<double>& tau,
    std::vector<double>& vol,
    std::vector<double>& rate)>;
```

B-spline refiner rules: empty `focus_intervals` ⇒ uniform refinement over
the axis; `changed = false` when at `max_points_per_dim` or no midpoint
inserted; never redirects (`changed_dim == requested_dim` when changed).
Bin→interval conversion moves into the loop: for each problematic bin `b` of
the chosen axis, interval =
`[sb_lo + (sb_hi-sb_lo)*b/N_BINS, sb_lo + (sb_hi-sb_lo)*(b+1)/N_BINS]` with
`(sb_lo, sb_hi)` from `ctx.sample_bounds` for that axis (this task may take
them from `ctx.bounds` until Task 5 lands sample_bounds, then Task 5 flips
the source — leave a `// Task 5 flips to sample_bounds` marker).

- [ ] **Step 1: Failing tests** — direct unit tests on the refiner:

```cpp
TEST(BSplineRefineFnTest, NoOpAtCapReturnsUnchanged) {
    mango::AdaptiveGridParams p; p.max_points_per_dim = 4;
    auto fn = /* make_bspline_refine_fn(p) — export via a test hook or move
                 the factory declaration into a header this task */;
    std::vector<double> m{0.0, 0.1, 0.2, 0.3}, t{0.1, 0.5, 1.0, 2.0},
                        v{0.1, 0.2, 0.3, 0.4}, r{0.01, 0.03, 0.05, 0.08};
    auto out = fn(0, {}, m, t, v, r);
    EXPECT_FALSE(out.changed);
    EXPECT_EQ(m.size(), 4u);
}
TEST(BSplineRefineFnTest, UniformWhenNoFocus) { /* grows toward
    grid.size()*refinement_factor, changed_dim == 0 */ }
TEST(BSplineRefineFnTest, FocusIntervalTargetsBin) { /* midpoints only
    inside the provided interval */ }
```

  `make_bspline_refine_fn` is file-static today — move its declaration to
  `src/option/table/bspline/bspline_adaptive.hpp` (or a small internal
  header) so tests reach it.
- [ ] **Step 2:** run — FAIL.
- [ ] **Step 3:** implement refiner + loop interval computation + Chebyshev
  signature adaptation (keep its current redirection internally for now,
  returning the ACTUAL changed dim in `changed_dim`).
- [ ] **Step 4:** run unit test + `bazel test //tests:adaptive_grid_builder_test` — PASS.
- [ ] **Step 5: Commit** `refactor: RefineFn returns outcome and takes focus intervals`

### Task 5: Domain separation + headroom fix + user-domain sampling

**Files:**
- Modify: `src/option/table/adaptive_refinement.hpp` (`RefinementContext`
  gains `SurfaceBounds sample_bounds;`; `extract_chain_domain(const
  OptionGrid&, size_t expected_m_knots)`)
- Modify: `src/option/table/adaptive_refinement.cpp` (`extract_chain_domain`
  builds both bounds — sample = user ranges + existing `expand_domain_bounds`
  min-spreads; fit = sample + `spline_support_headroom(width,
  expected_m_knots)` on moneyness only, as today; `run_refinement` samples
  fresh validation from `sample_bounds`; bin normalization + focus intervals
  from `sample_bounds`)
- Modify: `src/option/table/bspline/bspline_adaptive.cpp` — 4D caller passes
  `max(chain.strikes.size(), params.min_moneyness_points)`; segmented
  builder: split `domain_` into `sample_domain_`/`fit_domain_`, headroom
  applied in `build_adaptive()` (move it out of `create()`); its final
  validation LHS switches to the sample domain.
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.cpp` — populate
  `sample_bounds` before the CC-level extension; fit bounds from actual
  nodes; NO B-spline headroom (spec D3 "no double headroom").
- Test: `tests/adaptive_refinement_unit_test.cc` + existing
  `tests/adaptive_grid_builder_test.cc`

- [ ] **Step 1: Failing tests:**

```cpp
TEST(ExtractChainDomainTest, HeadroomUsesExpectedKnots) {
    mango::OptionGrid chain = /* 7 strikes spanning log-m width w, spot 100,
        maturities {0.1, 1.0}, vols {0.1, 0.4}, rates {0.02, 0.08} */;
    auto ctx = mango::extract_chain_domain(chain, 60).value();
    double w = ctx.sample_bounds.m_max - ctx.sample_bounds.m_min;
    EXPECT_NEAR(ctx.bounds.m_max - ctx.sample_bounds.m_max, 3.0 * w / 59.0, 1e-12);
    EXPECT_NEAR(ctx.sample_bounds.m_min - ctx.bounds.m_min, 3.0 * w / 59.0, 1e-12);
    // tau/vol/rate: fit == sample (no headroom on those axes)
    EXPECT_EQ(ctx.bounds.tau_min, ctx.sample_bounds.tau_min);
}
// Regression: 7 knots used to give 3*w/6 — assert the old value is NOT used.
```

  Loop-side (synthetic BuildFn records every holdout/fresh coordinate):
  all sampled `m` within `sample_bounds` even though grids span `bounds`.
- [ ] **Step 2:** run — FAIL (signature/fields absent).
- [ ] **Step 3:** implement (all call sites; `grep -rn extract_chain_domain
  src/` to find every caller).
- [ ] **Step 4:** full `bazel test //...` — the adaptive builder tests must
  still pass (grids change slightly from smaller headroom; if any golden
  values break, update them with a comment citing spec D3).
- [ ] **Step 5: Commit** `fix: sample adaptive validation from the user domain`

### Task 6: Core loop — holdout, retention, viability, measured walk

The centerpiece. Rewrite `run_refinement` per spec D4–D6 exactly. All
Global-Constraints values apply. Structure (complete skeleton — implement
with this shape):

```cpp
std::expected<RefinementResult, PriceTableError> run_refinement(
    const AdaptiveGridParams& params, BuildFn build_fn,
    PrepareRefsFn prepare_refs, ScoreErrorFn score,
    RefineFn refine_fn, const RefinementContext& ctx,
    const InitialGrids& initial_grids, const RefineStateHooks& hooks = {})
{
    // 1. Param validation (spec D3): target/floor finite&positive,
    //    refinement_factor finite&>1, max_iter>=1, validation_samples>=8.
    // 2. Seed grids (unchanged logic).
    // 3. Holdout: LHS(validation_samples, lhs_seed ^ 0x484F4C44) over
    //    sample_bounds; prepare_refs per point once; invalid points
    //    excluded+counted; < max(4, n/4) valid -> ValidationFailed.
    // 4. Candidate records: grids, snapshot (hooks.snapshot), holdout_max/
    //    avg, bins, iter, viable (finite holdout AND fresh evals, and
    //    holdout_max <= 0.20).
    // 5. Iterate (<= max_iter builds):
    //    a. build (seed fail -> error; trial fail -> mark tried, restore
    //       base+state, IterationStats{build_failed=true, refined_dim=axis},
    //       consume budget, build_failure_fallback=true, goto axis pick)
    //    b. fresh samples (from sample_bounds, iteration-seeded as today)
    //       -> prepare+score each; track non-finite -> candidate non-viable
    //    c. holdout eval: interp + score against cached refs (NO prepare)
    //    d. record candidate
    //    e. converged = fresh_max<=target && holdout_max<=target -> break
    //    f. axis pick: score[d]=concentration over BASE bins (0 if empty;
    //       ties by dim order); skip tried; none left -> break
    //    g. reset grids+state to exploration base; refine_fn(axis,
    //       intervals-from-base-bins); !changed -> tried[axis]=true, no
    //       build consumed, goto f
    //    h. after next evaluation: >=2% better than prev best -> tried={};
    //       else tried[changed_dim]=true. Any improvement -> new base.
    // 6. Exit: best viable candidate or NoViableSurface. If picked != last
    //    built: rebuild via build_fn (IterationStats refined_dim=-2, not
    //    counted in total_iterations; failure -> propagate error).
    // 7. Fill RefinementResult + BuildDiagnostics (monotonicity scan of
    //    returned candidate: 7 sigma points over sample_bounds at holdout
    //    (m,tau,r); tolerance max(1e-8*spot, target*vega_floor)).
}
```

**Files:**
- Modify: `src/option/table/adaptive_refinement.{hpp,cpp}`
  (+ `RefineStateHooks` typedef; `RefinementResult` gains
  `BuildDiagnostics diagnostics;`)
- Modify: callers pass hooks (`{}` for B-spline) — compile fixes.
- Test: `tests/adaptive_refinement_unit_test.cc` — the full synthetic
  battery from the spec's "Unit: run_refinement" list. Each spec bullet is
  one TEST. Synthetic harness sketch:

```cpp
struct FakeSurface { double err; };  // scripted per-iteration quality
// build_fn returns SurfaceHandle whose price() = analytic_ref + script[i].err
// prepare_refs returns {ref_price = analytic_ref(pt), vega = 1.0}
// score returns |interp - ref| (so holdout_max == script[i].err exactly)
// refine_fn: scripted RefineOutcome + optional grid mutation
```

  Required tests (names indicative — implement ALL spec bullets):
  `RetentionPicksBestViable`, `ViabilityBoundRejectsAll`,
  `NonFiniteHoldoutDisqualifies`, `NonFiniteFreshDisqualifies`,
  `NonViableSeedRecoveredViaLaterAxis`, `SubThresholdAdvancesBaseOnly`,
  `FivePercentRestartsWalk`, `TrialBuildFailureContinuesExploration`,
  `SeedBuildFailurePropagates`, `FinalRebuildFailurePropagates`,
  `BacktrackReachesThirdAxis`, `NoOpRefineConsumesNoBuild`,
  `StateHooksRestoredOnBacktrack`, `ConvergenceRequiresHoldout`,
  `HoldoutRefsPreparedOnce`, `HoldoutValidityThresholds`,
  `ParamValidation` (each invalid param), `MaxIterOneReturnsSeed`,
  `SamplesInsideSampleBounds`, `FinalRebuildStatsExcluded`.

- [ ] **Step 1:** write the harness + all tests; run — FAIL.
- [ ] **Step 2:** implement the loop.
- [ ] **Step 3:** `bazel test //tests:adaptive_refinement_unit_test
  --test_output=errors` — PASS.
- [ ] **Step 4:** `bazel test //...` — existing adaptive tests pass (they
  now run through retention; expected: same or better achieved errors; fix
  any golden expectations with spec citations).
- [ ] **Step 5: Commit** `feat: holdout retention and measured backtracking in run_refinement`
  (split into 2–3 commits if natural: holdout+retention, walk, diagnostics).

### Task 7: Chebyshev state hooks + redirection removal

**Files:**
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.cpp` — refiners
  advance EXACTLY the requested axis or return `changed=false` (delete the
  lowest-level-balance redirection, ~line 488); provide
  `RefineStateHooks` snapshotting `ChebyshevRefinementState` (levels + node
  vectors) as `shared_ptr<const State>`; continuous caller drops its
  unconditional post-loop `build_fn` call (loop guarantees last == picked).
- Test: extend `tests/adaptive_refinement_unit_test.cc` (hook contract with
  a fake counter state) + existing chebyshev adaptive tests.

- [ ] **Step 1:** failing test: fake state {level:int}; script: axis 0
  rejected, axis 1 accepted (restart), axis 0 retried → snapshot/restore
  sequence yields level as-of-base, not as-of-rejected-trial.
- [ ] **Step 2:** implement hooks + redirection removal.
- [ ] **Step 3:** `bazel test //tests:chebyshev* //tests:adaptive*` (list
  actual targets via `grep chebyshev tests/BUILD.bazel`) — PASS.
- [ ] **Step 4: Commit** `fix: Chebyshev refiners honor requested axis with state rollback`

### Task 8: Segmented final-surface contracts

**Files:**
- Modify: `src/option/table/bspline/bspline_adaptive.cpp`
  (`build_adaptive_bspline_segmented` final section, currently ~lines
  614–728): final validation via `prepare_refs` ONCE over sample-domain LHS
  (`lhs_seed + 999` as today), D4 validity rules
  (`ValidationFailed` when < max(4, n/4) valid); score original; retry when
  `orig_max > target || !orig_viable`; score retry on the SAME refs; return
  lowest-error viable (both non-viable → `NoViableSurface`); diagnostics
  describe the returned surface (+ its monotonicity scan).
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.cpp`
  (`build_adaptive_chebyshev_segmented`, ~line 858): add the same final
  validation + viability gate on the assembled all-K_ref surface (no retry).
- Test: `tests/adaptive_grid_builder_test.cc` (or the segmented builders'
  existing test file — locate via `grep -rln build_adaptive_bspline_segmented
  tests/`).

- [ ] **Step 1:** failing tests: (a) force retry (tiny max_points so target
  unmet) → returned diagnostics' `achieved_max_error` equals the returned
  surface's score on shared refs and `used_retry` truthful; (b) synthetic:
  retry worse than original → original returned; (c) segmented Chebyshev
  final gate: a config whose assembled surface scores above 0.20 →
  `NoViableSurface` (construct by corrupting... if hard to force, test the
  gate function directly by exposing a small internal helper
  `select_final_surface(orig_score, retry_score, ...)` and unit-test it).
- [ ] **Step 2:** implement.
- [ ] **Step 3:** segmented tests + `bazel test //...` — PASS.
- [ ] **Step 4: Commit** `fix: validate segmented final surfaces and retry selection`

### Task 9: Diagnostics + published bounds through the factory; Python

**Files:**
- Modify: `src/option/table/bspline/bspline_adaptive.hpp` (+ chebyshev
  equivalent): result structs gain `BuildDiagnostics diagnostics;` and
  `SurfaceBounds sample_bounds;`
- Modify: `src/option/price_table_factory.{hpp,cpp}`:
  `AnyPriceTable::Impl` + `std::optional<BuildDiagnostics> diagnostics`;
  `AnyPriceTable::build_diagnostics()`; `make_iv_solver` copies it into the
  solver; adaptive wrapping passes `sample_bounds` as the published
  `SurfaceBounds` (today `make_bspline_surface` derives bounds from spline
  knots — add an overload taking explicit bounds; same for the segmented
  wrap and Chebyshev paths). Continuous-Chebyshev factory path unchanged
  (diagnostics `nullopt`, documented).
- Modify: `src/option/interpolated_iv_solver.hpp`: `AnyInterpIVSolver`
  stores/exposes `std::optional<BuildDiagnostics> build_diagnostics()`.
- Modify: `src/python/mango_bindings.cpp`: `build_diagnostics` property on
  `PriceTable` + `InterpolatedIVSolver` returning a dict (or None).
- Test: `tests/price_table_factory_test.cc`, `tests/iv_solver_factory_test.cc`,
  Python test if a python test target exists (`grep -rn python tests/
  src/python/BUILD.bazel` — else cover via C++ only and note it).

- [ ] **Step 1:** failing tests: adaptive factory build →
  `build_diagnostics()` has value, `total_iterations >= 1`; manual build →
  `nullopt`; Parquet round-trip (existing serialization test file) →
  `nullopt` and byte-stream unchanged by this branch (existing golden
  serialization tests must pass untouched); bounds: query at
  `m = sample m_min - epsilon` (old headroom zone) → rejected
  (`InvalidGridConfig`/out-of-bounds per existing behavior).
- [ ] **Step 2:** implement.
- [ ] **Step 3:** `bazel test //... ` + `bazel build //src/python:mango_option` — PASS.
- [ ] **Step 4: Commit** `feat: surface adaptive build diagnostics; publish user-domain bounds`

### Task 10: IV inversion screen

**Files:**
- Modify: `src/option/interpolated_iv_solver.hpp`
  (`InterpolatedIVSolverConfig` gains `bool detect_multiple_roots = true;`
  — C++/Python only; the `solve` path per spec D8)
- Modify: `src/python/mango_bindings.cpp` (config field)
- Test: `tests/interpolated_iv_solver_test.cc` (locate actual name via
  `grep -rln InterpolatedIVSolver tests/`)

Implementation per spec D8, in `solve` after bounds check:

```cpp
// 1. Signed vega pre-check at bracket quartiles:
double probes[3] = {sigma_min + 0.25*(sigma_max-sigma_min),
                    sigma_min + 0.50*(sigma_max-sigma_min),
                    sigma_min + 0.75*(sigma_max-sigma_min)};
double max_vega = -std::numeric_limits<double>::infinity();
for (double sv : probes) {
    double v = surface_.vega(...);          // SIGNED — no std::abs
    if (!std::isfinite(v)) return NumericalInstability;
    max_vega = std::max(max_vega, v);
}
if (max_vega < config_.vega_threshold) return VegaTooSmall;
// 2. Screen (config_.detect_multiple_roots):
//    f[i] at 17 uniform sigmas; zero_tol = 1e-9 * spot;
//    classify zeros -> runs; count transitions per spec (tangency ->
//    MultipleRoots; all-zero -> MultipleRoots{final_error=0,last_vol=
//    sigma_min}; endpoint-zero boundary root only if |f| <= config_.tolerance
//    else BracketingFailed; >1 -> MultipleRoots{count, low endpoint};
//    ==1 -> Brent on that subinterval, then post-hoc: FD slope across the
//    narrowed interval's endpoints > 0 else MultipleRoots;
//    ==0 -> Brent on full bracket (today's path)).
```

- [ ] **Step 1:** failing tests — every spec "Unit: IV screen" bullet, one
  TEST each. The synthetic non-monotone surface: build a small manual
  B-spline table whose vol axis has an engineered dip (fit a table from a
  crafted price tensor via `PriceTableBuilder` with a non-monotone-in-σ
  price array; if builder-side EEP flooring fights the shape, instead test
  through a minimal `PriceTable` stub type satisfying the surface concept —
  read `surface_concepts.hpp` and mimic `tests/` existing fakes). The
  defended-failure test (`detect_multiple_roots=false` returns the wrong
  root) documents the bug class.
- [ ] **Step 2:** implement.
- [ ] **Step 3:** targeted test + `bazel test //...` — PASS; verify the
  monotone-surface path count: exactly 17 extra `price` evals (count via
  the stub).
- [ ] **Step 4: Commit** `feat: multi-root screen and signed vega pre-check for interpolated IV`

### Task 11: Integration regressions, docs, benchmark evidence

**Files:**
- Test: `tests/adaptive_grid_builder_test.cc` (q0 regression),
  segmented tests per Task 8 file
- Modify: `docs/ARCHITECTURE.md` (§4: exact projection wording — July spec
  D1; adaptive loop description: holdout/retention/walk; the IV screen)
- Modify: `docs/API_GUIDE.md` (build_diagnostics usage snippet;
  detect_multiple_roots)
- Benchmarks: build + run `interp_iv_safety` before (main) and after
  (branch), `--path=q0`, `--path=bspline`, `--path=dividends`; store logs
  under the PR description per spec "Benchmark evidence".

- [ ] **Step 1:** q0 regression test (spec wording):

```cpp
// Regression: adaptive refinement returned its catastrophically-degraded
// final iteration (issue #434); retention must return the best candidate
// and IV inversion must never return a spurious low root.
TEST(AdaptiveRegressionTest, Q0BifurcationRetainedAndScreened) {
    // Scaled-down q=0 PUT config (moneyness .8-1.2 x 5, vols .1-.4 x 4,
    //  rates .02-.08 x 3, maturity_grid {.05,.1,.3,.6,1.}, target 2e-5,
    //  validation_samples 16, max_iter 4) — keep under ~60s.
    auto solver = mango::make_interpolated_iv_solver(config).value();
    auto diag = solver.build_diagnostics();
    ASSERT_TRUE(diag.has_value());
    EXPECT_LE(diag->achieved_max_error, 0.01);  // 100 bps sanity bound
    // Wrong-root region probe: price a sigma=0.30 30d K/S=0.8 put via FDM,
    // query IV: either near 0.30 (2e-2 abs) or MultipleRoots — never < 0.15.
}
```

- [ ] **Step 2:** run new + FULL suite: `bazel test //...`;
  `bazel build //benchmarks/... //src/python:mango_option`.
- [ ] **Step 3:** benchmark evidence runs (use `-c opt
  --copt=-Wno-error=maybe-uninitialized`; detached, logs kept); compare
  against the recorded prototype/base numbers; investigate any TV/K≥1e-3
  deterioration > 0.1 bps (spec gate); write the explanation paragraph for
  the PR (vanilla σ=30% slice is the expected watch item).
- [ ] **Step 4:** docs edits.
- [ ] **Step 5: Commit** `test: q0 bifurcation regression + docs and benchmark evidence`

---

## Verification (whole branch)

- `bazel test //...` — all green.
- `bazel build //benchmarks/... //src/python:mango_option` — clean.
- Benchmark evidence per spec (q0 ≤ 10 bps at σ=30% TV/K≥1e-4).
- Serialization goldens untouched.
- Spec acceptance criteria 1–8 walked one by one against the diff.
