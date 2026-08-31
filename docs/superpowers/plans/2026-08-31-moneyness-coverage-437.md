# Moneyness Coverage for Adaptive Cached Build (#437) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every auto-estimated batch PDE solve behind the B-spline price
table builders materializes a concrete shared grid that covers the full
log-moneyness axis, so `extract_tensor` never extrapolates tail tensor
values (GitHub issue #437).

**Architecture:** Hoist `ensure_moneyness_coverage` to
`mango::detail`, add a `materialize_covering_grid` helper that widens
accuracy → estimates the batch grid → wraps it as a concrete
`PDEGridConfig`, and pass that as `custom_grid` in both
`solve_missing_slices` auto-estimation branches (adaptive) and the
non-adaptive explicit-grid fallback. A gridless `solve_batch` is routed
per normalized (σ,r,q,T) group and re-estimates a per-σ-width grid, so
only a concrete custom grid realizes the coverage. Also add `build()`'s
upfront explicit-grid coverage rejection to the adaptive cached path.

**Tech Stack:** C++23, Bazel, GoogleTest. All work happens in the worktree
`/home/kai/work/mango-option/.worktrees/fix-437-moneyness-coverage` on
branch `fix/437-moneyness-coverage`.

**Spec:** `docs/superpowers/specs/2026-08-31-moneyness-coverage-437-design.md`
(read it first; its Decisions section D1–D6 binds this plan).

## Global Constraints

- Every solve on an estimated grid must go through a **concrete
  `PDEGridConfig` passed as `custom_grid`** — never `set_grid_accuracy` +
  gridless `solve_batch` (spec §2; normalized-chain routing defeats
  accuracy-only widening).
- Coverage/`n_sigma` derivations use the batch **actually being solved**
  (`missing_params` on the adaptive path), never `all_params` (spec §3).
- The `grid_meets_constraints` check keeps using `all_params` (spec §3).
- No new SPDX headers needed (no new files); keep existing ones intact.
- Library code must not printf/fprintf (repo rule); none is added.
- Commit messages: imperative mood, ≤50-char subject.
- Run all bazel commands from the worktree root.

---

### Task 1: Hoist `ensure_moneyness_coverage` + add `materialize_covering_grid`

**Files:**
- Modify: `src/option/table/bspline/bspline_builder.hpp` (add `<span>`
  include; declare two functions in the existing `namespace detail` block
  that ends at the `}  // namespace detail` before the grid-estimation
  doc comment)
- Modify: `src/option/table/bspline/bspline_builder.cpp` (define the two
  functions; delete the file-static template `ensure_moneyness_coverage`
  at lines ~233-257; update its two call sites at ~:266 and ~:365)
- Test: `tests/price_table_builder_test.cc`

**Interfaces:**
- Consumes: `GridAccuracyParams`, `PricingParams`, `PDEGridSpec`,
  `PDEGridConfig`, `estimate_batch_pde_grid` (all from
  `mango/option/grid_spec_types.hpp`, already included transitively).
- Produces (later tasks call these exactly):
  ```cpp
  namespace mango::detail {
  void ensure_moneyness_coverage(GridAccuracyParams& accuracy,
                                 std::span<const PricingParams> batch,
                                 std::span<const double> log_moneyness_grid);
  PDEGridSpec materialize_covering_grid(GridAccuracyParams accuracy,
                                        std::span<const PricingParams> batch,
                                        std::span<const double> log_moneyness_grid);
  }
  ```

- [ ] **Step 1: Write the failing tests**

Append to `tests/price_table_builder_test.cc` (match the file's existing
namespace/using conventions; it already includes the builder header):

```cpp
// ===========================================================================
// Regression tests for issue #437 (moneyness coverage helpers)
// ===========================================================================

// Regression: adaptive cached path skipped moneyness-coverage widening
// Bug (#437): solve_missing_slices never raised n_sigma, so the PDE domain
// could undershoot the moneyness axis and extract_tensor extrapolated tails.
TEST(EnsureMoneynessCoverage, WidensNSigmaWhenAxisUndershoots) {
    mango::GridAccuracyParams accuracy;  // default n_sigma = 5.0
    std::vector<mango::PricingParams> batch{mango::PricingParams(
        mango::OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.1,
                          .rate = 0.05, .dividend_yield = 0.0,
                          .option_type = mango::OptionType::PUT},
        0.20)};
    std::vector<double> log_m = {-0.51, 0.0, 0.51};
    mango::detail::ensure_moneyness_coverage(accuracy, batch, log_m);
    const double expected = 0.51 / (0.20 * std::sqrt(0.1)) * 1.1;
    EXPECT_NEAR(accuracy.n_sigma, expected, 1e-12);
}

TEST(EnsureMoneynessCoverage, LeavesNSigmaWhenCovered) {
    mango::GridAccuracyParams accuracy;  // default n_sigma = 5.0
    std::vector<mango::PricingParams> batch{mango::PricingParams(
        mango::OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                          .rate = 0.05, .dividend_yield = 0.0,
                          .option_type = mango::OptionType::PUT},
        0.50)};
    std::vector<double> log_m = {-0.51, 0.0, 0.51};
    mango::detail::ensure_moneyness_coverage(accuracy, batch, log_m);
    EXPECT_DOUBLE_EQ(accuracy.n_sigma, 5.0);  // 0.51/0.5*1.1 = 1.12 < 5
}

// Regression: exported helper must not read front()/back() of empty spans
// or divide a wide axis by the 1e-10 floor for an empty batch.
TEST(EnsureMoneynessCoverage, EmptyInputsAreNoOps) {
    mango::GridAccuracyParams accuracy;
    std::vector<mango::PricingParams> batch{mango::PricingParams(
        mango::OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.1,
                          .rate = 0.05, .dividend_yield = 0.0,
                          .option_type = mango::OptionType::PUT},
        0.20)};
    std::vector<double> log_m = {-0.51, 0.0, 0.51};

    mango::detail::ensure_moneyness_coverage(accuracy, {}, log_m);
    EXPECT_DOUBLE_EQ(accuracy.n_sigma, 5.0);

    mango::detail::ensure_moneyness_coverage(accuracy, batch, {});
    EXPECT_DOUBLE_EQ(accuracy.n_sigma, 5.0);
}

// Regression: widened accuracy alone is bypassed by per-normalized-group
// grid estimation; the materialized concrete grid is what guarantees
// coverage for EVERY slice of a multi-sigma batch.
TEST(MaterializeCoveringGrid, ConcreteGridCoversAxisForMultiSigmaBatch) {
    mango::GridAccuracyParams accuracy;
    std::vector<mango::PricingParams> batch;
    for (double sigma : {0.10, 0.15, 0.20}) {
        batch.push_back(mango::PricingParams(
            mango::OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.1,
                              .rate = 0.05, .dividend_yield = 0.0,
                              .option_type = mango::OptionType::PUT},
            sigma));
    }
    std::vector<double> log_m = {-0.51, 0.0, 0.51};
    auto spec = mango::detail::materialize_covering_grid(accuracy, batch, log_m);
    auto* config = std::get_if<mango::PDEGridConfig>(&spec);
    ASSERT_NE(config, nullptr);
    EXPECT_LE(config->grid_spec.x_min(), -0.51);
    EXPECT_GE(config->grid_spec.x_max(), 0.51);
    EXPECT_GT(config->n_time, 0u);
    EXPECT_TRUE(config->mandatory_times.empty());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `bazel test //tests:price_table_builder_test --test_output=errors`
Expected: FAIL to compile — `mango::detail::ensure_moneyness_coverage`
and `materialize_covering_grid` do not exist yet.

- [ ] **Step 3: Implement**

In `src/option/table/bspline/bspline_builder.hpp`:
1. Add `#include <span>` to the include block.
2. Inside the existing `namespace detail { ... }` block (the one holding
   `uniform_grid` / `log_uniform_grid` / `sqrt_uniform_grid`), append:

```cpp
/// Ensure accuracy.n_sigma is large enough that a shared grid estimated by
/// estimate_batch_pde_grid(batch, accuracy) for a normalized batch
/// (spot = strike = K_ref, x0 = 0) spans the whole log-moneyness axis:
/// that grid's baseline half-width is n_sigma * max(sigma*sqrt(T)) over
/// `batch`, and if it undershoots max(|m_front|, |m_back|), extract_tensor
/// would extrapolate. Callers must actually solve on such an estimated
/// grid (passed as a concrete custom grid): BatchAmericanOptionSolver's
/// gridless routing estimates per normalized group instead and does NOT
/// realize this width. No-op when either span is empty; expects the grid
/// sorted ascending (as validated table axes are).
void ensure_moneyness_coverage(GridAccuracyParams& accuracy,
                               std::span<const PricingParams> batch,
                               std::span<const double> log_moneyness_grid);

/// Widen `accuracy` for moneyness coverage, estimate the shared batch
/// grid, and return it as a concrete PDEGridConfig suitable for
/// solve_batch's custom_grid parameter (which propagates it into every
/// normalized group). mandatory_times stays empty: the batch solver
/// reconstructs per-contract dividend taus itself.
PDEGridSpec materialize_covering_grid(GridAccuracyParams accuracy,
                                      std::span<const PricingParams> batch,
                                      std::span<const double> log_moneyness_grid);
```

In `src/option/table/bspline/bspline_builder.cpp`:
1. Delete the file-static `template <size_t N> static void
   ensure_moneyness_coverage(...)` definition (currently lines ~233-257)
   and replace it with (non-template, at namespace `mango` scope, before
   `PriceTableBuilderND<N>::estimate_pde_grid`):

```cpp
namespace detail {

void ensure_moneyness_coverage(GridAccuracyParams& accuracy,
                               std::span<const PricingParams> batch,
                               std::span<const double> log_moneyness_grid)
{
    if (batch.empty() || log_moneyness_grid.empty()) return;

    const double log_m_min = log_moneyness_grid.front();
    const double log_m_max = log_moneyness_grid.back();
    const double required_half_width =
        std::max(std::abs(log_m_min), std::abs(log_m_max));

    // Compute max σ√T across the batch (floor to avoid division by zero)
    double max_sigma_sqrt_T = 0.0;
    for (const auto& p : batch) {
        max_sigma_sqrt_T = std::max(max_sigma_sqrt_T,
                                    p.volatility * std::sqrt(p.maturity));
    }
    max_sigma_sqrt_T = std::max(max_sigma_sqrt_T, 1e-10);

    constexpr double MARGIN = 1.1;  // 10% margin for boundary effects
    double required_n_sigma = (required_half_width / max_sigma_sqrt_T) * MARGIN;
    accuracy.n_sigma = std::max(accuracy.n_sigma, required_n_sigma);
}

PDEGridSpec materialize_covering_grid(GridAccuracyParams accuracy,
                                      std::span<const PricingParams> batch,
                                      std::span<const double> log_moneyness_grid)
{
    ensure_moneyness_coverage(accuracy, batch, log_moneyness_grid);
    auto [grid_spec, time_domain] = estimate_batch_pde_grid(batch, accuracy);
    return PDEGridConfig{grid_spec, time_domain.n_steps(), {}};
}

}  // namespace detail
```

2. Update the call site in `estimate_pde_grid` (~:266) from
   `ensure_moneyness_coverage<N>(accuracy, batch, axes);` to
   `detail::ensure_moneyness_coverage(accuracy, batch, axes.grids[0]);`
3. Update the call site in the explicit-grid fallback (~:365) the same
   way for now (`detail::ensure_moneyness_coverage(accuracy, batch,
   axes.grids[0]);` — Task 4 replaces this branch's solve with the
   materialized grid; this task only keeps behavior compiling).

- [ ] **Step 4: Run tests to verify they pass**

Run: `bazel test //tests:price_table_builder_test //tests:price_table_builder_custom_grid_test --test_output=errors`
Expected: PASS (new tests green, existing builder tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/option/table/bspline/bspline_builder.hpp \
        src/option/table/bspline/bspline_builder.cpp \
        tests/price_table_builder_test.cc
git commit -m "Hoist moneyness coverage into shared detail helpers"
```

---

### Task 2: Upfront explicit-grid coverage rejection in the adaptive path

**Files:**
- Modify: `src/option/table/bspline/bspline_adaptive.cpp`
  (`build_cached_surface`, after the `builder_result` error check)
- Test: `tests/adaptive_surface_build_integration_test.cc`

**Interfaces:**
- Consumes: `PDEGridConfig` / `PDEGridSpec` variant access;
  `PriceTableError{PriceTableErrorCode::InvalidConfig}`.
- Produces: `build_adaptive_bspline` now fails with `InvalidConfig` for
  explicit grids whose bounds do not cover the (headroom-widened) fit
  moneyness axis — matching `PriceTableBuilderND::build()`
  (`bspline_builder.cpp:73-84`).

- [ ] **Step 1: Write the failing test**

Append to `tests/adaptive_surface_build_integration_test.cc` (inside
`namespace mango { namespace { ... } }` like its neighbors):

```cpp
// Regression (#437): the adaptive cached path bypasses build()'s upfront
// explicit-grid coverage validation (bspline_builder.cpp:73-84), so an
// explicit PDE grid narrower than the moneyness fit axis was silently
// accepted and its tails cubic-spline-extrapolated by extract_tensor.
TEST(AdaptiveGridBuilderTest, RejectsExplicitGridNotCoveringMoneyness) {
    OptionGrid chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;
    chain.strikes = {60.0, 80.0, 100.0, 120.0, 140.0};
    chain.maturities = {0.05, 0.1};
    chain.implied_vols = {0.10, 0.15, 0.20};
    chain.rates = {0.03, 0.05};

    AdaptiveGridParams params;
    params.target_iv_error = 0.002;
    params.max_iter = 1;
    params.validation_samples = 4;

    // Half-width 0.25 vs required |ln(100/60)| ~= 0.51 (+ headroom).
    auto grid_spec = GridSpec<double>::sinh_spaced(-0.25, 0.25, 101, 2.0).value();
    auto result = build_adaptive_bspline(params, chain,
        PDEGridConfig{grid_spec, 200, {}}, OptionType::PUT);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bazel test //tests:adaptive_surface_build_integration_test --test_output=errors --test_filter='*RejectsExplicitGridNotCoveringMoneyness*'`
(If the target's sharding interferes with `--test_filter`, run without the
filter and check this test's row.)
Expected: FAIL — the unfixed build succeeds (extrapolating), so
`ASSERT_FALSE(result.has_value())` fires.

- [ ] **Step 3: Implement**

In `build_cached_surface` (bspline_adaptive.cpp), right after the
`builder_result` error check and before the cache/tau-grid handling, add:

```cpp
    // Upfront explicit-grid coverage check, mirroring
    // PriceTableBuilderND::build(): an explicit grid narrower than the
    // moneyness fit axis would be silently extrapolated by
    // extract_tensor.  Auto-estimated grids are widened instead
    // (solve_missing_slices).
    if (const auto* explicit_grid = std::get_if<PDEGridConfig>(&pde_grid)) {
        if (!m_grid.empty() &&
            (m_grid.front() < explicit_grid->grid_spec.x_min() ||
             m_grid.back() > explicit_grid->grid_spec.x_max())) {
            return std::unexpected(
                PriceTableError{PriceTableErrorCode::InvalidConfig});
        }
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `bazel test //tests:adaptive_surface_build_integration_test --test_output=errors`
Expected: PASS, including the pre-existing `BuildsWithSyntheticChain`
(its `[-3, 3]` grid covers that chain's axis).

- [ ] **Step 5: Commit**

```bash
git add src/option/table/bspline/bspline_adaptive.cpp \
        tests/adaptive_surface_build_integration_test.cc
git commit -m "Reject non-covering explicit grids in adaptive build"
```

---

### Task 3: Materialize covering grids in `solve_missing_slices`

**Files:**
- Modify: `src/option/table/bspline/bspline_adaptive.cpp`
  (`solve_missing_slices` — both branches — and its call in
  `build_cached_surface`)
- Test: `tests/adaptive_surface_build_integration_test.cc`

**Interfaces:**
- Consumes: `detail::materialize_covering_grid` (Task 1),
  `detail::ensure_moneyness_coverage` (Task 1).
- Produces: `solve_missing_slices` gains a `std::span<const double>
  m_grid` parameter (log-moneyness fit axis), inserted before
  `const PDEGridSpec& pde_grid`; `build_cached_surface` passes its
  `m_grid` vector.

- [ ] **Step 1: Write the failing end-to-end regression test**

Append to `tests/adaptive_surface_build_integration_test.cc`. Includes
needed at the top of the file (add if missing):
`#include "mango/option/american_option.hpp"` and `<cmath>`.

```cpp
namespace {

// Reference FDM price with a PINNED explicit configuration (spec: the
// tolerance floor must not drift if solve_american_option defaults
// change).
double fdm_reference_price(double spot, double strike, double tau,
                           double sigma, double rate) {
    PricingParams ref_params(
        OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
                   .rate = rate, .dividend_yield = 0.0,
                   .option_type = OptionType::PUT},
        sigma);
    auto solver = AmericanOptionSolver::create(
        ref_params, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::High)});
    EXPECT_TRUE(solver.has_value());
    auto ref = solver->solve();
    EXPECT_TRUE(ref.has_value());
    return ref->value_at(spot);
}

}  // namespace

// Regression (#437): the adaptive cached path (GridAccuracyParams branch)
// solved gridless, so per-normalized-group estimation gave each sigma
// slice half-width n_sigma * sigma * sqrt(tau); with tau_max = 0.1 and
// vols <= 0.20 that is ~0.316 at best against a fit axis reaching
// |ln(100/60)| ~= 0.51 (+ headroom), so extract_tensor extrapolated the
// tails.  Min-sigma assertions guard the routing defect specifically: a
// widening-only fix covers the max-sigma slice while every lower-sigma
// slice still extrapolates.
// Pre-fix max abs error on this branch's parent: <RECORD IN STEP 2>.
TEST(AdaptiveGridBuilderTest, TensorTailsMatchFdmAtExtremeMoneyness) {
    OptionGrid chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;
    chain.strikes = {60.0, 80.0, 100.0, 120.0, 140.0};
    chain.maturities = {0.05, 0.1};
    chain.implied_vols = {0.10, 0.15, 0.20};
    chain.rates = {0.03, 0.05};

    AdaptiveGridParams params;
    params.target_iv_error = 0.002;  // relaxed: accuracy is asserted below
    params.max_iter = 2;
    params.validation_samples = 8;

    auto result = build_adaptive_bspline(
        params, chain, make_grid_accuracy(GridAccuracyProfile::High),
        OptionType::PUT);
    ASSERT_TRUE(result.has_value());

    auto wrapper = make_bspline_surface(
        result->spline, result->K_ref, result->dividend_yield,
        OptionType::PUT);
    ASSERT_TRUE(wrapper.has_value());

    const auto& m_axis = result->axes.grids[0];
    const auto& tau_axis = result->axes.grids[1];
    const auto& vol_axis = result->axes.grids[2];
    const auto& rate_axis = result->axes.grids[3];
    const double K = result->K_ref;
    const double tau = tau_axis.back();
    const double r = rate_axis.front();

    // Tolerance in $ per K_ref=100 strike: pinned empirically in Step 4
    // (must sit well below the recorded pre-fix error).
    constexpr double TOL = 0.05;  // placeholder-pin: tighten in Step 4

    for (double m : {m_axis.front(), m_axis.back()}) {
        for (double sigma : {vol_axis.front(), vol_axis.back()}) {
            const double S = K * std::exp(m);
            const double ref = fdm_reference_price(S, K, tau, sigma, r);
            const double got = wrapper->price(S, K, tau, sigma, r);
            EXPECT_NEAR(got, ref, TOL)
                << "m=" << m << " sigma=" << sigma;
        }
    }
}
```

Also append the adaptive fallback-branch regression (round-3 review):

```cpp
// Regression (#437, fallback branch): an explicit grid that covers the
// fit axis but violates MAX_DX falls back to accuracy estimation, which
// also solved gridless.  The explicit bounds [-0.7, 0.7] are chosen snug:
// old fallback n_sigma = 0.7/(0.2*sqrt(0.1)) * 1.1 ~= 12.2 gives the
// min-sigma (0.10) group half-width ~0.39 < the ~0.55 fit-axis reach, so
// the min-sigma tail was extrapolated pre-fix while max-sigma was not.
TEST(AdaptiveGridBuilderTest, FallbackExplicitGridCoversMoneynessTails) {
    OptionGrid chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;
    chain.strikes = {60.0, 80.0, 100.0, 120.0, 140.0};
    chain.maturities = {0.05, 0.1};
    chain.implied_vols = {0.10, 0.15, 0.20};
    chain.rates = {0.03, 0.05};

    AdaptiveGridParams params;
    params.target_iv_error = 0.002;
    params.max_iter = 2;
    params.validation_samples = 8;

    // Covers the fit axis (upfront check passes) but 17 points over
    // width 1.4 makes max_dx > 0.05, forcing the fallback branch.
    auto grid_spec = GridSpec<double>::sinh_spaced(-0.7, 0.7, 17, 2.0).value();
    auto result = build_adaptive_bspline(params, chain,
        PDEGridConfig{grid_spec, 200, {}}, OptionType::PUT);
    ASSERT_TRUE(result.has_value());

    auto wrapper = make_bspline_surface(
        result->spline, result->K_ref, result->dividend_yield,
        OptionType::PUT);
    ASSERT_TRUE(wrapper.has_value());

    const auto& m_axis = result->axes.grids[0];
    const auto& vol_axis = result->axes.grids[2];
    const double K = result->K_ref;
    const double tau = result->axes.grids[1].back();
    const double r = result->axes.grids[3].front();
    constexpr double TOL = 0.05;  // placeholder-pin: tighten in Step 4

    for (double m : {m_axis.front(), m_axis.back()}) {
        for (double sigma : {vol_axis.front(), vol_axis.back()}) {
            const double S = K * std::exp(m);
            const double ref = fdm_reference_price(S, K, tau, sigma, r);
            const double got = wrapper->price(S, K, tau, sigma, r);
            EXPECT_NEAR(got, ref, TOL)
                << "m=" << m << " sigma=" << sigma;
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail; record pre-fix errors**

Run: `bazel test //tests:adaptive_surface_build_integration_test --test_output=all`
Expected: both new tests FAIL with `EXPECT_NEAR` violations at
extrapolated endpoints (min-σ at minimum). Record the largest reported
`|got - ref|` per test and write it into the test comments
(`Pre-fix max abs error ...`). If a test unexpectedly PASSES, the scenario
is not reproducing the domain undershoot — verify the fit-axis bounds and
grid choices against the numbers in the comments before proceeding (do
not weaken TOL to force a failure).

- [ ] **Step 3: Implement**

In `bspline_adaptive.cpp`:

1. Change the signature:

```cpp
/// Solve missing PDE slices, dispatching on PDEGridSpec variant.
BatchAmericanOptionResult solve_missing_slices(
    BatchAmericanOptionSolver& batch_solver,
    const std::vector<PricingParams>& missing_params,
    const std::vector<PricingParams>& all_params,
    std::span<const double> m_grid,
    const PDEGridSpec& pde_grid,
    const std::vector<double>& tau_grid)
```

2. Replace the `GridAccuracyParams` branch:

```cpp
    if (const auto* accuracy_grid = std::get_if<GridAccuracyParams>(&pde_grid)) {
        // A concrete covering grid, not set_grid_accuracy: gridless solves
        // are routed per normalized (sigma, r) group and re-estimate a
        // per-sigma-width grid there, undershooting the moneyness axis
        // for every sub-max sigma (issue #437).
        auto covering = detail::materialize_covering_grid(
            *accuracy_grid, missing_params, m_grid);
        return batch_solver.solve_batch(missing_params, true, nullptr,
                                        covering);
    }
```

3. In the `PDEGridConfig` fallback branch (the `!grid_meets_constraints`
   tail), keep the constraint check computed from `all_params` as-is, but:
   - compute a second max over the solved batch:

```cpp
        const double missing_max_sigma_sqrt_tau = std::ranges::max(
            missing_params | std::views::transform(sigma_sqrt_tau));
```

   - derive `required_n_sigma` from it (replacing the `all_params`-based
     `max_sigma_sqrt_tau` in that formula only):

```cpp
        if (missing_max_sigma_sqrt_tau >= 1e-10) {
            double required_n_sigma =
                (max_abs_x / missing_max_sigma_sqrt_tau) * DOMAIN_MARGIN_FACTOR;
            accuracy.n_sigma = std::max(5.0, required_n_sigma);
        }
```

   - replace the branch's tail (`batch_solver.set_grid_accuracy(accuracy);
     return batch_solver.solve_batch(missing_params, true);`) with:

```cpp
        auto covering = detail::materialize_covering_grid(
            accuracy, missing_params, m_grid);
        return batch_solver.solve_batch(missing_params, true, nullptr,
                                        covering);
```

4. Update the caller in `build_cached_surface`:

```cpp
        fresh_results = solve_missing_slices(
            batch_solver, missing_params, all_params, m_grid, pde_grid,
            tau_grid);
```

- [ ] **Step 4: Run tests, pin tolerances, verify green**

Run: `bazel test //tests:adaptive_surface_build_integration_test --test_output=all`
Expected: both new tests PASS. From the run's reported deviations, pin
each `TOL` at roughly 10× the observed post-fix max error, and confirm
that value is ≤ 1/5 of the recorded pre-fix error (spec: the tolerance
must discriminate domain coverage from ordinary interpolation error).
Replace the `placeholder-pin` comments with the final values and rerun to
confirm PASS.

- [ ] **Step 5: Run the neighboring suites**

Run: `bazel test //tests:adaptive_grid_builder_test //tests:price_table_builder_test //tests:price_table_builder_custom_grid_test --test_output=errors`
Expected: PASS (the adaptive standard path's PDE-solve counts may differ
from before — only failures matter here).

- [ ] **Step 6: Commit**

```bash
git add src/option/table/bspline/bspline_adaptive.cpp \
        tests/adaptive_surface_build_integration_test.cc
git commit -m "Materialize covering grids in adaptive cached solves"
```

---

### Task 4: Materialize the covering grid in the non-adaptive fallback (D6)

**Files:**
- Modify: `src/option/table/bspline/bspline_builder.cpp`
  (`PriceTableBuilderND<N>::solve_batch`, explicit-grid fallback tail,
  ~:364-368)
- Test: `tests/price_table_builder_custom_grid_test.cc`

**Interfaces:**
- Consumes: `detail::materialize_covering_grid` (Task 1);
  `mango::testing::PriceTableBuilderAccess<4>::{make_batch, solve_batch}`
  (existing test access shim, `tests/price_table_builder_test_access.hpp`).
- Produces: no signature changes; the fallback now returns results whose
  every slice grid spans the moneyness axis.

- [ ] **Step 1: Write the failing test**

Append to `tests/price_table_builder_custom_grid_test.cc` (it already
includes the access shim; match its namespace conventions):

```cpp
// Regression (#437 / D6): the non-adaptive explicit-grid fallback solved
// gridless (set_grid_accuracy + solve_batch), so normalized routing
// re-estimated per-sigma-width grids per group: with sigmas {0.10..0.30},
// tau_max 0.1, axis +/-0.51 and explicit bounds +/-0.6, the widened
// n_sigma ~= 6.96 gave the min-sigma group half-width ~0.22 < 0.51 and
// its tails were extrapolated.  The fallback must materialize one
// concrete covering grid for the whole batch.
TEST(PriceTableBuilderCustomGridTest, FallbackGridCoversAxisForAllSlices) {
    std::vector<double> m = {-0.51, -0.2, 0.0, 0.2, 0.51};
    std::vector<double> tau = {0.025, 0.05, 0.075, 0.1};
    std::vector<double> vol = {0.10, 0.15, 0.20, 0.30};
    std::vector<double> rate = {0.02, 0.03, 0.04, 0.05};

    // Covers the axis (passes build()'s upfront check) but 15 points over
    // width 1.2 makes max_dx > 0.05 -> stability constraints fail ->
    // fallback branch.
    auto grid_spec = mango::GridSpec<double>::sinh_spaced(-0.6, 0.6, 15, 2.0).value();

    auto setup = mango::PriceTableBuilder::from_vectors(
        m, tau, vol, rate, /*K_ref=*/100.0,
        mango::PDEGridSpec{mango::PDEGridConfig{grid_spec, 200, {}}},
        mango::OptionType::PUT);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = setup.value();

    auto batch = mango::testing::PriceTableBuilderAccess<4>::make_batch(
        builder, axes);
    auto results = mango::testing::PriceTableBuilderAccess<4>::solve_batch(
        builder, batch, axes);

    ASSERT_EQ(results.results.size(), batch.size());
    for (size_t i = 0; i < results.results.size(); ++i) {
        ASSERT_TRUE(results.results[i].has_value()) << "slice " << i;
        auto x = results.results[i]->grid()->x();
        EXPECT_LE(x.front(), axes.grids[0].front()) << "slice " << i;
        EXPECT_GE(x.back(), axes.grids[0].back()) << "slice " << i;
    }
}
```

(If `from_vectors` requires the dividend-yield argument positionally in
this codebase revision, pass `0.0` for it — check the declaration in
`bspline_builder.hpp:457` and match it exactly.)

- [ ] **Step 2: Run test to verify it fails**

Run: `bazel test //tests:price_table_builder_custom_grid_test --test_output=all --test_filter='*FallbackGridCoversAxisForAllSlices*'`
Expected: FAIL — the min-σ slices' `x.front()/x.back()` do not span
`[-0.51, 0.51]`.

- [ ] **Step 3: Implement**

In `PriceTableBuilderND<N>::solve_batch`'s fallback tail
(bspline_builder.cpp), replace:

```cpp
                // Also ensure coverage of the moneyness axis
                detail::ensure_moneyness_coverage(accuracy, batch, axes.grids[0]);

                solver.set_grid_accuracy(accuracy);
                return solver.solve_batch(batch, true, nullptr);
```

with:

```cpp
                // Materialize one concrete covering grid for the whole
                // batch: a gridless solve is routed per normalized
                // (sigma, r) group and would re-estimate per-sigma-width
                // grids there, undershooting the moneyness axis for every
                // sub-max sigma (issue #437).
                auto covering = detail::materialize_covering_grid(
                    accuracy, batch, axes.grids[0]);
                return solver.solve_batch(batch, true, nullptr, covering);
```

(All table batch params share `maturity = axes.grids[1].back()`, so the
estimated time domain already spans the maturity axis — the equal-maturity
premise from the spec.)

- [ ] **Step 4: Run tests to verify green**

Run: `bazel test //tests:price_table_builder_custom_grid_test //tests:price_table_builder_custom_grid_advanced_test //tests:price_table_builder_custom_grid_diagnosis_test //tests:price_table_builder_test --test_output=errors`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/option/table/bspline/bspline_builder.cpp \
        tests/price_table_builder_custom_grid_test.cc
git commit -m "Materialize covering grid in builder fallback"
```

---

### Task 5: Full verification (pre-PR checklist)

**Files:** none modified (verification only; fix regressions if any).

- [ ] **Step 1: Full test suite**

Run: `bazel test //... --test_output=errors`
Expected: all tests pass (baseline on this worktree before changes:
148/148). Investigate any failure — the adaptive standard path and the
builder fallback now solve on different (wider) grids, so a legitimately
affected accuracy pin must be understood, not blindly re-pinned; anything
surprising goes through superpowers:systematic-debugging.

- [ ] **Step 2: Benchmarks + Python bindings compile (CI parity)**

Run: `bazel build //benchmarks/... //src/python:mango_option`
Expected: builds succeed without new warnings.

- [ ] **Step 3: Commit any stragglers**

Only if fixes were needed in Steps 1-2; otherwise nothing to commit.

## Deviations from plan

- **Task 3 fallback-test explicit grid bounds**: changed from the planned
  `[-0.7, 0.7]` to `[-0.6, 0.6]` because `make_batch()` pins batch maturity
  to the widened fit tau axis (~0.5), not the chain's raw max maturity; the
  planned bounds passed even pre-fix under that widened maturity, so the
  bounds were narrowed to reproduce the bug.
- **Task 3/final-review tolerances**: pinned at `1e-5` and `1e-3` rather
  than the "~10x post-fix" guideline, to stay robust to cross-toolchain
  numerical noise. Both remain far below the recorded pre-fix errors
  (0.4263 and 0.0569 respectively).
- **Task 4 test recalibration**: recalibrated to `tau={0.25, 0.5, 0.75,
  1.0}`, `vol={0.08, 0.12, 0.20, 0.30}` because the originally planned
  config was normalized-chain-INELIGIBLE (margin 0.22 < 0.35) and could not
  reproduce the bug. With the recalibrated config, eligibility now passes
  (0.40 >= 0.35) while the sigma=0.08 groups undershoot (0.40 < 0.51).
