# NaN Input Guards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every NaN entry point in the interpolation/pricing stack fail loudly (error on build paths, NaN on query paths) instead of silently producing 0.0 — closing issues #425, #426, #466.

**Architecture:** Build-time guards return errors through the existing error types (`std::optional<std::string_view>` for CubicSpline, new `std::expected<..., InterpolationError>` for ChebyshevInterpolant, `SolverError` for the FDM solve). Query-time NaN propagates naturally through the unchanged B-spline/Chebyshev kernels; three `std::max(0.0, v)` masking boundaries become NaN-preserving floors. The adaptive Chebyshev cache fails extraction on invalid slices instead of zero-filling.

**Tech Stack:** C++23, Bazel, GoogleTest, Google Benchmark.

**Spec:** `docs/plans/2026-08-31-nan-input-guards-design.md` (rev 4, design-review-approved). Read it before starting — decisions D1–D8 bind this plan.

## Global Constraints

- **Perf (D4):** no query-path changes beyond the NaN-preserving floors (one predictable `isnan` branch each). Acceptance: `latency_sweep` price-query medians regress < 3% vs the Task-1 baseline.
- **Signed zero (D8):** floors stay `+0.0`-canonical for all non-NaN input: `std::isnan(v) ? v : std::max(0.0, v)`. Never a bare `std::max(v, 0.0)`.
- **Enum ordinals:** `SolverErrorCode::NonFiniteSolution` is appended AFTER `Unknown` (Python exposes values; ordinals must not shift).
- Every new test follows the CLAUDE.md regression-test format (comment: `// Regression: ...` / `// Bug: ...`).
- Commit messages: imperative mood, ≤50-char subject.
- All commands run from the worktree root. Full CI parity before PR: `bazel test //...`, `bazel build //benchmarks/...`, `bazel build //src/python:mango_option`.

---

### Task 1: Capture performance baseline

**Files:** none modified (output goes to `/tmp/codex-skills/5d8b8f12-dce6-4646-afc8-3120ada90939/` — NOT committed).

**Interfaces:** Produces `baseline-latency.txt` and `baseline-bspline.txt` consumed by Task 7.

- [ ] **Step 1: Run the surface-price benchmark on the unmodified tree**

Run:
```bash
bazel run -c opt //benchmarks:latency_sweep -- --benchmark_repetitions=3 --benchmark_report_aggregates_only=true > /tmp/codex-skills/5d8b8f12-dce6-4646-afc8-3120ada90939/baseline-latency.txt 2>&1
tail -30 /tmp/codex-skills/5d8b8f12-dce6-4646-afc8-3120ada90939/baseline-latency.txt
```
Expected: benchmark table with median timings (this benchmark takes minutes; that's normal).

- [ ] **Step 2: Run the raw B-spline eval control benchmark**

Run:
```bash
bazel run -c opt //benchmarks:bspline_template_vs_hardcoded -- --benchmark_repetitions=3 --benchmark_report_aggregates_only=true > /tmp/codex-skills/5d8b8f12-dce6-4646-afc8-3120ada90939/baseline-bspline.txt 2>&1
tail -20 /tmp/codex-skills/5d8b8f12-dce6-4646-afc8-3120ada90939/baseline-bspline.txt
```
Expected: benchmark table. No commit for this task.

---

### Task 2: CubicSpline finiteness guards (#425)

**Files:**
- Modify: `src/math/cubic_spline_solver.hpp` (`build()` ~line 74, `rebuild_same_grid()` ~line 153)
- Test: `tests/thomas_cubic_spline_test.cc`

**Interfaces:**
- Produces: `build()`/`rebuild_same_grid()` return an error `std::optional<std::string_view>` for non-finite input. Error strings: `"X contains non-finite values (NaN or Inf)"`, `"Y contains non-finite values (NaN or Inf)"`. Tasks 3 and 6 rely on NaN input marking a spline build as failed.

- [ ] **Step 1: Write the failing tests** (append to `tests/thomas_cubic_spline_test.cc`, matching its existing TEST style and includes; add `#include <cmath>` if absent)

```cpp
// ===========================================================================
// Regression tests for issue #425 (silent NaN coefficients)
// ===========================================================================

// Regression: build() silently produced NaN coefficients on NaN y input
// Bug: no finiteness validation; Thomas solve propagated NaN, build returned success
TEST(CubicSplineNaNGuardTest, BuildRejectsNaNY) {
    mango::CubicSpline<double> spline;
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> y = {0.0, 1.0, std::nan(""), 3.0};
    auto err = spline.build(x, y);
    EXPECT_TRUE(err.has_value());
}

TEST(CubicSplineNaNGuardTest, BuildRejectsInfY) {
    mango::CubicSpline<double> spline;
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> y = {0.0, 1.0, std::numeric_limits<double>::infinity(), 3.0};
    EXPECT_TRUE(spline.build(x, y).has_value());
}

// Regression: NaN x passed the strictly-increasing check
// Bug: every comparison with NaN is false, so {0, NaN, 2} passed monotonicity
TEST(CubicSplineNaNGuardTest, BuildRejectsNaNX) {
    mango::CubicSpline<double> spline;
    std::vector<double> x = {0.0, std::nan(""), 2.0, 3.0};
    std::vector<double> y = {0.0, 1.0, 2.0, 3.0};
    EXPECT_TRUE(spline.build(x, y).has_value());
}

TEST(CubicSplineNaNGuardTest, BuildRejectsInfX) {
    mango::CubicSpline<double> spline;
    std::vector<double> x = {0.0, 1.0, 2.0, std::numeric_limits<double>::infinity()};
    std::vector<double> y = {0.0, 1.0, 2.0, 3.0};
    EXPECT_TRUE(spline.build(x, y).has_value());
}

TEST(CubicSplineNaNGuardTest, RebuildSameGridRejectsNaNY) {
    mango::CubicSpline<double> spline;
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> y = {0.0, 1.0, 2.0, 3.0};
    ASSERT_FALSE(spline.build(x, y).has_value());
    std::vector<double> y_bad = {0.0, std::nan(""), 2.0, 3.0};
    EXPECT_TRUE(spline.rebuild_same_grid(std::span<const double>(y_bad)).has_value());
}

TEST(CubicSplineNaNGuardTest, RebuildSameGridRejectsInfY) {
    mango::CubicSpline<double> spline;
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> y = {0.0, 1.0, 2.0, 3.0};
    ASSERT_FALSE(spline.build(x, y).has_value());
    std::vector<double> y_bad = {0.0, 1.0, -std::numeric_limits<double>::infinity(), 3.0};
    EXPECT_TRUE(spline.rebuild_same_grid(std::span<const double>(y_bad)).has_value());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `bazel test //tests:thomas_cubic_spline_test --test_output=all --test_filter='*NaNGuard*'`
Expected: FAIL (guards don't exist yet; builds currently succeed).

- [ ] **Step 3: Implement the guards** in `src/math/cubic_spline_solver.hpp`

In `build()`, immediately after the `n < 2` check and BEFORE the strictly-increasing loop:
```cpp
        // Non-finite input silently poisons the Thomas solve (issue #425);
        // NaN also defeats the monotonicity check below (all comparisons false)
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(x[i])) {
                return "X contains non-finite values (NaN or Inf)";
            }
        }
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(y[i])) {
                return "Y contains non-finite values (NaN or Inf)";
            }
        }
```
In `rebuild_same_grid()`, after the size check and BEFORE `std::copy`:
```cpp
        for (size_t i = 0; i < y.size(); ++i) {
            if (!std::isfinite(y[i])) {
                return "Y contains non-finite values (NaN or Inf)";
            }
        }
```
Confirm the header includes `<cmath>` (add if missing).

- [ ] **Step 4: Run tests to verify they pass**

Run: `bazel test //tests:thomas_cubic_spline_test --test_output=errors`
Expected: PASS (all tests in the target, old and new).

- [ ] **Step 5: Verify no downstream breakage and commit**

Run: `bazel test //tests:cubic_spline_2d_test //tests:american_option_test //tests:chebyshev_pde_cache_test --test_output=errors`
Expected: PASS.
```bash
git add src/math/cubic_spline_solver.hpp tests/thomas_cubic_spline_test.cc
git commit -m "Reject non-finite input in CubicSpline build paths"
```

---

### Task 3: FDM output validation — D7 (`NonFiniteSolution`)

**Files:**
- Modify: `src/support/error_types.hpp` (SolverErrorCode enum, ~line 14)
- Modify: `src/option/american_option.hpp` (add `detail` declaration)
- Modify: `src/option/american_option.cpp` (implement + call before `return AmericanOptionResult(grid, params_);` ~line 517)
- Modify: `src/python/mango_bindings.cpp` (~line 698)
- Test: `tests/american_option_test.cc`

**Interfaces:**
- Produces: `SolverErrorCode::NonFiniteSolution` (appended after `Unknown`); `mango::detail::validate_finite_solution(std::span<const double>, std::span<const double>) -> std::optional<SolverError>` declared in `american_option.hpp`. Task 4's `value_at` NaN test relies on solve() never returning a result with non-finite solution data.

- [ ] **Step 1: Write the failing tests** (append to `tests/american_option_test.cc`)

```cpp
// ===========================================================================
// Regression tests for issue #425/#466 family: D7 solver output validation
// ===========================================================================

// Regression: a NaN PDE solution built an AmericanOptionResult whose empty
// spline evaluated to 0.0 — a plausible price — in opt builds
// Bug: build_spline() checked the CubicSpline error only via assert
TEST(ValidateFiniteSolutionTest, RejectsNaNInFinalSolution) {
    std::vector<double> final_u = {1.0, std::nan(""), 3.0};
    std::vector<double> prev_u = {1.0, 2.0, 3.0};
    auto err = mango::detail::validate_finite_solution(final_u, prev_u);
    ASSERT_TRUE(err.has_value());
    EXPECT_EQ(err->code, mango::SolverErrorCode::NonFiniteSolution);
}

TEST(ValidateFiniteSolutionTest, RejectsInfInPrevSolution) {
    std::vector<double> final_u = {1.0, 2.0, 3.0};
    std::vector<double> prev_u = {1.0, std::numeric_limits<double>::infinity(), 3.0};
    auto err = mango::detail::validate_finite_solution(final_u, prev_u);
    ASSERT_TRUE(err.has_value());
    EXPECT_EQ(err->code, mango::SolverErrorCode::NonFiniteSolution);
}

TEST(ValidateFiniteSolutionTest, AcceptsFiniteSolution) {
    std::vector<double> final_u = {1.0, 2.0, 3.0};
    std::vector<double> prev_u = {0.5, 1.5, 2.5};
    EXPECT_FALSE(mango::detail::validate_finite_solution(final_u, prev_u).has_value());
}
```

- [ ] **Step 2: Run tests to verify they fail to compile**

Run: `bazel test //tests:american_option_test --test_output=errors --test_filter='*ValidateFiniteSolution*'`
Expected: BUILD FAILURE (`validate_finite_solution` not declared).

- [ ] **Step 3: Implement**

`src/support/error_types.hpp` — append to the enum (keep `Unknown` where it is):
```cpp
enum class SolverErrorCode {
    ConvergenceFailure,
    LinearSolveFailure,
    InvalidConfiguration,
    Unknown,
    NonFiniteSolution  ///< PDE solve produced non-finite solution values
                       ///< (appended after Unknown: ordinals are exposed to Python)
};
```

`src/option/american_option.hpp` — in `namespace mango`, near the other free-function declarations:
```cpp
namespace detail {
/// D7 (issues #425/#466 family): validate the vectors used to build the
/// result's value/theta splines. Non-finite PDE output must fail the solve,
/// not silently become an empty spline that evaluates to 0.
[[nodiscard]] std::optional<SolverError> validate_finite_solution(
    std::span<const double> final_solution,
    std::span<const double> prev_solution);
}  // namespace detail
```
(Ensure `<optional>` and `<span>` are included; `SolverError` comes via existing includes.)

`src/option/american_option.cpp` — implementation (anywhere in the `mango` namespace scope) and call site:
```cpp
std::optional<SolverError> detail::validate_finite_solution(
    std::span<const double> final_solution,
    std::span<const double> prev_solution)
{
    for (double v : final_solution) {
        if (!std::isfinite(v)) {
            return SolverError{SolverErrorCode::NonFiniteSolution};
        }
    }
    for (double v : prev_solution) {
        if (!std::isfinite(v)) {
            return SolverError{SolverErrorCode::NonFiniteSolution};
        }
    }
    return std::nullopt;
}
```
Replace `return AmericanOptionResult(grid, params_);` (~line 517) with:
```cpp
    if (auto err = detail::validate_finite_solution(
            grid->solution(), grid->solution_prev())) {
        return std::unexpected(*err);
    }

    return AmericanOptionResult(grid, params_);
```

`src/python/mango_bindings.cpp` (~line 698) — after the `Unknown` value line:
```cpp
        .value("NonFiniteSolution", mango::SolverErrorCode::NonFiniteSolution)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `bazel test //tests:american_option_test --test_output=errors && bazel build //src/python:mango_option`
Expected: PASS / builds.

- [ ] **Step 5: Commit**

```bash
git add src/support/error_types.hpp src/option/american_option.hpp src/option/american_option.cpp src/python/mango_bindings.cpp tests/american_option_test.cc
git commit -m "Fail solve() when PDE output is non-finite"
```

---

### Task 4: NaN-preserving floors + raw-eval locks (#466)

**Files:**
- Modify: `src/option/table/eep/eep_decomposer.hpp:17-19` (`eep_floor`)
- Modify: `src/option/table/transform_leaf.hpp:28-33` (`TransformLeaf::price`)
- Modify: `src/option/american_option_result.cpp:45` (`value_at`)
- Test: `tests/eep_decomposer_test.cc`, `tests/bspline_nd_test.cc`, `tests/chebyshev_surface_test.cc`, `tests/american_option_result_test.cc`

**Interfaces:**
- Consumes: nothing from earlier tasks (independent).
- Produces: NaN-in→NaN-out at `eep_floor`, `TransformLeaf::price`, `AmericanOptionResult::value_at`. Finite behavior (incl. `+0.0` canonicalization) unchanged. Task 5's Chebyshev guard relies on `eep_floor` no longer flooring NaN to 0.0 at build time.

- [ ] **Step 1: Write the failing tests**

`tests/eep_decomposer_test.cc` (append; keep existing signed-zero tests untouched):
```cpp
// Regression: eep_floor masked NaN as +0.0 at table-build time (issue #466)
// Bug: std::max(0.0, NaN) returns its first argument, hiding NaN from the
// downstream build_from_values finiteness guard
TEST(EEPFloorTest, NaNPropagates) {
    EXPECT_TRUE(std::isnan(mango::eep_floor(std::nan(""))));
}
```

`tests/bspline_nd_test.cc` (append; uses the existing fixture helpers):
```cpp
// ===========================================================================
// Regression tests for issue #466 (NaN query coordinates)
// ===========================================================================

// Regression: issue #466 claimed eval returns 0.0 for NaN queries; empirical
// check showed raw eval already propagates NaN (degree-1 Cox-de Boor terms
// compute (NaN - t)/den * 0 = NaN). This locks that behavior against future
// basis "optimizations" — the 0.0 masking was at the std::max boundaries above.
TEST_F(BSplineNDTest, EvalPropagatesNaNQuery) {
    auto grid = create_uniform_grid(0.0, 1.0, 10);
    auto knots = create_clamped_knots(grid);
    std::vector<double> coeffs(grid.size(), 1.5);
    auto spline = BSplineND<double, 1>::create({grid}, {knots}, coeffs);
    ASSERT_TRUE(spline.has_value());
    const double nan = std::nan("");
    EXPECT_TRUE(std::isnan(spline->eval({nan})));
    EXPECT_TRUE(std::isnan(spline->eval_partial(0, {nan})));
    EXPECT_TRUE(std::isnan(spline->eval_second_partial(0, {nan})));

    // NaN in each coordinate position of a 2D spline
    auto grid2 = create_uniform_grid(0.0, 2.0, 8);
    auto knots2 = create_clamped_knots(grid2);
    std::vector<double> c2(grid.size() * grid2.size(), 2.0);
    auto sp2 = BSplineND<double, 2>::create({grid, grid2}, {knots, knots2}, c2);
    ASSERT_TRUE(sp2.has_value());
    EXPECT_TRUE(std::isnan(sp2->eval({nan, 0.5})));
    EXPECT_TRUE(std::isnan(sp2->eval({0.5, nan})));
}

// ±Inf keeps clamp-to-edge semantics (only NaN propagates)
TEST_F(BSplineNDTest, EvalStillClampsInfQuery) {
    auto grid = create_uniform_grid(0.0, 1.0, 10);
    auto knots = create_clamped_knots(grid);
    std::vector<double> coeffs(grid.size(), 1.5);
    auto spline = BSplineND<double, 1>::create({grid}, {knots}, coeffs);
    ASSERT_TRUE(spline.has_value());
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_DOUBLE_EQ(spline->eval({inf}), spline->eval({1.0}));
    EXPECT_DOUBLE_EQ(spline->eval({-inf}), spline->eval({0.0}));
}
```

`tests/chebyshev_surface_test.cc` (append; mirror the file's existing interpolant-building style — it already constructs `ChebyshevInterpolant<4, RawTensor<4>>` and `TransformLeaf`; note after Task 5 the `::build` factory returns `std::expected`, but this task runs first, so call the factory exactly as the file's existing tests do):
```cpp
// ===========================================================================
// Regression tests for issue #466 (TransformLeaf masked NaN as 0.0)
// ===========================================================================

// Regression: TransformLeaf::price returned 0.0 for NaN interpolant output
// Bug: std::max(0.0, raw) returns 0.0 when raw is NaN
TEST(TransformLeafNaNTest, PricePropagatesNaNAndKeepsFloor) {
    auto f = [](std::array<double, 4>) { return -1.0; };  // always-negative raw
    mango::Domain<4> dom{.lo = {-0.7, 0.05, 0.1, 0.0}, .hi = {0.7, 2.0, 0.5, 0.08}};
    std::array<size_t, 4> npts = {5, 5, 5, 5};
    auto interp = mango::ChebyshevInterpolant<4, mango::RawTensor<4>>::build(f, dom, npts);
    mango::ChebyshevTransformLeaf leaf(std::move(interp),
                                       mango::StandardTransform4D{}, 100.0);

    // Finite query over a negative raw value: floored to +0.0 (not -0.0)
    double p = leaf.price(100.0, 100.0, 1.0, 0.2, 0.05);
    EXPECT_DOUBLE_EQ(p, 0.0);
    EXPECT_FALSE(std::signbit(p));

    // NaN spot propagates instead of masking to 0.0
    EXPECT_TRUE(std::isnan(leaf.price(std::nan(""), 100.0, 1.0, 0.2, 0.05)));

    // Inf spot still clamps to the domain edge (finite output)
    EXPECT_TRUE(std::isfinite(
        leaf.price(std::numeric_limits<double>::infinity(), 100.0, 1.0, 0.2, 0.05)));
}
```
(If `ChebyshevTransformLeaf` / `StandardTransform4D` are named differently in this test file's includes, follow the file's existing leaf-construction test — the assertions above are what matter.)

`tests/american_option_result_test.cc` (append; use the file's existing solve fixture/params style):
```cpp
// Regression: value_at(NaN) returned 0.0 (issue #466 family)
// Bug: std::max(0.0, spline eval) masked the NaN from log(NaN/K)
TEST(AmericanOptionResultNaNTest, ValueAtNaNSpotReturnsNaN) {
    mango::PricingParams params(
        mango::OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                          .rate = 0.05, .dividend_yield = 0.02,
                          .option_type = mango::OptionType::PUT},
        0.20);
    auto result = mango::solve_american_option(params);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(std::isnan(result->value_at(std::nan(""))));
}
```

- [ ] **Step 2: Run tests — raw-eval locks should PASS already, floor tests should FAIL**

Run: `bazel test //tests:bspline_nd_test //tests:eep_decomposer_test //tests:chebyshev_surface_test //tests:american_option_result_test --test_output=errors --cache_test_results=no`
Expected: `bspline_nd_test` PASSES (locks existing behavior); `EEPFloorTest.NaNPropagates`, `TransformLeafNaNTest`, `AmericanOptionResultNaNTest` FAIL (0.0 instead of NaN).

- [ ] **Step 3: Implement the three floors**

`src/option/table/eep/eep_decomposer.hpp`:
```cpp
inline double eep_floor(double eep_raw) {
    // NaN-preserving (issue #466): max(0.0, NaN) would mask NaN as +0.0 and
    // hide it from build-time finiteness guards. +0.0 canonicalization for
    // finite input is contractual (EEPFloorTest.BothSignedZerosProducePositiveZero).
    return std::isnan(eep_raw) ? eep_raw : std::max(0.0, eep_raw);
}
```
(Add `#include <cmath>` if absent.)

`src/option/table/transform_leaf.hpp` — in `price()`:
```cpp
        double raw = interp_.eval(coords);
        // NaN-preserving floor (issue #466): keep +0.0 canonicalization for
        // finite raw, propagate NaN instead of masking it as a 0.0 price
        double floored = std::isnan(raw) ? raw : std::max(0.0, raw);
        return floored * strike / K_ref_;
```
Also update the class doc comment line `/// Produces: max(0, interp(coords)) * strike/K_ref.` to mention NaN propagation. (Add `#include <cmath>` if absent.)

`src/option/american_option_result.cpp` (~line 45):
```cpp
    double raw = spline_.eval(x);
    // NaN-preserving floor (issue #466): NaN spot must not price as 0.0
    double value_normalized = std::isnan(raw) ? raw : std::max(0.0, raw);
```

- [ ] **Step 4: Run tests to verify everything passes**

Run: `bazel test //tests:bspline_nd_test //tests:eep_decomposer_test //tests:chebyshev_surface_test //tests:american_option_result_test --test_output=errors`
Expected: PASS (including the pre-existing signed-zero tests).

- [ ] **Step 5: Commit**

```bash
git add src/option/table/eep/eep_decomposer.hpp src/option/table/transform_leaf.hpp src/option/american_option_result.cpp tests/eep_decomposer_test.cc tests/bspline_nd_test.cc tests/chebyshev_surface_test.cc tests/american_option_result_test.cc
git commit -m "Propagate NaN through price floors instead of masking as 0"
```

---

### Task 5: ChebyshevInterpolant `std::expected` build validation (#426)

**Files:**
- Modify: `src/math/chebyshev/chebyshev_interpolant.hpp` (both factories)
- Modify: `src/math/chebyshev/BUILD.bazel` (`chebyshev_interpolant` target deps)
- Modify (call sites): `src/option/table/chebyshev/chebyshev_adaptive.cpp` (3 sites + `build_segment_leaves` signature + its 2 callers), `src/option/table/chebyshev/chebyshev_table_builder.cpp:185`, `src/option/price_table_factory.cpp:648`, `src/option/table/serialization/reconstruct.hpp:125`, `benchmarks/latency_sweep.cc:416`, `benchmarks/greek_latency.cc:511`
- Modify (test call sites — unwrap EVERY factory expression in each file, not just one): `tests/chebyshev_interpolant_test.cc` (7 sampling uses), `tests/chebyshev_surface_test.cc` (incl. the Task-4 test), `tests/parquet_io_test.cc` (2 uses)
- Test: `tests/chebyshev_interpolant_test.cc`

**Interfaces:**
- Consumes: `convert_to_price_table_error(const InterpolationError&)` and `detail::to_validation_error(...)` (both exist).
- Produces:
```cpp
template <typename... Args>
[[nodiscard]] static std::expected<ChebyshevInterpolant, InterpolationError>
build_from_values(std::span<const double> values, const Domain<N>& domain,
                  const std::array<size_t, N>& num_pts, Args&&... storage_args);
// same expected return for the sampling build(f, domain, num_pts, ...)
```
`detail`-free public contract; class gains `requires (N >= 1)`. Task 6 relies on `build_segment_leaves` returning `std::expected<std::vector<ChebyshevSegmentedLeaf>, PriceTableError>`.

- [ ] **Step 1: Write the failing unit tests** (append to `tests/chebyshev_interpolant_test.cc`)

```cpp
// ===========================================================================
// Regression tests for issue #426 (build_from_values silently fit NaN input)
// ===========================================================================

// Regression: builds succeeded with 15-20% NaN input during the #419 incident
// Bug: no input validation and no error path (object returned directly)
TEST(ChebyshevInterpolantGuardTest, BuildFromValuesRejectsNaN) {
    mango::Domain<2> dom{.lo = {0.0, 0.0}, .hi = {1.0, 1.0}};
    std::vector<double> values(9, 0.0);
    values[4] = std::nan("");
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build_from_values(
        values, dom, {3, 3});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::NaNInput);
    EXPECT_EQ(r.error().index, 4u);
}

TEST(ChebyshevInterpolantGuardTest, BuildFromValuesRejectsInf) {
    mango::Domain<2> dom{.lo = {0.0, 0.0}, .hi = {1.0, 1.0}};
    std::vector<double> values(9, 0.0);
    values[2] = std::numeric_limits<double>::infinity();
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build_from_values(
        values, dom, {3, 3});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::InfInput);
    EXPECT_EQ(r.error().index, 2u);
}

TEST(ChebyshevInterpolantGuardTest, RejectsSizeMismatch) {
    mango::Domain<2> dom{.lo = {0.0, 0.0}, .hi = {1.0, 1.0}};
    std::vector<double> values(8, 0.0);  // needs 9
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build_from_values(
        values, dom, {3, 3});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::ValueSizeMismatch);
}

TEST(ChebyshevInterpolantGuardTest, RejectsNumPtsBelowTwo) {
    mango::Domain<2> dom{.lo = {0.0, 0.0}, .hi = {1.0, 1.0}};
    std::vector<double> values(3, 0.0);
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build_from_values(
        values, dom, {1, 3});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::InsufficientGridPoints);
}

TEST(ChebyshevInterpolantGuardTest, RejectsNaNDomain) {
    mango::Domain<2> dom{.lo = {std::nan(""), 0.0}, .hi = {1.0, 1.0}};
    std::vector<double> values(9, 0.0);
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build_from_values(
        values, dom, {3, 3});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::NaNInput);
}

TEST(ChebyshevInterpolantGuardTest, RejectsInfDomain) {
    mango::Domain<2> dom{.lo = {0.0, 0.0},
                         .hi = {std::numeric_limits<double>::infinity(), 1.0}};
    std::vector<double> values(9, 0.0);
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build_from_values(
        values, dom, {3, 3});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::InfInput);
}

TEST(ChebyshevInterpolantGuardTest, RejectsReversedDomain) {
    mango::Domain<2> dom{.lo = {0.0, 1.0}, .hi = {1.0, 0.5}};  // axis 1 reversed
    std::vector<double> values(9, 0.0);
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build_from_values(
        values, dom, {3, 3});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::GridNotSorted);
}

TEST(ChebyshevInterpolantGuardTest, RejectsZeroWidthDomain) {
    mango::Domain<2> dom{.lo = {0.0, 0.5}, .hi = {1.0, 0.5}};
    std::vector<double> values(9, 0.0);
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build_from_values(
        values, dom, {3, 3});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::ZeroWidthGrid);
}

// Regression: unchecked product of num_pts could overflow size_t
TEST(ChebyshevInterpolantGuardTest, RejectsShapeProductOverflow) {
    mango::Domain<2> dom{.lo = {0.0, 0.0}, .hi = {1.0, 1.0}};
    std::array<size_t, 2> huge = {std::numeric_limits<size_t>::max() / 2, 4};
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build_from_values(
        std::span<const double>{}, dom, huge);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::ValueSizeMismatch);
}

// Sampling overload: NaN from the sampled function is rejected too
TEST(ChebyshevInterpolantGuardTest, BuildRejectsNaNSampledFunction) {
    mango::Domain<2> dom{.lo = {0.0, 0.0}, .hi = {1.0, 1.0}};
    auto f = [](std::array<double, 2> c) {
        return (c[0] > 0.5) ? std::nan("") : 1.0;
    };
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build(f, dom, {4, 4});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::NaNInput);
}

// Sampling overload must validate shape/domain BEFORE invoking f (or allocating)
TEST(ChebyshevInterpolantGuardTest, SamplingValidatesBeforeInvokingF) {
    int calls = 0;
    auto f = [&calls](std::array<double, 2>) { ++calls; return 1.0; };

    mango::Domain<2> reversed{.lo = {0.0, 1.0}, .hi = {1.0, 0.5}};
    EXPECT_FALSE(mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build(
        f, reversed, {3, 3}).has_value());
    EXPECT_EQ(calls, 0);

    mango::Domain<2> ok{.lo = {0.0, 0.0}, .hi = {1.0, 1.0}};
    EXPECT_FALSE(mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build(
        f, ok, {1, 3}).has_value());
    EXPECT_EQ(calls, 0);

    std::array<size_t, 2> huge = {std::numeric_limits<size_t>::max() / 2, 4};
    EXPECT_FALSE(mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build(
        f, ok, huge).has_value());
    EXPECT_EQ(calls, 0);
}

// Locks existing behavior: NaN queries propagate through barycentric eval
TEST(ChebyshevInterpolantGuardTest, EvalPropagatesNaNQuery) {
    mango::Domain<2> dom{.lo = {0.0, 0.0}, .hi = {1.0, 1.0}};
    auto f = [](std::array<double, 2> c) { return c[0] + c[1]; };
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build(f, dom, {5, 5});
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(std::isnan(r->eval({std::nan(""), 0.5})));
}
```
(Match the file's existing include/namespace style; it already uses `ChebyshevTensor<N>` aliases in places — the explicit `ChebyshevInterpolant<2, RawTensor<2>>` spelling is fine alongside.)

- [ ] **Step 2: Run to verify build failure**

Run: `bazel test //tests:chebyshev_interpolant_test --test_output=errors`
Expected: BUILD FAILURE (`has_value` on a non-expected type / no matching overload).

- [ ] **Step 3: Implement the new factory contract** in `src/math/chebyshev/chebyshev_interpolant.hpp`

- Add includes: `<expected>`, `<limits>`, `"mango/support/error_types.hpp"` (`<cmath>` is present).
- Constrain the class: `template <size_t N, typename Storage> requires (N >= 1) class ChebyshevInterpolant {`.
- Add a private static validator (shape + domain only; usable before sampling):
```cpp
    /// Validate num_pts/domain and compute the tensor size (overflow-checked).
    /// Returns the total point count, or the error to surface.
    [[nodiscard]] static std::expected<size_t, InterpolationError>
    validate_shape(const Domain<N>& domain, const std::array<size_t, N>& num_pts) {
        size_t total = 1;
        for (size_t d = 0; d < N; ++d) {
            if (num_pts[d] < 2) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::InsufficientGridPoints, num_pts[d], d});
            }
            if (std::isnan(domain.lo[d]) || std::isnan(domain.hi[d])) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::NaNInput, 0, d});
            }
            if (std::isinf(domain.lo[d]) || std::isinf(domain.hi[d])) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::InfInput, 0, d});
            }
            if (domain.lo[d] == domain.hi[d]) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::ZeroWidthGrid, 0, d});
            }
            if (domain.lo[d] > domain.hi[d]) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::GridNotSorted, 0, d});
            }
            if (total > std::numeric_limits<size_t>::max() / num_pts[d]) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::ValueSizeMismatch, 0, d});
            }
            total *= num_pts[d];
        }
        return total;
    }
```
- `build_from_values` becomes:
```cpp
    template <typename... Args>
    [[nodiscard]] static std::expected<ChebyshevInterpolant, InterpolationError>
    build_from_values(std::span<const double> values,
                      const Domain<N>& domain,
                      const std::array<size_t, N>& num_pts,
                      Args&&... storage_args) {
        auto total = validate_shape(domain, num_pts);
        if (!total.has_value()) {
            return std::unexpected(total.error());
        }
        if (values.size() != *total) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::ValueSizeMismatch, values.size(), 0});
        }
        for (size_t i = 0; i < values.size(); ++i) {
            if (std::isnan(values[i])) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::NaNInput, values.size(), i});
            }
            if (std::isinf(values[i])) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::InfInput, values.size(), i});
            }
        }
        // ... existing body unchanged (nodes/weights generation, Storage::build) ...
        return interp;
    }
```
- The sampling `build()` calls `validate_shape` FIRST (before computing strides, allocating `values`, or invoking `f`), returns `std::unexpected(total.error())` on failure, then samples and delegates to `build_from_values` (whose re-validation is cheap and catches NaN from `f`).

- Update `src/math/chebyshev/BUILD.bazel` `chebyshev_interpolant` deps: add `"//src/support:error_types"` (check the exact target name with `grep -n 'name = ' src/support/BUILD.bazel`; use the target that exports `error_types.hpp`).

- [ ] **Step 4: Update all call sites** (tree must build green in this same task)

`src/option/table/serialization/reconstruct.hpp:125`:
```cpp
    auto interp = ChebyshevInterpolant<N, RawTensor<N>>::build_from_values(
        std::span<const double>(seg.values), domain, num_pts);
    if (!interp.has_value()) {
        return std::unexpected(convert_to_price_table_error(interp.error()));
    }
    return std::move(*interp);
```

`src/option/price_table_factory.cpp:648`:
```cpp
    auto cheb = ChebyshevInterpolant<3, RawTensor<3>>::build_from_values(
        std::span<const double>(pde->values), domain, backend.chebyshev_pts);
    if (!cheb.has_value()) {
        return std::unexpected(detail::to_validation_error(
            convert_to_price_table_error(cheb.error())));
    }
```
…and pass `std::move(*cheb)` where `cheb` was used.

`src/option/table/chebyshev/chebyshev_table_builder.cpp:185`:
```cpp
    auto interp = ChebyshevInterpolant<4, RawTensor<4>>::build_from_values(
        eep_span, config.domain, config.num_pts);
    if (!interp.has_value()) {
        return std::unexpected(convert_to_price_table_error(interp.error()));
    }
    ChebyshevTransformLeaf tleaf(std::move(*interp), StandardTransform4D{},
                                 config.K_ref);
```

`src/option/table/chebyshev/chebyshev_adaptive.cpp`:
- `build_segment_leaves` return type → `std::expected<std::vector<ChebyshevSegmentedLeaf>, PriceTableError>`; its two internal `build_from_values` calls (placeholder zeros ~283 and per-segment values ~324) become:
```cpp
            auto interp = ChebyshevInterpolant<4, RawTensor<4>>::
                build_from_values(..., domain, num_pts);
            if (!interp.has_value()) {
                return std::unexpected(convert_to_price_table_error(interp.error()));
            }
            leaves.emplace_back(std::move(*interp), StandardTransform4D{}, K_ref);
```
(the placeholder site propagates too — no `.value()`, per spec);
final `return leaves;` still works (implicit expected conversion).
- Caller ~499 (BuildFn lambda):
```cpp
        auto leaves = build_segment_leaves(...);
        if (!leaves.has_value()) {
            return std::unexpected(leaves.error());
        }
```
then `std::move(*leaves)` into `leaves_shared`.
- Caller ~595 (`build_chebyshev_segmented_pieces`): same unwrap; use `std::move(*leaves)` in the returned `ChebyshevSegmentedPieces`.
- Site ~441 (4D EEP values, inside the standard BuildFn lambda): same unwrap-and-map pattern as the table-builder site; then `ChebyshevRawTransformLeaf tleaf(std::move(*interp), ...)`.
- Add `#include "mango/support/error_types.hpp"` where `convert_to_price_table_error` is used if not already visible.

`benchmarks/latency_sweep.cc:416` and `benchmarks/greek_latency.cc:511`: append `.value()` to the `build_from_values(...)` expression (adjust the variable to `auto cheb = ...build_from_values(...).value();`).

Test files — unwrap EVERY factory expression: `tests/chebyshev_interpolant_test.cc` (7 pre-existing `::build(` uses → `.value()` or `ASSERT_TRUE(r.has_value())` + use `*r`), `tests/chebyshev_surface_test.cc` (the `::build(` use at ~line 21 and the Task-4 `TransformLeafNaNTest` → `.value()` / `std::move(*interp)`), `tests/parquet_io_test.cc` (2 `build_from_values` uses → `.value()`).

- [ ] **Step 5: Run the full affected set**

Run: `bazel test //tests:chebyshev_interpolant_test //tests:chebyshev_surface_test //tests:parquet_io_test //tests:chebyshev_pde_cache_test --test_output=errors && bazel build //benchmarks:latency_sweep //benchmarks:greek_latency //src/option/table/chebyshev/... //src/option:price_table_factory 2>&1 | tail -5`
Expected: all PASS / build OK. (If the `price_table_factory` label differs, `bazel build //src/...` the owning package.)

- [ ] **Step 6: Commit**

```bash
git add src/math/chebyshev/chebyshev_interpolant.hpp src/math/chebyshev/BUILD.bazel src/option/table/chebyshev/chebyshev_adaptive.cpp src/option/table/chebyshev/chebyshev_table_builder.cpp src/option/price_table_factory.cpp src/option/table/serialization/reconstruct.hpp benchmarks/latency_sweep.cc benchmarks/greek_latency.cc tests/chebyshev_interpolant_test.cc tests/chebyshev_surface_test.cc tests/parquet_io_test.cc
git commit -m "Validate ChebyshevInterpolant input via std::expected"
```

---

### Task 6: Adaptive cache fails loudly on invalid slices (D6) + reconstruction rejection test

**Files:**
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.hpp` (add `detail::build_segment_leaves` declaration)
- Modify: `src/option/table/chebyshev/chebyshev_adaptive.cpp` (move `build_segment_leaves` out of the anonymous namespace into `mango::detail`; `get_slice == nullptr` → error at ~301 and ~410)
- Modify: `tests/BUILD.bazel` (`chebyshev_pde_cache_test` deps: add the adaptive library target)
- Test: `tests/chebyshev_pde_cache_test.cc`, `tests/parquet_io_test.cc`

**Interfaces:**
- Consumes: Task 5's expected-valued `build_segment_leaves`; Task 2's CubicSpline NaN rejection (which marks the cache slice invalid).
- Produces: `mango::detail::build_segment_leaves(...)` declared in `chebyshev_adaptive.hpp` with the exact Task-5 signature; extraction returns `PriceTableErrorCode::ExtractionFailed` on missing/invalid needed slices.

- [ ] **Step 1: Write the failing regression tests**

`tests/chebyshev_pde_cache_test.cc` (append; add `#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"`):
```cpp
// ===========================================================================
// Regression tests for the #419 incident closure (D6)
// ===========================================================================

// Regression: a NaN PDE slice was stored as invalid, then extraction did
// `if (!spline) continue` over a zero-initialized tensor — the surface built
// "successfully" out of silent zeros
// Bug: ChebyshevPDECache::store_slice discarded the CubicSpline build error
// and build_segment_leaves treated missing slices as skippable
TEST(ChebyshevPDECacheTest, InvalidSliceFailsSegmentExtraction) {
    mango::ChebyshevPDECache cache;
    std::vector<double> x = {-0.5, 0.0, 0.5, 1.0};
    std::vector<double> bad = {0.1, std::nan(""), 0.2, 0.3};
    cache.store_slice(0.2, 0.05, 0, x, bad);
    ASSERT_EQ(cache.get_slice(0.2, 0.05, 0), nullptr);  // marked invalid

    std::vector<double> seg_bounds = {0.0, 1.0};
    std::vector<bool> seg_is_gap = {false};
    std::vector<double> m = {-0.5, 0.0, 0.5};
    std::vector<double> tau = {0.5};
    std::vector<double> sigma = {0.2};
    std::vector<double> rate = {0.05};

    auto leaves = mango::detail::build_segment_leaves(
        cache, /*K_ref=*/100.0, seg_bounds, seg_is_gap, /*include_gaps=*/false,
        m, tau, sigma, rate);
    ASSERT_FALSE(leaves.has_value());
    EXPECT_EQ(leaves.error().code, mango::PriceTableErrorCode::ExtractionFailed);
}
```

`tests/parquet_io_test.cc` (append near the existing reconstruction tests, using their serialize-then-load fixture style):
```cpp
// Regression: a persisted table containing NaN values used to load and
// produce garbage prices (issue #426 deserialization policy)
// Bug: reconstruction did not validate segment values for finiteness
```
Concretely: copy the file's smallest existing round-trip test, patch one value
in the serialized segment's `values` array to `std::nan("")` before the load
step, and assert the load fails (`ASSERT_FALSE(loaded.has_value())`). If the
fixture only round-trips through in-memory structs, poison the struct value
directly before calling the reconstruct entry point the test already uses.

- [ ] **Step 2: Run to verify failure**

Run: `bazel test //tests:chebyshev_pde_cache_test //tests:parquet_io_test --test_output=errors`
Expected: `chebyshev_pde_cache_test` BUILD FAILURE (`detail::build_segment_leaves` not declared). The parquet test may already PASS (Task 5 wired the guard into reconstruct) — if so, it's a lock, keep it.

- [ ] **Step 3: Implement**

`chebyshev_adaptive.hpp` — after the existing public declarations (`ChebyshevSegmentedLeaf` comes from the already-included `chebyshev_surface.hpp`; add `#include "mango/option/table/chebyshev/chebyshev_pde_cache.hpp"` and `<expected>`/`<span>` if missing):
```cpp
namespace detail {
/// Exposed for testing (D6): builds per-segment Chebyshev leaves from cached
/// PDE slices. A needed slice that is missing or invalid is
/// PriceTableErrorCode::ExtractionFailed — never a silent zero region.
[[nodiscard]] std::expected<std::vector<ChebyshevSegmentedLeaf>, PriceTableError>
build_segment_leaves(ChebyshevPDECache& cache,
                     double K_ref,
                     const std::vector<double>& seg_bounds,
                     const std::vector<bool>& seg_is_gap,
                     bool include_gaps,
                     std::span<const double> m_nodes,
                     std::span<const double> tau_nodes,
                     std::span<const double> sigma_nodes,
                     std::span<const double> rate_nodes);
}  // namespace detail
```
`chebyshev_adaptive.cpp`:
- Move `build_segment_leaves` out of the anonymous namespace, define as `mango::detail::build_segment_leaves` (drop `static`); update both internal callers to `detail::build_segment_leaves`.
- In the extraction loop (~301) and the 4D loop (~410), replace `if (!spline) continue;` with:
```cpp
                    auto* spline = cache.get_slice(sigma, rate, tau_idx[jt]);
                    if (!spline) {
                        // A slice needed here was never solved or failed its
                        // spline build — fail loudly, never zero-fill (D6)
                        return std::unexpected(PriceTableError{
                            PriceTableErrorCode::ExtractionFailed});
                    }
```
(The ~410 site is inside the lambda already returning `std::expected<SurfaceHandle, PriceTableError>` — same pattern, indices per that loop.)

`tests/BUILD.bazel` — add the adaptive library (the target providing `chebyshev_adaptive.hpp`; find it with `grep -n "chebyshev_adaptive" src/option/table/chebyshev/BUILD.bazel`) to `chebyshev_pde_cache_test` deps.

- [ ] **Step 4: Run tests to verify they pass**

Run: `bazel test //tests:chebyshev_pde_cache_test //tests:parquet_io_test //tests:adaptive_grid_builder_test --test_output=errors`
Expected: PASS (adaptive_grid_builder_test guards against regressions in the adaptive build path).

- [ ] **Step 5: Commit**

```bash
git add src/option/table/chebyshev/chebyshev_adaptive.hpp src/option/table/chebyshev/chebyshev_adaptive.cpp tests/chebyshev_pde_cache_test.cc tests/parquet_io_test.cc tests/BUILD.bazel
git commit -m "Fail adaptive extraction on invalid cached slices"
```

---

### Task 7: Full verification + perf comparison

**Files:** none (verification only; fixes discovered here belong to the task that owns the file).

- [ ] **Step 1: Full CI parity**

Run (each must succeed):
```bash
bazel test //...
bazel build //benchmarks/...
bazel build //src/python:mango_option
```
Expected: 148 pre-existing tests + all new tests pass. Any failure: fix in the owning task's files, commit there, re-run.

- [ ] **Step 2: Perf after** (same machine state as Task 1)

```bash
bazel run -c opt //benchmarks:latency_sweep -- --benchmark_repetitions=3 --benchmark_report_aggregates_only=true > /tmp/codex-skills/5d8b8f12-dce6-4646-afc8-3120ada90939/after-latency.txt 2>&1
bazel run -c opt //benchmarks:bspline_template_vs_hardcoded -- --benchmark_repetitions=3 --benchmark_report_aggregates_only=true > /tmp/codex-skills/5d8b8f12-dce6-4646-afc8-3120ada90939/after-bspline.txt 2>&1
```
Compare medians against the Task-1 baselines. Acceptance (D4): price-query medians regress < 3%; the raw B-spline control should be pure noise. Record the numbers — they go in the PR body.

- [ ] **Step 3: Spec cross-check**

Re-read `docs/plans/2026-08-31-nan-input-guards-design.md` acceptance criteria 1–4 and confirm each holds. Criterion 4 spot-check is the new tests themselves (all run in the default build; none are assert/`EXPECT_DEATH`-based, so they hold in opt).
