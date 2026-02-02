# Code Review #251 Remaining Items Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address the 10 remaining items from code review issue #251 to improve defensive coding, test quality, and numerical robustness.

**Architecture:** All changes are localized defensive improvements — no API changes, no new types. Each task modifies 1-2 files plus tests. Changes are independent and can be committed separately.

**Tech Stack:** C++23, GoogleTest, Bazel

---

### Task 1: Replace thread-local unordered_map with single grow-only buffer

**Files:**
- Modify: `src/option/iv_solver_fdm.cpp:89-100`

**Context:** The IV solver's `objective_function()` is called many times during Brent iterations. Each call needs a workspace buffer. Currently uses `thread_local std::unordered_map<size_t, std::pmr::vector<double>>` keyed by grid size — over-engineered since within a single Brent solve the grid size only varies slightly (different volatilities → different auto-estimated grid sizes). A single grow-only `std::vector<double>` suffices.

**Step 1: Replace the workspace cache**

In `src/option/iv_solver_fdm.cpp`, replace lines 89-100:

```cpp
// OLD:
thread_local std::unordered_map<size_t, std::pmr::vector<double>> workspace_cache;

size_t n = grid_spec.n_points();
size_t required_size = PDEWorkspace::required_size(n);

auto& buffer = workspace_cache[n];
if (buffer.size() < required_size) {
    buffer.resize(required_size);
}

auto pde_workspace_result = PDEWorkspace::from_buffer(
    std::span<double>(buffer.data(), buffer.size()), n);
```

with:

```cpp
// Single grow-only buffer per thread — avoids unordered_map overhead.
// Grid size may vary across Brent iterations (volatility changes grid estimation)
// but a single buffer that grows to the max is simpler and faster.
thread_local std::vector<double> workspace_buffer;

size_t n = grid_spec.n_points();
size_t required_size = PDEWorkspace::required_size(n);

if (workspace_buffer.size() < required_size) {
    workspace_buffer.resize(required_size);
}

auto pde_workspace_result = PDEWorkspace::from_buffer(
    std::span<double>(workspace_buffer.data(), workspace_buffer.size()), n);
```

Also remove the `#include <unordered_map>` if present (it may be pulled in transitively — check).

**Step 2: Run tests to verify**

Run: `bazel test //tests:iv_solver_test //tests:iv_solver_interpolated_test --test_output=errors`
Expected: All pass (behavior unchanged)

**Step 3: Commit**

```bash
git add src/option/iv_solver_fdm.cpp
git commit -m "Simplify IV solver thread-local cache to single buffer"
```

---

### Task 2: Add NaN/Inf validation to BSplineND::create()

**Files:**
- Modify: `src/math/bspline_nd.hpp:84-125`
- Test: `tests/bspline_nd_test.cc`

**Context:** `BSplineNDSeparable::fit()` already validates NaN/Inf in input values (lines 196-209 of bspline_nd_separable.hpp). But `BSplineND::create()` accepts pre-computed coefficients and does not check them. If a spline fitting upstream produced NaN coefficients silently, `BSplineND::eval()` would return NaN without any clear error.

**Step 1: Write failing test**

Add to `tests/bspline_nd_test.cc`:

```cpp
TEST(BSplineNDTest, CreateRejectsNaNCoefficients) {
    // Create valid 1D grids and knots
    std::array<std::vector<double>, 1> grids = {{{0.0, 1.0, 2.0, 3.0}}};
    auto knots_vec = mango::clamped_knots_cubic(grids[0]);
    std::array<std::vector<double>, 1> knots = {{knots_vec}};

    // Coefficients with NaN
    std::vector<double> coeffs = {1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), 4.0};

    auto result = mango::BSplineND<double, 1>::create(grids, knots, std::move(coeffs));
    EXPECT_FALSE(result.has_value());
}

TEST(BSplineNDTest, CreateRejectsInfCoefficients) {
    std::array<std::vector<double>, 1> grids = {{{0.0, 1.0, 2.0, 3.0}}};
    auto knots_vec = mango::clamped_knots_cubic(grids[0]);
    std::array<std::vector<double>, 1> knots = {{knots_vec}};

    std::vector<double> coeffs = {1.0, std::numeric_limits<double>::infinity(), 3.0, 4.0};

    auto result = mango::BSplineND<double, 1>::create(grids, knots, std::move(coeffs));
    EXPECT_FALSE(result.has_value());
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:bspline_nd_test --test_output=all --test_filter="*RejectsNaN*:*RejectsInf*"`
Expected: FAIL (create() currently accepts NaN/Inf)

**Step 3: Add validation in create()**

In `src/math/bspline_nd.hpp`, after the coefficient size check (line 118-122), add:

```cpp
        if (coeffs.size() != expected_size) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::CoefficientSizeMismatch,
                coeffs.size()});
        }

        // Validate coefficients for NaN/Inf
        for (size_t i = 0; i < coeffs.size(); ++i) {
            if (!std::isfinite(coeffs[i])) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::NaNInput,
                    coeffs.size(),
                    i});
            }
        }
```

Note: Reuse `NaNInput` error code for both NaN and Inf (they're both non-finite). The `index` field records the position of the first bad value.

**Step 4: Run tests**

Run: `bazel test //tests:bspline_nd_test --test_output=errors`
Expected: All pass including new tests

**Step 5: Commit**

```bash
git add src/math/bspline_nd.hpp tests/bspline_nd_test.cc
git commit -m "Add NaN/Inf validation to BSplineND coefficient input"
```

---

### Task 3: Add overflow guard to BSplineNDSeparable stride calculation

**Files:**
- Modify: `src/math/bspline_nd_separable.hpp:264-266`

**Context:** The constructor computes `strides_[i-1] = strides_[i] * dims_[i]` without overflow checking. While overflow is unlikely with realistic grids (even 20×20×20×20 = 160,000), the rest of the codebase uses `safe_multiply` consistently. The `safe_multiply` header is already included.

However, the constructor is called after `create()` validation succeeds, and `create()` doesn't check for stride overflow either. The simplest fix: use `safe_multiply` in `fit()` where the size is already checked with `safe_multiply`, and add the same check to the constructor. But since the constructor can't return errors (it's private, called from factory), we should add the check to `create()`.

**Step 1: Add overflow check in create()**

In `src/math/bspline_nd_separable.hpp`, in the `create()` method (around line 128-143), after grid validation but before construction, add:

```cpp
        // Verify stride computation won't overflow
        size_t stride_check = 1;
        for (size_t i = N; i > 0; --i) {
            auto result = safe_multiply(stride_check, grids[i - 1].size());
            if (!result.has_value()) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::ValueSizeMismatch,
                    grids[i - 1].size(),
                    i - 1});
            }
            stride_check = result.value();
        }
```

**Step 2: Run tests**

Run: `bazel test //tests:bspline_nd_test --test_output=errors`
Expected: All pass (no behavior change for realistic grids)

**Step 3: Commit**

```bash
git add src/math/bspline_nd_separable.hpp
git commit -m "Add overflow guard to BSplineNDSeparable stride computation"
```

---

### Task 4: Replace magic epsilon in bspline_basis.hpp

**Files:**
- Modify: `src/math/bspline_basis.hpp:78`

**Context:** Line 78 uses `T{1e-12}` as epsilon for clamping knot positions. This should use `std::numeric_limits<T>::epsilon()` for type-correctness. The existing code already partially does this (line 79-80), but the `T{1e-12}` constant should be replaced.

Current code (line 78-80):
```cpp
const T eps = std::max(T{1e-12} * spacing,
                      std::numeric_limits<T>::epsilon() *
                          std::max(std::abs(right), T{1}));
```

The expression takes `max(1e-12 * spacing, eps_mach * max(|right|, 1))`. The `1e-12` is approximately `128 * epsilon` for double (where `epsilon ≈ 2.2e-16`). For float, `1e-12` is much smaller than float epsilon (`1.19e-7`), making the first branch meaningless.

**Step 1: Replace magic constant**

In `src/math/bspline_basis.hpp`, replace line 78:

```cpp
// OLD:
const T eps = std::max(T{1e-12} * spacing,
                      std::numeric_limits<T>::epsilon() *
                          std::max(std::abs(right), T{1}));

// NEW:
const T eps = std::max(T{128} * std::numeric_limits<T>::epsilon() * spacing,
                      std::numeric_limits<T>::epsilon() *
                          std::max(std::abs(right), T{1}));
```

**Step 2: Run tests**

Run: `bazel test //... --test_output=errors`
Expected: All pass (128 * epsilon ≈ 2.84e-14 for double, close to original 1e-12 behavior)

**Step 3: Commit**

```bash
git add src/math/bspline_basis.hpp
git commit -m "Replace magic epsilon with type-safe numeric_limits expression"
```

---

### Task 5: Replace magic epsilon in grid.hpp uniformity checks

**Files:**
- Modify: `src/pde/core/grid.hpp:328,737`

**Context:** Both `GridView::is_uniform()` and `GridSpacing::compute_spacing()` use hardcoded `T(1e-10)` tolerance. This was already flagged — the grid epsilon was fixed in PR #309 for `GridSpec::merge_nearby_clusters`, but these two were missed.

**Step 1: Fix GridView::is_uniform (line 328)**

```cpp
// OLD:
bool is_uniform(T tolerance = T(1e-10)) const {

// NEW:
bool is_uniform(T tolerance = T(100) * std::numeric_limits<T>::epsilon()) const {
```

**Step 2: Fix GridSpacing::compute_spacing (line 737)**

```cpp
// OLD:
constexpr T tolerance = T(1e-10);

// NEW:
constexpr T tolerance = T(100) * std::numeric_limits<T>::epsilon();
```

Note: `<limits>` is already included in grid.hpp (was added in PR #309).

**Step 3: Run tests**

Run: `bazel test //... --test_output=errors`
Expected: All pass (100 * epsilon ≈ 2.2e-14, stricter than 1e-10 but uniform grids are generated analytically so tolerance is plenty)

**Step 4: Commit**

```bash
git add src/pde/core/grid.hpp
git commit -m "Replace magic tolerance in grid uniformity checks"
```

---

### Task 6: Delete disabled Neumann BC test

**Files:**
- Modify: `tests/pde_solver_test.cc:119-187`

**Context:** `DISABLED_HeatEquationNeumannBC` tests Neumann BC with analytical Jacobian support that doesn't exist yet. The test has been disabled since it was written, providing no value. The comment says "Future work: Implement analytical Jacobian for Neumann ghost-point coupling at boundaries." This future work should be tracked as an issue, not dead test code.

**Step 1: Delete the test**

Remove lines 119-187 from `tests/pde_solver_test.cc` (the entire `DISABLED_HeatEquationNeumannBC` test and its preceding comment block).

**Step 2: Run tests**

Run: `bazel test //tests:pde_solver_test --test_output=errors`
Expected: All pass (test was disabled anyway)

**Step 3: Commit**

```bash
git add tests/pde_solver_test.cc
git commit -m "Remove disabled Neumann BC test

Neumann BC with analytical boundary Jacobian is tracked
as future work. Dead test code provides no value."
```

---

### Task 7: Add spline failure logging in price_table_builder

**Files:**
- Modify: `src/option/table/price_table_builder.cpp:435-446`

**Context:** When cubic spline fitting fails during price table construction, the failure is tracked in `failed_spline` vector but never logged. The codebase uses USDT probes for tracing — but the `failed_spline` vector is checked later for error reporting. The actual issue is that failures are silently filled with NaN, then the `failed_spline` count is only used for a coarse "X splines failed" message. We should add a USDT probe at the failure point so tracing reveals exactly which (σ, r, τ) combination failed.

**Step 1: Check if USDT probe infrastructure exists for price table**

Read `src/support/ivcalc_trace.h` and check for existing price table probes.

If a probe like `MANGO_TRACE_PRICE_TABLE_SPLINE_FAILURE` doesn't exist, add one. If adding USDT probes is complex, a simpler approach: ensure the `failed_spline` information is propagated to the build result (it likely already is).

Actually, re-reading the code: the `failed_spline` vector IS used after the parallel region. The build result already contains failure information. The concern was just that failures aren't logged at the point they occur. Since USDT is the logging mechanism and adding probes is non-trivial, skip the probe for now. Instead, verify the failure count is properly reported in the build result.

**Step 1: Verify failure propagation**

Read `src/option/table/price_table_builder.cpp` to confirm `failed_spline.size()` is checked after the parallel loop and propagated to the result.

If it's already properly propagated (likely), this task is **already addressed** — close it.

If not, add a check after the parallel loop:

```cpp
if (!failed_spline.empty()) {
    // Return error with count of failed splines
    return std::unexpected(PriceTableError{
        PriceTableErrorCode::SplineFittingFailed,
        failed_spline.size()});
}
```

**Step 2: Run tests**

Run: `bazel test //tests:price_table_test --test_output=errors`
Expected: All pass

**Step 3: Commit (if changes were needed)**

```bash
git add src/option/table/price_table_builder.cpp
git commit -m "Propagate spline failure count in price table build result"
```

---

### Task 8: Simplify monotonicity enforcement in grid generation

**Files:**
- Modify: `src/pde/core/grid.hpp:240-288` (the `enforce_monotonicity` method)

**Context:** The `enforce_monotonicity` static method uses an iterative loop with max 100 passes, then a separate forward+backward pass. This is over-engineered for a grid generation utility. A single forward pass with minimum spacing enforcement is sufficient since multi-sinh grids are approximately monotonic to begin with — the enforcement is just a safety net.

**Step 1: Simplify enforce_monotonicity**

Replace the current implementation (lines 240-288) with:

```cpp
    static void enforce_monotonicity(std::vector<T>& points, T x_min, T x_max) {
        const size_t n = points.size();
        if (n < 2) return;

        const T min_spacing = (x_max - x_min) / static_cast<T>(n * 100);

        // Clamp endpoints
        points[0] = x_min;
        points[n-1] = x_max;

        // Forward pass: ensure strictly increasing with minimum spacing
        for (size_t i = 1; i < n - 1; ++i) {
            if (points[i] <= points[i-1] + min_spacing) {
                points[i] = points[i-1] + min_spacing;
            }
        }

        // Backward pass: ensure last interior point doesn't exceed x_max - min_spacing
        for (size_t i = n - 2; i > 0; --i) {
            if (points[i] >= points[i+1] - min_spacing) {
                points[i] = points[i+1] - min_spacing;
            }
        }
    }
```

This is a single forward + backward pass (O(n)) instead of up to 100 iterative passes.

**Step 2: Run tests**

Run: `bazel test //... --test_output=errors`
Expected: All pass (grid generation produces same results for well-behaved inputs)

**Step 3: Commit**

```bash
git add src/pde/core/grid.hpp
git commit -m "Simplify grid monotonicity enforcement to single pass"
```

---

### Task 9: Replace magic epsilon in bspline_basis.hpp boundary check

**Files:**
- Modify: `src/math/bspline_basis.hpp:173`

**Context:** Line 173 uses `T{1e-14}` for right-boundary exact interpolation check. Should use `std::numeric_limits<T>::epsilon()` scaled appropriately.

```cpp
// OLD (line 173):
if (std::abs(x - t.back()) < T{1e-14}) {

// NEW:
if (std::abs(x - t.back()) < T{64} * std::numeric_limits<T>::epsilon() * std::max(std::abs(t.back()), T{1})) {
```

This scales relative to the knot value magnitude, which is correct for floating-point comparison.

**Step 1: Apply fix**

Make the change in `src/math/bspline_basis.hpp` line 173.

**Step 2: Run tests**

Run: `bazel test //... --test_output=errors`
Expected: All pass

**Step 3: Commit (combine with Task 4)**

If Task 4 hasn't been committed yet, include this in the same commit. Otherwise:

```bash
git add src/math/bspline_basis.hpp
git commit -m "Replace magic epsilon in B-spline boundary check"
```

---

### Task 10: Tighten normalized chain delta tolerance

**Files:**
- Modify: `tests/normalized_chain_accuracy_test.cc:156`

**Context:** The QuantLib delta comparison uses 10% relative tolerance, which is very loose. Run the test first to see actual error, then tighten if warranted.

**Step 1: Run test with verbose output to see actual errors**

Run: `bazel test //tests:normalized_chain_accuracy_test --test_output=all`

Examine the actual delta errors reported. If they're consistently under 5%, tighten to 5%. If under 3%, tighten to 3%.

**Step 2: Tighten tolerance based on observed errors**

In `tests/normalized_chain_accuracy_test.cc` line 156:

```cpp
// OLD:
EXPECT_LT(delta_rel, 10.0)  // Within 10%

// NEW (adjust based on actual errors — target ~2x headroom over observed):
EXPECT_LT(delta_rel, 5.0)  // Within 5%
```

**Step 3: Run tests to confirm**

Run: `bazel test //tests:normalized_chain_accuracy_test --test_output=errors`
Expected: All pass with tightened tolerance

**Step 4: Commit**

```bash
git add tests/normalized_chain_accuracy_test.cc
git commit -m "Tighten normalized chain delta tolerance"
```

---

## Summary

| Task | Description | Files | Risk |
|------|-------------|-------|------|
| 1 | Thread-local cache simplification | iv_solver_fdm.cpp | Low |
| 2 | BSplineND NaN/Inf validation | bspline_nd.hpp, test | Low |
| 3 | Stride overflow guard | bspline_nd_separable.hpp | Low |
| 4 | Magic epsilon in knot clamping | bspline_basis.hpp | Low |
| 5 | Magic epsilon in grid uniformity | grid.hpp | Low |
| 6 | Delete disabled test | pde_solver_test.cc | None |
| 7 | Spline failure logging | price_table_builder.cpp | Low |
| 8 | Monotonicity simplification | grid.hpp | Medium |
| 9 | Magic epsilon in boundary check | bspline_basis.hpp | Low |
| 10 | Tighten delta tolerance | normalized_chain_accuracy_test.cc | Low |

**Dependencies:** Tasks 4 and 9 touch the same file — commit together. Tasks 5 and 8 touch grid.hpp — commit together or sequentially. All other tasks are independent.
