# Price Table Builder Refactor - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace specialized `PriceTable4DBuilder` with generic `PriceTableBuilder<4>`, achieving feature parity then deleting the old builder.

**Architecture:** Phase 1 migration extends the generic builder with diagnostics (`PriceTableResult<N>`), fixes the grid configuration bug (custom_grid parameter), ports validation logic, adds helper factories, updates all consumers, then deletes the specialized builder.

**Tech Stack:** C++23, Bazel, GoogleTest, OpenMP, std::expected, PMR allocators

**Design Document:** `docs/plans/2025-11-24-price-table-builder-refactor.md` (10 review rounds, APPROVED)

---

## Task 1: Add OptionChain to Shared Header

Extract `OptionChain` from specialized builder before any other changes (Step 5 in design).

**Files:**
- Create: `src/option/option_chain.hpp`
- Modify: `src/option/price_table_4d_builder.hpp:87` (remove inline definition)
- Test: `tests/option_chain_test.cc`

**Step 1.1: Write the failing test**

```cpp
// tests/option_chain_test.cc
#include <gtest/gtest.h>
#include "src/option/option_chain.hpp"

TEST(OptionChainTest, DefaultConstruction) {
    mango::OptionChain chain;
    EXPECT_TRUE(chain.ticker.empty());
    EXPECT_EQ(chain.spot, 0.0);
    EXPECT_TRUE(chain.strikes.empty());
    EXPECT_TRUE(chain.maturities.empty());
    EXPECT_TRUE(chain.implied_vols.empty());
    EXPECT_TRUE(chain.rates.empty());
    EXPECT_EQ(chain.dividend_yield, 0.0);
}

TEST(OptionChainTest, FieldPopulation) {
    mango::OptionChain chain;
    chain.ticker = "AAPL";
    chain.spot = 150.0;
    chain.strikes = {140.0, 150.0, 160.0};
    chain.maturities = {0.25, 0.5, 1.0};
    chain.implied_vols = {0.20, 0.22, 0.25};
    chain.rates = {0.05, 0.05, 0.05};
    chain.dividend_yield = 0.01;

    EXPECT_EQ(chain.ticker, "AAPL");
    EXPECT_EQ(chain.spot, 150.0);
    EXPECT_EQ(chain.strikes.size(), 3);
    EXPECT_EQ(chain.dividend_yield, 0.01);
}
```

**Step 1.2: Run test to verify it fails**

Run: `bazel test //tests:option_chain_test --test_output=all`
Expected: FAIL with "file not found" or "OptionChain not defined"

**Step 1.3: Create the header file**

```cpp
// src/option/option_chain.hpp
#pragma once

#include <vector>
#include <string>

namespace mango {

/// Market option chain data (from exchanges)
///
/// Represents raw option chain data as typically received from market data
/// feeds or exchanges. Can contain duplicate strikes/maturities (e.g., multiple
/// options with same parameters but different bid/ask spreads).
///
/// Extracted from PriceTable4DBuilder for reusability.
struct OptionChain {
    std::string ticker;                  ///< Underlying ticker symbol
    double spot = 0.0;                   ///< Current underlying price
    std::vector<double> strikes;         ///< Strike prices (may have duplicates)
    std::vector<double> maturities;      ///< Times to expiration in years (may have duplicates)
    std::vector<double> implied_vols;    ///< Market implied volatilities (for grid)
    std::vector<double> rates;           ///< Risk-free rates (may have duplicates)
    double dividend_yield = 0.0;         ///< Continuous dividend yield
};

} // namespace mango
```

**Step 1.4: Add BUILD target**

```python
# In src/option/BUILD.bazel, add:
cc_library(
    name = "option_chain",
    hdrs = ["option_chain.hpp"],
    visibility = ["//visibility:public"],
)
```

**Step 1.5: Add test BUILD target**

```python
# In tests/BUILD.bazel, add:
cc_test(
    name = "option_chain_test",
    srcs = ["option_chain_test.cc"],
    deps = [
        "//src/option:option_chain",
        "@googletest//:gtest_main",
    ],
)
```

**Step 1.6: Run test to verify it passes**

Run: `bazel test //tests:option_chain_test --test_output=all`
Expected: PASS

**Step 1.7: Update specialized builder to use shared header**

```cpp
// In src/option/price_table_4d_builder.hpp, replace inline OptionChain with:
#include "src/option/option_chain.hpp"

// Remove the struct OptionChain { ... } definition (lines ~87-95)
```

**Step 1.8: Verify existing tests still pass**

Run: `bazel test //tests:price_table_4d_integration_test --test_output=all`
Expected: PASS

**Step 1.9: Commit**

```bash
git add src/option/option_chain.hpp src/option/BUILD.bazel tests/option_chain_test.cc tests/BUILD.bazel src/option/price_table_4d_builder.hpp
git commit -m "Extract OptionChain to shared header

Prepares for specialized builder deletion by moving OptionChain
to its own header file that both builders can use.

- Add src/option/option_chain.hpp with full struct definition
- Update price_table_4d_builder.hpp to include shared header
- Add unit tests for OptionChain construction"
```

---

## Task 2: Add BSplineFittingStats Struct

Add diagnostic struct for B-spline fitting results (Step 2 in design, part 1).

**Files:**
- Modify: `src/option/price_table_builder.hpp` (add struct)
- Test: `tests/bspline_fitting_stats_test.cc`

**Step 2.1: Write the failing test**

```cpp
// tests/bspline_fitting_stats_test.cc
#include <gtest/gtest.h>
#include "src/option/price_table_builder.hpp"

TEST(BSplineFittingStatsTest, DefaultConstruction) {
    mango::BSplineFittingStats stats;
    EXPECT_EQ(stats.max_residual_axis0, 0.0);
    EXPECT_EQ(stats.max_residual_axis1, 0.0);
    EXPECT_EQ(stats.max_residual_axis2, 0.0);
    EXPECT_EQ(stats.max_residual_axis3, 0.0);
    EXPECT_EQ(stats.max_residual_overall, 0.0);
    EXPECT_EQ(stats.condition_axis0, 0.0);
    EXPECT_EQ(stats.condition_axis1, 0.0);
    EXPECT_EQ(stats.condition_axis2, 0.0);
    EXPECT_EQ(stats.condition_axis3, 0.0);
    EXPECT_EQ(stats.condition_max, 0.0);
    EXPECT_EQ(stats.failed_slices_axis0, 0);
    EXPECT_EQ(stats.failed_slices_axis1, 0);
    EXPECT_EQ(stats.failed_slices_axis2, 0);
    EXPECT_EQ(stats.failed_slices_axis3, 0);
    EXPECT_EQ(stats.failed_slices_total, 0);
}
```

**Step 2.2: Run test to verify it fails**

Run: `bazel test //tests:bspline_fitting_stats_test --test_output=all`
Expected: FAIL with "BSplineFittingStats not defined"

**Step 2.3: Add BSplineFittingStats struct**

```cpp
// In src/option/price_table_builder.hpp, add after includes:

/// B-spline fitting diagnostics (extracted from BSplineNDSeparable)
struct BSplineFittingStats {
    double max_residual_axis0 = 0.0;
    double max_residual_axis1 = 0.0;
    double max_residual_axis2 = 0.0;
    double max_residual_axis3 = 0.0;
    double max_residual_overall = 0.0;

    double condition_axis0 = 0.0;
    double condition_axis1 = 0.0;
    double condition_axis2 = 0.0;
    double condition_axis3 = 0.0;
    double condition_max = 0.0;

    size_t failed_slices_axis0 = 0;
    size_t failed_slices_axis1 = 0;
    size_t failed_slices_axis2 = 0;
    size_t failed_slices_axis3 = 0;
    size_t failed_slices_total = 0;
};
```

**Step 2.4: Add test BUILD target**

```python
# In tests/BUILD.bazel, add:
cc_test(
    name = "bspline_fitting_stats_test",
    srcs = ["bspline_fitting_stats_test.cc"],
    deps = [
        "//src/option:price_table_builder",
        "@googletest//:gtest_main",
    ],
)
```

**Step 2.5: Run test to verify it passes**

Run: `bazel test //tests:bspline_fitting_stats_test --test_output=all`
Expected: PASS

**Step 2.6: Commit**

```bash
git add src/option/price_table_builder.hpp tests/bspline_fitting_stats_test.cc tests/BUILD.bazel
git commit -m "Add BSplineFittingStats struct for fitting diagnostics

Provides per-axis tracking of:
- Max residuals from B-spline fitting
- Condition numbers per axis
- Failed slice counts

Part of PriceTableResult<N> instrumentation."
```

---

## Task 3: Add PriceTableResult<N> Struct

Add the result struct with surface pointer and diagnostics (Step 2 in design, part 2).

**Files:**
- Modify: `src/option/price_table_builder.hpp` (add struct)
- Test: `tests/price_table_result_test.cc`

**Step 3.1: Write the failing test**

```cpp
// tests/price_table_result_test.cc
#include <gtest/gtest.h>
#include "src/option/price_table_builder.hpp"

TEST(PriceTableResultTest, DefaultConstruction) {
    mango::PriceTableResult<4> result;
    EXPECT_EQ(result.surface, nullptr);
    EXPECT_EQ(result.n_pde_solves, 0);
    EXPECT_EQ(result.precompute_time_seconds, 0.0);
    EXPECT_EQ(result.fitting_stats.max_residual_overall, 0.0);
}

TEST(PriceTableResultTest, FieldAssignment) {
    mango::PriceTableResult<4> result;
    result.n_pde_solves = 200;
    result.precompute_time_seconds = 5.5;
    result.fitting_stats.max_residual_overall = 1e-6;

    EXPECT_EQ(result.n_pde_solves, 200);
    EXPECT_EQ(result.precompute_time_seconds, 5.5);
    EXPECT_EQ(result.fitting_stats.max_residual_overall, 1e-6);
}
```

**Step 3.2: Run test to verify it fails**

Run: `bazel test //tests:price_table_result_test --test_output=all`
Expected: FAIL with "PriceTableResult not defined"

**Step 3.3: Add PriceTableResult<N> struct**

```cpp
// In src/option/price_table_builder.hpp, add after BSplineFittingStats:

/// Result from price table build with diagnostics
template <size_t N>
struct PriceTableResult {
    std::shared_ptr<const PriceTableSurface<N>> surface = nullptr;  ///< Immutable surface
    size_t n_pde_solves = 0;                    ///< Number of PDE solves performed
    double precompute_time_seconds = 0.0;       ///< Wall-clock build time
    BSplineFittingStats fitting_stats;          ///< B-spline fitting diagnostics
};
```

**Step 3.4: Add test BUILD target**

```python
# In tests/BUILD.bazel, add:
cc_test(
    name = "price_table_result_test",
    srcs = ["price_table_result_test.cc"],
    deps = [
        "//src/option:price_table_builder",
        "@googletest//:gtest_main",
    ],
)
```

**Step 3.5: Run test to verify it passes**

Run: `bazel test //tests:price_table_result_test --test_output=all`
Expected: PASS

**Step 3.6: Commit**

```bash
git add src/option/price_table_builder.hpp tests/price_table_result_test.cc tests/BUILD.bazel
git commit -m "Add PriceTableResult<N> struct with diagnostics

Contains:
- shared_ptr to immutable surface
- PDE solve count
- Wall-clock build time
- BSplineFittingStats for quality metrics

Replaces raw shared_ptr<Surface> return from build()."
```

---

## Task 4: Add FitCoeffsResult Internal Struct

Add return type for fit_coeffs() that includes both coefficients and stats (Step 2 in design, part 3).

**Files:**
- Modify: `src/option/price_table_builder.hpp` (add struct, update fit_coeffs signature)
- Modify: `src/option/price_table_builder.cpp` (update fit_coeffs implementation)
- Test: Existing tests should still pass

**Step 4.1: Add FitCoeffsResult struct to header**

```cpp
// In src/option/price_table_builder.hpp, add inside PriceTableBuilder class (private section):

private:
    /// Internal result from B-spline coefficient fitting
    struct FitCoeffsResult {
        std::vector<double> coefficients;
        BSplineFittingStats stats;
    };
```

**Step 4.2: Update fit_coeffs signature in header**

```cpp
// Change from:
std::expected<std::vector<double>, std::string> fit_coeffs(
    const PriceTensor<N>& tensor,
    const PriceTableAxes<N>& axes) const;

// To:
std::expected<FitCoeffsResult, std::string> fit_coeffs(
    const PriceTensor<N>& tensor,
    const PriceTableAxes<N>& axes) const;
```

**Step 4.3: Update fit_coeffs implementation**

```cpp
// In src/option/price_table_builder.cpp, update fit_coeffs:

template <size_t N>
std::expected<typename PriceTableBuilder<N>::FitCoeffsResult, std::string>
PriceTableBuilder<N>::fit_coeffs(
    const PriceTensor<N>& tensor,
    const PriceTableAxes<N>& axes) const
{
    // ... existing fitter creation code ...

    auto fit_result = fitter_result->fit(values);
    if (!fit_result.has_value()) {
        return std::unexpected("B-spline fitting failed: " + fit_result.error());
    }

    const auto& result = fit_result.value();

    // Map BSplineNDSeparableResult to BSplineFittingStats
    BSplineFittingStats stats;
    stats.max_residual_axis0 = result.max_residual_per_axis[0];
    stats.max_residual_axis1 = result.max_residual_per_axis[1];
    stats.max_residual_axis2 = result.max_residual_per_axis[2];
    stats.max_residual_axis3 = result.max_residual_per_axis[3];
    stats.max_residual_overall = *std::max_element(
        result.max_residual_per_axis.begin(),
        result.max_residual_per_axis.end()
    );

    stats.condition_axis0 = result.condition_per_axis[0];
    stats.condition_axis1 = result.condition_per_axis[1];
    stats.condition_axis2 = result.condition_per_axis[2];
    stats.condition_axis3 = result.condition_per_axis[3];
    stats.condition_max = *std::max_element(
        result.condition_per_axis.begin(),
        result.condition_per_axis.end()
    );

    stats.failed_slices_axis0 = result.failed_slices[0];
    stats.failed_slices_axis1 = result.failed_slices[1];
    stats.failed_slices_axis2 = result.failed_slices[2];
    stats.failed_slices_axis3 = result.failed_slices[3];
    stats.failed_slices_total = std::accumulate(
        result.failed_slices.begin(),
        result.failed_slices.end(),
        size_t(0)
    );

    return FitCoeffsResult{
        .coefficients = std::move(result.coefficients),
        .stats = stats
    };
}
```

**Step 4.4: Update build() to use new return type**

```cpp
// In build() method, change from:
auto coeffs_result = fit_coeffs(tensor_result.value(), axes);
if (!coeffs_result.has_value()) {
    return std::unexpected("fit_coeffs failed: " + coeffs_result.error());
}
auto coefficients = std::move(coeffs_result.value());

// To:
auto coeffs_result = fit_coeffs(tensor_result.value(), axes);
if (!coeffs_result.has_value()) {
    return std::unexpected("fit_coeffs failed: " + coeffs_result.error());
}
auto& fit_result = coeffs_result.value();
auto coefficients = std::move(fit_result.coefficients);
auto fitting_stats = fit_result.stats;  // Captured for PriceTableResult
```

**Step 4.5: Verify existing tests still pass**

Run: `bazel test //tests:price_table_builder_test --test_output=all`
Expected: PASS

**Step 4.6: Commit**

```bash
git add src/option/price_table_builder.hpp src/option/price_table_builder.cpp
git commit -m "Add FitCoeffsResult to capture B-spline stats

fit_coeffs() now returns both coefficients and BSplineFittingStats,
enabling build() to populate PriceTableResult diagnostics.

Maps BSplineNDSeparableResult fields to user-facing stats struct."
```

---

## Task 5: Update build() Return Type

Change build() to return PriceTableResult<N> with full instrumentation (Step 2 in design, part 4).

**Files:**
- Modify: `src/option/price_table_builder.hpp` (update signature)
- Modify: `src/option/price_table_builder.cpp` (update implementation)
- Test: `tests/price_table_builder_result_test.cc`

**Step 5.1: Write the failing test**

```cpp
// tests/price_table_builder_result_test.cc
#include <gtest/gtest.h>
#include "src/option/price_table_builder.hpp"
#include "src/option/price_table_axes.hpp"
#include "src/option/price_table_config.hpp"

TEST(PriceTableBuilderResultTest, BuildReturnsDiagnostics) {
    // Create minimal valid axes (4 points per axis for B-spline)
    mango::PriceTableAxes<4> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1};  // moneyness
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};  // maturity
    axes.grids[2] = {0.15, 0.20, 0.25, 0.30};  // volatility
    axes.grids[3] = {0.03, 0.04, 0.05, 0.06};  // rate
    axes.names = {"moneyness", "maturity", "volatility", "rate"};

    mango::PriceTableConfig config;
    config.option_type = mango::OptionType::PUT;
    config.K_ref = 100.0;
    config.grid_estimator = mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value();
    config.n_time = 100;

    mango::PriceTableBuilder<4> builder(config);
    auto result = builder.build(axes);

    ASSERT_TRUE(result.has_value()) << "Build failed: " << result.error();

    // Check diagnostics are populated
    EXPECT_NE(result->surface, nullptr);
    EXPECT_GT(result->n_pde_solves, 0);
    EXPECT_GT(result->precompute_time_seconds, 0.0);
    // Fitting stats should be populated (exact values depend on data)
    EXPECT_GE(result->fitting_stats.max_residual_overall, 0.0);
}
```

**Step 5.2: Run test to verify it fails**

Run: `bazel test //tests:price_table_builder_result_test --test_output=all`
Expected: FAIL (build() returns wrong type)

**Step 5.3: Update build() signature in header**

```cpp
// In src/option/price_table_builder.hpp, change from:
std::expected<std::shared_ptr<PriceTableSurface<N>>, std::string>
build(const PriceTableAxes<N>& axes);

// To:
std::expected<PriceTableResult<N>, std::string>
build(const PriceTableAxes<N>& axes);
```

**Step 5.4: Update build() implementation with timing**

```cpp
// In src/option/price_table_builder.cpp:

template <size_t N>
std::expected<PriceTableResult<N>, std::string>
PriceTableBuilder<N>::build(const PriceTableAxes<N>& axes)
{
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // ... existing validation and batch creation ...

    // ... solve_batch() call ...

    // Count PDE solves
    size_t n_pde_solves = batch_result.results.size() - batch_result.failed_count;

    // ... extract_tensor() call ...

    // ... fit_coeffs() call (now returns FitCoeffsResult) ...
    auto& fit_result = coeffs_result.value();
    auto coefficients = std::move(fit_result.coefficients);
    BSplineFittingStats fitting_stats = fit_result.stats;

    // ... create surface ...

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    // Return full result
    return PriceTableResult<N>{
        .surface = std::move(surface),
        .n_pde_solves = n_pde_solves,
        .precompute_time_seconds = elapsed,
        .fitting_stats = fitting_stats
    };
}
```

**Step 5.5: Add test BUILD target**

```python
# In tests/BUILD.bazel, add:
cc_test(
    name = "price_table_builder_result_test",
    srcs = ["price_table_builder_result_test.cc"],
    deps = [
        "//src/option:price_table_builder",
        "//src/option:price_table_axes",
        "//src/option:price_table_config",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5.6: Run test to verify it passes**

Run: `bazel test //tests:price_table_builder_result_test --test_output=all`
Expected: PASS

**Step 5.7: Commit**

```bash
git add src/option/price_table_builder.hpp src/option/price_table_builder.cpp tests/price_table_builder_result_test.cc tests/BUILD.bazel
git commit -m "Update build() to return PriceTableResult<N>

Now returns full diagnostics:
- Surface pointer (shared_ptr<const PriceTableSurface<N>>)
- PDE solve count
- Wall-clock timing
- BSplineFittingStats

Breaking change: callers must update from shared_ptr to PriceTableResult."
```

---

## Task 6: Extend BatchAmericanOptionSolver Public API (Step 3a)

Add custom_grid parameter to public solve_batch() methods.

**Files:**
- Modify: `src/option/american_option_batch.hpp` (public signatures)
- Modify: `src/option/american_option_batch.cpp` (implementations)
- Test: `tests/batch_solver_custom_grid_test.cc`

**Step 6.1: Write the failing test**

```cpp
// tests/batch_solver_custom_grid_test.cc
#include <gtest/gtest.h>
#include "src/option/american_option_batch.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"

TEST(BatchSolverCustomGridTest, AcceptsCustomGrid) {
    std::vector<mango::AmericanOptionParams> batch;
    batch.emplace_back(100.0, 100.0, 1.0, 0.05, 0.02, mango::OptionType::PUT, 0.20);

    auto grid_spec = mango::GridSpec<double>::uniform(-2.0, 2.0, 51).value();
    auto time_domain = mango::TimeDomain::from_n_steps(0.0, 1.0, 100);

    std::optional<std::pair<mango::GridSpec<double>, mango::TimeDomain>> custom_grid =
        std::make_pair(grid_spec, time_domain);

    mango::BatchAmericanOptionSolver solver;
    auto result = solver.solve_batch(batch, true, nullptr, custom_grid);

    EXPECT_TRUE(result.all_succeeded());
    EXPECT_EQ(result.results.size(), 1);
}

TEST(BatchSolverCustomGridTest, NulloptUsesAutoEstimation) {
    std::vector<mango::AmericanOptionParams> batch;
    batch.emplace_back(100.0, 100.0, 1.0, 0.05, 0.02, mango::OptionType::PUT, 0.20);

    mango::BatchAmericanOptionSolver solver;
    // Pass nullopt explicitly - should use auto-estimation
    auto result = solver.solve_batch(batch, true, nullptr, std::nullopt);

    EXPECT_TRUE(result.all_succeeded());
}
```

**Step 6.2: Run test to verify it fails**

Run: `bazel test //tests:batch_solver_custom_grid_test --test_output=all`
Expected: FAIL (solve_batch doesn't accept 4th parameter)

**Step 6.3: Update public API in header (lines 187-199)**

```cpp
// In src/option/american_option_batch.hpp, update solve_batch signatures:

/// Solve a batch of American options with automatic routing
BatchAmericanOptionResult solve_batch(
    std::span<const AmericanOptionParams> params,
    bool use_shared_grid = false,
    SetupCallback setup = nullptr,
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid = std::nullopt);

/// Solve a batch of American options (vector overload)
BatchAmericanOptionResult solve_batch(
    const std::vector<AmericanOptionParams>& params,
    bool use_shared_grid = false,
    SetupCallback setup = nullptr,
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid = std::nullopt)
{
    return solve_batch(std::span{params}, use_shared_grid, setup, custom_grid);
}
```

**Step 6.4: Update private method signatures (lines 213-241)**

```cpp
// In src/option/american_option_batch.hpp, update private methods:

BatchAmericanOptionResult solve_regular_batch(
    std::span<const AmericanOptionParams> params,
    bool use_shared_grid = false,
    SetupCallback setup = nullptr,
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid = std::nullopt);

BatchAmericanOptionResult solve_normalized_chain(
    std::span<const AmericanOptionParams> params,
    SetupCallback setup,
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid = std::nullopt);
```

**Step 6.5: Update span overload dispatch in .cpp**

```cpp
// In src/option/american_option_batch.cpp, update solve_batch span overload:

BatchAmericanOptionResult BatchAmericanOptionSolver::solve_batch(
    std::span<const AmericanOptionParams> params,
    bool use_shared_grid,
    SetupCallback setup,
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid)
{
    if (params.empty()) {
        return BatchAmericanOptionResult{.results = {}, .failed_count = 0};
    }

    // Route to appropriate solver
    if (use_shared_grid && !setup && use_normalized_ && is_normalized_eligible(params, use_shared_grid)) {
        return solve_normalized_chain(params, setup, custom_grid);
    } else {
        return solve_regular_batch(params, use_shared_grid, setup, custom_grid);
    }
}
```

**Step 6.6: Update solve_regular_batch implementation**

```cpp
// In src/option/american_option_batch.cpp:

BatchAmericanOptionResult BatchAmericanOptionSolver::solve_regular_batch(
    std::span<const AmericanOptionParams> params,
    bool use_shared_grid,
    SetupCallback setup,
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid)
{
    // ... existing code ...

    // Grid determination
    GridSpec<double> grid_spec;
    TimeDomain time_domain;

    if (custom_grid.has_value()) {
        // Use provided grid directly (bypass auto-estimation)
        auto [gs, td] = custom_grid.value();
        grid_spec = gs;
        time_domain = td;
    } else {
        // Existing path: use grid_accuracy_ member to estimate grid
        // ... existing auto-estimation code ...
    }

    // ... rest of implementation using grid_spec and time_domain ...
}
```

**Step 6.7: Update solve_normalized_chain implementation**

```cpp
// In src/option/american_option_batch.cpp:

BatchAmericanOptionResult BatchAmericanOptionSolver::solve_normalized_chain(
    std::span<const AmericanOptionParams> params,
    SetupCallback setup,
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid)
{
    // ... existing code ...

    // Grid determination
    GridSpec<double> grid_spec;
    TimeDomain time_domain;

    if (custom_grid.has_value()) {
        // Use provided grid directly (bypass auto-estimation)
        auto [gs, td] = custom_grid.value();
        grid_spec = gs;
        time_domain = td;
    } else {
        // Existing path: use grid_accuracy_ member to estimate grid
        // ... existing auto-estimation code ...
    }

    // ... rest of implementation using grid_spec and time_domain ...
}
```

**Step 6.8: Add test BUILD target**

```python
# In tests/BUILD.bazel, add:
cc_test(
    name = "batch_solver_custom_grid_test",
    srcs = ["batch_solver_custom_grid_test.cc"],
    deps = [
        "//src/option:american_option_batch",
        "//src/pde/core:grid",
        "//src/pde/core:time_domain",
        "@googletest//:gtest_main",
    ],
)
```

**Step 6.9: Run test to verify it passes**

Run: `bazel test //tests:batch_solver_custom_grid_test --test_output=all`
Expected: PASS

**Step 6.10: Run full test suite**

Run: `bazel test //...`
Expected: All tests pass

**Step 6.11: Commit**

```bash
git add src/option/american_option_batch.hpp src/option/american_option_batch.cpp tests/batch_solver_custom_grid_test.cc tests/BUILD.bazel
git commit -m "Add custom_grid parameter to BatchAmericanOptionSolver

Extends solve_batch() with optional GridSpec+TimeDomain parameter:
- When provided, bypasses auto-estimation and uses exact grid
- When nullopt, uses existing grid_accuracy_ estimation

Enables PriceTableBuilder to pass user's exact grid configuration."
```

---

## Task 7: Update PriceTableBuilder to Use custom_grid (Step 3b)

Fix the grid configuration bug by passing complete GridSpec to solver.

**Files:**
- Modify: `src/option/price_table_builder.cpp` (lines 131-150)
- Test: `tests/price_table_builder_grid_test.cc`

**Step 7.1: Write the failing test**

```cpp
// tests/price_table_builder_grid_test.cc
#include <gtest/gtest.h>
#include "src/option/price_table_builder.hpp"
#include "src/option/price_table_axes.hpp"
#include "src/option/price_table_config.hpp"

TEST(PriceTableBuilderGridTest, RespectsUserGridBounds) {
    // Create axes with specific moneyness range
    mango::PriceTableAxes<4> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1, 1.2};  // moneyness range [0.8, 1.2]
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};
    axes.grids[2] = {0.15, 0.20, 0.25, 0.30};
    axes.grids[3] = {0.03, 0.04, 0.05, 0.06};
    axes.names = {"moneyness", "maturity", "volatility", "rate"};

    // Configure grid with specific bounds that cover log(0.8) to log(1.2)
    // log(0.8) ≈ -0.223, log(1.2) ≈ 0.182
    mango::PriceTableConfig config;
    config.option_type = mango::OptionType::PUT;
    config.K_ref = 100.0;
    config.grid_estimator = mango::GridSpec<double>::uniform(-0.5, 0.5, 51).value();
    config.n_time = 100;

    mango::PriceTableBuilder<4> builder(config);
    auto result = builder.build(axes);

    // Should succeed because grid bounds cover moneyness range
    ASSERT_TRUE(result.has_value()) << "Build failed: " << result.error();
    EXPECT_NE(result->surface, nullptr);
}

TEST(PriceTableBuilderGridTest, RejectsInsufficientGridBounds) {
    mango::PriceTableAxes<4> axes;
    axes.grids[0] = {0.5, 0.75, 1.0, 1.5, 2.0};  // Wide moneyness range
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};
    axes.grids[2] = {0.15, 0.20, 0.25, 0.30};
    axes.grids[3] = {0.03, 0.04, 0.05, 0.06};
    axes.names = {"moneyness", "maturity", "volatility", "rate"};

    // Configure grid with narrow bounds that don't cover the range
    // log(0.5) ≈ -0.693, log(2.0) ≈ 0.693, but grid only [-0.1, 0.1]
    mango::PriceTableConfig config;
    config.option_type = mango::OptionType::PUT;
    config.K_ref = 100.0;
    config.grid_estimator = mango::GridSpec<double>::uniform(-0.1, 0.1, 51).value();
    config.n_time = 100;

    mango::PriceTableBuilder<4> builder(config);
    auto result = builder.build(axes);

    // Should fail validation because grid bounds don't cover moneyness range
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("exceeds PDE grid bounds") != std::string::npos ||
                result.error().find("moneyness") != std::string::npos);
}
```

**Step 7.2: Run test to verify it fails**

Run: `bazel test //tests:price_table_builder_grid_test --test_output=all`
Expected: FAIL (grid bounds not checked/respected)

**Step 7.3: Update build() to pass custom_grid**

```cpp
// In src/option/price_table_builder.cpp, in build() method:

// BEFORE (buggy - only forwards point counts):
// GridAccuracyParams accuracy;
// accuracy.min_spatial_points = ...
// solver.set_grid_accuracy(accuracy);
// return solver.solve_batch(batch, true);

// AFTER (fixed - passes complete GridSpec with domain bounds):
BatchAmericanOptionSolver solver;

// Build custom grid config with user's exact GridSpec
GridSpec<double> user_grid = config_.grid_estimator;

// Use TimeDomain factory
auto time_domain = TimeDomain::from_n_steps(
    0.0,                        // t_start
    axes.grids[1].back(),       // t_end (max maturity)
    config_.n_time              // n_steps
);

// Pass complete grid specification (bypasses auto-estimation)
std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid =
    std::make_pair(user_grid, time_domain);

// CRITICAL: Register maturity grid as snapshot times
solver.set_snapshot_times(axes.grids[1]);

// Solve with custom grid (x_min/x_max preserved)
auto batch_result = solver.solve_batch(batch, true, nullptr, custom_grid);
```

**Step 7.4: Add test BUILD target**

```python
# In tests/BUILD.bazel, add:
cc_test(
    name = "price_table_builder_grid_test",
    srcs = ["price_table_builder_grid_test.cc"],
    deps = [
        "//src/option:price_table_builder",
        "//src/option:price_table_axes",
        "//src/option:price_table_config",
        "@googletest//:gtest_main",
    ],
)
```

**Step 7.5: Run test to verify it passes**

Run: `bazel test //tests:price_table_builder_grid_test --test_output=all`
Expected: PASS

**Step 7.6: Commit**

```bash
git add src/option/price_table_builder.cpp tests/price_table_builder_grid_test.cc tests/BUILD.bazel
git commit -m "Fix grid configuration bug in PriceTableBuilder

Now passes user's complete GridSpec to BatchAmericanOptionSolver
via custom_grid parameter, preserving x_min/x_max domain bounds.

Previously, GridAccuracyParams only forwarded point counts,
causing auto-estimation to override user's spatial domain."
```

---

## Task 8: Port Validation Logic (Step 4)

Add comprehensive input validation from specialized builder.

**Files:**
- Modify: `src/option/price_table_builder.cpp` (add validation in build())
- Test: `tests/price_table_builder_validation_test.cc`

**Step 8.1: Write the failing tests**

```cpp
// tests/price_table_builder_validation_test.cc
#include <gtest/gtest.h>
#include "src/option/price_table_builder.hpp"
#include "src/option/price_table_axes.hpp"
#include "src/option/price_table_config.hpp"

class PriceTableValidationTest : public ::testing::Test {
protected:
    mango::PriceTableConfig config;
    mango::PriceTableAxes<4> axes;

    void SetUp() override {
        config.option_type = mango::OptionType::PUT;
        config.K_ref = 100.0;
        config.grid_estimator = mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value();
        config.n_time = 100;

        // Valid default axes
        axes.grids[0] = {0.8, 0.9, 1.0, 1.1};
        axes.grids[1] = {0.25, 0.5, 0.75, 1.0};
        axes.grids[2] = {0.15, 0.20, 0.25, 0.30};
        axes.grids[3] = {0.03, 0.04, 0.05, 0.06};
        axes.names = {"moneyness", "maturity", "volatility", "rate"};
    }
};

TEST_F(PriceTableValidationTest, RejectsTooFewPointsAxis0) {
    axes.grids[0] = {0.9, 1.0, 1.1};  // Only 3 points
    mango::PriceTableBuilder<4> builder(config);
    auto result = builder.build(axes);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("4") != std::string::npos);  // Needs 4 points
}

TEST_F(PriceTableValidationTest, RejectsNegativeMoneyness) {
    axes.grids[0] = {-0.1, 0.9, 1.0, 1.1};  // Negative moneyness
    mango::PriceTableBuilder<4> builder(config);
    auto result = builder.build(axes);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("positive") != std::string::npos);
}

TEST_F(PriceTableValidationTest, RejectsZeroMaturity) {
    axes.grids[1] = {0.0, 0.5, 0.75, 1.0};  // Zero maturity
    mango::PriceTableBuilder<4> builder(config);
    auto result = builder.build(axes);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("positive") != std::string::npos ||
                result.error().find("Maturity") != std::string::npos);
}

TEST_F(PriceTableValidationTest, RejectsNegativeVolatility) {
    axes.grids[2] = {-0.1, 0.20, 0.25, 0.30};  // Negative vol
    mango::PriceTableBuilder<4> builder(config);
    auto result = builder.build(axes);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("positive") != std::string::npos);
}

TEST_F(PriceTableValidationTest, RejectsZeroKRef) {
    config.K_ref = 0.0;
    mango::PriceTableBuilder<4> builder(config);
    auto result = builder.build(axes);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("K_ref") != std::string::npos);
}

TEST_F(PriceTableValidationTest, AcceptsValidInput) {
    mango::PriceTableBuilder<4> builder(config);
    auto result = builder.build(axes);
    EXPECT_TRUE(result.has_value()) << "Unexpected error: " << result.error();
}
```

**Step 8.2: Run tests to verify they fail**

Run: `bazel test //tests:price_table_builder_validation_test --test_output=all`
Expected: FAIL (validation not implemented)

**Step 8.3: Add validation in build()**

```cpp
// In src/option/price_table_builder.cpp, at start of build():

template <size_t N>
std::expected<PriceTableResult<N>, std::string>
PriceTableBuilder<N>::build(const PriceTableAxes<N>& axes)
{
    if constexpr (N != 4) {
        return std::unexpected("Only N=4 currently supported");
    }

    // Validate axes basic structure
    auto validation = axes.validate();
    if (!validation.has_value()) {
        return std::unexpected("Validation failed");
    }

    // Check minimum 4 points per axis (B-spline requirement)
    for (size_t i = 0; i < N; ++i) {
        if (axes.grids[i].size() < 4) {
            return std::unexpected("Axis " + std::to_string(i) +
                                   " has only " + std::to_string(axes.grids[i].size()) +
                                   " points (need >=4 for cubic B-splines)");
        }
    }

    // Check positive moneyness (needed for log)
    if (axes.grids[0].front() <= 0.0) {
        return std::unexpected("Moneyness must be positive (needed for log)");
    }

    // Check positive maturity (strict > 0)
    if (axes.grids[1].front() <= 0.0) {
        return std::unexpected("Maturity must be positive (tau > 0 required for PDE time domain)");
    }

    // Check positive volatility
    if (axes.grids[2].front() <= 0.0) {
        return std::unexpected("Volatility must be positive");
    }

    // Check K_ref > 0
    if (config_.K_ref <= 0.0) {
        return std::unexpected("Reference strike K_ref must be positive");
    }

    // Check PDE domain coverage
    const double x_min_requested = std::log(axes.grids[0].front());
    const double x_max_requested = std::log(axes.grids[0].back());
    const double x_min = config_.grid_estimator.x_min();
    const double x_max = config_.grid_estimator.x_max();

    if (x_min_requested < x_min || x_max_requested > x_max) {
        return std::unexpected(
            "Requested moneyness range [" + std::to_string(axes.grids[0].front()) + ", " +
            std::to_string(axes.grids[0].back()) + "] in spot ratios "
            "maps to log-moneyness [" + std::to_string(x_min_requested) + ", " +
            std::to_string(x_max_requested) + "], "
            "which exceeds PDE grid bounds [" + std::to_string(x_min) + ", " +
            std::to_string(x_max) + "]. "
            "Narrow the moneyness grid or expand the PDE domain."
        );
    }

    // ... rest of build() implementation ...
}
```

**Step 8.4: Add test BUILD target**

```python
# In tests/BUILD.bazel, add:
cc_test(
    name = "price_table_builder_validation_test",
    srcs = ["price_table_builder_validation_test.cc"],
    deps = [
        "//src/option:price_table_builder",
        "//src/option:price_table_axes",
        "//src/option:price_table_config",
        "@googletest//:gtest_main",
    ],
)
```

**Step 8.5: Run tests to verify they pass**

Run: `bazel test //tests:price_table_builder_validation_test --test_output=all`
Expected: PASS

**Step 8.6: Commit**

```bash
git add src/option/price_table_builder.cpp tests/price_table_builder_validation_test.cc tests/BUILD.bazel
git commit -m "Port validation logic to generic PriceTableBuilder

Validates:
- Minimum 4 points per axis (B-spline requirement)
- Positive moneyness (needed for log transform)
- Positive maturity (tau > 0 for PDE)
- Positive volatility
- Positive K_ref
- PDE domain covers requested moneyness range

Catches invalid inputs before expensive PDE work."
```

---

## Task 9: Add Helper Factories (Step 6)

Add from_vectors, from_strikes, from_chain factory methods.

**Files:**
- Modify: `src/option/price_table_builder.hpp` (add factory declarations)
- Modify: `src/option/price_table_builder.cpp` (add implementations)
- Test: `tests/price_table_builder_factories_test.cc`

**Step 9.1: Write the failing test for from_vectors**

```cpp
// tests/price_table_builder_factories_test.cc
#include <gtest/gtest.h>
#include "src/option/price_table_builder.hpp"
#include "src/option/option_chain.hpp"

TEST(PriceTableFactoriesTest, FromVectorsCreatesBuilderAndAxes) {
    std::vector<double> moneyness = {0.8, 0.9, 1.0, 1.1};
    std::vector<double> maturity = {0.25, 0.5, 0.75, 1.0};
    std::vector<double> vol = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate = {0.03, 0.04, 0.05, 0.06};

    auto grid_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value();

    auto result = mango::PriceTableBuilder<4>::from_vectors(
        moneyness, maturity, vol, rate,
        100.0,      // K_ref
        grid_spec,
        100,        // n_time
        mango::OptionType::PUT
    );

    ASSERT_TRUE(result.has_value()) << "Factory failed";
    auto& [builder, axes] = result.value();

    EXPECT_EQ(axes.grids[0].size(), 4);
    EXPECT_EQ(axes.grids[1].size(), 4);
}

TEST(PriceTableFactoriesTest, FromStrikesComputesMoneyness) {
    double spot = 100.0;
    std::vector<double> strikes = {80.0, 90.0, 100.0, 110.0, 120.0};
    std::vector<double> maturities = {0.25, 0.5, 0.75, 1.0};
    std::vector<double> vols = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rates = {0.03, 0.04, 0.05, 0.06};

    auto grid_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value();

    auto result = mango::PriceTableBuilder<4>::from_strikes(
        spot, strikes, maturities, vols, rates,
        grid_spec, 100, mango::OptionType::PUT
    );

    ASSERT_TRUE(result.has_value()) << "Factory failed";
    auto& [builder, axes] = result.value();

    // Moneyness = spot/strike, sorted ascending
    // strikes [80, 90, 100, 110, 120] → moneyness [0.833, 0.909, 1.0, 1.111, 1.25]
    EXPECT_EQ(axes.grids[0].size(), 5);
    EXPECT_NEAR(axes.grids[0][0], 100.0/120.0, 1e-6);  // 0.833
    EXPECT_NEAR(axes.grids[0][4], 100.0/80.0, 1e-6);   // 1.25
}

TEST(PriceTableFactoriesTest, FromChainExtractsFields) {
    mango::OptionChain chain;
    chain.ticker = "AAPL";
    chain.spot = 150.0;
    chain.strikes = {140.0, 145.0, 150.0, 155.0, 160.0};
    chain.maturities = {0.25, 0.5, 0.75, 1.0};
    chain.implied_vols = {0.20, 0.22, 0.25, 0.28};
    chain.rates = {0.04, 0.045, 0.05, 0.055};
    chain.dividend_yield = 0.01;

    auto grid_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value();

    auto result = mango::PriceTableBuilder<4>::from_chain(
        chain, grid_spec, 100, mango::OptionType::PUT
    );

    ASSERT_TRUE(result.has_value()) << "Factory failed";
    auto& [builder, axes] = result.value();

    EXPECT_EQ(axes.grids[0].size(), 5);  // 5 strikes → 5 moneyness
    EXPECT_EQ(axes.grids[1].size(), 4);  // 4 maturities
}
```

**Step 9.2: Run test to verify it fails**

Run: `bazel test //tests:price_table_builder_factories_test --test_output=all`
Expected: FAIL (factories not defined)

**Step 9.3: Add factory declarations to header**

```cpp
// In src/option/price_table_builder.hpp, add to PriceTableBuilder class:

#include "src/option/option_chain.hpp"

template <size_t N>
class PriceTableBuilder {
public:
    // ... existing members ...

    /// Factory from vectors (returns builder AND axes)
    static std::expected<std::pair<PriceTableBuilder<4>, PriceTableAxes<4>>, std::string>
    from_vectors(
        std::vector<double> moneyness,
        std::vector<double> maturity,
        std::vector<double> volatility,
        std::vector<double> rate,
        double K_ref,
        GridSpec<double> grid_spec,
        size_t n_time,
        OptionType type = OptionType::PUT,
        double dividend_yield = 0.0);

    /// Factory from strikes (auto-computes moneyness)
    static std::expected<std::pair<PriceTableBuilder<4>, PriceTableAxes<4>>, std::string>
    from_strikes(
        double spot,
        std::vector<double> strikes,
        std::vector<double> maturities,
        std::vector<double> volatilities,
        std::vector<double> rates,
        GridSpec<double> grid_spec,
        size_t n_time,
        OptionType type = OptionType::PUT,
        double dividend_yield = 0.0);

    /// Factory from option chain
    static std::expected<std::pair<PriceTableBuilder<4>, PriceTableAxes<4>>, std::string>
    from_chain(
        const OptionChain& chain,
        GridSpec<double> grid_spec,
        size_t n_time,
        OptionType type = OptionType::PUT);
};
```

**Step 9.4: Implement factories in .cpp**

```cpp
// In src/option/price_table_builder.cpp:

#include <algorithm>

namespace {
// Helper: sort and dedupe a vector
std::vector<double> sort_and_dedupe(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}
}  // namespace

template <>
std::expected<std::pair<PriceTableBuilder<4>, PriceTableAxes<4>>, std::string>
PriceTableBuilder<4>::from_vectors(
    std::vector<double> moneyness,
    std::vector<double> maturity,
    std::vector<double> volatility,
    std::vector<double> rate,
    double K_ref,
    GridSpec<double> grid_spec,
    size_t n_time,
    OptionType type,
    double dividend_yield)
{
    // Sort and dedupe
    moneyness = sort_and_dedupe(std::move(moneyness));
    maturity = sort_and_dedupe(std::move(maturity));
    volatility = sort_and_dedupe(std::move(volatility));
    rate = sort_and_dedupe(std::move(rate));

    // Validate positivity
    if (!moneyness.empty() && moneyness.front() <= 0.0) {
        return std::unexpected("Moneyness must be positive");
    }
    if (!maturity.empty() && maturity.front() <= 0.0) {
        return std::unexpected("Maturity must be positive");
    }
    if (!volatility.empty() && volatility.front() <= 0.0) {
        return std::unexpected("Volatility must be positive");
    }
    if (K_ref <= 0.0) {
        return std::unexpected("K_ref must be positive");
    }

    // Build axes
    PriceTableAxes<4> axes;
    axes.grids[0] = std::move(moneyness);
    axes.grids[1] = std::move(maturity);
    axes.grids[2] = std::move(volatility);
    axes.grids[3] = std::move(rate);
    axes.names = {"moneyness", "maturity", "volatility", "rate"};

    // Build config
    PriceTableConfig config;
    config.option_type = type;
    config.K_ref = K_ref;
    config.grid_estimator = grid_spec;
    config.n_time = n_time;
    config.dividend_yield = dividend_yield;

    return std::make_pair(PriceTableBuilder<4>(config), std::move(axes));
}

template <>
std::expected<std::pair<PriceTableBuilder<4>, PriceTableAxes<4>>, std::string>
PriceTableBuilder<4>::from_strikes(
    double spot,
    std::vector<double> strikes,
    std::vector<double> maturities,
    std::vector<double> volatilities,
    std::vector<double> rates,
    GridSpec<double> grid_spec,
    size_t n_time,
    OptionType type,
    double dividend_yield)
{
    if (spot <= 0.0) {
        return std::unexpected("Spot must be positive");
    }

    // Sort and dedupe
    strikes = sort_and_dedupe(std::move(strikes));
    maturities = sort_and_dedupe(std::move(maturities));
    volatilities = sort_and_dedupe(std::move(volatilities));
    rates = sort_and_dedupe(std::move(rates));

    // Validate strikes positive
    if (!strikes.empty() && strikes.front() <= 0.0) {
        return std::unexpected("Strikes must be positive");
    }

    // Compute moneyness = spot/strike
    std::vector<double> moneyness;
    moneyness.reserve(strikes.size());
    for (double K : strikes) {
        moneyness.push_back(spot / K);
    }
    // Note: if strikes are ascending, moneyness is descending
    // Sort to make ascending
    std::sort(moneyness.begin(), moneyness.end());

    return from_vectors(
        std::move(moneyness),
        std::move(maturities),
        std::move(volatilities),
        std::move(rates),
        spot,  // K_ref = spot
        grid_spec,
        n_time,
        type,
        dividend_yield
    );
}

template <>
std::expected<std::pair<PriceTableBuilder<4>, PriceTableAxes<4>>, std::string>
PriceTableBuilder<4>::from_chain(
    const OptionChain& chain,
    GridSpec<double> grid_spec,
    size_t n_time,
    OptionType type)
{
    return from_strikes(
        chain.spot,
        chain.strikes,
        chain.maturities,
        chain.implied_vols,
        chain.rates,
        grid_spec,
        n_time,
        type,
        chain.dividend_yield
    );
}
```

**Step 9.5: Add test BUILD target**

```python
# In tests/BUILD.bazel, add:
cc_test(
    name = "price_table_builder_factories_test",
    srcs = ["price_table_builder_factories_test.cc"],
    deps = [
        "//src/option:price_table_builder",
        "//src/option:option_chain",
        "@googletest//:gtest_main",
    ],
)
```

**Step 9.6: Run test to verify it passes**

Run: `bazel test //tests:price_table_builder_factories_test --test_output=all`
Expected: PASS

**Step 9.7: Commit**

```bash
git add src/option/price_table_builder.hpp src/option/price_table_builder.cpp tests/price_table_builder_factories_test.cc tests/BUILD.bazel
git commit -m "Add helper factories to PriceTableBuilder

from_vectors(): Direct axis creation from vectors
from_strikes(): Computes moneyness from spot/strike
from_chain(): Extracts from OptionChain struct

All factories return (builder, axes) pair, solving axes ownership.
Includes sort/dedupe and positivity validation."
```

---

## Task 10: Update IVSolverInterpolated (Step 7 Critical)

Migrate the most critical consumer before deletion.

**Files:**
- Modify: `src/option/iv_solver_interpolated.hpp`
- Modify: `src/option/iv_solver_interpolated.cpp`
- Test: Existing `tests/iv_solver_interpolated_test.cc`

**Step 10.1: Update includes**

```cpp
// In src/option/iv_solver_interpolated.hpp and .cpp, change:

// BEFORE:
#include "src/option/price_table_4d_builder.hpp"

// AFTER:
#include "src/option/price_table_builder.hpp"
#include "src/option/price_table_axes.hpp"
#include "src/option/price_table_surface.hpp"
```

**Step 10.2: Update surface storage type**

```cpp
// In iv_solver_interpolated.hpp, if surface is stored:

// BEFORE:
PriceTableSurface surface_;

// AFTER:
std::shared_ptr<const PriceTableSurface<4>> surface_;
```

**Step 10.3: Update surface construction**

```cpp
// In iv_solver_interpolated.cpp, where table is built:

// BEFORE:
auto builder_result = PriceTable4DBuilder::create(moneyness, maturity, vol, rate, K_ref);
auto result = builder_result->precompute(OptionType::PUT, 101, 1000);
surface_ = result->surface;

// AFTER:
auto grid_spec = GridSpec<double>::uniform(-3.0, 3.0, 101).value();
auto factory_result = PriceTableBuilder<4>::from_vectors(
    moneyness, maturity, vol, rate, K_ref, grid_spec, 1000, OptionType::PUT
);
if (!factory_result.has_value()) {
    // Handle error
    return std::unexpected(factory_result.error());
}
auto [builder, axes] = std::move(factory_result.value());

auto build_result = builder.build(axes);
if (!build_result.has_value()) {
    return std::unexpected(build_result.error());
}
surface_ = build_result->surface;
```

**Step 10.4: Update surface access**

```cpp
// BEFORE:
double price = surface_.eval(m, tau, sigma, r);

// AFTER:
double price = surface_->eval(m, tau, sigma, r);  // Note: -> not .
```

**Step 10.5: Verify tests pass**

Run: `bazel test //tests:iv_solver_interpolated_test --test_output=all`
Expected: PASS

**Step 10.6: Commit**

```bash
git add src/option/iv_solver_interpolated.hpp src/option/iv_solver_interpolated.cpp
git commit -m "Migrate IVSolverInterpolated to generic PriceTableBuilder

- Update includes from price_table_4d_builder to price_table_builder
- Change surface storage to shared_ptr<const PriceTableSurface<4>>
- Use from_vectors factory for table creation
- Update surface access from . to ->

Critical migration for Step 8 deletion."
```

---

## Task 11: Update Remaining Consumers (Step 7)

Search and update all other references to specialized builder.

**Files:**
- Various test files
- Benchmark files
- Example files
- Documentation

**Step 11.1: Search for all references**

Run:
```bash
rg -n "price_table_4d_builder" -g'*.[ch]pp' -g'*.cc'
rg -n "PriceTable4DBuilder"
rg -n "PriceTable4D" -g'*.md'
```

**Step 11.2: Update each file found**

For each file, apply the migration pattern from Task 10:
1. Update includes
2. Update type (shared_ptr)
3. Use factory method
4. Update access (. to ->)

**Step 11.3: Verify all tests pass**

Run: `bazel test //...`
Expected: All tests pass

**Step 11.4: Commit**

```bash
git add -A
git commit -m "Update all consumers to use generic PriceTableBuilder

Migrated files:
- tests/price_table_4d_integration_test.cc
- tests/price_table_end_to_end_performance_test.cc
- benchmarks/market_iv_e2e_benchmark.cc
- [list other files]

All references to PriceTable4DBuilder replaced."
```

---

## Task 12: Delete Specialized Builder (Step 8)

Remove the old builder after all consumers migrated.

**Files:**
- Delete: `src/option/price_table_4d_builder.hpp`
- Delete: `src/option/price_table_4d_builder.cpp`
- Modify: `src/option/BUILD.bazel` (remove targets)

**Step 12.1: Verify no remaining references**

Run:
```bash
rg -n "price_table_4d_builder" -g'*.[ch]pp' -g'*.cc'
rg -n "PriceTable4DBuilder"
```

Expected: No matches

**Step 12.2: Delete files**

```bash
rm src/option/price_table_4d_builder.hpp
rm src/option/price_table_4d_builder.cpp
```

**Step 12.3: Update BUILD file**

Remove the `price_table_4d_builder` target from `src/option/BUILD.bazel`.

**Step 12.4: Verify build succeeds**

Run: `bazel build //...`
Expected: SUCCESS

**Step 12.5: Verify all tests pass**

Run: `bazel test //...`
Expected: All tests pass

**Step 12.6: Commit**

```bash
git add -A
git commit -m "Delete specialized PriceTable4DBuilder

All consumers migrated to generic PriceTableBuilder<4>.
OptionChain preserved in shared header (src/option/option_chain.hpp).

Completes Phase 1 migration."
```

---

## Task 13: Update Documentation (Step 9)

Update all documentation to reflect new API.

**Files:**
- Modify: `docs/API_GUIDE.md`
- Modify: `CLAUDE.md`
- Modify: Inline doc comments

**Step 13.1: Update API_GUIDE.md**

Replace all examples using PriceTable4DBuilder with PriceTableBuilder<4>.

**Step 13.2: Update CLAUDE.md**

Update quick reference examples.

**Step 13.3: Verify documentation builds**

Run: `bazel build //docs/...` (if applicable)

**Step 13.4: Commit**

```bash
git add docs/API_GUIDE.md CLAUDE.md
git commit -m "Update documentation for generic PriceTableBuilder

- Replace PriceTable4DBuilder examples with PriceTableBuilder<4>
- Document new factory methods
- Document PriceTableResult<N> return type
- Add migration notes"
```

---

## Task 14: Final Verification

Run full test suite and verify clean build.

**Step 14.1: Clean build**

Run: `bazel clean && bazel build //...`
Expected: SUCCESS with no warnings

**Step 14.2: Full test suite**

Run: `bazel test //...`
Expected: All tests pass

**Step 14.3: Build examples**

Run: `bazel build //examples/...`
Expected: SUCCESS

**Step 14.4: Build benchmarks**

Run: `bazel build //benchmarks/...`
Expected: SUCCESS

**Step 14.5: Commit final state**

```bash
git add -A
git commit -m "Phase 1 complete: Generic PriceTableBuilder migration

Summary:
- Added PriceTableResult<N> with full diagnostics
- Fixed grid configuration bug (custom_grid parameter)
- Ported validation logic from specialized builder
- Added from_vectors/from_strikes/from_chain factories
- Migrated all consumers
- Deleted specialized PriceTable4DBuilder
- Updated documentation

All tests passing. Ready for Phase 2 improvements."
```

---

**Plan complete and saved to `docs/plans/2025-11-24-price-table-builder-impl.md`.**

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
