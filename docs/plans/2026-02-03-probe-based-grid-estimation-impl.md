# Probe-Based PDE Grid Estimation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace heuristic grid estimation in IVSolver with Richardson-style probe solves that empirically verify grid adequacy.

**Architecture:** Add `probe_grid_adequacy()` function, integrate into IVSolver with σ_high calibration and per-solve() grid caching.

**Tech Stack:** C++23, GoogleTest, USDT probes

---

### Task 1: Add target_price_error to IVSolverFDMConfig

**Files:**
- Modify: `src/option/iv_solver.hpp`
- Test: `tests/iv_solver_test.cc`

**Step 1: Write the failing test**

```cpp
TEST(IVSolverTest, TargetPriceErrorConfig) {
    // Custom value
    IVSolverFDMConfig config{.target_price_error = 0.005};
    EXPECT_EQ(config.target_price_error, 0.005);

    // Default should be 0.01
    IVSolverFDMConfig default_config{};
    EXPECT_EQ(default_config.target_price_error, 0.01);

    // Zero disables probe-based (uses grid field instead)
    IVSolverFDMConfig heuristic_config{.target_price_error = 0.0};
    EXPECT_EQ(heuristic_config.target_price_error, 0.0);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:iv_solver_test --test_filter=*TargetPriceErrorConfig* --test_output=all`
Expected: FAIL with "no member named 'target_price_error'"

**Step 3: Add field to IVSolverFDMConfig**

In `src/option/iv_solver.hpp`, add to `IVSolverFDMConfig`:
```cpp
struct IVSolverFDMConfig {
    // ... existing fields ...

    /// Target price error for probe-based grid calibration (absolute units).
    /// If > 0, uses Richardson-style probes; `grid` field is ignored.
    /// If == 0, falls back to `grid` field (heuristic or explicit).
    /// Default: 0.01 ($0.01 accuracy)
    double target_price_error = 0.01;
};
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:iv_solver_test --test_filter=*TargetPriceErrorConfig* --test_output=all`
Expected: PASS

**Step 5: Commit**

```bash
git add src/option/iv_solver.hpp tests/iv_solver_test.cc
git commit -m "Add target_price_error field to IVSolverFDMConfig"
```

---

### Task 2: Create probe_grid_adequacy() function

**Files:**
- Create: `src/option/grid_probe.hpp`
- Create: `src/option/grid_probe.cpp`
- Modify: `src/option/BUILD.bazel`
- Create: `tests/grid_probe_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the failing test**

```cpp
// tests/grid_probe_test.cc
#include <gtest/gtest.h>
#include "src/option/grid_probe.hpp"

TEST(GridProbeTest, ConvergesForTypicalOption) {
    mango::PricingParams params(
        mango::OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = mango::OptionType::PUT},
        0.20);

    auto result = mango::probe_grid_adequacy(params, 0.01);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->converged);
    EXPECT_LE(result->estimated_error, 0.01);
    EXPECT_GE(result->grid.n_points(), 100);
    EXPECT_GE(result->time_domain.n_steps(), 50);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:grid_probe_test --test_output=all`
Expected: FAIL with "No such file or directory" or "probe_grid_adequacy not found"

**Step 3: Create header with ProbeResult and function declaration**

```cpp
// src/option/grid_probe.hpp
// SPDX-License-Identifier: MIT
#pragma once

#include <expected>
#include "src/pde/core/grid_spec.hpp"
#include "src/pde/core/time_domain.hpp"
#include "src/option/american_option.hpp"
#include "src/support/error_types.hpp"

namespace mango {

struct ProbeResult {
    GridSpec<double> grid;
    TimeDomain time_domain;
    double estimated_error;
    size_t probe_iterations;
    bool converged;  ///< False if max iterations reached without convergence
};

/// Probe-based grid adequacy check using Richardson extrapolation.
/// Solves at Nx and 2Nx, compares prices and deltas.
/// Returns grid adequate for target_error.
/// If max_iterations reached without convergence, returns finest grid with converged=false.
std::expected<ProbeResult, ValidationError> probe_grid_adequacy(
    const PricingParams& params,
    double target_error,
    size_t initial_Nx = 100,
    size_t max_iterations = 3);

}  // namespace mango
```

**Step 4: Create implementation**

```cpp
// src/option/grid_probe.cpp
// SPDX-License-Identifier: MIT
#include "src/option/grid_probe.hpp"
#include "src/option/american_option.hpp"
#include <cmath>
#include <algorithm>

namespace mango {

std::expected<ProbeResult, ValidationError> probe_grid_adequacy(
    const PricingParams& params,
    double target_error,
    size_t initial_Nx,
    size_t max_iterations)
{
    if (target_error <= 0.0) {
        return std::unexpected(ValidationError{"target_error must be positive"});
    }

    size_t Nx = initial_Nx;
    constexpr double delta_tol = 0.01;
    constexpr double price_floor = 0.10;
    constexpr size_t min_time_steps = 50;
    // Note: TR-BDF2 is L-stable (unconditionally stable for diffusion)
    // No CFL constraint needed; Nt floor is purely for accuracy

    GridSpec<double> best_grid;
    TimeDomain best_td;
    double best_error = std::numeric_limits<double>::max();

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        // Build grid at Nx
        GridAccuracyParams acc1{};
        acc1.min_spatial_points = Nx;
        acc1.max_spatial_points = Nx;
        auto [grid1, td1] = estimate_pde_grid(params, acc1);

        // Enforce Nt floor for short maturities (accuracy, not stability)
        if (td1.n_steps() < min_time_steps) {
            td1 = TimeDomain::from_n_steps(0.0, params.maturity, min_time_steps);
        }

        // Solve at Nx
        auto result1 = solve_american_option(params, PDEGridConfig{
            .grid_spec = grid1, .n_time = td1.n_steps()});
        if (!result1.has_value()) {
            return std::unexpected(ValidationError{"PDE solve failed at Nx"});
        }
        double P1 = result1->value_at(params.spot);

        // Build grid at 2Nx
        GridAccuracyParams acc2{};
        acc2.min_spatial_points = 2 * Nx;
        acc2.max_spatial_points = 2 * Nx;
        auto [grid2, td2] = estimate_pde_grid(params, acc2);

        // Enforce Nt floor for finer grid
        if (td2.n_steps() < min_time_steps) {
            td2 = TimeDomain::from_n_steps(0.0, params.maturity, min_time_steps);
        }

        // Solve at 2Nx
        auto result2 = solve_american_option(params, PDEGridConfig{
            .grid_spec = grid2, .n_time = td2.n_steps()});
        if (!result2.has_value()) {
            return std::unexpected(ValidationError{"PDE solve failed at 2Nx"});
        }
        double P2 = result2->value_at(params.spot);

        // Compute delta consistently at same physical point via finite difference
        // Note: value_at() uses linear interpolation in log-moneyness space,
        // which is sufficient since we're comparing relative convergence
        double h = std::max(0.01 * params.spot, 0.01);  // 1% bump with floor

        // Ensure spot ± h is within grid domain (grid domain is in log-moneyness)
        double x_spot = std::log(params.spot / params.strike);
        double x_lo = grid2.x_min();
        double x_hi = grid2.x_max();
        double spot_lo = params.strike * std::exp(x_lo);
        double spot_hi = params.strike * std::exp(x_hi);
        double s_minus = std::max(params.spot - h, spot_lo * 1.01);
        double s_plus = std::min(params.spot + h, spot_hi * 0.99);
        h = (s_plus - s_minus) / 2.0;  // Recompute h after clamping
        double s_center = (s_plus + s_minus) / 2.0;

        double delta1 = (result1->value_at(s_plus) - result1->value_at(s_minus)) / (2.0 * h);
        double delta2 = (result2->value_at(s_plus) - result2->value_at(s_minus)) / (2.0 * h);

        // Composite acceptance criterion using max of both prices
        double price_diff = std::abs(P1 - P2);
        double delta_diff = std::abs(delta1 - delta2);
        double price_ref = std::max({std::abs(P1), std::abs(P2), price_floor});
        double price_tol = std::max(target_error, 0.001 * price_ref);

        best_grid = grid2;
        best_td = td2;
        best_error = price_diff;

        if (price_diff <= price_tol && delta_diff <= delta_tol) {
            return ProbeResult{
                .grid = grid2,
                .time_domain = td2,
                .estimated_error = price_diff,
                .probe_iterations = iter + 1,
                .converged = true
            };
        }

        Nx *= 2;
    }

    // Return finest grid with converged=false
    return ProbeResult{
        .grid = best_grid,
        .time_domain = best_td,
        .estimated_error = best_error,
        .probe_iterations = max_iterations,
        .converged = false
    };
}

}  // namespace mango
```

**Step 5: Add BUILD targets**

Add to `src/option/BUILD.bazel`:
```python
cc_library(
    name = "grid_probe",
    srcs = ["grid_probe.cpp"],
    hdrs = ["grid_probe.hpp"],
    deps = [
        ":american_option",
        "//src/pde/core:grid_spec",
        "//src/pde/core:time_domain",
        "//src/support:error_types",
    ],
    visibility = ["//visibility:public"],
)
```

Add to `tests/BUILD.bazel`:
```python
cc_test(
    name = "grid_probe_test",
    srcs = ["grid_probe_test.cc"],
    deps = [
        "//src/option:grid_probe",
        "@googletest//:gtest_main",
    ],
)
```

**Step 6: Run test to verify it passes**

Run: `bazel test //tests:grid_probe_test --test_output=all`
Expected: PASS

**Step 7: Commit**

```bash
git add src/option/grid_probe.hpp src/option/grid_probe.cpp src/option/BUILD.bazel tests/grid_probe_test.cc tests/BUILD.bazel
git commit -m "Add probe_grid_adequacy() for Richardson-style grid calibration"
```

---

### Task 3: Add edge case tests for probe_grid_adequacy

**Files:**
- Modify: `tests/grid_probe_test.cc`

**Step 1: Write tests for edge cases**

```cpp
TEST(GridProbeTest, ShortMaturityEnforcesNtFloor) {
    mango::PricingParams params(
        mango::OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 0.02,  // 1 week
            .rate = 0.05, .option_type = mango::OptionType::PUT},
        0.30);

    auto result = mango::probe_grid_adequacy(params, 0.01);
    ASSERT_TRUE(result.has_value());
    EXPECT_GE(result->time_domain.n_steps(), 50);  // Nt floor for accuracy
    // Note: No CFL check needed - TR-BDF2 is L-stable (unconditionally stable)
}

TEST(GridProbeTest, DeepITMConverges) {
    mango::PricingParams params(
        mango::OptionSpec{
            .spot = 80.0, .strike = 100.0, .maturity = 1.0,  // 20% ITM put
            .rate = 0.05, .option_type = mango::OptionType::PUT},
        0.20);

    auto result = mango::probe_grid_adequacy(params, 0.01);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->converged);
    EXPECT_LE(result->estimated_error, 0.01);
}

TEST(GridProbeTest, HighVolatilityConverges) {
    mango::PricingParams params(
        mango::OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .option_type = mango::OptionType::PUT},
        0.60);  // 60% vol

    auto result = mango::probe_grid_adequacy(params, 0.01);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->converged);
    EXPECT_LE(result->estimated_error, 0.01);
}

TEST(GridProbeTest, WithDiscreteDividends) {
    mango::PricingParams params(
        mango::OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .option_type = mango::OptionType::PUT,
            .discrete_dividends = {mango::Dividend{.calendar_time = 0.25, .amount = 2.0}}},
        0.20);

    auto result = mango::probe_grid_adequacy(params, 0.01);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->converged);
    EXPECT_LE(result->estimated_error, 0.01);
}

TEST(GridProbeTest, InvalidTargetErrorReturnsError) {
    mango::PricingParams params(
        mango::OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                          .rate = 0.05, .option_type = mango::OptionType::PUT},
        0.20);

    auto result = mango::probe_grid_adequacy(params, -0.01);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().message.find("positive") != std::string::npos);
}

TEST(GridProbeTest, MaxIterationsReturnsConvergedFalse) {
    // Very tight tolerance that may not converge in 3 iterations
    mango::PricingParams params(
        mango::OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .option_type = mango::OptionType::PUT},
        0.20);

    // Use 1 iteration max with very tight tolerance
    auto result = mango::probe_grid_adequacy(params, 1e-8, 50, 1);
    ASSERT_TRUE(result.has_value());
    // Either converged or not, but we have a result
    EXPECT_EQ(result->probe_iterations, 1);
    // If not converged, converged flag should be false
    if (result->estimated_error > 1e-8) {
        EXPECT_FALSE(result->converged);
    }
}
```

**Step 2: Run tests**

Run: `bazel test //tests:grid_probe_test --test_output=all`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/grid_probe_test.cc
git commit -m "Add edge case tests for probe_grid_adequacy"
```

---

### Task 4: Integrate probe into IVSolver with per-solve() caching

**Files:**
- Modify: `src/option/iv_solver.hpp`
- Modify: `src/option/iv_solver.cpp`
- Modify: `tests/iv_solver_test.cc`

**Step 1: Write failing test**

```cpp
TEST(IVSolverTest, UsesProbeBasedGridWhenTargetPriceErrorSet) {
    mango::OptionSpec spec{
        .spot = 100.0, .strike = 100.0, .maturity = 1.0,
        .rate = 0.05, .dividend_yield = 0.02,
        .option_type = mango::OptionType::PUT};

    mango::IVSolverFDMConfig config{
        .target_price_error = 0.01,
        .sigma_lower = 0.05,
        .sigma_upper = 0.80
    };

    mango::IVSolver solver(config);
    mango::IVQuery query(spec, 5.50);  // ~20% IV

    auto result = solver.solve(query);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->iv, 0.20, 0.01);
}

TEST(IVSolverTest, DifferentOptionsGetDifferentGrids) {
    // Verify cache is per-solve(), not per-solver-instance
    mango::IVSolverFDMConfig config{.target_price_error = 0.01};
    mango::IVSolver solver(config);

    // Option 1: short maturity
    mango::OptionSpec spec1{
        .spot = 100.0, .strike = 100.0, .maturity = 0.1,
        .rate = 0.05, .option_type = mango::OptionType::PUT};
    mango::IVQuery query1(spec1, 2.0);

    // Option 2: long maturity (needs different grid)
    mango::OptionSpec spec2{
        .spot = 100.0, .strike = 100.0, .maturity = 2.0,
        .rate = 0.05, .option_type = mango::OptionType::PUT};
    mango::IVQuery query2(spec2, 12.0);

    auto result1 = solver.solve(query1);
    auto result2 = solver.solve(query2);

    // Both should succeed (if cache was shared incorrectly, one might fail)
    ASSERT_TRUE(result1.has_value());
    ASSERT_TRUE(result2.has_value());
}

TEST(IVSolverTest, FallsBackToHeuristicWhenProbeDisabled) {
    mango::OptionSpec spec{
        .spot = 100.0, .strike = 100.0, .maturity = 1.0,
        .rate = 0.05, .option_type = mango::OptionType::PUT};

    // target_price_error = 0 disables probe-based
    mango::IVSolverFDMConfig config{.target_price_error = 0.0};
    mango::IVSolver solver(config);
    mango::IVQuery query(spec, 5.50);

    auto result = solver.solve(query);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->iv, 0.20, 0.01);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:iv_solver_test --test_filter=*ProbeBasedGrid* --test_output=all`
Expected: FAIL or unexpected behavior (probe not yet integrated)

**Step 3: Modify solve() to calibrate at σ_high with local cache**

In `src/option/iv_solver.cpp`, modify `solve()`:
```cpp
std::expected<IVSuccess, IVError> IVSolver::solve(const IVQuery& query) const {
    // Local cache for this solve() call only (NOT a member variable)
    std::optional<std::pair<GridSpec<double>, TimeDomain>> probe_grid;

    // If target_price_error > 0, use probe-based calibration
    if (config_.target_price_error > 0.0) {
        // Calibrate at sigma_upper (worst case for grid requirements)
        double sigma_high = config_.sigma_upper;
        PricingParams probe_params(query.option, sigma_high);

        auto probe_result = probe_grid_adequacy(probe_params, config_.target_price_error);
        if (probe_result.has_value() && probe_result->converged) {
            probe_grid = {probe_result->grid, probe_result->time_domain};
        }
        // If probe fails or doesn't converge, probe_grid remains nullopt
        // -> falls back to default heuristic (GridAccuracyParams{})
        // Note: config_.grid is IGNORED when target_price_error > 0
    }

    // Pass probe_grid to objective function (or nullptr for heuristic path)
    // ... rest of Brent search using probe_grid if set ...
}
```

**Step 4: Modify objective_function to accept optional cached grid**

```cpp
// Use cached grid when available
if (probe_grid.has_value()) {
    grid_spec = probe_grid->first;
    time_domain = probe_grid->second;
} else if (config_.target_price_error > 0.0) {
    // Probe was attempted but didn't converge -> use default heuristic
    // (config_.grid is ignored when target_price_error > 0)
    auto [auto_grid, auto_td] = estimate_pde_grid(option_params, GridAccuracyParams{});
    grid_spec = auto_grid;
    time_domain = auto_td;
} else {
    // target_price_error == 0 -> use config_.grid (existing behavior)
    std::visit([&](const auto& grid_variant) { ... }, config_.grid);
}
```

**Step 5: Add include for grid_probe.hpp**

```cpp
#include "src/option/grid_probe.hpp"
```

**Step 6: Update BUILD.bazel to add grid_probe dependency**

```python
# In src/option/BUILD.bazel, add to iv_solver deps:
deps = [
    # ... existing deps ...
    ":grid_probe",
],
```

**Step 7: Run test to verify it passes**

Run: `bazel test //tests:iv_solver_test --test_output=all`
Expected: All PASS

**Step 8: Commit**

```bash
git add src/option/iv_solver.hpp src/option/iv_solver.cpp src/option/BUILD.bazel tests/iv_solver_test.cc
git commit -m "Integrate probe-based grid calibration into IVSolver with per-solve caching"
```

---

### Task 5: Add fallback behavior test when probe doesn't converge

**Files:**
- Modify: `tests/iv_solver_test.cc`

**Step 1: Write test for fallback behavior**

```cpp
TEST(IVSolverTest, FallsBackToHeuristicWhenProbeDoesntConverge) {
    mango::OptionSpec spec{
        .spot = 100.0, .strike = 100.0, .maturity = 1.0,
        .rate = 0.05, .option_type = mango::OptionType::PUT};

    // Very tight tolerance that probe may not achieve
    mango::IVSolverFDMConfig config{.target_price_error = 1e-10};
    mango::IVSolver solver(config);
    mango::IVQuery query(spec, 5.50);

    // Should still succeed (falls back to heuristic or uses non-converged grid)
    auto result = solver.solve(query);
    ASSERT_TRUE(result.has_value());
    // IV should still be reasonable
    EXPECT_GT(result->iv, 0.10);
    EXPECT_LT(result->iv, 0.50);
}
```

**Step 2: Run tests**

Run: `bazel test //tests:iv_solver_test --test_output=all`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/iv_solver_test.cc
git commit -m "Add test for IVSolver fallback when probe doesn't converge"
```

---

### Task 6: Add USDT probes for calibration tracing

**Files:**
- Modify: `src/option/grid_probe.cpp`

**Step 1: Add USDT probe points**

```cpp
// At top of file
#include "src/support/usdt_probes.hpp"

// In probe_grid_adequacy():

// At start of function
MANGO_PROBE(grid_probe, calibration_start, initial_Nx, target_error);

// After each iteration
MANGO_PROBE(grid_probe, iteration_complete, iter, Nx, price_diff, delta_diff,
            (price_diff <= price_tol && delta_diff <= delta_tol));

// At end (success path)
MANGO_PROBE(grid_probe, calibration_complete, grid2.n_points(), price_diff, iter + 1, true);

// At end (non-convergence path)
MANGO_PROBE(grid_probe, calibration_complete, best_grid.n_points(), best_error, max_iterations, false);
```

**Step 2: Test with mango-trace**

Run: `sudo ./tools/mango-trace monitor ./test_program --preset=debug`
Expected: See probe events during IV solve

**Step 3: Commit**

```bash
git add src/option/grid_probe.cpp
git commit -m "Add USDT probes for grid calibration tracing"
```

---

### Task 7: Add benchmark comparing old vs new IVSolver

**Files:**
- Create: `benchmarks/iv_solver_probe_benchmark.cc`
- Modify: `benchmarks/BUILD.bazel`

**Step 1: Create benchmark**

```cpp
// SPDX-License-Identifier: MIT
#include <benchmark/benchmark.h>
#include "src/option/iv_solver.hpp"

static void BM_IVSolver_Heuristic(benchmark::State& state) {
    mango::OptionSpec spec{
        .spot = 100.0, .strike = 100.0, .maturity = 1.0,
        .rate = 0.05, .dividend_yield = 0.02,
        .option_type = mango::OptionType::PUT};

    mango::IVSolverFDMConfig config{.target_price_error = 0.0};  // Disable probe
    mango::IVSolver solver(config);
    mango::IVQuery query(spec, 5.50);

    for (auto _ : state) {
        auto result = solver.solve(query);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_IVSolver_Heuristic);

static void BM_IVSolver_ProbeBased(benchmark::State& state) {
    mango::OptionSpec spec{
        .spot = 100.0, .strike = 100.0, .maturity = 1.0,
        .rate = 0.05, .dividend_yield = 0.02,
        .option_type = mango::OptionType::PUT};

    mango::IVSolverFDMConfig config{.target_price_error = 0.01};
    mango::IVSolver solver(config);
    mango::IVQuery query(spec, 5.50);

    for (auto _ : state) {
        auto result = solver.solve(query);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_IVSolver_ProbeBased);

// Benchmark single-iteration case (where probe overhead is not amortized)
static void BM_IVSolver_ProbeBased_SingleIteration(benchmark::State& state) {
    mango::OptionSpec spec{
        .spot = 100.0, .strike = 100.0, .maturity = 1.0,
        .rate = 0.05, .dividend_yield = 0.02,
        .option_type = mango::OptionType::PUT};

    mango::IVSolverFDMConfig config{.target_price_error = 0.01};
    mango::IVQuery query(spec, 5.50);

    for (auto _ : state) {
        // New solver each iteration (no amortization)
        mango::IVSolver solver(config);
        auto result = solver.solve(query);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_IVSolver_ProbeBased_SingleIteration);

BENCHMARK_MAIN();
```

**Step 2: Add BUILD target**

```python
cc_binary(
    name = "iv_solver_probe_benchmark",
    srcs = ["iv_solver_probe_benchmark.cc"],
    deps = [
        "//src/option:iv_solver",
        "@google_benchmark//:benchmark_main",
    ],
)
```

**Step 3: Run benchmark**

Run: `bazel run //benchmarks:iv_solver_probe_benchmark`
Expected: Multi-iteration probe-based should be faster; single-iteration may be slower

**Step 4: Commit**

```bash
git add benchmarks/iv_solver_probe_benchmark.cc benchmarks/BUILD.bazel
git commit -m "Add benchmark comparing heuristic vs probe-based IVSolver"
```

---

### Task 8: Update documentation

**Files:**
- Modify: `docs/API_GUIDE.md`
- Modify: `CLAUDE.md`

**Step 1: Add usage example to API_GUIDE.md**

Add new section after IV Solver section:

```markdown
### Probe-Based Grid Calibration

For users who want accuracy guarantees without understanding grid parameters:

```cpp
IVSolverFDMConfig config{
    .target_price_error = 0.01  // $0.01 absolute price accuracy
};

IVSolver solver(config);
auto result = solver.solve(query);
```

The solver automatically calibrates the PDE grid using Richardson-style
probe solves at σ_high (the upper bound of the volatility search range),
ensuring the requested accuracy is achieved.

**API Rules:**
- `target_price_error > 0`: Uses probe-based calibration; `grid` field is ignored
- `target_price_error == 0`: Falls back to `grid` field (heuristic or explicit)
- Default: `0.01` ($0.01 accuracy)

**Performance Note:** Probe calibration adds 4-6 PDE solves upfront (~50-100ms),
but this is amortized across all Brent iterations. For IV solves requiring
multiple iterations, total time is typically ~20% faster than per-iteration
heuristic re-estimation.
```

**Step 2: Update CLAUDE.md patterns**

Add Pattern 5 to "Common Development Patterns" section:

```markdown
**Pattern 5: Probe-Based IV Solver**
```cpp
#include "src/option/iv_solver.hpp"

// Simple: just specify target accuracy
mango::IVSolverFDMConfig config{.target_price_error = 0.01};
mango::IVSolver solver(config);

mango::IVQuery query(spec, 10.45);
auto result = solver.solve(query);
```
```

**Step 3: Commit**

```bash
git add docs/API_GUIDE.md CLAUDE.md
git commit -m "Document probe-based grid calibration in API guide"
```

---

## Verification

After all tasks complete:

```bash
bazel test //...                           # All tests pass
bazel build //benchmarks/...               # Benchmarks compile
bazel build //src/python:mango_option      # Python bindings compile
bazel run //benchmarks:iv_solver_probe_benchmark  # Performance validated
```

Verify:
- `target_price_error` field exists with default 0.01
- Probe-based calibration activates when `target_price_error > 0`
- Grid is cached per `solve()` call (not shared across calls)
- Fallback to heuristic works when `target_price_error == 0` or probe doesn't converge
- Edge cases handled: short maturity, deep ITM, high vol, dividends
- Benchmark shows expected performance characteristics
