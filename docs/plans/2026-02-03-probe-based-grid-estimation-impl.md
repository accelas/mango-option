# Probe-Based PDE Grid Estimation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace heuristic grid estimation in IVSolver with Richardson-style probe solves that empirically verify grid adequacy.

**Architecture:** Add `probe_grid_adequacy()` function, integrate into IVSolver with σ_high calibration and grid caching.

**Tech Stack:** C++23, GoogleTest, USDT probes

---

### Task 1: Add target_price_error to IVSolverFDMConfig

**Files:**
- Modify: `src/option/iv_solver.hpp`
- Test: `tests/iv_solver_test.cc`

**Step 1: Write the failing test**

```cpp
TEST(IVSolverTest, TargetPriceErrorConfig) {
    IVSolverFDMConfig config{.target_price_error = 0.005};
    EXPECT_EQ(config.target_price_error, 0.005);

    // Default should be 0.01
    IVSolverFDMConfig default_config{};
    EXPECT_EQ(default_config.target_price_error, 0.01);
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

    /// Target price error for probe-based grid calibration.
    /// When set, uses Richardson-style probes instead of heuristic.
    /// Default: 0.01 ($0.01 per $100 notional)
    std::optional<double> target_price_error = std::nullopt;
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
};

/// Probe-based grid adequacy check using Richardson extrapolation.
/// Solves at Nx and 2Nx, compares prices and deltas.
/// Returns grid adequate for target_error, or error if max iterations exceeded.
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
#include "src/option/grid_probe.hpp"
#include "src/option/american_option.hpp"
#include <cmath>

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

    GridSpec<double> best_grid;
    TimeDomain best_td;
    double best_error = std::numeric_limits<double>::max();

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        // Build grid at Nx
        GridAccuracyParams acc1{};
        acc1.min_spatial_points = Nx;
        acc1.max_spatial_points = Nx;
        auto [grid1, td1] = estimate_pde_grid(params, acc1);

        // Enforce Nt floor for short maturities
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
        double delta1 = result1->delta();

        // Build grid at 2Nx
        GridAccuracyParams acc2{};
        acc2.min_spatial_points = 2 * Nx;
        acc2.max_spatial_points = 2 * Nx;
        auto [grid2, td2] = estimate_pde_grid(params, acc2);

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
        double delta2 = result2->delta();

        // Composite acceptance criterion
        double price_diff = std::abs(P1 - P2);
        double delta_diff = std::abs(delta1 - delta2);
        double price_tol = std::max(target_error, 0.001 * std::max(P1, price_floor));

        best_grid = grid2;
        best_td = td2;
        best_error = price_diff;

        if (price_diff <= price_tol && delta_diff <= delta_tol) {
            return ProbeResult{
                .grid = grid2,
                .time_domain = td2,
                .estimated_error = price_diff,
                .probe_iterations = iter + 1
            };
        }

        Nx *= 2;
    }

    // Return finest grid with warning (via USDT probe in production)
    return ProbeResult{
        .grid = best_grid,
        .time_domain = best_td,
        .estimated_error = best_error,
        .probe_iterations = max_iterations
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

**Step 1: Write failing tests for edge cases**

```cpp
TEST(GridProbeTest, ShortMaturityEnforcesNtFloor) {
    mango::PricingParams params(
        mango::OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 0.02,  // 1 week
            .rate = 0.05, .option_type = mango::OptionType::PUT},
        0.30);

    auto result = mango::probe_grid_adequacy(params, 0.01);
    ASSERT_TRUE(result.has_value());
    EXPECT_GE(result->time_domain.n_steps(), 50);  // Nt floor
}

TEST(GridProbeTest, DeepITMConverges) {
    mango::PricingParams params(
        mango::OptionSpec{
            .spot = 80.0, .strike = 100.0, .maturity = 1.0,  // 20% ITM put
            .rate = 0.05, .option_type = mango::OptionType::PUT},
        0.20);

    auto result = mango::probe_grid_adequacy(params, 0.01);
    ASSERT_TRUE(result.has_value());
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

### Task 4: Integrate probe into IVSolver with grid caching

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
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:iv_solver_test --test_filter=*ProbeBasedGrid* --test_output=all`
Expected: FAIL or unexpected behavior (probe not yet integrated)

**Step 3: Add grid caching member to IVSolver**

In `src/option/iv_solver.hpp`:
```cpp
class IVSolver {
    // ... existing members ...

private:
    /// Cached grid from probe calibration (reused across Brent iterations)
    mutable std::optional<std::pair<GridSpec<double>, TimeDomain>> cached_probe_grid_;
};
```

**Step 4: Modify solve() to calibrate at σ_high and cache**

In `src/option/iv_solver.cpp`, at the start of `solve()`:
```cpp
// If target_price_error is set, use probe-based calibration
if (config_.target_price_error.has_value() && !cached_probe_grid_.has_value()) {
    double sigma_high = std::max(config_.sigma_upper, 2.0 * initial_sigma_guess);
    PricingParams probe_params(query.option, sigma_high);

    auto probe_result = probe_grid_adequacy(probe_params, *config_.target_price_error);
    if (probe_result.has_value()) {
        cached_probe_grid_ = {probe_result->grid, probe_result->time_domain};
    }
    // If probe fails, fall back to heuristic (existing behavior)
}
```

**Step 5: Use cached grid in objective function**

Modify `objective_function()` to use cached grid when available:
```cpp
if (cached_probe_grid_.has_value()) {
    grid_spec = cached_probe_grid_->first;
    time_domain = cached_probe_grid_->second;
} else {
    // Existing heuristic path
    std::visit([&](const auto& grid_variant) { ... }, config_.grid);
}
```

**Step 6: Run test to verify it passes**

Run: `bazel test //tests:iv_solver_test --test_output=all`
Expected: All PASS

**Step 7: Commit**

```bash
git add src/option/iv_solver.hpp src/option/iv_solver.cpp tests/iv_solver_test.cc
git commit -m "Integrate probe-based grid calibration into IVSolver"
```

---

### Task 5: Add USDT probes for calibration tracing

**Files:**
- Modify: `src/option/grid_probe.cpp`
- Modify: `src/support/usdt_probes.hpp` (if exists) or create probe definitions

**Step 1: Add USDT probe points**

```cpp
// In grid_probe.cpp
MANGO_PROBE(grid_probe, calibration_start, Nx, target_error);
MANGO_PROBE(grid_probe, iteration_complete, iter, Nx, price_diff, delta_diff, converged);
MANGO_PROBE(grid_probe, calibration_complete, final_Nx, estimated_error, iterations);
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

### Task 6: Add benchmark comparing old vs new IVSolver

**Files:**
- Create: `benchmarks/iv_solver_probe_benchmark.cc`
- Modify: `benchmarks/BUILD.bazel`

**Step 1: Create benchmark**

```cpp
#include <benchmark/benchmark.h>
#include "src/option/iv_solver.hpp"

static void BM_IVSolver_Heuristic(benchmark::State& state) {
    mango::OptionSpec spec{...};
    mango::IVSolverFDMConfig config{};  // Uses heuristic
    mango::IVSolver solver(config);
    mango::IVQuery query(spec, 5.50);

    for (auto _ : state) {
        auto result = solver.solve(query);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_IVSolver_Heuristic);

static void BM_IVSolver_ProbeBased(benchmark::State& state) {
    mango::OptionSpec spec{...};
    mango::IVSolverFDMConfig config{.target_price_error = 0.01};
    mango::IVSolver solver(config);
    mango::IVQuery query(spec, 5.50);

    for (auto _ : state) {
        auto result = solver.solve(query);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_IVSolver_ProbeBased);

BENCHMARK_MAIN();
```

**Step 2: Run benchmark**

Run: `bazel run //benchmarks:iv_solver_probe_benchmark`
Expected: Probe-based should be ~20% faster after calibration warmup

**Step 3: Commit**

```bash
git add benchmarks/iv_solver_probe_benchmark.cc benchmarks/BUILD.bazel
git commit -m "Add benchmark comparing heuristic vs probe-based IVSolver"
```

---

### Task 7: Update documentation

**Files:**
- Modify: `docs/API_GUIDE.md`
- Modify: `CLAUDE.md`

**Step 1: Add usage example to API_GUIDE.md**

```markdown
### Probe-Based Grid Calibration

For users who want accuracy guarantees without understanding grid parameters:

```cpp
IVSolverFDMConfig config{
    .target_price_error = 0.01  // $0.01 accuracy per $100 notional
};

IVSolver solver(config);
auto result = solver.solve(query);
```

The solver automatically calibrates the PDE grid using Richardson-style
probe solves, ensuring the requested accuracy is achieved.
```

**Step 2: Update CLAUDE.md patterns**

Add to "Common Development Patterns" section.

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
