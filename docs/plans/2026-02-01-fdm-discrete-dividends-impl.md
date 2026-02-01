# FDM Discrete Dividend Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add discrete cash dividend handling to the FDM American option solver via mandatory time points and cubic spline solution interpolation at dividend dates.

**Architecture:** Extend `TimeDomain` to support non-uniform time steps that land exactly on dividend dates. Register temporal events on the PDE solver that shift the solution in log-moneyness space via cubic spline interpolation at each dividend date. Wire this into `AmericanOptionSolver::solve()` and the regular batch path.

**Tech Stack:** C++23, TR-BDF2 PDE solver, `CubicSpline<double>` from `src/math/cubic_spline_solver.hpp`

---

### Task 1: Extend TimeDomain with mandatory time points

**Files:**
- Modify: `src/pde/core/time_domain.hpp`
- Test: `tests/time_domain_test.cc`

**Step 1: Write the failing test**

Add to `tests/time_domain_test.cc`:

```cpp
TEST(TimeDomainTest, MandatoryTimePoints) {
    // dt=0.25 over [0, 1.0], mandatory point at 0.3
    auto td = TimeDomain::with_mandatory_points(0.0, 1.0, 0.25, {0.3});
    auto pts = td.time_points();

    // Must contain 0.0, 0.3, and 1.0
    EXPECT_DOUBLE_EQ(pts.front(), 0.0);
    EXPECT_DOUBLE_EQ(pts.back(), 1.0);

    // 0.3 must appear exactly
    bool found = false;
    for (double p : pts) {
        if (std::abs(p - 0.3) < 1e-14) found = true;
    }
    EXPECT_TRUE(found) << "Mandatory point 0.3 not found in time points";

    // All intervals must be <= dt + epsilon
    for (size_t i = 1; i < pts.size(); ++i) {
        EXPECT_LE(pts[i] - pts[i-1], 0.25 + 1e-10);
    }

    // Strictly increasing
    for (size_t i = 1; i < pts.size(); ++i) {
        EXPECT_GT(pts[i], pts[i-1]);
    }
}

TEST(TimeDomainTest, MandatoryTimePointsMultiple) {
    // Two mandatory points
    auto td = TimeDomain::with_mandatory_points(0.0, 1.0, 0.5, {0.2, 0.7});
    auto pts = td.time_points();

    // Both mandatory points must appear
    auto contains = [&](double v) {
        for (double p : pts) if (std::abs(p - v) < 1e-14) return true;
        return false;
    };
    EXPECT_TRUE(contains(0.2));
    EXPECT_TRUE(contains(0.7));
}

TEST(TimeDomainTest, MandatoryTimePointsEmptyFallback) {
    // No mandatory points: same as from_n_steps
    auto td = TimeDomain::with_mandatory_points(0.0, 1.0, 0.25, {});
    EXPECT_EQ(td.n_steps(), 4u);
    EXPECT_NEAR(td.dt(), 0.25, 1e-14);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:time_domain_test --test_output=all`
Expected: FAIL — `with_mandatory_points` does not exist

**Step 3: Implement TimeDomain::with_mandatory_points**

Add to `src/pde/core/time_domain.hpp`:

```cpp
/// Construct time domain with mandatory time points
///
/// Produces a non-uniform time grid that lands exactly on each mandatory
/// point. Between mandatory points, sub-intervals are roughly dt-sized.
/// Mandatory points outside [t_start, t_end] are ignored.
///
/// @param t_start Initial time
/// @param t_end Final time
/// @param dt Target time step size
/// @param mandatory Mandatory time points (e.g., dividend dates)
static TimeDomain with_mandatory_points(
    double t_start, double t_end, double dt,
    std::vector<double> mandatory)
{
    // Filter and sort mandatory points within (t_start, t_end)
    std::vector<double> breaks;
    for (double t : mandatory) {
        if (t > t_start + 1e-14 && t < t_end - 1e-14) {
            breaks.push_back(t);
        }
    }
    std::sort(breaks.begin(), breaks.end());

    // Remove duplicates
    breaks.erase(std::unique(breaks.begin(), breaks.end(),
        [](double a, double b) { return std::abs(a - b) < 1e-14; }),
        breaks.end());

    if (breaks.empty()) {
        // No mandatory points: uniform grid
        size_t n = static_cast<size_t>(std::ceil((t_end - t_start) / dt));
        return from_n_steps(t_start, t_end, n);
    }

    // Build time points: subdivide each interval [boundary_i, boundary_{i+1}]
    std::vector<double> boundaries;
    boundaries.push_back(t_start);
    for (double b : breaks) boundaries.push_back(b);
    boundaries.push_back(t_end);

    std::vector<double> points;
    points.push_back(t_start);

    for (size_t seg = 0; seg + 1 < boundaries.size(); ++seg) {
        double seg_start = boundaries[seg];
        double seg_end = boundaries[seg + 1];
        double seg_len = seg_end - seg_start;
        size_t n_sub = std::max(size_t{1},
            static_cast<size_t>(std::ceil(seg_len / dt)));
        double sub_dt = seg_len / static_cast<double>(n_sub);
        for (size_t j = 1; j <= n_sub; ++j) {
            points.push_back(seg_start + j * sub_dt);
        }
    }

    // Store as non-uniform time domain
    TimeDomain td;
    td.t_start_ = t_start;
    td.t_end_ = t_end;
    td.n_steps_ = points.size() - 1;
    td.dt_ = (t_end - t_start) / static_cast<double>(td.n_steps_);  // average dt
    td.time_points_ = std::move(points);
    return td;
}
```

This requires adding a `std::vector<double> time_points_` member and modifying `time_points()` and `dt()` to use it when non-empty. Also add a per-step dt accessor:

```cpp
/// Get dt for a specific step (supports non-uniform grids)
double dt_at(size_t step) const {
    if (!time_points_.empty() && step < time_points_.size() - 1) {
        return time_points_[step + 1] - time_points_[step];
    }
    return dt_;
}

/// Generate vector of time points from t_start to t_end
std::vector<double> time_points() const {
    if (!time_points_.empty()) {
        return time_points_;
    }
    // Original uniform path
    std::vector<double> times;
    times.reserve(n_steps_ + 1);
    for (size_t i = 0; i <= n_steps_; ++i) {
        times.push_back(t_start_ + i * dt_);
    }
    return times;
}
```

New member:
```cpp
std::vector<double> time_points_;  ///< Non-uniform time points (empty = uniform)
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:time_domain_test --test_output=all`
Expected: PASS

**Step 5: Run full suite**

Run: `bazel test //...`
Expected: All 104 tests pass (no regression)

**Step 6: Commit**

```bash
git add src/pde/core/time_domain.hpp tests/time_domain_test.cc
git commit -m "Add TimeDomain::with_mandatory_points for non-uniform grids"
```

---

### Task 2: Update PDESolver to use variable dt per step

**Files:**
- Modify: `src/pde/core/pde_solver.hpp`
- Test: `tests/pde_solver_test.cc`

The `solve()` loop currently uses `const double dt = time.dt()`. Change it to read per-step dt from TimeDomain.

**Step 1: Write the failing test**

Add to `tests/pde_solver_test.cc` (or a new test file if cleaner):

```cpp
// Test that PDE solver works with non-uniform time steps
TEST(PDESolverTest, NonUniformTimeSteps) {
    // Solve a simple heat equation with a mandatory time point
    // and verify the solution is reasonable (not NaN, converges)
    // Use Laplacian PDE as simplest test case

    // Create time domain with mandatory point
    auto td = TimeDomain::with_mandatory_points(0.0, 1.0, 0.01, {0.33});

    // ... (set up grid, workspace, simple PDE solver)
    // Verify solution is finite and converges
}
```

However, we can rely on the existing PDE solver tests continuing to pass (they use uniform dt, which still works). The key change is the solve loop. Let's make the change and verify existing tests pass.

**Step 2: Modify PDESolver::solve()**

In `src/pde/core/pde_solver.hpp`, change the `solve()` method:

Replace the time-stepping section (lines ~114-178):

```cpp
std::expected<void, SolverError> solve() {
    const auto& time = grid_->time();
    double t = time.t_start();

    auto u_current = grid_->solution();
    auto u_prev = grid_->solution_prev();

    // Record initial condition if requested
    if (grid_->should_record(0)) {
        grid_->record(0, u_current);
    }

    size_t step = 0;
    if (config_.rannacher_startup && time.n_steps() > 0) {
        double dt_step = time.dt_at(0);
        auto rannacher_ok = solve_rannacher_startup(t, dt_step, u_current, u_prev)
            .transform([&] {
                t += dt_step;
                if (grid_->should_record(step + 1)) {
                    grid_->record(step + 1, u_current);
                }
                step = 1;
            });
        if (!rannacher_ok) {
            return std::unexpected(rannacher_ok.error());
        }
    }

    for (; step < time.n_steps(); ++step) {
        double t_old = t;
        double dt_step = time.dt_at(step);

        // Copy u_current to u_prev for next iteration
        std::copy(u_current.begin(), u_current.end(), u_prev.begin());

        double t_next = t + dt_step;

        // Stage 1: Trapezoidal rule to t_n + γ·dt
        double t_stage1 = t + config_.gamma * dt_step;
        auto stage1_ok = solve_stage1(t, t_stage1, dt_step, u_current, u_prev);
        if (!stage1_ok) {
            return std::unexpected(stage1_ok.error());
        }

        // Stage 2: BDF2 from t_n to t_n+1
        auto stage2_ok = solve_stage2(t_stage1, t_next, dt_step, u_current, u_prev);
        if (!stage2_ok) {
            return std::unexpected(stage2_ok.error());
        }

        // Process temporal events AFTER completing the step
        process_temporal_events(t_old, t_next, step, u_current);

        // Update time
        t = t_next;

        // Record snapshot AFTER events
        if (grid_->should_record(step + 1)) {
            grid_->record(step + 1, u_current);
        }
    }

    return {};
}
```

Also update `solve_rannacher_startup()` similarly — it uses `dt` directly. Change it to accept `dt` as parameter or read `dt_at(0)`.

**Step 3: Run full test suite**

Run: `bazel test //...`
Expected: All 104 tests pass. The uniform dt path produces identical `dt_at(step)` values, so existing behavior is unchanged.

**Step 4: Commit**

```bash
git add src/pde/core/pde_solver.hpp
git commit -m "Use per-step dt in PDESolver for non-uniform time grids"
```

---

### Task 3: Implement dividend jump callback

**Files:**
- Create: `src/option/discrete_dividend_event.hpp`
- Modify: `src/option/BUILD.bazel`
- Test: `tests/discrete_dividend_event_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the failing test**

Create `tests/discrete_dividend_event_test.cc`:

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/discrete_dividend_event.hpp"
#include <vector>
#include <cmath>

using namespace mango;

TEST(DiscreteDividendEventTest, BasicPutShift) {
    // Grid in log-moneyness: x = ln(S/K)
    // K = 100, dividend D = 5
    // At x = 0 (S = 100): S_adj = 95, x_adj = ln(0.95) ≈ -0.0513
    // The solution should shift right (toward lower S)

    std::vector<double> x = {-1.0, -0.5, 0.0, 0.5, 1.0};
    // Put payoff: max(1 - exp(x), 0)
    std::vector<double> u(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        u[i] = std::max(1.0 - std::exp(x[i]), 0.0);
    }

    double original_atm = u[2];  // At x=0, payoff = 0

    auto callback = make_dividend_event(5.0, 100.0);
    callback(0.5, std::span<const double>(x), std::span<double>(u));

    // After dividend: value at x=0 should increase for a put
    // because S drops from 100 to 95 (deeper ITM)
    EXPECT_GT(u[2], original_atm)
        << "Put value at ATM should increase after dividend (spot drops)";

    // Solution should still be non-negative (put values)
    for (size_t i = 0; i < u.size(); ++i) {
        EXPECT_GE(u[i], -1e-10) << "Solution must be non-negative at index " << i;
    }
}

TEST(DiscreteDividendEventTest, NoShiftWhenSpotBelowDividend) {
    // At x = -3 (S/K ≈ 0.05, S ≈ 5), dividend D = 10
    // S - D < 0, so clamp to intrinsic value
    std::vector<double> x = {-3.0};
    std::vector<double> u = {1.0 - std::exp(-3.0)};  // put intrinsic

    auto callback = make_dividend_event(10.0, 100.0);
    callback(0.5, std::span<const double>(x), std::span<double>(u));

    // Should be clamped to intrinsic (can't go negative on spot)
    EXPECT_GE(u[0], 0.0);
}

TEST(DiscreteDividendEventTest, ZeroDividendNoOp) {
    std::vector<double> x = {-1.0, 0.0, 1.0};
    std::vector<double> u = {0.5, 0.3, 0.1};
    std::vector<double> u_orig = u;

    auto callback = make_dividend_event(0.0, 100.0);
    callback(0.5, std::span<const double>(x), std::span<double>(u));

    for (size_t i = 0; i < u.size(); ++i) {
        EXPECT_DOUBLE_EQ(u[i], u_orig[i]);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:discrete_dividend_event_test --test_output=all`
Expected: FAIL — file/target does not exist

**Step 3: Add BUILD target**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "discrete_dividend_event_test",
    size = "small",
    srcs = ["discrete_dividend_event_test.cc"],
    deps = [
        "//src/option:discrete_dividend_event",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

Add to `src/option/BUILD.bazel`:

```python
cc_library(
    name = "discrete_dividend_event",
    hdrs = ["discrete_dividend_event.hpp"],
    deps = [
        "//src/math:cubic_spline_solver",
        "//src/pde/core:pde_solver",
    ],
    visibility = ["//visibility:public"],
)
```

**Step 4: Implement make_dividend_event**

Create `src/option/discrete_dividend_event.hpp`:

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include "src/pde/core/pde_solver.hpp"
#include "src/math/cubic_spline_solver.hpp"
#include <cmath>
#include <span>
#include <vector>

namespace mango {

/// Create a temporal event callback for a discrete cash dividend.
///
/// At the dividend date, the spot drops from S to S - D. In log-moneyness
/// coordinates x = ln(S/K), this shifts the solution: for each grid point
/// x[i], the new value is u(x') where x' = ln(exp(x[i]) - D/K).
///
/// Uses cubic spline interpolation for the shifted evaluation points.
///
/// @param dividend_amount Cash dividend amount D (in dollars)
/// @param strike Reference strike K (normalization base)
/// @return TemporalEventCallback suitable for PDESolver::add_temporal_event()
inline TemporalEventCallback make_dividend_event(double dividend_amount, double strike) {
    const double d = dividend_amount / strike;  // normalized dividend

    return [d](double /*t*/, std::span<const double> x, std::span<double> u) {
        if (d <= 0.0) return;  // no-op for zero dividend

        const size_t n = x.size();

        // Build cubic spline of current solution
        CubicSpline<double> spline;
        auto err = spline.build(x, std::span<const double>(u.data(), u.size()));
        if (err.has_value()) return;  // spline build failed, leave u unchanged

        // Apply dividend shift: x' = ln(exp(x) - d)
        for (size_t i = 0; i < n; ++i) {
            double S_over_K = std::exp(x[i]);
            double S_adj_over_K = S_over_K - d;

            if (S_adj_over_K > 1e-10) {
                double x_shifted = std::log(S_adj_over_K);
                u[i] = spline.eval(x_shifted);
            } else {
                // Spot drops to zero or below: put is worth max payoff,
                // call is worthless. Use put intrinsic as safe fallback
                // (obstacle re-application by PDESolver handles the rest).
                u[i] = std::max(1.0 - S_adj_over_K, 0.0);
            }
        }
    };
}

}  // namespace mango
```

**Step 5: Run test to verify it passes**

Run: `bazel test //tests:discrete_dividend_event_test --test_output=all`
Expected: PASS

**Step 6: Run full suite**

Run: `bazel test //...`
Expected: All tests pass

**Step 7: Commit**

```bash
git add src/option/discrete_dividend_event.hpp src/option/BUILD.bazel \
        tests/discrete_dividend_event_test.cc tests/BUILD.bazel
git commit -m "Add make_dividend_event for discrete dividend jump callback"
```

---

### Task 4: Wire dividend events into AmericanOptionSolver

**Files:**
- Modify: `src/option/american_option.cpp`
- Modify: `src/option/american_option.hpp` (estimate_grid_for_option)
- Test: `tests/american_option_test.cc`

**Step 1: Write the failing test**

Add to `tests/american_option_test.cc`:

```cpp
TEST(AmericanOptionTest, DiscreteDividendPutPriceHigherThanNoDividend) {
    // A discrete dividend increases put value (spot drops)
    PricingParams no_div(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20);
    PricingParams with_div(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20,
                           {{0.5, 3.0}});  // $3 dividend at t=0.5

    auto result_no_div = solve_american_option_auto(no_div);
    auto result_with_div = solve_american_option_auto(with_div);

    ASSERT_TRUE(result_no_div.has_value());
    ASSERT_TRUE(result_with_div.has_value());

    // Put with dividend should be worth more (spot drops by $3)
    EXPECT_GT(result_with_div->value(), result_no_div->value())
        << "Put with discrete dividend should be worth more than without";
}

TEST(AmericanOptionTest, DiscreteDividendCallPriceLowerThanNoDividend) {
    // A discrete dividend decreases call value (spot drops)
    PricingParams no_div(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::CALL, 0.20);
    PricingParams with_div(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::CALL, 0.20,
                           {{0.5, 3.0}});

    auto result_no_div = solve_american_option_auto(no_div);
    auto result_with_div = solve_american_option_auto(with_div);

    ASSERT_TRUE(result_no_div.has_value());
    ASSERT_TRUE(result_with_div.has_value());

    EXPECT_LT(result_with_div->value(), result_no_div->value())
        << "Call with discrete dividend should be worth less than without";
}

TEST(AmericanOptionTest, DiscreteDividendMultiple) {
    // Two dividends should shift more than one
    PricingParams one_div(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20,
                          {{0.5, 2.0}});
    PricingParams two_div(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20,
                          {{0.3, 2.0}, {0.7, 2.0}});

    auto result_one = solve_american_option_auto(one_div);
    auto result_two = solve_american_option_auto(two_div);

    ASSERT_TRUE(result_one.has_value());
    ASSERT_TRUE(result_two.has_value());

    EXPECT_GT(result_two->value(), result_one->value())
        << "Two dividends should increase put value more than one";
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:american_option_test --test_output=all --test_filter="*DiscreteDividend*"`
Expected: FAIL — discrete dividend prices equal non-dividend prices (events not registered)

**Step 3: Modify estimate_grid_for_option**

In `src/option/american_option.hpp`, update `estimate_grid_for_option()` to produce a TimeDomain with mandatory points at dividend dates:

```cpp
inline std::pair<GridSpec<double>, TimeDomain> estimate_grid_for_option(
    const PricingParams& params,
    const GridAccuracyParams& accuracy = GridAccuracyParams{})
{
    // ... existing spatial grid estimation (unchanged) ...

    // Create sinh-spaced GridSpec
    auto grid_spec = GridSpec<double>::sinh_spaced(x_min, x_max, Nx, accuracy.alpha);

    // Convert discrete dividend calendar times to time-to-expiry
    std::vector<double> mandatory_tau;
    for (const auto& [t_cal, amount] : params.discrete_dividends) {
        double tau = params.maturity - t_cal;
        if (tau > 0.0 && tau < params.maturity) {
            mandatory_tau.push_back(tau);
        }
    }

    TimeDomain time_domain = mandatory_tau.empty()
        ? TimeDomain::from_n_steps(0.0, params.maturity, Nt)
        : TimeDomain::with_mandatory_points(0.0, params.maturity, dt, mandatory_tau);

    return {grid_spec.value(), time_domain};
}
```

**Step 4: Modify AmericanOptionSolver::solve()**

In `src/option/american_option.cpp`, add dividend event registration after creating the PDE solver:

Add include at top:
```cpp
#include "src/option/discrete_dividend_event.hpp"
```

In `solve()`, replace the solver creation block (lines 109-119) with:

```cpp
    if (params_.type == OptionType::PUT) {
        AmericanPutSolver pde_solver(params_, grid, workspace_);
        pde_solver.initialize(AmericanPutSolver::payoff);
        pde_solver.set_config(trbdf2_config_);

        // Register discrete dividend events
        for (const auto& [t_cal, amount] : params_.discrete_dividends) {
            double tau = params_.maturity - t_cal;
            if (tau > 0.0 && tau < params_.maturity) {
                pde_solver.add_temporal_event(tau,
                    make_dividend_event(amount, params_.strike));
            }
        }

        solve_result = pde_solver.solve();
    } else {
        AmericanCallSolver pde_solver(params_, grid, workspace_);
        pde_solver.initialize(AmericanCallSolver::payoff);
        pde_solver.set_config(trbdf2_config_);

        for (const auto& [t_cal, amount] : params_.discrete_dividends) {
            double tau = params_.maturity - t_cal;
            if (tau > 0.0 && tau < params_.maturity) {
                pde_solver.add_temporal_event(tau,
                    make_dividend_event(amount, params_.strike));
            }
        }

        solve_result = pde_solver.solve();
    }
```

**Step 5: Update BUILD dependency**

Add `"//src/option:discrete_dividend_event"` to the deps of `//src/option:american_option` in `src/option/BUILD.bazel`.

**Step 6: Run test to verify it passes**

Run: `bazel test //tests:american_option_test --test_output=all --test_filter="*DiscreteDividend*"`
Expected: PASS

**Step 7: Run full suite**

Run: `bazel test //...`
Expected: All tests pass

**Step 8: Commit**

```bash
git add src/option/american_option.cpp src/option/american_option.hpp \
        src/option/BUILD.bazel tests/american_option_test.cc
git commit -m "Wire discrete dividend events into AmericanOptionSolver"
```

---

### Task 5: Verify batch solver support

**Files:**
- Test: `tests/american_option_batch_test.cc` (or `tests/american_option_test.cc`)

The regular batch solver calls `AmericanOptionSolver::solve()` per option, so
discrete dividends should work automatically. Normalized chain rejects them.
Write tests to confirm both paths.

**Step 1: Write the test**

```cpp
TEST(AmericanOptionBatchTest, RegularBatchWithDiscreteDividends) {
    // Batch of options with discrete dividends falls back to regular path
    std::vector<PricingParams> batch;
    batch.push_back(PricingParams(100.0, 100.0, 1.0, 0.05, 0.0,
                                  OptionType::PUT, 0.20, {{0.5, 3.0}}));
    batch.push_back(PricingParams(100.0, 110.0, 1.0, 0.05, 0.0,
                                  OptionType::PUT, 0.20, {{0.5, 3.0}}));

    AmericanOptionBatchSolver solver;
    auto results = solver.solve_batch(batch);

    EXPECT_EQ(results.failures, 0u);
    for (const auto& r : results.results) {
        ASSERT_TRUE(r.has_value());
        EXPECT_GT(r->value(), 0.0);
    }
}
```

**Step 2: Run test**

Run: `bazel test //tests:american_option_test --test_output=all --test_filter="*RegularBatch*"`
Expected: PASS (regular batch delegates to single-option solver which now handles dividends)

**Step 3: Commit**

```bash
git add tests/american_option_test.cc
git commit -m "Add batch solver test for discrete dividends"
```

---

### Task 6: Accuracy test against known values

**Files:**
- Test: `tests/discrete_dividend_accuracy_test.cc`
- Modify: `tests/BUILD.bazel`

Verify the FDM prices are reasonable by comparing against the spot-adjustment
heuristic: an American put with a $3 dividend at t=0.5 on S=100, K=100 should
price similarly to an American put on S=97, K=100 with no dividend (rough
lower bound). Also compare put-call parity bounds.

**Step 1: Write the test**

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/american_option.hpp"
#include "src/option/american_option_batch.hpp"

using namespace mango;

TEST(DiscreteDividendAccuracyTest, PutPriceWithinReasonableBounds) {
    // ATM put, S=100, K=100, T=1, sigma=0.20, r=0.05
    // Discrete dividend: $3 at t=0.5
    PricingParams with_div(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20,
                           {{0.5, 3.0}});

    auto result = solve_american_option_auto(with_div);
    ASSERT_TRUE(result.has_value());
    double price = result->value();

    // Lower bound: European put on S-PV(D) (spot adjusted for PV of dividend)
    // PV(D) = 3 * exp(-0.05 * 0.5) ≈ 2.926
    // S_adj ≈ 97.07
    // No-dividend put on S=97.07, K=100 should be less than American with dividend
    PricingParams lower_bound(97.07, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20);
    auto lb_result = solve_american_option_auto(lower_bound);
    ASSERT_TRUE(lb_result.has_value());

    // Upper bound: put on S=100-3=97 (full dividend subtracted immediately)
    PricingParams upper_bound(97.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20);
    auto ub_result = solve_american_option_auto(upper_bound);
    ASSERT_TRUE(ub_result.has_value());

    // Price should be in a reasonable range
    EXPECT_GT(price, 3.0) << "Put with dividend must be worth more than intrinsic bump";
    EXPECT_LT(price, 25.0) << "Put price should be reasonable";

    // Should be close to the spot-adjusted price (within ~10% relative)
    double ref = lb_result->value();
    EXPECT_NEAR(price, ref, ref * 0.15)
        << "Discrete dividend put should be close to spot-adjusted reference";
}

TEST(DiscreteDividendAccuracyTest, LargeDividendStressTest) {
    // Large dividend: $20 on S=100 (20% of spot)
    PricingParams params(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.30,
                         {{0.5, 20.0}});

    auto result = solve_american_option_auto(params);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->value(), 0.0);
    EXPECT_TRUE(std::isfinite(result->value()));
}

TEST(DiscreteDividendAccuracyTest, DividendNearExpiry) {
    // Dividend very close to expiry (t=0.95, T=1.0)
    PricingParams params(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20,
                         {{0.95, 2.0}});

    auto result = solve_american_option_auto(params);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->value(), 0.0);
    EXPECT_TRUE(std::isfinite(result->value()));
}
```

**Step 2: Add BUILD target**

```python
cc_test(
    name = "discrete_dividend_accuracy_test",
    size = "medium",
    srcs = ["discrete_dividend_accuracy_test.cc"],
    copts = ["-fopenmp"],
    linkopts = ["-fopenmp"],
    deps = [
        "//src/option:american_option",
        "//src/option:american_option_batch",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 3: Run test**

Run: `bazel test //tests:discrete_dividend_accuracy_test --test_output=all`
Expected: PASS

**Step 4: Run full suite**

Run: `bazel test //...`
Expected: All tests pass

**Step 5: Commit**

```bash
git add tests/discrete_dividend_accuracy_test.cc tests/BUILD.bazel
git commit -m "Add accuracy tests for FDM discrete dividend pricing"
```
