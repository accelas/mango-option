<!-- SPDX-License-Identifier: MIT -->
# American Option Pricing - C++20 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement American option pricing with discrete dividends, temporal events, and obstacle constraints for early exercise.

**Architecture:** Four-layer design: (1) Temporal event system for dividends, (2) Obstacle condition interface via std::function, (3) LogMoneynessBlackScholesOperator for PDE, (4) AmericanOptionSolver high-level API with boundary auto-detection. Uses existing PDESolver with extensions for obstacles and events.

**Tech Stack:** C++20, GoogleTest, USDT, Bazel, TR-BDF2 PDE solver

**Prerequisites:**
- `PDESolver` template exists in `src/cpp/pde_solver.hpp`
- `WorkspaceStorage` exists in `src/cpp/workspace.hpp`
- `NewtonWorkspace` exists for Newton iteration
- Verify these files exist before starting

---

## Task 1: Extend WorkspaceStorage for Obstacle Buffer

**Files:**
- Modify: `src/cpp/workspace.hpp` (add psi buffer)

**Step 1: Check current buffer allocation**

```bash
grep "buffer_.*n\*" src/cpp/workspace.hpp
```

Expected: Find current allocation (likely 5*n or similar)

**Step 2: Write test for psi buffer access**

Add to `tests/workspace_test.cc` (create if needed):

```cpp
#include <gtest/gtest.h>
#include "src/cpp/workspace.hpp"

namespace mango {
namespace {

TEST(WorkspaceStorageTest, PsiBufferAvailable) {
    const size_t n = 100;
    WorkspaceStorage ws(n);

    auto psi = ws.psi_buffer();
    EXPECT_EQ(psi.size(), n);

    // Verify writable
    for (size_t i = 0; i < n; ++i) {
        psi[i] = static_cast<double>(i);
    }

    // Verify no overlap with other buffers
    auto u_current = ws.u_current();
    u_current[0] = 999.0;
    EXPECT_NE(psi[0], 999.0);
}

}  // namespace
}  // namespace mango
```

**Step 3: Add test target if needed**

If `tests/BUILD.bazel` doesn't have workspace_test:

```python
cc_test(
    name = "workspace_test",
    size = "small",
    srcs = ["workspace_test.cc"],
    deps = [
        "//src:pde_solver",
        "@googletest//:gtest_main",
    ],
)
```

**Step 4: Run test to verify it fails**

```bash
bazel test //tests:workspace_test --test_output=errors
```

Expected: FAIL with "no member named 'psi_buffer'"

**Step 5: Update WorkspaceStorage buffer allocation**

In `src/cpp/workspace.hpp`, change buffer size from `5*n` to `6*n`:

```cpp
explicit WorkspaceStorage(size_t n)
    : n_(n)
    , buffer_(6 * n)  // Updated from 5*n for obstacle support
{
    size_t offset = 0;

    // Existing buffers (order must match!)
    u_current_ = std::span{buffer_.data() + offset, n}; offset += n;
    u_next_ = std::span{buffer_.data() + offset, n}; offset += n;
    u_stage_ = std::span{buffer_.data() + offset, n}; offset += n;
    rhs_ = std::span{buffer_.data() + offset, n}; offset += n;
    Lu_ = std::span{buffer_.data() + offset, n}; offset += n;

    // New: obstacle buffer
    psi_ = std::span{buffer_.data() + offset, n}; offset += n;

    assert(offset == 6 * n);
}

std::span<double> psi_buffer() { return psi_; }

private:
    std::span<double> psi_;  // Add member
```

**Step 6: Run test to verify it passes**

```bash
bazel test //tests:workspace_test --test_output=all
```

Expected: PASS

**Step 7: Commit**

```bash
git add src/cpp/workspace.hpp tests/workspace_test.cc tests/BUILD.bazel
git commit -m "feat(american): add obstacle buffer to WorkspaceStorage"
```

---

## Task 2: Add Temporal Event System to PDESolver

**Files:**
- Modify: `src/cpp/pde_solver.hpp` (add event system)
- Create: `tests/temporal_event_test.cc`

**Step 1: Write test for temporal event**

Create `tests/temporal_event_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "src/cpp/pde_solver.hpp"
#include "src/cpp/boundary_conditions.hpp"
#include "src/cpp/spatial_operators.hpp"
#include <cmath>

namespace mango {
namespace {

TEST(TemporalEventTest, EventAppliedAfterStep) {
    // Simple 1D grid
    GridSpec grid{
        .n = 51,
        .x_min = -1.0,
        .x_max = 1.0
    };

    TimeSpec time{
        .t_start = 0.0,
        .t_end = 1.0,
        .dt = 0.1,
        .n_steps = 10
    };

    // Zero spatial operator (no PDE evolution)
    auto spatial_op = [](auto x, auto t, auto u, auto Lu) {
        std::fill(Lu.begin(), Lu.end(), 0.0);
    };

    DirichletBC left_bc{0.0};
    DirichletBC right_bc{0.0};

    TRBDF2Config trbdf2_config{};
    RootFindingConfig root_config{};

    PDESolver solver(grid, time, trbdf2_config, root_config,
                     left_bc, right_bc, spatial_op);

    // Initial condition: u = 1 everywhere
    solver.initialize([](auto x) { return 1.0; });

    // Add event at t=0.5 that doubles all values
    bool event_fired = false;
    solver.add_temporal_event(0.5, [&](double t, auto x, auto u) {
        event_fired = true;
        for (size_t i = 0; i < u.size(); ++i) {
            u[i] *= 2.0;
        }
    });

    ASSERT_TRUE(solver.solve());

    // Verify event was applied
    EXPECT_TRUE(event_fired);

    auto solution = solver.solution();
    // After event at t=0.5, all values should be 2.0
    EXPECT_NEAR(solution[25], 2.0, 1e-10);
}

}  // namespace
}  // namespace mango
```

**Step 2: Add test target**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "temporal_event_test",
    size = "small",
    srcs = ["temporal_event_test.cc"],
    deps = [
        "//src:pde_solver",
        "@googletest//:gtest_main",
    ],
)
```

**Step 3: Run test to verify it fails**

```bash
bazel test //tests:temporal_event_test --test_output=errors
```

Expected: FAIL with "no member named 'add_temporal_event'"

**Step 4: Add event types to pde_solver.hpp**

Add before PDESolver class:

```cpp
// Temporal event callback signature
using TemporalEventCallback = std::function<void(double t,
                                                  std::span<const double> x,
                                                  std::span<double> u)>;

// Temporal event definition
struct TemporalEvent {
    double time;
    TemporalEventCallback callback;

    auto operator<=>(const TemporalEvent& other) const {
        return time <=> other.time;
    }
};
```

**Step 5: Add event system to PDESolver class**

Add to PDESolver class (in public section):

```cpp
void add_temporal_event(double time, TemporalEventCallback callback) {
    events_.push_back({time, std::move(callback)});
    std::ranges::sort(events_);
}
```

Add to PDESolver class (in private section):

```cpp
std::vector<TemporalEvent> events_;
size_t next_event_idx_ = 0;

void process_temporal_events(double t_old, double t_new, size_t step);
```

**Step 6: Implement process_temporal_events()**

Add after PDESolver class methods:

```cpp
template<typename BoundaryL, typename BoundaryR, typename SpatialOp>
void PDESolver<BoundaryL, BoundaryR, SpatialOp>::process_temporal_events(
    double t_old, double t_new, size_t step) {

    while (next_event_idx_ < events_.size()) {
        const auto& event = events_[next_event_idx_];

        if (event.time <= t_old) {
            next_event_idx_++;
            continue;
        }

        if (event.time > t_new) {
            break;
        }

        // Event is in (t_old, t_new] - apply it
        event.callback(event.time, grid_.x(), workspace_.u_current());
        next_event_idx_++;

        MANGO_TRACE_ALGO_PROGRESS(MODULE_PDE_SOLVER, step,
                                   "Processed temporal event");
    }
}
```

**Step 7: Update solve() to call process_temporal_events()**

Modify PDESolver::solve() to add event processing after each timestep:

```cpp
bool solve() {
    double t = time_.t_start();
    const double dt = time_.dt();

    for (size_t step = 0; step < time_.n_steps(); ++step) {
        double t_next = t + dt;

        // TR-BDF2 step
        std::copy(workspace_.u_current().begin(),
                  workspace_.u_current().end(),
                  workspace_.u_old().begin());

        if (!solve_stage1(t, t + config_.gamma * dt, dt)) return false;
        if (!solve_stage2(t, t_next, dt)) return false;

        std::copy(workspace_.u_next().begin(),
                  workspace_.u_next().end(),
                  workspace_.u_current().begin());

        // Process events AFTER completing the step
        process_temporal_events(t, t_next, step);

        t = t_next;
    }

    return true;
}
```

**Step 8: Run test to verify it passes**

```bash
bazel test //tests:temporal_event_test --test_output=all
```

Expected: PASS

**Step 9: Commit**

```bash
git add src/cpp/pde_solver.hpp tests/temporal_event_test.cc tests/BUILD.bazel
git commit -m "feat(american): add temporal event system to PDESolver"
```

---

## Task 3: Add Obstacle Condition Support to PDESolver

**Files:**
- Modify: `src/cpp/pde_solver.hpp` (add obstacle interface)
- Create: `tests/obstacle_test.cc`

**Step 1: Write test for obstacle projection**

Create `tests/obstacle_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "src/cpp/pde_solver.hpp"
#include "src/cpp/boundary_conditions.hpp"
#include <algorithm>

namespace mango {
namespace {

TEST(ObstacleTest, ProjectionEnforced) {
    GridSpec grid{.n = 51, .x_min = -1.0, .x_max = 1.0};
    TimeSpec time{.t_start = 0.0, .t_end = 0.1, .dt = 0.01, .n_steps = 10};

    auto spatial_op = [](auto x, auto t, auto u, auto Lu) {
        std::fill(Lu.begin(), Lu.end(), 0.0);
    };

    // Obstacle: ψ(x) = 0.5 everywhere
    ObstacleCallback obstacle = [](auto x, auto t, auto psi) {
        std::fill(psi.begin(), psi.end(), 0.5);
    };

    DirichletBC left_bc{0.0};
    DirichletBC right_bc{0.0};

    TRBDF2Config trbdf2_config{};
    RootFindingConfig root_config{};

    PDESolver solver(grid, time, trbdf2_config, root_config,
                     left_bc, right_bc, spatial_op, obstacle);

    // Initial condition: u = 0.3 (below obstacle)
    solver.initialize([](auto x) { return 0.3; });

    ASSERT_TRUE(solver.solve());

    // Solution should be projected to obstacle
    auto solution = solver.solution();
    for (size_t i = 1; i < solution.size() - 1; ++i) {
        EXPECT_GE(solution[i], 0.5 - 1e-10);
    }
}

}  // namespace
}  // namespace mango
```

**Step 2: Add test target**

```python
cc_test(
    name = "obstacle_test",
    size = "small",
    srcs = ["obstacle_test.cc"],
    deps = [
        "//src:pde_solver",
        "@googletest//:gtest_main",
    ],
)
```

**Step 3: Run test to verify it fails**

```bash
bazel test //tests:obstacle_test --test_output=errors
```

Expected: FAIL (obstacle not defined)

**Step 4: Add ObstacleCallback type**

Add to `src/cpp/pde_solver.hpp` before PDESolver:

```cpp
// Obstacle callback signature
using ObstacleCallback = std::function<void(std::span<const double> x,
                                            double t,
                                            std::span<double> psi)>;
```

**Step 5: Update PDESolver constructor signature**

Modify PDESolver constructor to accept optional obstacle:

```cpp
PDESolver(const GridSpec& grid,
          const TimeSpec& time,
          const TRBDF2Config& config,
          const RootFindingConfig& root_config,
          BoundaryL&& left_bc,
          BoundaryR&& right_bc,
          SpatialOp&& spatial_op,
          std::optional<ObstacleCallback> obstacle = std::nullopt)
    : grid_(grid)
    , time_(time)
    , config_(config)
    , left_bc_(std::forward<BoundaryL>(left_bc))
    , right_bc_(std::forward<BoundaryR>(right_bc))
    , spatial_op_(std::forward<SpatialOp>(spatial_op))
    , obstacle_(std::move(obstacle))
    , workspace_(grid.n)
    , newton_ws_(/* ... */)
{ }
```

**Step 6: Add obstacle member and apply method**

Add to PDESolver private section:

```cpp
std::optional<ObstacleCallback> obstacle_;

void apply_obstacle(std::span<double> u, double t);
```

**Step 7: Implement apply_obstacle()**

```cpp
template<typename BoundaryL, typename BoundaryR, typename SpatialOp>
void PDESolver<BoundaryL, BoundaryR, SpatialOp>::apply_obstacle(
    std::span<double> u, double t) {

    if (!obstacle_) return;

    auto psi = workspace_.psi_buffer();

    if (workspace_.cache_config().enabled) {
        // Cache-blocked version
        for (size_t block = 0; block < workspace_.cache_config().n_blocks; ++block) {
            auto [start, end] = workspace_.get_block_interior_range(block);

            auto x_block = grid_.x().subspan(start, end - start);
            auto psi_block = psi.subspan(start, end - start);
            auto u_block = u.subspan(start, end - start);

            (*obstacle_)(x_block, t, psi_block);

            #pragma omp simd
            for (size_t i = 0; i < u_block.size(); ++i) {
                u_block[i] = std::max(u_block[i], psi_block[i]);
            }
        }
    } else {
        // Full-array version
        (*obstacle_)(grid_.x(), t, psi);

        #pragma omp simd
        for (size_t i = 0; i < u.size(); ++i) {
            u[i] = std::max(u[i], psi[i]);
        }
    }
}
```

**Step 8: Call apply_obstacle in Newton iteration**

Modify `solve_newton_stage()` to apply obstacle after update:

```cpp
// Update: u_new = u_old + Δu
newton_ws_.apply_update(u_stage);

// PROJECT ONTO FEASIBLE SET
apply_obstacle(u_stage, t_stage);

// Apply boundary conditions
apply_boundary_conditions(u_stage, t_stage);

// Check convergence AFTER projection
if (newton_ws_.check_convergence()) {
    return true;
}
```

**Step 9: Run test to verify it passes**

```bash
bazel test //tests:obstacle_test --test_output=all
```

Expected: PASS

**Step 10: Commit**

```bash
git add src/cpp/pde_solver.hpp tests/obstacle_test.cc tests/BUILD.bazel
git commit -m "feat(american): add obstacle condition support"
```

---

## Task 4: Implement LogMoneynessBlackScholesOperator

**Files:**
- Create: `src/cpp/american_option.hpp`
- Create: `tests/american_option_test.cc`

**Step 1: Write test for operator evaluation**

Create `tests/american_option_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "src/cpp/american_option.hpp"
#include <cmath>

namespace mango {
namespace {

TEST(LogMoneynessBlackScholesOperatorTest, OperatorEvaluation) {
    double sigma = 0.20;
    double rate = 0.05;

    LogMoneynessBlackScholesOperator op(sigma, rate);

    // Grid in log-moneyness space
    std::vector<double> x = {-0.1, 0.0, 0.1};
    std::vector<double> u = {5.0, 10.0, 5.0};  // Parabola
    std::vector<double> Lu(3);

    op(x, 0.0, u, Lu);

    // L(u) = (σ²/2)∂²u/∂x² + (r - σ²/2)∂u/∂x - ru
    // For parabola u = 10 - 50*(x-0)^2:
    // ∂u/∂x = -100*x, at x=0: ∂u/∂x = 0
    // ∂²u/∂x² = -100
    // L(u) at x=0: (0.04/2)*(-100) + 0 - 0.05*10 = -2 - 0.5 = -2.5

    EXPECT_NEAR(Lu[1], -2.5, 0.5);  // Approximate due to finite differences
}

}  // namespace
}  // namespace mango
```

**Step 2: Add test target**

```python
cc_test(
    name = "american_option_test",
    size = "small",
    srcs = ["american_option_test.cc"],
    deps = [
        "//src:pde_solver",
        "@googletest//:gtest_main",
    ],
)
```

**Step 3: Run test to verify it fails**

```bash
bazel test //tests:american_option_test --test_output=errors
```

Expected: FAIL with "american_option.hpp: No such file"

**Step 4: Create american_option.hpp with OptionType enum**

Create `src/cpp/american_option.hpp`:

```cpp
#pragma once

#include <span>
#include <vector>
#include <cmath>
#include <algorithm>

namespace mango {

// Option type enumeration
enum class OptionType {
    CALL,
    PUT
};

// Forward declarations
class LogMoneynessBlackScholesOperator;
class AmericanPutObstacle;
class AmericanCallObstacle;
class DividendJump;

}  // namespace mango
```

**Step 5: Implement LogMoneynessBlackScholesOperator**

Add to `src/cpp/american_option.hpp`:

```cpp
// Black-Scholes operator in log-moneyness coordinates
// L(V) = (σ²/2)∂²V/∂x² + (r - σ²/2)∂V/∂x - rV
class LogMoneynessBlackScholesOperator {
public:
    LogMoneynessBlackScholesOperator(double volatility, double rate)
        : sigma_(volatility)
        , r_(rate)
        , sigma2_half_(0.5 * volatility * volatility)
        , drift_(rate - sigma2_half_)
    { }

    void operator()(std::span<const double> x,
                   double t,
                   std::span<const double> u,
                   std::span<double> Lu) const {
        const size_t n = x.size();
        const double dx = x[1] - x[0];  // Uniform grid
        const double dx2_inv = 1.0 / (dx * dx);
        const double dx_inv_half = 0.5 / dx;

        // Boundaries: zero (Dirichlet will override)
        Lu[0] = 0.0;
        Lu[n-1] = 0.0;

        // Interior points: centered finite differences
        #pragma omp simd
        for (size_t i = 1; i < n - 1; ++i) {
            // Second derivative: ∂²u/∂x²
            double d2u_dx2 = (u[i-1] - 2.0*u[i] + u[i+1]) * dx2_inv;

            // First derivative: ∂u/∂x (centered)
            double du_dx = (u[i+1] - u[i-1]) * dx_inv_half;

            // L(u) = (σ²/2)∂²u/∂x² + (r - σ²/2)∂u/∂x - ru
            Lu[i] = sigma2_half_ * d2u_dx2 + drift_ * du_dx - r_ * u[i];
        }
    }

    // Block-aware version for cache-blocking
    void apply_block(std::span<const double> x_block,
                     double t,
                     std::span<const double> u_extended,
                     std::span<double> Lu_block,
                     size_t global_start) const {
        // For cache-blocking: u_extended includes overlap
        // Apply operator to block interior
        const size_t n_block = x_block.size();
        const double dx = x_block[1] - x_block[0];
        const double dx2_inv = 1.0 / (dx * dx);
        const double dx_inv_half = 0.5 / dx;

        #pragma omp simd
        for (size_t i = 0; i < n_block; ++i) {
            // u_extended[0] is u[global_start-1]
            // u_extended[i+1] is u[global_start+i]
            double d2u_dx2 = (u_extended[i] - 2.0*u_extended[i+1] + u_extended[i+2]) * dx2_inv;
            double du_dx = (u_extended[i+2] - u_extended[i]) * dx_inv_half;

            Lu_block[i] = sigma2_half_ * d2u_dx2 + drift_ * du_dx - r_ * u_extended[i+1];
        }
    }

private:
    double sigma_;
    double r_;
    double sigma2_half_;
    double drift_;
};
```

**Step 6: Run test to verify it passes**

```bash
bazel test //tests:american_option_test --test_output=all
```

Expected: PASS (approximately, finite difference errors)

**Step 7: Commit**

```bash
git add src/cpp/american_option.hpp tests/american_option_test.cc tests/BUILD.bazel
git commit -m "feat(american): implement LogMoneynessBlackScholesOperator"
```

---

## Task 5: Implement Obstacle Functions (Put and Call)

**Files:**
- Modify: `src/cpp/american_option.hpp` (add obstacles)

**Step 1: Write test for American put obstacle**

Add to `tests/american_option_test.cc`:

```cpp
TEST(AmericanPutObstacleTest, PayoffCorrect) {
    double strike = 100.0;
    AmericanPutObstacle obstacle(strike);

    // Grid in log-moneyness: x = ln(S/K)
    std::vector<double> x = {
        std::log(80.0/100.0),   // OTM: S=80
        std::log(100.0/100.0),  // ATM: S=100
        std::log(120.0/100.0)   // ITM: S=120
    };

    std::vector<double> psi(3);
    obstacle(x, 0.0, psi);

    // Put payoff: max(K - S, 0)
    EXPECT_NEAR(psi[0], 20.0, 1e-10);  // max(100-80, 0) = 20
    EXPECT_NEAR(psi[1], 0.0, 1e-10);   // max(100-100, 0) = 0
    EXPECT_NEAR(psi[2], 0.0, 1e-10);   // max(100-120, 0) = 0
}

TEST(AmericanCallObstacleTest, PayoffCorrect) {
    double strike = 100.0;
    AmericanCallObstacle obstacle(strike);

    std::vector<double> x = {
        std::log(80.0/100.0),
        std::log(100.0/100.0),
        std::log(120.0/100.0)
    };

    std::vector<double> psi(3);
    obstacle(x, 0.0, psi);

    // Call payoff: max(S - K, 0)
    EXPECT_NEAR(psi[0], 0.0, 1e-10);   // max(80-100, 0) = 0
    EXPECT_NEAR(psi[1], 0.0, 1e-10);   // max(100-100, 0) = 0
    EXPECT_NEAR(psi[2], 20.0, 1e-10);  // max(120-100, 0) = 20
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:american_option_test --test_filter="*Obstacle*" --test_output=errors
```

Expected: FAIL (AmericanPutObstacle not defined)

**Step 3: Implement AmericanPutObstacle**

Add to `src/cpp/american_option.hpp`:

```cpp
// American put obstacle: ψ(x,t) = max(K - S, 0)
class AmericanPutObstacle {
public:
    explicit AmericanPutObstacle(double strike)
        : strike_(strike)
    { }

    void operator()(std::span<const double> x, double t, std::span<double> psi) const {
        #pragma omp simd
        for (size_t i = 0; i < x.size(); ++i) {
            double S = strike_ * std::exp(x[i]);
            psi[i] = std::max(strike_ - S, 0.0);
        }
    }

private:
    double strike_;
};

// American call obstacle: ψ(x,t) = max(S - K, 0)
class AmericanCallObstacle {
public:
    explicit AmericanCallObstacle(double strike)
        : strike_(strike)
    { }

    void operator()(std::span<const double> x, double t, std::span<double> psi) const {
        #pragma omp simd
        for (size_t i = 0; i < x.size(); ++i) {
            double S = strike_ * std::exp(x[i]);
            psi[i] = std::max(S - strike_, 0.0);
        }
    }

private:
    double strike_;
};
```

**Step 4: Run test to verify it passes**

```bash
bazel test //tests:american_option_test --test_filter="*Obstacle*" --test_output=all
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/cpp/american_option.hpp tests/american_option_test.cc
git commit -m "feat(american): implement put/call obstacle functions"
```

---

## Task 6: Implement DividendJump Class

**Files:**
- Modify: `src/cpp/american_option.hpp` (add DividendJump)

**Step 1: Write test for dividend jump**

Add to `tests/american_option_test.cc`:

```cpp
TEST(DividendJumpTest, StockPriceJump) {
    double strike = 100.0;
    double dividend = 5.0;
    DividendJump div_jump(strike, dividend, OptionType::PUT);

    // Grid before dividend: S ∈ [80, 100, 120]
    std::vector<double> x_grid = {
        std::log(80.0/100.0),
        std::log(100.0/100.0),
        std::log(120.0/100.0)
    };

    // Option values before dividend
    std::vector<double> u = {20.0, 10.0, 5.0};

    div_jump(0.5, x_grid, u);

    // After dividend: S → S-D, so S ∈ [75, 95, 115]
    // u_new(x) should map from x_post = ln((S-D)/K)
    // At x=ln(80/100): S=80, S_post=75, x_post=ln(75/100)
    // We're checking that u got updated (exact values depend on interpolation)

    // Just verify u changed
    EXPECT_NE(u[0], 20.0);
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:american_option_test --test_filter="*DividendJump*" --test_output=errors
```

Expected: FAIL (DividendJump not defined)

**Step 3: Implement DividendJump**

Add to `src/cpp/american_option.hpp`:

```cpp
// Dividend jump: applies stock price drop S → S - D
class DividendJump {
public:
    DividendJump(double strike, double dividend_amount, OptionType option_type)
        : strike_(strike)
        , dividend_(dividend_amount)
        , option_type_(option_type)
    { }

    void operator()(double t, std::span<const double> x_grid, std::span<double> u) {
        const size_t n = x_grid.size();

        // Copy current solution
        std::vector<double> u_old(u.begin(), u.end());

        // For each grid point, compute post-dividend moneyness
        for (size_t i = 0; i < n; ++i) {
            double S = strike_ * std::exp(x_grid[i]);
            double S_post = S - dividend_;

            if (S_post <= 0.0) {
                // Stock went to zero - use intrinsic value
                u[i] = (option_type_ == OptionType::PUT) ? strike_ : 0.0;
                continue;
            }

            double x_post = std::log(S_post / strike_);

            // Find x_post in grid and interpolate
            if (x_post <= x_grid[0]) {
                u[i] = u_old[0];
            } else if (x_post >= x_grid[n-1]) {
                u[i] = u_old[n-1];
            } else {
                // Linear interpolation
                auto it = std::ranges::lower_bound(x_grid, x_post);
                size_t j = std::distance(x_grid.begin(), it);

                double x0 = x_grid[j-1];
                double x1 = x_grid[j];
                double weight = (x_post - x0) / (x1 - x0);

                u[i] = (1.0 - weight) * u_old[j-1] + weight * u_old[j];
            }
        }
    }

private:
    double strike_;
    double dividend_;
    OptionType option_type_;
};
```

**Step 4: Run test to verify it passes**

```bash
bazel test //tests:american_option_test --test_filter="*DividendJump*" --test_output=all
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/cpp/american_option.hpp tests/american_option_test.cc
git commit -m "feat(american): implement DividendJump for discrete dividends"
```

---

## Task 7: Implement AmericanOptionSolver High-Level API (Part 1: Structure)

**Files:**
- Modify: `src/cpp/american_option.hpp` (add solver structures)

**Step 1: Add parameter structures**

Add to `src/cpp/american_option.hpp`:

```cpp
// American option parameters
struct AmericanOptionParams {
    double strike;
    double volatility;
    double risk_free_rate;
    double time_to_maturity;
    OptionType option_type;

    // Discrete dividends
    std::span<const double> dividend_times;     // Time until dividend (years)
    std::span<const double> dividend_amounts;   // Dividend amount ($)

    // LIFETIME CONTRACT: Backing vectors must remain valid
};

// Grid specification in log-moneyness space
struct AmericanOptionGrid {
    double x_min = -0.7;   // ln(S/K) min
    double x_max = 0.7;    // ln(S/K) max
    size_t n_points = 101;
    double dt = 0.001;
    size_t n_steps = 1000;

    // Auto-scale grid based on current spot
    static AmericanOptionGrid auto_scale(double spot, double strike,
                                         double time_to_maturity);
};

// American option result
struct AmericanOptionResult {
    bool converged;
    std::string error;
};
```

**Step 2: Add AmericanOptionSolver class declaration**

```cpp
// High-level American option solver
class AmericanOptionSolver {
public:
    AmericanOptionSolver(const AmericanOptionParams& params,
                        const AmericanOptionGrid& grid);

    bool solve();

    // Get price at specific moneyness (m = S/K)
    double price_at(double moneyness) const;

    // Get solution array (in log-moneyness space)
    std::span<const double> solution() const;

    // Greeks computation
    double compute_delta_at(double x) const;
    double compute_gamma_at(double x) const;
    double compute_theta_at(double x) const;

private:
    AmericanOptionParams params_;
    AmericanOptionGrid grid_;

    // PDE solver (unique_ptr for pImpl)
    struct SolverImpl;
    std::unique_ptr<SolverImpl> solver_;

    std::vector<DividendJump> dividend_jumps_;

    void register_dividends();
};
```

**Step 3: Run build to verify structure compiles**

```bash
bazel build //src:pde_solver
```

Expected: SUCCESS (header-only so far)

**Step 4: Commit structure**

```bash
git add src/cpp/american_option.hpp
git commit -m "feat(american): add AmericanOptionSolver structure"
```

---

## Task 8: Implement AmericanOptionSolver (Part 2: Constructor and solve)

**Files:**
- Modify: `src/cpp/american_option.hpp` (implement solver)
- Add test

**Step 1: Write integration test**

Add to `tests/american_option_test.cc`:

```cpp
TEST(AmericanOptionSolverTest, SimpleAmericanPut) {
    AmericanOptionParams params{
        .strike = 100.0,
        .volatility = 0.20,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OptionType::PUT,
        .dividend_times = {},
        .dividend_amounts = {}
    };

    AmericanOptionGrid grid{
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 101,
        .dt = 0.001,
        .n_steps = 1000
    };

    AmericanOptionSolver solver(params, grid);
    ASSERT_TRUE(solver.solve());

    // ATM put should be worth something
    double price_atm = solver.price_at(1.0);
    EXPECT_GT(price_atm, 0.0);
    EXPECT_LT(price_atm, 100.0);

    // Deep ITM put should be close to intrinsic
    double price_itm = solver.price_at(0.8);  // S/K = 0.8
    double intrinsic = 20.0;  // K - S = 100 - 80
    EXPECT_GT(price_itm, intrinsic);  // Should have time value
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:american_option_test --test_filter="*Solver*" --test_output=errors
```

Expected: FAIL (not implemented)

**Step 3: Implement SolverImpl structure**

Add after AmericanOptionSolver class:

```cpp
// pImpl for type-erased PDESolver
struct AmericanOptionSolver::SolverImpl {
    using SolverType = PDESolver<DirichletBC, DirichletBC,
                                  LogMoneynessBlackScholesOperator>;

    std::unique_ptr<SolverType> solver;
    std::vector<double> x_grid;
};
```

**Step 4: Implement constructor**

```cpp
inline AmericanOptionSolver::AmericanOptionSolver(
    const AmericanOptionParams& params,
    const AmericanOptionGrid& grid)
    : params_(params)
    , grid_(grid)
    , solver_(std::make_unique<SolverImpl>())
{
    // Create uniform grid in log-moneyness space
    solver_->x_grid.resize(grid.n_points);
    double dx = (grid.x_max - grid.x_min) / (grid.n_points - 1);
    for (size_t i = 0; i < grid.n_points; ++i) {
        solver_->x_grid[i] = grid.x_min + i * dx;
    }

    GridSpec grid_spec{
        .n = grid.n_points,
        .x_min = grid.x_min,
        .x_max = grid.x_max
    };

    TimeSpec time_spec{
        .t_start = 0.0,
        .t_end = params.time_to_maturity,
        .dt = grid.dt,
        .n_steps = grid.n_steps
    };

    // Spatial operator
    LogMoneynessBlackScholesOperator spatial_op(params.volatility,
                                                params.risk_free_rate);

    // Boundary conditions (simple Dirichlet for now)
    DirichletBC left_bc(0.0);
    DirichletBC right_bc(params.strike);  // Far OTM put worth strike

    // Obstacle
    ObstacleCallback obstacle;
    if (params.option_type == OptionType::PUT) {
        AmericanPutObstacle put_obs(params.strike);
        obstacle = [put_obs](auto x, auto t, auto psi) {
            put_obs(x, t, psi);
        };
    } else {
        AmericanCallObstacle call_obs(params.strike);
        obstacle = [call_obs](auto x, auto t, auto psi) {
            call_obs(x, t, psi);
        };
    }

    TRBDF2Config trbdf2_config{};
    RootFindingConfig root_config{};

    solver_->solver = std::make_unique<SolverImpl::SolverType>(
        grid_spec, time_spec, trbdf2_config, root_config,
        left_bc, right_bc, spatial_op, obstacle
    );

    // Register dividends
    register_dividends();
}
```

**Step 5: Implement register_dividends**

```cpp
inline void AmericanOptionSolver::register_dividends() {
    struct DividendInfo {
        double solver_time;
        DividendJump jump;
    };

    std::vector<DividendInfo> dividend_info;
    dividend_info.reserve(params_.dividend_times.size());

    for (size_t i = 0; i < params_.dividend_times.size(); ++i) {
        double div_time = params_.dividend_times[i];
        double div_amount = params_.dividend_amounts[i];

        // Convert to solver time (backward time)
        double solver_time = params_.time_to_maturity - div_time;

        if (solver_time < 1e-10) continue;  // Skip dividends at maturity

        dividend_info.push_back({
            solver_time,
            DividendJump{params_.strike, div_amount, params_.option_type}
        });
    }

    dividend_jumps_.reserve(dividend_info.size());
    for (const auto& info : dividend_info) {
        dividend_jumps_.push_back(info.jump);
    }

    for (size_t i = 0; i < dividend_info.size(); ++i) {
        double solver_time = dividend_info[i].solver_time;

        TemporalEventCallback callback = [this, i](double t, auto x, auto u) {
            dividend_jumps_[i](t, x, u);
        };

        solver_->solver->add_temporal_event(solver_time, callback);
    }
}
```

**Step 6: Implement solve and accessors**

```cpp
inline bool AmericanOptionSolver::solve() {
    // Initial condition: payoff at maturity
    auto init = [this](double x) {
        double S = params_.strike * std::exp(x);
        if (params_.option_type == OptionType::PUT) {
            return std::max(params_.strike - S, 0.0);
        } else {
            return std::max(S - params_.strike, 0.0);
        }
    };

    solver_->solver->initialize(init);
    return solver_->solver->solve();
}

inline std::span<const double> AmericanOptionSolver::solution() const {
    return solver_->solver->solution();
}

inline double AmericanOptionSolver::price_at(double moneyness) const {
    double x = std::log(moneyness);
    const auto& x_grid = solver_->x_grid;
    const auto& u = solution();

    // Linear interpolation
    if (x <= x_grid[0]) return u[0];
    if (x >= x_grid.back()) return u.back();

    auto it = std::ranges::lower_bound(x_grid, x);
    size_t j = std::distance(x_grid.begin(), it);

    double x0 = x_grid[j-1];
    double x1 = x_grid[j];
    double weight = (x - x0) / (x1 - x0);

    return (1.0 - weight) * u[j-1] + weight * u[j];
}
```

**Step 7: Run test to verify it passes**

```bash
bazel test //tests:american_option_test --test_filter="*Solver*" --test_output=all
```

Expected: PASS

**Step 8: Commit**

```bash
git add src/cpp/american_option.hpp tests/american_option_test.cc
git commit -m "feat(american): implement AmericanOptionSolver core"
```

---

## Task 9: Implement Greeks Computation

**Files:**
- Modify: `src/cpp/american_option.hpp` (implement Greeks)

**Step 1: Write test for delta**

Add to `tests/american_option_test.cc`:

```cpp
TEST(AmericanOptionSolverTest, DeltaComputation) {
    AmericanOptionParams params{
        .strike = 100.0,
        .volatility = 0.20,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OptionType::PUT,
        .dividend_times = {},
        .dividend_amounts = {}
    };

    AmericanOptionGrid grid{
        .x_min = -0.7, .x_max = 0.7, .n_points = 101,
        .dt = 0.001, .n_steps = 1000
    };

    AmericanOptionSolver solver(params, grid);
    ASSERT_TRUE(solver.solve());

    // ATM put delta should be around -0.5
    double x_atm = 0.0;
    double delta = solver.compute_delta_at(x_atm);

    EXPECT_LT(delta, 0.0);  // Put delta negative
    EXPECT_GT(delta, -1.0);
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:american_option_test --test_filter="*Delta*" --test_output=errors
```

Expected: FAIL (returns 0.0)

**Step 3: Implement compute_delta_at**

```cpp
inline double AmericanOptionSolver::compute_delta_at(double x) const {
    const auto& solution = solver_->solver->solution();
    const auto& x_grid = solver_->x_grid;
    const size_t n = x_grid.size();

    // Find position in grid
    auto it = std::ranges::lower_bound(x_grid, x);

    if (it == x_grid.end()) {
        double dx = x_grid[n-1] - x_grid[n-2];
        double dV_dx = (solution[n-1] - solution[n-2]) / dx;
        double S = params_.strike * std::exp(x);
        return dV_dx / S;
    }

    size_t j = std::distance(x_grid.begin(), it);

    if (j == 0) {
        double dx = x_grid[1] - x_grid[0];
        double dV_dx = (solution[1] - solution[0]) / dx;
        double S = params_.strike * std::exp(x);
        return dV_dx / S;
    }

    if (j >= n - 1) {
        double dx = x_grid[n-1] - x_grid[n-2];
        double dV_dx = (solution[n-1] - solution[n-2]) / dx;
        double S = params_.strike * std::exp(x);
        return dV_dx / S;
    }

    // Centered difference
    double dx = x_grid[j+1] - x_grid[j-1];
    double dV_dx = (solution[j+1] - solution[j-1]) / dx;
    double S = params_.strike * std::exp(x);

    return dV_dx / S;
}
```

**Step 4: Implement stub Greeks**

```cpp
inline double AmericanOptionSolver::compute_gamma_at(double x) const {
    // Placeholder - similar to delta but second derivative
    return 0.0;
}

inline double AmericanOptionSolver::compute_theta_at(double x) const {
    // Placeholder - uses spatial operator
    return 0.0;
}
```

**Step 5: Run test to verify it passes**

```bash
bazel test //tests:american_option_test --test_filter="*Delta*" --test_output=all
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/cpp/american_option.hpp tests/american_option_test.cc
git commit -m "feat(american): implement delta computation (gamma/theta stubs)"
```

---

## Task 10: Add Dividend Test

**Files:**
- Modify: `tests/american_option_test.cc`

**Step 1: Add dividend test**

```cpp
TEST(AmericanOptionSolverTest, WithDividends) {
    std::vector<double> div_times = {0.5};
    std::vector<double> div_amounts = {2.0};

    AmericanOptionParams params{
        .strike = 100.0,
        .volatility = 0.20,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OptionType::PUT,
        .dividend_times = div_times,
        .dividend_amounts = div_amounts
    };

    AmericanOptionGrid grid{
        .x_min = -0.7, .x_max = 0.7, .n_points = 101,
        .dt = 0.001, .n_steps = 1000
    };

    AmericanOptionSolver solver(params, grid);
    ASSERT_TRUE(solver.solve());

    // With dividend, option should be more valuable
    double price = solver.price_at(1.0);
    EXPECT_GT(price, 0.0);
}
```

**Step 2: Run test**

```bash
bazel test //tests:american_option_test --test_filter="*Dividend*" --test_output=all
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/american_option_test.cc
git commit -m "test(american): add dividend handling test"
```

---

## Task 11: Run Full Test Suite

**Files:**
- None (verification)

**Step 1: Run all tests**

```bash
bazel test //tests:american_option_test --test_output=summary
```

Expected: All tests PASS

**Step 2: Build entire project**

```bash
bazel build //...
```

Expected: SUCCESS

---

## Completion Checklist

- [ ] Task 1: WorkspaceStorage extended
- [ ] Task 2: Temporal event system added
- [ ] Task 3: Obstacle condition support added
- [ ] Task 4: LogMoneynessBlackScholesOperator implemented
- [ ] Task 5: Put/Call obstacles implemented
- [ ] Task 6: DividendJump implemented
- [ ] Task 7-8: AmericanOptionSolver API complete
- [ ] Task 9: Greeks (delta) implemented
- [ ] Task 10: Dividend test added
- [ ] Task 11: All tests passing

**Estimated time:** 4-6 hours (11 focused tasks)

**Success criteria:**
- All tests pass
- AmericanOptionSolver prices American puts correctly
- Dividend jumps applied properly
- Obstacle constraints enforced
- Ready for IV calculation dependency
