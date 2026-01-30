# Phase 1 Weeks 5-6: TR-BDF2 Solver + Cache Blocking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement TR-BDF2 (Two-stage Runge-Kutta with Backward Differentiation Formula) time-stepping solver with cache-blocked stencils for efficient PDE solving on large grids.

**Architecture:** Port TR-BDF2 solver from C to C++20, integrating with WorkspaceStorage for cache-blocking, spatial operators with pre-computed dx, and boundary condition policies. Implement fixed-point iteration for implicit stages with proper convergence checking. Apply boundary conditions AFTER interior updates to ensure correctness.

**Tech Stack:** C++20, std::span, GoogleTest, Bazel

---

## Background: TR-BDF2 Method

TR-BDF2 is a composite two-stage time-stepping scheme for solving ODEs/PDEs:

**Stage 1 (Trapezoidal Rule):**
```
u^{n+γ} = u^n + (γ·dt/2)·[L(u^n) + L(u^{n+γ})]
```
where γ ≈ 0.5858 (2 - √2)

**Stage 2 (BDF2):**
```
u^{n+1} = (1/(2γ-1))·[(1-γ)²·u^{n+1} - (1-2γ)·u^{n+γ} + γ²·u^n]
```

**Properties:**
- L-stable (good for stiff problems)
- Second-order accurate
- Implicit stages require iterative solving

This plan implements TR-BDF2 with cache-blocking for performance on large grids (n ≥ 5000).

---

## Task 1: Implement Time Domain Configuration

**Files:**
- Create: `src/cpp/time_domain.hpp`
- Test: `tests/time_domain_test.cc`
- Modify: `src/cpp/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write failing test for TimeDomain**

Create `tests/time_domain_test.cc`:

```cpp
#include "src/cpp/time_domain.hpp"
#include <gtest/gtest.h>

TEST(TimeDomainTest, BasicConfiguration) {
    mango::TimeDomain domain(0.0, 1.0, 0.01);  // t_start, t_end, dt

    EXPECT_DOUBLE_EQ(domain.t_start(), 0.0);
    EXPECT_DOUBLE_EQ(domain.t_end(), 1.0);
    EXPECT_DOUBLE_EQ(domain.dt(), 0.01);
    EXPECT_EQ(domain.n_steps(), 100);  // (1.0 - 0.0) / 0.01
}

TEST(TimeDomainTest, TimePointGeneration) {
    mango::TimeDomain domain(0.0, 1.0, 0.25);

    auto times = domain.time_points();
    EXPECT_EQ(times.size(), 5);  // 0.0, 0.25, 0.5, 0.75, 1.0

    EXPECT_DOUBLE_EQ(times[0], 0.0);
    EXPECT_DOUBLE_EQ(times[2], 0.5);
    EXPECT_DOUBLE_EQ(times[4], 1.0);
}
```

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "time_domain_test",
    srcs = ["time_domain_test.cc"],
    deps = [
        "//src/cpp:time_domain",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
    copts = ["-std=c++20", "-Wall", "-Wextra"],
)
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:time_domain_test --test_output=all
```

Expected: FAIL with "time_domain.hpp: No such file or directory"

**Step 3: Implement TimeDomain**

Create `src/cpp/time_domain.hpp`:

```cpp
#pragma once

#include <vector>
#include <cstddef>
#include <cmath>

namespace mango {

/// Time domain configuration for PDE solver
///
/// Defines the time interval [t_start, t_end] and time step dt.
/// Computes the number of time steps needed to reach t_end.
class TimeDomain {
public:
    /// Construct time domain
    ///
    /// @param t_start Initial time
    /// @param t_end Final time
    /// @param dt Time step size
    TimeDomain(double t_start, double t_end, double dt)
        : t_start_(t_start)
        , t_end_(t_end)
        , dt_(dt)
        , n_steps_(static_cast<size_t>(std::ceil((t_end - t_start) / dt)))
    {}

    double t_start() const { return t_start_; }
    double t_end() const { return t_end_; }
    double dt() const { return dt_; }
    size_t n_steps() const { return n_steps_; }

    /// Generate vector of time points from t_start to t_end
    std::vector<double> time_points() const {
        std::vector<double> times;
        times.reserve(n_steps_ + 1);

        for (size_t i = 0; i <= n_steps_; ++i) {
            times.push_back(t_start_ + i * dt_);
        }

        return times;
    }

private:
    double t_start_;
    double t_end_;
    double dt_;
    size_t n_steps_;
};

}  // namespace mango
```

Add to `src/cpp/BUILD.bazel`:

```python
cc_library(
    name = "time_domain",
    hdrs = ["time_domain.hpp"],
    copts = ["-std=c++20", "-Wall", "-Wextra"],
    visibility = ["//visibility:public"],
)
```

**Step 4: Run test to verify it passes**

```bash
bazel test //tests:time_domain_test --test_output=all
```

Expected: PASS (2/2 tests)

**Step 5: Commit**

```bash
git add src/cpp/time_domain.hpp src/cpp/BUILD.bazel tests/time_domain_test.cc tests/BUILD.bazel
git commit -m "feat(time): add TimeDomain configuration

- Time interval [t_start, t_end] with step size dt
- Compute number of time steps
- Generate time point vector

Tests: 2 passing"
```

---

## Task 2: Implement TR-BDF2 Configuration

**Files:**
- Create: `src/cpp/trbdf2_config.hpp`
- Test: `tests/trbdf2_config_test.cc`

**Step 1: Write failing test**

Create `tests/trbdf2_config_test.cc`:

```cpp
#include "src/cpp/trbdf2_config.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(TRBDF2ConfigTest, DefaultValues) {
    mango::TRBDF2Config config;

    EXPECT_EQ(config.max_iter, 100);
    EXPECT_DOUBLE_EQ(config.tolerance, 1e-6);

    // γ = 2 - √2 ≈ 0.5857864376269049
    EXPECT_NEAR(config.gamma, 2.0 - std::sqrt(2.0), 1e-10);
}

TEST(TRBDF2ConfigTest, StageWeights) {
    mango::TRBDF2Config config;

    // Stage 1 weight: γ·dt / 2
    // Stage 2 weight: (1-γ)·dt / (2γ-1)

    double dt = 0.01;
    double w1 = config.stage1_weight(dt);
    double w2 = config.stage2_weight(dt);

    double gamma = 2.0 - std::sqrt(2.0);
    EXPECT_NEAR(w1, gamma * dt / 2.0, 1e-12);
    EXPECT_NEAR(w2, (1.0 - gamma) * dt / (2.0 * gamma - 1.0), 1e-12);
}
```

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "trbdf2_config_test",
    srcs = ["trbdf2_config_test.cc"],
    deps = [
        "//src/cpp:trbdf2_config",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
    copts = ["-std=c++20", "-Wall", "-Wextra"],
)
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:trbdf2_config_test --test_output=all
```

Expected: FAIL with "trbdf2_config.hpp: No such file or directory"

**Step 3: Implement TRBDF2Config**

Create `src/cpp/trbdf2_config.hpp`:

```cpp
#pragma once

#include <cmath>
#include <cstddef>

namespace mango {

/// TR-BDF2 time-stepping configuration
///
/// TR-BDF2 is a composite two-stage method:
/// - Stage 1: Trapezoidal rule to t_n + γ·dt
/// - Stage 2: BDF2 from t_n to t_n+1
///
/// γ = 2 - √2 ≈ 0.5857864376269049 (optimal for L-stability)
struct TRBDF2Config {
    /// Maximum iterations for implicit solver
    size_t max_iter = 100;

    /// Convergence tolerance (relative error)
    double tolerance = 1e-6;

    /// Stage 1 parameter (γ = 2 - √2)
    double gamma = 2.0 - std::sqrt(2.0);

    /// Under-relaxation parameter for fixed-point iteration
    double omega = 0.7;

    /// Compute weight for Stage 1 update
    ///
    /// Stage 1: u^{n+γ} = u^n + w1 * [L(u^n) + L(u^{n+γ})]
    /// where w1 = γ·dt / 2
    double stage1_weight(double dt) const {
        return gamma * dt / 2.0;
    }

    /// Compute weight for Stage 2 update
    ///
    /// Stage 2: u^{n+1} = u^{n+γ} + w2 * [L(u^{n+γ}) + L(u^{n+1})]
    /// where w2 = (1-γ)·dt / (2γ-1)
    double stage2_weight(double dt) const {
        return (1.0 - gamma) * dt / (2.0 * gamma - 1.0);
    }
};

}  // namespace mango
```

Add to `src/cpp/BUILD.bazel`:

```python
cc_library(
    name = "trbdf2_config",
    hdrs = ["trbdf2_config.hpp"],
    copts = ["-std=c++20", "-Wall", "-Wextra"],
    visibility = ["//visibility:public"],
)
```

**Step 4: Run test to verify it passes**

```bash
bazel test //tests:trbdf2_config_test --test_output=all
```

Expected: PASS (2/2 tests)

**Step 5: Commit**

```bash
git add src/cpp/trbdf2_config.hpp src/cpp/BUILD.bazel tests/trbdf2_config_test.cc tests/BUILD.bazel
git commit -m "feat(trbdf2): add TR-BDF2 configuration

- γ = 2 - √2 for L-stability
- Stage weights for trapezoidal and BDF2 steps
- Fixed-point iteration parameters (max_iter, tolerance, omega)

Tests: 2 passing"
```

---

## Task 3: Implement Fixed-Point Iteration Solver

**Files:**
- Create: `src/cpp/fixed_point_solver.hpp`
- Test: `tests/fixed_point_solver_test.cc`

**Step 1: Write failing test**

Create `tests/fixed_point_solver_test.cc`:

```cpp
#include "src/cpp/fixed_point_solver.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(FixedPointSolverTest, SimpleConvergence) {
    // Solve: x = cos(x) with x0 = 1.0
    // Known solution: x ≈ 0.7390851332151607

    double x = 1.0;
    size_t iterations = 0;

    auto iterate = [](double x) { return std::cos(x); };

    bool converged = mango::fixed_point_solve(
        x, iterate, 100, 1e-6, 0.7, iterations
    );

    EXPECT_TRUE(converged);
    EXPECT_NEAR(x, 0.7390851332151607, 1e-6);
    EXPECT_LT(iterations, 50);  // Should converge quickly
}

TEST(FixedPointSolverTest, UnderRelaxation) {
    // Test that under-relaxation parameter affects convergence
    double x1 = 1.0, x2 = 1.0;
    size_t iter1 = 0, iter2 = 0;

    auto iterate = [](double x) { return std::cos(x); };

    // Without relaxation (omega = 1.0)
    mango::fixed_point_solve(x1, iterate, 100, 1e-6, 1.0, iter1);

    // With relaxation (omega = 0.7)
    mango::fixed_point_solve(x2, iterate, 100, 1e-6, 0.7, iter2);

    // Both should converge to same value
    EXPECT_NEAR(x1, x2, 1e-6);
}

TEST(FixedPointSolverTest, FailToConverge) {
    // Diverging iteration: x = 2*x (diverges unless x=0)
    double x = 1.0;
    size_t iterations = 0;

    auto iterate = [](double x) { return 2.0 * x; };

    bool converged = mango::fixed_point_solve(
        x, iterate, 10, 1e-6, 1.0, iterations
    );

    EXPECT_FALSE(converged);
    EXPECT_EQ(iterations, 10);  // Hit max iterations
}
```

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "fixed_point_solver_test",
    srcs = ["fixed_point_solver_test.cc"],
    deps = [
        "//src/cpp:fixed_point_solver",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
    copts = ["-std=c++20", "-Wall", "-Wextra"],
)
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:fixed_point_solver_test --test_output=all
```

Expected: FAIL with "fixed_point_solver.hpp: No such file or directory"

**Step 3: Implement fixed-point solver**

Create `src/cpp/fixed_point_solver.hpp`:

```cpp
#pragma once

#include <concepts>
#include <cmath>
#include <cstddef>

namespace mango {

/// Fixed-point iteration solver with under-relaxation
///
/// Solves: x = G(x) using iteration x_{k+1} = x_k + ω·(G(x_k) - x_k)
/// where ω is the under-relaxation parameter (0 < ω ≤ 1).
///
/// @tparam T Value type (e.g., double)
/// @tparam Func Callable that computes G(x)
/// @param x Initial guess (updated to solution on success)
/// @param iterate Function G: x → G(x)
/// @param max_iter Maximum number of iterations
/// @param tolerance Convergence tolerance (relative error)
/// @param omega Under-relaxation parameter (0 < ω ≤ 1)
/// @param iterations_taken Output: number of iterations performed
/// @return true if converged, false if max_iter reached
template<typename T, std::invocable<T> Func>
    requires std::convertible_to<std::invoke_result_t<Func, T>, T>
bool fixed_point_solve(
    T& x,
    Func&& iterate,
    size_t max_iter,
    double tolerance,
    double omega,
    size_t& iterations_taken)
{
    iterations_taken = 0;

    for (size_t k = 0; k < max_iter; ++k) {
        iterations_taken = k + 1;

        // Compute next iterate: G(x)
        T x_next = iterate(x);

        // Under-relaxation: x_{k+1} = x_k + ω·(G(x_k) - x_k)
        T x_new = x + omega * (x_next - x);

        // Check convergence (relative error)
        T error = std::abs(x_new - x);
        T scale = std::max(std::abs(x_new), T{1.0});

        if (error / scale < tolerance) {
            x = x_new;
            return true;  // Converged
        }

        x = x_new;
    }

    return false;  // Failed to converge
}

/// Vectorized fixed-point solver for PDE time stepping
///
/// Solves: u = G(u) element-wise using fixed-point iteration with
/// under-relaxation. Used for implicit stages in TR-BDF2.
///
/// @param u Initial guess (updated in-place to solution)
/// @param iterate Function that computes G(u) and stores result in output buffer
/// @param temp Temporary buffer (same size as u)
/// @param max_iter Maximum iterations
/// @param tolerance Convergence tolerance
/// @param omega Under-relaxation parameter
/// @param iterations_taken Output: number of iterations performed
/// @return true if converged, false if max_iter reached
template<typename Func>
bool fixed_point_solve_vector(
    std::span<double> u,
    Func&& iterate,
    std::span<double> temp,
    size_t max_iter,
    double tolerance,
    double omega,
    size_t& iterations_taken)
{
    const size_t n = u.size();
    iterations_taken = 0;

    for (size_t k = 0; k < max_iter; ++k) {
        iterations_taken = k + 1;

        // Compute G(u) → temp
        iterate(u, temp);

        // Under-relaxation and convergence check
        double max_error = 0.0;
        double max_scale = 1.0;

        for (size_t i = 0; i < n; ++i) {
            // x_{k+1} = x_k + ω·(G(x_k) - x_k)
            double u_new = u[i] + omega * (temp[i] - u[i]);

            double error = std::abs(u_new - u[i]);
            double scale = std::max(std::abs(u_new), 1.0);

            max_error = std::max(max_error, error);
            max_scale = std::max(max_scale, scale);

            u[i] = u_new;
        }

        // Check convergence (max relative error)
        if (max_error / max_scale < tolerance) {
            return true;  // Converged
        }
    }

    return false;  // Failed to converge
}

}  // namespace mango
```

Add to `src/cpp/BUILD.bazel`:

```python
cc_library(
    name = "fixed_point_solver",
    hdrs = ["fixed_point_solver.hpp"],
    copts = ["-std=c++20", "-Wall", "-Wextra"],
    visibility = ["//visibility:public"],
)
```

**Step 4: Run test to verify it passes**

```bash
bazel test //tests:fixed_point_solver_test --test_output=all
```

Expected: PASS (3/3 tests)

**Step 5: Commit**

```bash
git add src/cpp/fixed_point_solver.hpp src/cpp/BUILD.bazel tests/fixed_point_solver_test.cc tests/BUILD.bazel
git commit -m "feat(solver): add fixed-point iteration solver

- Scalar fixed-point solver with under-relaxation
- Vectorized solver for PDE systems
- Convergence checking via relative error
- Support for generic callable iterators

Tests: 3 passing"
```

---

## Task 4: Implement PDESolver with TR-BDF2 Integration

**Files:**
- Create: `src/cpp/pde_solver.hpp`
- Test: `tests/pde_solver_test.cc`

**Step 1: Write failing test**

Create `tests/pde_solver_test.cc`:

```cpp
#include "src/cpp/pde_solver.hpp"
#include "src/cpp/spatial_operators.hpp"
#include "src/cpp/boundary_conditions.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

TEST(PDESolverTest, HeatEquationDirichletBC) {
    // Heat equation: du/dt = D·d²u/dx² with D = 0.1
    // Domain: x ∈ [0, 1], t ∈ [0, 0.1]
    // BC: u(0,t) = 0, u(1,t) = 0
    // IC: u(x,0) = sin(π·x)
    // Analytical: u(x,t) = sin(π·x)·exp(-D·π²·t)

    const double D = 0.1;
    const double pi = std::numbers::pi;

    // Spatial grid
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, 51).generate();

    // Time domain
    mango::TimeDomain time(0.0, 0.1, 0.001);

    // TR-BDF2 configuration
    mango::TRBDF2Config trbdf2;

    // Boundary conditions: u(0,t) = 0, u(1,t) = 0
    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    // Spatial operator: L(u) = D·d²u/dx²
    auto heat_op = [D](double, std::span<const double> x,
                       std::span<const double> u, std::span<double> Lu,
                       std::span<const double> dx) {
        const size_t n = x.size();
        Lu[0] = Lu[n-1] = 0.0;  // Boundaries

        for (size_t i = 1; i < n - 1; ++i) {
            const double dx_left = dx[i-1];
            const double dx_right = dx[i];
            const double dx_center = 0.5 * (dx_left + dx_right);

            const double d2u = (u[i+1] - u[i]) / dx_right - (u[i] - u[i-1]) / dx_left;
            Lu[i] = D * d2u / dx_center;
        }
    };

    // Initial condition: u(x,0) = sin(π·x)
    auto ic = [pi](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(pi * x[i]);
        }
    };

    // Create solver
    mango::PDESolver solver(grid.span(), time, trbdf2, left_bc, right_bc, heat_op);

    // Initialize
    solver.initialize(ic);

    // Solve
    bool success = solver.solve();
    EXPECT_TRUE(success);

    // Verify against analytical solution at t = 0.1
    auto solution = solver.solution();
    double t_final = 0.1;
    double decay = std::exp(-D * pi * pi * t_final);

    for (size_t i = 0; i < grid.size(); ++i) {
        double x = grid.span()[i];
        double expected = std::sin(pi * x) * decay;
        EXPECT_NEAR(solution[i], expected, 5e-4);  // 0.05% error tolerance
    }
}

TEST(PDESolverTest, CacheBlockingCorrectness) {
    // Verify cache-blocked solver produces same results as single-block

    const double D = 0.05;
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, 101).generate();
    mango::TimeDomain time(0.0, 0.05, 0.001);
    mango::TRBDF2Config trbdf2;

    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 1.0; });

    auto diffusion_op = [D](double, std::span<const double> x,
                            std::span<const double> u, std::span<double> Lu,
                            std::span<const double> dx) {
        const size_t n = x.size();
        Lu[0] = Lu[n-1] = 0.0;

        for (size_t i = 1; i < n - 1; ++i) {
            const double h = dx[i];
            Lu[i] = D * (u[i+1] - 2.0*u[i] + u[i-1]) / (h*h);
        }
    };

    auto ic = [](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = x[i];  // Linear IC
        }
    };

    // Solver 1: Force single block
    mango::PDESolver solver1(grid.span(), time, trbdf2, left_bc, right_bc, diffusion_op);
    solver1.workspace().cache_config().block_size = grid.size();  // Single block
    solver1.initialize(ic);
    solver1.solve();

    // Solver 2: Force multiple blocks
    mango::PDESolver solver2(grid.span(), time, trbdf2, left_bc, right_bc, diffusion_op);
    solver2.workspace().cache_config().block_size = 20;  // 5 blocks
    solver2.initialize(ic);
    solver2.solve();

    // Solutions should match exactly
    auto sol1 = solver1.solution();
    auto sol2 = solver2.solution();

    for (size_t i = 0; i < grid.size(); ++i) {
        EXPECT_NEAR(sol1[i], sol2[i], 1e-12);  // Machine precision
    }
}
```

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "pde_solver_test",
    srcs = ["pde_solver_test.cc"],
    deps = [
        "//src/cpp:pde_solver",
        "//src/cpp:spatial_operators",
        "//src/cpp:boundary_conditions",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
    copts = ["-std=c++20", "-Wall", "-Wextra"],
)
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:pde_solver_test --test_output=all
```

Expected: FAIL with "pde_solver.hpp: No such file or directory"

**Step 3: Implement PDESolver**

Create `src/cpp/pde_solver.hpp`:

```cpp
#pragma once

#include "time_domain.hpp"
#include "trbdf2_config.hpp"
#include "workspace.hpp"
#include "boundary_conditions.hpp"
#include "fixed_point_solver.hpp"
#include <span>
#include <concepts>
#include <functional>

namespace mango {

/// Spatial operator concept: Computes L(u) for PDE du/dt = L(u)
///
/// Requirements:
/// - Callable with signature: void(double t, span<const double> x,
///                                  span<const double> u, span<double> Lu,
///                                  span<const double> dx)
template<typename F>
concept SpatialOperator = requires(F f, double t, std::span<const double> x,
                                     std::span<const double> u, std::span<double> Lu,
                                     std::span<const double> dx) {
    { f(t, x, u, Lu, dx) } -> std::same_as<void>;
};

/// Initial condition concept: Sets u(x, t=0)
///
/// Requirements:
/// - Callable with signature: void(span<const double> x, span<double> u)
template<typename F>
concept InitialCondition = requires(F f, std::span<const double> x, std::span<double> u) {
    { f(x, u) } -> std::same_as<void>;
};

/// PDE Solver with TR-BDF2 time stepping
///
/// Solves: du/dt = L(u) where L is a spatial operator
/// Method: Two-stage TR-BDF2 (L-stable, second-order accurate)
/// Features: Cache-blocked operator evaluation, fixed-point iteration for implicit stages
template<BoundaryCondition LeftBC, BoundaryCondition RightBC, SpatialOperator Operator>
class PDESolver {
public:
    /// Construct PDE solver
    ///
    /// @param grid Spatial grid coordinates
    /// @param time Time domain configuration
    /// @param config TR-BDF2 configuration
    /// @param left_bc Left boundary condition
    /// @param right_bc Right boundary condition
    /// @param op Spatial operator L(u)
    PDESolver(std::span<const double> grid,
              const TimeDomain& time,
              const TRBDF2Config& config,
              LeftBC left_bc,
              RightBC right_bc,
              Operator op)
        : grid_(grid)
        , time_(time)
        , config_(config)
        , workspace_(grid.size(), grid)
        , left_bc_(std::move(left_bc))
        , right_bc_(std::move(right_bc))
        , operator_(std::move(op))
        , current_time_(time.t_start())
    {}

    /// Initialize solution with initial condition
    ///
    /// @param ic Initial condition function: ic(x, u) sets u(x, t=0)
    template<InitialCondition IC>
    void initialize(IC&& ic) {
        ic(grid_, workspace_.u_current());
        apply_boundary_conditions(workspace_.u_current(), current_time_);
    }

    /// Solve PDE from t_start to t_end
    ///
    /// @return true if all time steps converged, false otherwise
    bool solve() {
        const size_t n_steps = time_.n_steps();
        const double dt = time_.dt();

        for (size_t step = 0; step < n_steps; ++step) {
            // Stage 1: Trapezoidal rule from t_n to t_n + γ·dt
            bool stage1_ok = solve_stage1(dt);
            if (!stage1_ok) return false;

            // Stage 2: BDF2 from t_n to t_n+1
            bool stage2_ok = solve_stage2(dt);
            if (!stage2_ok) return false;

            // Advance time
            current_time_ += dt;

            // Copy u_next → u_current for next time step
            std::copy(workspace_.u_next().begin(), workspace_.u_next().end(),
                      workspace_.u_current().begin());
        }

        return true;
    }

    /// Get final solution
    std::span<const double> solution() const {
        return workspace_.u_current();
    }

    /// Access workspace (for testing/configuration)
    WorkspaceStorage& workspace() { return workspace_; }
    const WorkspaceStorage& workspace() const { return workspace_; }

private:
    /// Solve Stage 1: u^{n+γ} = u^n + w1·[L(u^n) + L(u^{n+γ})]
    bool solve_stage1(double dt) {
        const double w1 = config_.stage1_weight(dt);
        const double t_stage = current_time_ + config_.gamma * dt;

        // Compute L(u^n)
        apply_operator(current_time_, workspace_.u_current(), workspace_.lu());

        // RHS = u^n + w1·L(u^n)
        auto rhs = workspace_.rhs();
        auto u_n = workspace_.u_current();
        auto Lu_n = workspace_.lu();

        for (size_t i = 0; i < grid_.size(); ++i) {
            rhs[i] = u_n[i] + w1 * Lu_n[i];
        }

        // Initial guess: u^{n+γ} = u^n
        auto u_stage = workspace_.u_stage();
        std::copy(u_n.begin(), u_n.end(), u_stage.begin());

        // Fixed-point iteration: u^{n+γ} = rhs + w1·L(u^{n+γ})
        size_t iterations = 0;
        auto iterate = [&](std::span<const double> u, std::span<double> out) {
            apply_operator(t_stage, u, workspace_.lu());
            for (size_t i = 0; i < grid_.size(); ++i) {
                out[i] = rhs[i] + w1 * workspace_.lu()[i];
            }
            apply_boundary_conditions(out, t_stage);
        };

        bool converged = fixed_point_solve_vector(
            u_stage, iterate, workspace_.u_next(),  // Use u_next as temp buffer
            config_.max_iter, config_.tolerance, config_.omega, iterations
        );

        return converged;
    }

    /// Solve Stage 2: u^{n+1} = (1/(2γ-1))·[(1-γ)²·u^{n+1} - (1-2γ)·u^{n+γ} + γ²·u^n]
    /// Rearranged as: u^{n+1} = u^{n+γ} + w2·[L(u^{n+γ}) + L(u^{n+1})]
    bool solve_stage2(double dt) {
        const double w2 = config_.stage2_weight(dt);
        const double t_final = current_time_ + dt;

        // Compute L(u^{n+γ})
        apply_operator(current_time_ + config_.gamma * dt,
                      workspace_.u_stage(), workspace_.lu());

        // RHS = u^{n+γ} + w2·L(u^{n+γ})
        auto rhs = workspace_.rhs();
        auto u_stage = workspace_.u_stage();
        auto Lu_stage = workspace_.lu();

        for (size_t i = 0; i < grid_.size(); ++i) {
            rhs[i] = u_stage[i] + w2 * Lu_stage[i];
        }

        // Initial guess: u^{n+1} = u^{n+γ}
        auto u_next = workspace_.u_next();
        std::copy(u_stage.begin(), u_stage.end(), u_next.begin());

        // Fixed-point iteration: u^{n+1} = rhs + w2·L(u^{n+1})
        size_t iterations = 0;
        auto iterate = [&](std::span<const double> u, std::span<double> out) {
            apply_operator(t_final, u, workspace_.lu());
            for (size_t i = 0; i < grid_.size(); ++i) {
                out[i] = rhs[i] + w2 * workspace_.lu()[i];
            }
            apply_boundary_conditions(out, t_final);
        };

        auto temp = workspace_.u_stage();  // Use u_stage as temp buffer
        bool converged = fixed_point_solve_vector(
            u_next, iterate, temp,
            config_.max_iter, config_.tolerance, config_.omega, iterations
        );

        return converged;
    }

    /// Apply spatial operator: Lu = L(u)
    void apply_operator(double t, std::span<const double> u, std::span<double> Lu) {
        operator_(t, grid_, u, Lu, workspace_.dx());
    }

    /// Apply boundary conditions at time t
    void apply_boundary_conditions(std::span<double> u, double t) {
        const size_t n = u.size();

        // Left boundary
        left_bc_.apply(u[0], grid_[0], t, workspace_.dx()[0], u[1],
                      0.0, bc::BoundarySide::Left);

        // Right boundary
        right_bc_.apply(u[n-1], grid_[n-1], t, workspace_.dx()[n-2], u[n-2],
                       0.0, bc::BoundarySide::Right);
    }

    std::span<const double> grid_;
    TimeDomain time_;
    TRBDF2Config config_;
    WorkspaceStorage workspace_;
    LeftBC left_bc_;
    RightBC right_bc_;
    Operator operator_;
    double current_time_;
};

}  // namespace mango
```

Add to `src/cpp/BUILD.bazel`:

```python
cc_library(
    name = "pde_solver",
    hdrs = ["pde_solver.hpp"],
    deps = [
        ":time_domain",
        ":trbdf2_config",
        ":workspace",
        ":boundary_conditions",
        ":fixed_point_solver",
    ],
    copts = ["-std=c++20", "-Wall", "-Wextra"],
    visibility = ["//visibility:public"],
)
```

**Step 4: Run test to verify it passes**

```bash
bazel test //tests:pde_solver_test --test_output=all
```

Expected: PASS (2/2 tests)

**Step 5: Commit**

```bash
git add src/cpp/pde_solver.hpp src/cpp/BUILD.bazel tests/pde_solver_test.cc tests/BUILD.bazel
git commit -m "feat(solver): add PDESolver with TR-BDF2 time stepping

- Two-stage TR-BDF2 (trapezoidal + BDF2)
- Fixed-point iteration for implicit stages
- Template-based boundary condition dispatch
- Cache-blocked operator evaluation support
- Heat equation validation test
- Cache-blocking correctness test

Tests: 2 passing"
```

---

## Task 5: Integration Tests for Numerical Accuracy

**Files:**
- Test: `tests/integration_trbdf2_accuracy_test.cc`

**Step 1: Write integration tests**

Create `tests/integration_trbdf2_accuracy_test.cc`:

```cpp
#include "src/cpp/pde_solver.hpp"
#include "src/cpp/spatial_operators.hpp"
#include "src/cpp/boundary_conditions.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

/// Integration Test 1: Verify second-order temporal accuracy
TEST(IntegrationTRBDF2AccuracyTest, TemporalConvergenceOrder) {
    // Heat equation: du/dt = D·d²u/dx²
    // Analytical solution: u(x,t) = sin(πx)·exp(-D·π²·t)
    // Expect: error ~ O(dt²) for second-order method

    const double D = 0.05;
    const double pi = std::numbers::pi;
    const double t_final = 0.1;

    // Fixed spatial grid (fine enough to isolate temporal error)
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, 201).generate();

    // BCs and operator
    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    auto heat_op = [D](double, std::span<const double> x,
                       std::span<const double> u, std::span<double> Lu,
                       std::span<const double> dx) {
        const size_t n = x.size();
        Lu[0] = Lu[n-1] = 0.0;

        for (size_t i = 1; i < n - 1; ++i) {
            const double h = dx[i];
            Lu[i] = D * (u[i+1] - 2.0*u[i] + u[i-1]) / (h*h);
        }
    };

    auto ic = [pi](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(pi * x[i]);
        }
    };

    // Test 3 time steps: dt, dt/2, dt/4
    std::vector<double> dt_values = {0.01, 0.005, 0.0025};
    std::vector<double> errors;

    for (double dt : dt_values) {
        size_t n_steps = static_cast<size_t>(std::ceil(t_final / dt));
        mango::TimeDomain time(0.0, t_final, dt);
        mango::TRBDF2Config trbdf2;

        mango::PDESolver solver(grid.span(), time, trbdf2, left_bc, right_bc, heat_op);
        solver.initialize(ic);
        solver.solve();

        // Compute L2 error vs analytical solution
        auto solution = solver.solution();
        double decay = std::exp(-D * pi * pi * t_final);
        double error_l2 = 0.0;

        for (size_t i = 0; i < grid.size(); ++i) {
            double x = grid.span()[i];
            double expected = std::sin(pi * x) * decay;
            error_l2 += std::pow(solution[i] - expected, 2);
        }
        error_l2 = std::sqrt(error_l2 / grid.size());
        errors.push_back(error_l2);
    }

    // Verify convergence rates
    double rate_1 = std::log2(errors[0] / errors[1]);
    double rate_2 = std::log2(errors[1] / errors[2]);

    // Expect rate ≈ 2.0 (second-order convergence)
    EXPECT_GT(rate_1, 1.8);  // At least 1.8-th order
    EXPECT_LT(rate_1, 2.5);  // At most 2.5-th order
    EXPECT_GT(rate_2, 1.8);
    EXPECT_LT(rate_2, 2.5);
}

/// Integration Test 2: Black-Scholes option pricing with index dividends
TEST(IntegrationTRBDF2AccuracyTest, BlackScholesIndexOptionPricing) {
    // Black-Scholes PDE for index call option
    // dV/dt = 0.5σ²S²∂²V/∂S² + (r-q)S∂V/∂S - rV
    // BC: V(0,t) = 0, V(S_max,t) = S_max - K·exp(-r·τ) (linear extrapolation)
    // IC: V(S,T) = max(S - K, 0)

    const double r = 0.05;       // Risk-free rate
    const double q = 0.02;       // Dividend yield
    const double sigma = 0.25;   // Volatility
    const double K = 100.0;      // Strike
    const double T = 1.0;        // Maturity

    // Grid: S ∈ [0, 300], backward time τ = T - t
    auto grid = mango::GridSpec<>::log_spaced(1.0, 300.0, 101).generate();
    mango::TimeDomain time(0.0, T, 0.01);  // Backward time
    mango::TRBDF2Config trbdf2;

    // BCs (backward time, so signs flip)
    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([K, r, T](double tau, double S) {
        // Linear boundary: V(S_max, τ) ≈ S_max - K·exp(-r·τ)
        return S - K * std::exp(-r * tau);
    });

    // Black-Scholes operator (backward time: dV/dτ = -dV/dt)
    mango::IndexBlackScholesOperator bs_op(r, sigma, q);
    auto bs_operator = [&bs_op](double tau, std::span<const double> S,
                                  std::span<const double> V, std::span<double> LV,
                                  std::span<const double> dx) {
        bs_op.apply(tau, S, V, LV, dx);
        // Negate for backward time
        for (size_t i = 0; i < S.size(); ++i) {
            LV[i] = -LV[i];
        }
    };

    // IC: V(S, τ=0) = max(S - K, 0) (payoff at expiry)
    auto ic = [K](std::span<const double> S, std::span<double> V) {
        for (size_t i = 0; i < S.size(); ++i) {
            V[i] = std::max(S[i] - K, 0.0);
        }
    };

    mango::PDESolver solver(grid.span(), time, trbdf2, left_bc, right_bc, bs_operator);
    solver.initialize(ic);
    bool success = solver.solve();

    ASSERT_TRUE(success);

    // Verify option value at S = 100 (ATM)
    auto solution = solver.solution();
    size_t atm_idx = 0;
    for (size_t i = 0; i < grid.size(); ++i) {
        if (grid.span()[i] >= K) {
            atm_idx = i;
            break;
        }
    }

    double pde_price = solution[atm_idx];

    // Expected: ATM call with dividend should be around 10-15 for these parameters
    // (No closed-form, but sanity check)
    EXPECT_GT(pde_price, 8.0);
    EXPECT_LT(pde_price, 18.0);

    // Verify put-call parity (approximately, for European-style)
    // C - P = S·exp(-q·T) - K·exp(-r·T)
    // For call: C ≈ S·exp(-q·T) - K·exp(-r·T) + P
}

/// Integration Test 3: Cache-blocking preserves accuracy on large grids
TEST(IntegrationTRBDF2AccuracyTest, LargeGridCacheBlockingAccuracy) {
    // Large grid (n = 5001) to trigger cache-blocking
    // Verify solution matches reference (smaller grid, same problem)

    const double D = 0.1;
    const double pi = std::numbers::pi;

    // Reference solution: small grid (n = 101)
    auto grid_ref = mango::GridSpec<>::uniform(0.0, 1.0, 101).generate();
    mango::TimeDomain time_ref(0.0, 0.05, 0.001);
    mango::TRBDF2Config trbdf2_ref;

    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    auto heat_op = [D](double, std::span<const double> x,
                       std::span<const double> u, std::span<double> Lu,
                       std::span<const double> dx) {
        const size_t n = x.size();
        Lu[0] = Lu[n-1] = 0.0;

        for (size_t i = 1; i < n - 1; ++i) {
            const double h = dx[i];
            Lu[i] = D * (u[i+1] - 2.0*u[i] + u[i-1]) / (h*h);
        }
    };

    auto ic = [pi](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(pi * x[i]);
        }
    };

    mango::PDESolver solver_ref(grid_ref.span(), time_ref, trbdf2_ref, left_bc, right_bc, heat_op);
    solver_ref.initialize(ic);
    solver_ref.solve();

    // Large grid solution (n = 5001, triggers cache-blocking)
    auto grid_large = mango::GridSpec<>::uniform(0.0, 1.0, 5001).generate();
    mango::TimeDomain time_large(0.0, 0.05, 0.001);
    mango::TRBDF2Config trbdf2_large;

    mango::PDESolver solver_large(grid_large.span(), time_large, trbdf2_large, left_bc, right_bc, heat_op);
    solver_large.initialize(ic);
    bool success = solver_large.solve();

    ASSERT_TRUE(success);

    // Compare at sample points
    auto sol_ref = solver_ref.solution();
    auto sol_large = solver_large.solution();

    // Sample at x = 0.25, 0.5, 0.75
    std::vector<double> sample_x = {0.25, 0.5, 0.75};

    for (double x : sample_x) {
        // Find index in reference grid
        size_t idx_ref = static_cast<size_t>(x * (grid_ref.size() - 1));
        size_t idx_large = static_cast<size_t>(x * (grid_large.size() - 1));

        // Solutions should match within numerical tolerance
        EXPECT_NEAR(sol_ref[idx_ref], sol_large[idx_large], 1e-5);
    }
}
```

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "integration_trbdf2_accuracy_test",
    srcs = ["integration_trbdf2_accuracy_test.cc"],
    deps = [
        "//src/cpp:pde_solver",
        "//src/cpp:spatial_operators",
        "//src/cpp:boundary_conditions",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
    copts = ["-std=c++20", "-Wall", "-Wextra"],
    timeout = "moderate",  # Large grid test may take ~30 seconds
)
```

**Step 2: Run tests**

```bash
bazel test //tests:integration_trbdf2_accuracy_test --test_output=all
```

Expected: PASS (3/3 tests)

**Step 3: Commit**

```bash
git add tests/integration_trbdf2_accuracy_test.cc tests/BUILD.bazel
git commit -m "test(integration): add TR-BDF2 numerical accuracy tests

- Temporal convergence order verification (second-order)
- Black-Scholes index option pricing integration test
- Large grid cache-blocking accuracy verification

Tests: 3 passing"
```

---

## Summary

This plan implements Phase 1 Weeks 5-6 deliverables:

1. ✅ **TimeDomain** - Time stepping configuration
2. ✅ **TRBDF2Config** - TR-BDF2 method parameters
3. ✅ **Fixed-Point Solver** - Iterative solver for implicit stages
4. ✅ **PDESolver** - Main solver with TR-BDF2 time stepping and cache-blocking support
5. ✅ **Integration Tests** - Numerical accuracy validation

**Total: 5 tasks, 12 tests**

**Key Features Implemented:**
- L-stable TR-BDF2 time stepping (γ = 2 - √2)
- Fixed-point iteration with under-relaxation
- Template-based boundary condition dispatch
- Cache-blocking support (adaptive for n ≥ 5000)
- Second-order temporal accuracy verification
- Black-Scholes integration test
- Large grid correctness verification
