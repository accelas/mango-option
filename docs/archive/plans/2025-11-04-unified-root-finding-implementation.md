# Unified Root-Finding API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement unified root-finding API with Newton-Raphson workspace integration per approved design (docs/plans/2025-11-04-unified-root-finding-api.md)

**Architecture:** Three-layer design: (1) Root-finding types (RootFindingConfig, RootFindingResult), (2) NewtonWorkspace with hybrid allocation (8n owned + 2n borrowed from WorkspaceStorage), (3) NewtonSolver as persistent member in PDESolver. Memory reduction: 15n → 13n (13%).

**Tech Stack:** C++20, std::span, GoogleTest, Bazel

---

## Task 1: Create Root-Finding Types

**Files:**
- Create: `src/cpp/root_finding.hpp`
- Modify: `src/cpp/BUILD.bazel` (add library target)
- Test: `tests/root_finding_test.cc`

**Step 1: Write test for RootFindingConfig defaults**

Create `tests/root_finding_test.cc`:

```cpp
#include "mango/cpp/root_finding.hpp"
#include <gtest/gtest.h>

TEST(RootFindingConfigTest, DefaultValues) {
    mango::RootFindingConfig config;

    EXPECT_EQ(config.max_iter, 100);
    EXPECT_DOUBLE_EQ(config.tolerance, 1e-6);
    EXPECT_DOUBLE_EQ(config.jacobian_fd_epsilon, 1e-7);
    EXPECT_DOUBLE_EQ(config.brent_tol_abs, 1e-6);
}

TEST(RootFindingConfigTest, CustomValues) {
    mango::RootFindingConfig config{
        .max_iter = 50,
        .tolerance = 1e-8,
        .jacobian_fd_epsilon = 1e-9,
        .brent_tol_abs = 1e-8
    };

    EXPECT_EQ(config.max_iter, 50);
    EXPECT_DOUBLE_EQ(config.tolerance, 1e-8);
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:root_finding_test --test_output=all
```

Expected: Build fails with "root_finding.hpp not found"

**Step 3: Create root_finding.hpp with types**

Create `src/cpp/root_finding.hpp`:

```cpp
#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace mango {

/// Configuration for all root-finding methods
///
/// Unified configuration allowing different methods to coexist.
/// Each method uses only its relevant parameters.
struct RootFindingConfig {
    /// Maximum iterations for any method
    size_t max_iter = 100;

    /// Relative convergence tolerance
    double tolerance = 1e-6;

    // Newton-specific parameters
    double jacobian_fd_epsilon = 1e-7;  ///< Finite difference step for Jacobian

    // Brent-specific parameters
    double brent_tol_abs = 1e-6;  ///< Absolute tolerance for Brent's method

    // Future methods can add parameters here
};

/// Result from any root-finding method
///
/// Provides consistent interface for convergence status,
/// iteration count, and diagnostic information.
struct RootFindingResult {
    /// Convergence status
    bool converged;

    /// Number of iterations performed
    size_t iterations;

    /// Final error measure (method-dependent)
    double final_error;

    /// Optional failure diagnostic message
    std::optional<std::string> failure_reason;
};

}  // namespace mango
```

**Step 4: Add library target to BUILD.bazel**

Edit `src/cpp/BUILD.bazel`, add after workspace library:

```python
cc_library(
    name = "root_finding",
    hdrs = ["root_finding.hpp"],
    copts = [
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-O3",
    ],
    visibility = ["//visibility:public"],
)
```

**Step 5: Add test target to tests/BUILD.bazel**

Add after workspace_test:

```python
cc_test(
    name = "root_finding_test",
    srcs = ["root_finding_test.cc"],
    deps = [
        "//src/cpp:root_finding",
        "@googletest//:gtest_main",
    ],
    copts = ["-std=c++20"],
)
```

**Step 6: Run test to verify it passes**

```bash
bazel test //tests:root_finding_test --test_output=all
```

Expected: All tests PASS

**Step 7: Commit Task 1**

```bash
git add src/cpp/root_finding.hpp src/cpp/BUILD.bazel tests/root_finding_test.cc tests/BUILD.bazel
git commit -m "Add unified root-finding configuration types

Adds RootFindingConfig and RootFindingResult for consistent API across
Newton-Raphson, Brent's method, and future root-finding algorithms.

Supports method-specific parameters (jacobian_fd_epsilon for Newton,
brent_tol_abs for Brent) coexisting in single configuration struct.

Part of unified root-finding API design (Phase 1, Step 1).

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Create NewtonWorkspace

**Files:**
- Create: `src/cpp/newton_workspace.hpp`
- Modify: `src/cpp/BUILD.bazel` (add library target)
- Test: `tests/newton_workspace_test.cc`

**Step 1: Write test for workspace allocation**

Create `tests/newton_workspace_test.cc`:

```cpp
#include "mango/cpp/newton_workspace.hpp"
#include "mango/cpp/workspace.hpp"
#include "mango/cpp/grid.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(NewtonWorkspaceTest, CorrectAllocationSizes) {
    const size_t n = 101;
    std::vector<double> grid_data(n);
    for (size_t i = 0; i < n; ++i) {
        grid_data[i] = static_cast<double>(i) / (n - 1);
    }

    mango::WorkspaceStorage pde_ws(n, std::span{grid_data});
    mango::NewtonWorkspace newton_ws(n, pde_ws);

    // Owned arrays
    EXPECT_EQ(newton_ws.jacobian_diag().size(), n);
    EXPECT_EQ(newton_ws.jacobian_lower().size(), n - 1);
    EXPECT_EQ(newton_ws.jacobian_upper().size(), n - 1);
    EXPECT_EQ(newton_ws.residual().size(), n);
    EXPECT_EQ(newton_ws.delta_u().size(), n);
    EXPECT_EQ(newton_ws.u_old().size(), n);
    EXPECT_EQ(newton_ws.tridiag_workspace().size(), 2 * n);  // CRITICAL: 2n

    // Borrowed arrays
    EXPECT_EQ(newton_ws.Lu().size(), n);
    EXPECT_EQ(newton_ws.u_perturb().size(), n);
    EXPECT_EQ(newton_ws.Lu_perturb().size(), n);
}

TEST(NewtonWorkspaceTest, BorrowedArraysPointToWorkspace) {
    const size_t n = 101;
    std::vector<double> grid_data(n);
    for (size_t i = 0; i < n; ++i) {
        grid_data[i] = static_cast<double>(i) / (n - 1);
    }

    mango::WorkspaceStorage pde_ws(n, std::span{grid_data});
    mango::NewtonWorkspace newton_ws(n, pde_ws);

    // Verify borrowed arrays point to correct workspace arrays
    EXPECT_EQ(newton_ws.Lu().data(), pde_ws.lu().data());
    EXPECT_EQ(newton_ws.u_perturb().data(), pde_ws.u_stage().data());
    EXPECT_EQ(newton_ws.Lu_perturb().data(), pde_ws.rhs().data());
}

TEST(NewtonWorkspaceTest, OwnedArraysAreDistinct) {
    const size_t n = 101;
    std::vector<double> grid_data(n);
    for (size_t i = 0; i < n; ++i) {
        grid_data[i] = static_cast<double>(i) / (n - 1);
    }

    mango::WorkspaceStorage pde_ws(n, std::span{grid_data});
    mango::NewtonWorkspace newton_ws(n, pde_ws);

    // Owned arrays should not overlap
    EXPECT_NE(newton_ws.jacobian_diag().data(), newton_ws.residual().data());
    EXPECT_NE(newton_ws.residual().data(), newton_ws.delta_u().data());
    EXPECT_NE(newton_ws.delta_u().data(), newton_ws.u_old().data());
    EXPECT_NE(newton_ws.u_old().data(), newton_ws.tridiag_workspace().data());
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:newton_workspace_test --test_output=all
```

Expected: Build fails with "newton_workspace.hpp not found"

**Step 3: Create newton_workspace.hpp**

Create `src/cpp/newton_workspace.hpp`:

```cpp
#pragma once

#include "workspace.hpp"
#include <vector>
#include <span>
#include <cstddef>

namespace mango {

/// Workspace for Newton-Raphson iteration
///
/// **Memory Strategy (Hybrid Allocation):**
/// - Allocates: 8n doubles (Jacobian: 3n-2, residual: n, delta_u: n, u_old: n, tridiag: 2n)
/// - Borrows: 2n doubles from WorkspaceStorage as scratch space (u_stage, rhs)
/// - Total: 8n allocated + 2n borrowed (vs. 11n if everything owned)
///
/// **Safety of borrowing:**
/// - u_stage: Not used during Newton (operates on u_current)
/// - rhs: Passed as const to Newton solve(), safe to reuse for Lu_perturb scratch
/// - Lu: Read-only during Jacobian build, safe to reference
///
/// **Memory reduction:** 11n → 8n allocated (27% reduction in Newton-specific memory)
class NewtonWorkspace {
public:
    /// Construct workspace borrowing scratch arrays from PDE workspace
    ///
    /// @param n Grid size
    /// @param pde_ws PDE workspace to borrow scratch space from
    NewtonWorkspace(size_t n, WorkspaceStorage& pde_ws)
        : n_(n)
        , buffer_(compute_buffer_size(n))
        , Lu_(pde_ws.lu())
        , u_perturb_(pde_ws.u_stage())
        , Lu_perturb_(pde_ws.rhs())
    {
        setup_owned_arrays();
    }

    // Owned arrays (allocated in buffer_)
    std::span<double> jacobian_lower() { return jacobian_lower_; }
    std::span<double> jacobian_diag() { return jacobian_diag_; }
    std::span<double> jacobian_upper() { return jacobian_upper_; }
    std::span<double> residual() { return residual_; }
    std::span<double> delta_u() { return delta_u_; }
    std::span<double> u_old() { return u_old_; }
    std::span<double> tridiag_workspace() { return tridiag_workspace_; }

    // Borrowed arrays (spans into PDE workspace)
    std::span<const double> Lu() const { return Lu_; }
    std::span<double> u_perturb() { return u_perturb_; }
    std::span<double> Lu_perturb() { return Lu_perturb_; }

private:
    size_t n_;
    std::vector<double> buffer_;  // Single allocation for owned arrays

    // Owned spans (point into buffer_)
    std::span<double> jacobian_lower_;      // n-1
    std::span<double> jacobian_diag_;       // n
    std::span<double> jacobian_upper_;      // n-1
    std::span<double> residual_;            // n
    std::span<double> delta_u_;             // n
    std::span<double> u_old_;               // n
    std::span<double> tridiag_workspace_;   // 2n (CRITICAL: Thomas needs 2n)

    // Borrowed spans (point into WorkspaceStorage)
    std::span<double> Lu_;          // n (read-only during Jacobian)
    std::span<double> u_perturb_;   // n (scratch, from u_stage)
    std::span<double> Lu_perturb_;  // n (scratch, from rhs)

    static constexpr size_t compute_buffer_size(size_t n) {
        // jacobian: (n-1) + n + (n-1) = 3n - 2
        // residual: n
        // delta_u: n
        // u_old: n
        // tridiag_workspace: 2n (CRITICAL FIX from design review)
        return 3*n - 2 + n + n + n + 2*n;  // = 8n - 2
    }

    void setup_owned_arrays() {
        size_t offset = 0;
        jacobian_lower_      = std::span{buffer_.data() + offset, n_ - 1}; offset += n_ - 1;
        jacobian_diag_       = std::span{buffer_.data() + offset, n_};     offset += n_;
        jacobian_upper_      = std::span{buffer_.data() + offset, n_ - 1}; offset += n_ - 1;
        residual_            = std::span{buffer_.data() + offset, n_};     offset += n_;
        delta_u_             = std::span{buffer_.data() + offset, n_};     offset += n_;
        u_old_               = std::span{buffer_.data() + offset, n_};     offset += n_;
        tridiag_workspace_   = std::span{buffer_.data() + offset, 2*n_};
    }
};

}  // namespace mango
```

**Step 4: Add library target to BUILD.bazel**

Edit `src/cpp/BUILD.bazel`, add after root_finding:

```python
cc_library(
    name = "newton_workspace",
    hdrs = ["newton_workspace.hpp"],
    deps = [":workspace"],
    copts = [
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-O3",
    ],
    visibility = ["//visibility:public"],
)
```

**Step 5: Add test target to tests/BUILD.bazel**

Add after root_finding_test:

```python
cc_test(
    name = "newton_workspace_test",
    srcs = ["newton_workspace_test.cc"],
    deps = [
        "//src/cpp:newton_workspace",
        "//src/cpp:workspace",
        "//src/cpp:grid",
        "@googletest//:gtest_main",
    ],
    copts = ["-std=c++20"],
)
```

**Step 6: Run test to verify it passes**

```bash
bazel test //tests:newton_workspace_test --test_output=all
```

Expected: All tests PASS

**Step 7: Commit Task 2**

```bash
git add src/cpp/newton_workspace.hpp src/cpp/BUILD.bazel tests/newton_workspace_test.cc tests/BUILD.bazel
git commit -m "Add NewtonWorkspace with hybrid allocation strategy

Implements workspace for Newton-Raphson with:
- 8n doubles allocated (Jacobian, residual, delta_u, u_old, tridiag)
- 2n doubles borrowed from WorkspaceStorage (u_stage, rhs as scratch)

Memory reduction: 11n → 8n allocated (27% savings).

Critical fix from design review: Allocate 2n for tridiag_workspace
(Thomas algorithm requires 2n, not n).

Part of unified root-finding API design (Phase 1, Step 2).

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Create NewtonSolver (Core Logic)

**Files:**
- Create: `src/cpp/newton_solver.hpp`
- Modify: `src/cpp/BUILD.bazel` (add library target)
- Test: `tests/newton_solver_test.cc`

**Step 1: Write test for Newton convergence**

Create `tests/newton_solver_test.cc`:

```cpp
#include "mango/cpp/newton_solver.hpp"
#include "mango/cpp/root_finding.hpp"
#include "mango/cpp/workspace.hpp"
#include "mango/cpp/boundary_conditions.hpp"
#include "mango/cpp/spatial_operators.hpp"
#include "mango/cpp/grid.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

// Test fixture for Newton solver
class NewtonSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        n = 101;
        grid_data.resize(n);
        for (size_t i = 0; i < n; ++i) {
            grid_data[i] = static_cast<double>(i) / (n - 1);
        }
    }

    size_t n;
    std::vector<double> grid_data;
};

TEST_F(NewtonSolverTest, ConvergesForLinearProblem) {
    // Setup: Solve u = rhs + coeff_dt·∂²u/∂x² with Dirichlet BCs
    // This is a linear problem, Newton should converge in 1-2 iterations

    mango::RootFindingConfig config{.max_iter = 10, .tolerance = 1e-8};
    mango::WorkspaceStorage workspace(n, std::span{grid_data});

    // Dirichlet boundaries: u(0) = 0, u(1) = 0
    mango::bc::Dirichlet left_bc{[](double, double) { return 0.0; }};
    mango::bc::Dirichlet right_bc{[](double, double) { return 0.0; }};

    // Spatial operator: L(u) = ∂²u/∂x²
    mango::op::Laplacian spatial_op{1.0};  // Diffusion coefficient D = 1.0

    mango::NewtonSolver solver(n, config, workspace, left_bc, right_bc,
                              spatial_op, std::span{grid_data});

    // Initial guess: u = sin(πx)
    std::vector<double> u(n);
    for (size_t i = 0; i < n; ++i) {
        u[i] = std::sin(M_PI * grid_data[i]);
    }

    // RHS: rhs = u (i.e., solve u = u + 0·L(u), trivial fixed point)
    std::vector<double> rhs(u);

    double t = 0.0;
    double coeff_dt = 0.01;

    auto result = solver.solve(t, coeff_dt, std::span{u}, std::span{rhs});

    EXPECT_TRUE(result.converged);
    EXPECT_LE(result.iterations, 5);  // Should converge quickly
    EXPECT_LT(result.final_error, config.tolerance);
    EXPECT_FALSE(result.failure_reason.has_value());
}

TEST_F(NewtonSolverTest, RespectsDirichletBoundaries) {
    mango::RootFindingConfig config{.max_iter = 20, .tolerance = 1e-6};
    mango::WorkspaceStorage workspace(n, std::span{grid_data});

    mango::bc::Dirichlet left_bc{[](double, double) { return 1.0; }};
    mango::bc::Dirichlet right_bc{[](double, double) { return 2.0; }};

    mango::op::Laplacian spatial_op{1.0};

    mango::NewtonSolver solver(n, config, workspace, left_bc, right_bc,
                              spatial_op, std::span{grid_data});

    std::vector<double> u(n, 1.5);  // Initial guess
    std::vector<double> rhs(n, 1.5);

    auto result = solver.solve(0.0, 0.01, std::span{u}, std::span{rhs});

    EXPECT_TRUE(result.converged);
    EXPECT_DOUBLE_EQ(u[0], 1.0);  // Left BC
    EXPECT_DOUBLE_EQ(u[n-1], 2.0);  // Right BC
}

TEST_F(NewtonSolverTest, ReportsConvergenceFailure) {
    mango::RootFindingConfig config{.max_iter = 2, .tolerance = 1e-12};
    mango::WorkspaceStorage workspace(n, std::span{grid_data});

    mango::bc::Dirichlet left_bc{[](double, double) { return 0.0; }};
    mango::bc::Dirichlet right_bc{[](double, double) { return 0.0; }};
    mango::op::Laplacian spatial_op{1.0};

    mango::NewtonSolver solver(n, config, workspace, left_bc, right_bc,
                              spatial_op, std::span{grid_data});

    std::vector<double> u(n, 1.0);
    std::vector<double> rhs(n, 0.0);

    auto result = solver.solve(0.0, 0.01, std::span{u}, std::span{rhs});

    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.iterations, 2);
    EXPECT_TRUE(result.failure_reason.has_value());
    EXPECT_EQ(result.failure_reason.value(), "Max iterations reached");
}

TEST_F(NewtonSolverTest, ReuseAcrossMultipleSolves) {
    mango::RootFindingConfig config{.max_iter = 20, .tolerance = 1e-6};
    mango::WorkspaceStorage workspace(n, std::span{grid_data});

    mango::bc::Dirichlet left_bc{[](double, double) { return 0.0; }};
    mango::bc::Dirichlet right_bc{[](double, double) { return 0.0; }};
    mango::op::Laplacian spatial_op{1.0};

    mango::NewtonSolver solver(n, config, workspace, left_bc, right_bc,
                              spatial_op, std::span{grid_data});

    // Solve twice with different RHS
    std::vector<double> u1(n, 1.0), rhs1(n, 1.0);
    auto result1 = solver.solve(0.0, 0.01, std::span{u1}, std::span{rhs1});

    std::vector<double> u2(n, 2.0), rhs2(n, 2.0);
    auto result2 = solver.solve(0.1, 0.01, std::span{u2}, std::span{rhs2});

    EXPECT_TRUE(result1.converged);
    EXPECT_TRUE(result2.converged);
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:newton_solver_test --test_output=all
```

Expected: Build fails with "newton_solver.hpp not found"

**Step 3: Create newton_solver.hpp (Part 1: Basic structure)**

Create `src/cpp/newton_solver.hpp`:

```cpp
#pragma once

#include "root_finding.hpp"
#include "newton_workspace.hpp"
#include "workspace.hpp"
#include "tridiagonal_solver.hpp"
#include "boundary_conditions.hpp"
#include <span>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <limits>

namespace mango {

/// Newton-Raphson solver for implicit PDE stages
///
/// Solves nonlinear system: F(u) = rhs - u + coeff_dt·L(u) = 0
/// where L is the spatial operator.
///
/// **Algorithm:**
/// 1. Build Jacobian J = ∂F/∂u via finite differences (quasi-Newton: once per solve)
/// 2. Iterate: Solve J·δu = F(u), update u ← u + δu
/// 3. Check convergence: ||u_new - u_old|| / ||u_new|| < tolerance
///
/// **Designed for reuse:** Create once, call solve() multiple times.
/// No allocation happens during solve() - all memory pre-allocated.
///
/// @tparam BoundaryL Left boundary condition type
/// @tparam BoundaryR Right boundary condition type
/// @tparam SpatialOp Spatial operator type
template<typename BoundaryL, typename BoundaryR, typename SpatialOp>
class NewtonSolver {
public:
    NewtonSolver(size_t n,
                 const RootFindingConfig& config,
                 WorkspaceStorage& workspace,
                 const BoundaryL& left_bc,
                 const BoundaryR& right_bc,
                 const SpatialOp& spatial_op,
                 std::span<const double> grid)
        : n_(n)
        , config_(config)
        , workspace_(workspace)
        , left_bc_(left_bc)
        , right_bc_(right_bc)
        , spatial_op_(spatial_op)
        , grid_(grid)
        , newton_ws_(n, workspace)
    {}

    /// Solve implicit stage equation
    ///
    /// Solves: u = rhs + coeff_dt·L(u)
    /// Equivalently: F(u) = rhs - u + coeff_dt·L(u) = 0
    ///
    /// @param t Time at which to evaluate operators
    /// @param coeff_dt TR-BDF2 weight (stage1_weight or stage2_weight)
    /// @param u Solution vector (input: initial guess, output: converged solution)
    /// @param rhs Right-hand side from previous stage
    /// @return Result with convergence status
    RootFindingResult solve(double t, double coeff_dt,
                           std::span<double> u,
                           std::span<const double> rhs);

    const RootFindingConfig& config() const { return config_; }

private:
    size_t n_;
    RootFindingConfig config_;
    WorkspaceStorage& workspace_;
    const BoundaryL& left_bc_;
    const BoundaryR& right_bc_;
    const SpatialOp& spatial_op_;
    std::span<const double> grid_;

    NewtonWorkspace newton_ws_;

    // Helper methods (to be implemented)
    void compute_residual(std::span<const double> u, double coeff_dt,
                         std::span<const double> Lu,
                         std::span<const double> rhs,
                         std::span<double> residual);

    double compute_step_delta_error(std::span<const double> u_new,
                                    std::span<const double> u_old);

    void apply_bc_to_residual(std::span<double> residual,
                              std::span<const double> u,
                              double t);

    void apply_boundary_conditions(std::span<double> u, double t);

    void build_jacobian(double t, double coeff_dt,
                       std::span<const double> u, double eps);

    void build_jacobian_boundaries(double t, double coeff_dt,
                                   std::span<const double> u, double eps);
};

}  // namespace mango
```

**Step 4: Implement solve() method**

Add to `newton_solver.hpp` after class declaration:

```cpp
template<typename BoundaryL, typename BoundaryR, typename SpatialOp>
RootFindingResult NewtonSolver<BoundaryL, BoundaryR, SpatialOp>::solve(
    double t, double coeff_dt,
    std::span<double> u,
    std::span<const double> rhs)
{
    const double eps = config_.jacobian_fd_epsilon;

    // Apply BCs to initial guess
    apply_boundary_conditions(u, t);

    // Quasi-Newton: Build Jacobian once and reuse
    build_jacobian(t, coeff_dt, u, eps);

    // Copy initial guess
    std::copy(u.begin(), u.end(), newton_ws_.u_old().begin());

    // Newton iteration
    for (size_t iter = 0; iter < config_.max_iter; ++iter) {
        // Evaluate L(u)
        spatial_op_(t, grid_, u, workspace_.lu(), workspace_.dx());

        // Compute residual: F(u) = rhs - u + coeff_dt·L(u)
        compute_residual(u, coeff_dt, workspace_.lu(), rhs,
                       newton_ws_.residual());

        // CRITICAL FIX: Pass u explicitly to avoid reading stale workspace
        apply_bc_to_residual(newton_ws_.residual(), u, t);

        // Solve J·δu = F(u) using Thomas algorithm
        bool success = solve_tridiagonal(
            newton_ws_.jacobian_lower(),
            newton_ws_.jacobian_diag(),
            newton_ws_.jacobian_upper(),
            newton_ws_.residual(),
            newton_ws_.delta_u(),
            newton_ws_.tridiag_workspace()
        );

        if (!success) {
            return {false, iter, std::numeric_limits<double>::infinity(),
                   "Singular Jacobian"};
        }

        // Update: u ← u + δu
        for (size_t i = 0; i < n_; ++i) {
            u[i] += newton_ws_.delta_u()[i];
        }

        apply_boundary_conditions(u, t);

        // Check convergence via step delta
        double error = compute_step_delta_error(u, newton_ws_.u_old());

        if (error < config_.tolerance) {
            return {true, iter + 1, error, std::nullopt};
        }

        // Prepare for next iteration
        std::copy(u.begin(), u.end(), newton_ws_.u_old().begin());
    }

    return {false, config_.max_iter,
           compute_step_delta_error(u, newton_ws_.u_old()),
           "Max iterations reached"};
}
```

**Step 5: Implement helper methods**

Add helper method implementations to `newton_solver.hpp`:

```cpp
template<typename BoundaryL, typename BoundaryR, typename SpatialOp>
void NewtonSolver<BoundaryL, BoundaryR, SpatialOp>::compute_residual(
    std::span<const double> u, double coeff_dt,
    std::span<const double> Lu,
    std::span<const double> rhs,
    std::span<double> residual)
{
    for (size_t i = 0; i < n_; ++i) {
        residual[i] = rhs[i] - u[i] + coeff_dt * Lu[i];
    }
}

template<typename BoundaryL, typename BoundaryR, typename SpatialOp>
double NewtonSolver<BoundaryL, BoundaryR, SpatialOp>::compute_step_delta_error(
    std::span<const double> u_new,
    std::span<const double> u_old)
{
    double sum_sq_error = 0.0;
    double sum_sq_norm = 0.0;
    for (size_t i = 0; i < n_; ++i) {
        double diff = u_new[i] - u_old[i];
        sum_sq_error += diff * diff;
        sum_sq_norm += u_new[i] * u_new[i];
    }
    double rms_error = std::sqrt(sum_sq_error / n_);
    double rms_norm = std::sqrt(sum_sq_norm / n_);
    const double epsilon = 1e-12;
    return (rms_norm > epsilon) ? rms_error / (rms_norm + epsilon) : rms_error;
}

template<typename BoundaryL, typename BoundaryR, typename SpatialOp>
void NewtonSolver<BoundaryL, BoundaryR, SpatialOp>::apply_bc_to_residual(
    std::span<double> residual,
    std::span<const double> u,  // CRITICAL FIX: explicit parameter
    double t)
{
    // Left boundary
    if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::dirichlet_tag>) {
        double g = left_bc_.value(t, grid_[0]);
        residual[0] = g - u[0];  // Read from passed u, not workspace
    }

    // Right boundary
    if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::dirichlet_tag>) {
        double g = right_bc_.value(t, grid_[n_ - 1]);
        residual[n_ - 1] = g - u[n_ - 1];
    }
}

template<typename BoundaryL, typename BoundaryR, typename SpatialOp>
void NewtonSolver<BoundaryL, BoundaryR, SpatialOp>::apply_boundary_conditions(
    std::span<double> u, double t)
{
    // Left BC
    if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::dirichlet_tag>) {
        u[0] = left_bc_.value(t, grid_[0]);
    } else if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::neumann_tag>) {
        double dx = workspace_.dx()[0];
        double g = left_bc_.gradient(t, grid_[0]);
        u[0] = u[1] - g * dx;
    }

    // Right BC
    if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::dirichlet_tag>) {
        u[n_ - 1] = right_bc_.value(t, grid_[n_ - 1]);
    } else if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::neumann_tag>) {
        double dx = workspace_.dx()[n_ - 2];
        double g = right_bc_.gradient(t, grid_[n_ - 1]);
        u[n_ - 1] = u[n_ - 2] + g * dx;
    }
}

template<typename BoundaryL, typename BoundaryR, typename SpatialOp>
void NewtonSolver<BoundaryL, BoundaryR, SpatialOp>::build_jacobian(
    double t, double coeff_dt,
    std::span<const double> u, double eps)
{
    // Initialize u_perturb and compute baseline L(u)
    std::copy(u.begin(), u.end(), newton_ws_.u_perturb().begin());
    spatial_op_(t, grid_, u, workspace_.lu(), workspace_.dx());

    // Interior points: tridiagonal structure via finite differences
    for (size_t i = 1; i < n_ - 1; ++i) {
        // Diagonal: ∂F/∂u_i
        newton_ws_.u_perturb()[i] = u[i] + eps;
        spatial_op_(t, grid_, newton_ws_.u_perturb(), newton_ws_.Lu_perturb(), workspace_.dx());
        double dLi_dui = (newton_ws_.Lu_perturb()[i] - workspace_.lu()[i]) / eps;
        newton_ws_.jacobian_diag()[i] = -1.0 + coeff_dt * dLi_dui;
        newton_ws_.u_perturb()[i] = u[i];

        // Lower diagonal: ∂F_i/∂u_{i-1}
        newton_ws_.u_perturb()[i - 1] = u[i - 1] + eps;
        spatial_op_(t, grid_, newton_ws_.u_perturb(), newton_ws_.Lu_perturb(), workspace_.dx());
        double dLi_duim1 = (newton_ws_.Lu_perturb()[i] - workspace_.lu()[i]) / eps;
        newton_ws_.jacobian_lower()[i - 1] = coeff_dt * dLi_duim1;
        newton_ws_.u_perturb()[i - 1] = u[i - 1];

        // Upper diagonal: ∂F_i/∂u_{i+1}
        newton_ws_.u_perturb()[i + 1] = u[i + 1] + eps;
        spatial_op_(t, grid_, newton_ws_.u_perturb(), newton_ws_.Lu_perturb(), workspace_.dx());
        double dLi_duip1 = (newton_ws_.Lu_perturb()[i] - workspace_.lu()[i]) / eps;
        newton_ws_.jacobian_upper()[i] = coeff_dt * dLi_duip1;
        newton_ws_.u_perturb()[i + 1] = u[i + 1];
    }

    // Boundary rows
    build_jacobian_boundaries(t, coeff_dt, u, eps);
}

template<typename BoundaryL, typename BoundaryR, typename SpatialOp>
void NewtonSolver<BoundaryL, BoundaryR, SpatialOp>::build_jacobian_boundaries(
    double t, double coeff_dt,
    std::span<const double> u, double eps)
{
    // Left boundary
    if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::dirichlet_tag>) {
        newton_ws_.jacobian_diag()[0] = 1.0;
        newton_ws_.jacobian_upper()[0] = 0.0;
    } else if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::neumann_tag>) {
        newton_ws_.u_perturb()[0] = u[0] + eps;
        spatial_op_(t, grid_, newton_ws_.u_perturb(), newton_ws_.Lu_perturb(), workspace_.dx());
        double dL0_du0 = (newton_ws_.Lu_perturb()[0] - workspace_.lu()[0]) / eps;
        newton_ws_.jacobian_diag()[0] = -1.0 + coeff_dt * dL0_du0;
        newton_ws_.u_perturb()[0] = u[0];

        newton_ws_.u_perturb()[1] = u[1] + eps;
        spatial_op_(t, grid_, newton_ws_.u_perturb(), newton_ws_.Lu_perturb(), workspace_.dx());
        double dL0_du1 = (newton_ws_.Lu_perturb()[0] - workspace_.lu()[0]) / eps;
        newton_ws_.jacobian_upper()[0] = coeff_dt * dL0_du1;
        newton_ws_.u_perturb()[1] = u[1];
    }

    // Right boundary
    if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::dirichlet_tag>) {
        newton_ws_.jacobian_diag()[n_ - 1] = 1.0;
    } else if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::neumann_tag>) {
        size_t i = n_ - 1;
        newton_ws_.u_perturb()[i] = u[i] + eps;
        spatial_op_(t, grid_, newton_ws_.u_perturb(), newton_ws_.Lu_perturb(), workspace_.dx());
        double dLi_dui = (newton_ws_.Lu_perturb()[i] - workspace_.lu()[i]) / eps;
        newton_ws_.jacobian_diag()[i] = -1.0 + coeff_dt * dLi_dui;
        newton_ws_.u_perturb()[i] = u[i];
    }
}
```

**Step 6: Add library target to BUILD.bazel**

Edit `src/cpp/BUILD.bazel`, add after newton_workspace:

```python
cc_library(
    name = "newton_solver",
    hdrs = ["newton_solver.hpp"],
    deps = [
        ":root_finding",
        ":newton_workspace",
        ":workspace",
        ":tridiagonal_solver",
        ":boundary_conditions",
    ],
    copts = [
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-O3",
    ],
    visibility = ["//visibility:public"],
)
```

**Step 7: Add test target to tests/BUILD.bazel**

Add after newton_workspace_test:

```python
cc_test(
    name = "newton_solver_test",
    srcs = ["newton_solver_test.cc"],
    deps = [
        "//src/cpp:newton_solver",
        "//src/cpp:root_finding",
        "//src/cpp:workspace",
        "//src/cpp:boundary_conditions",
        "//src/cpp:spatial_operators",
        "//src/cpp:grid",
        "@googletest//:gtest_main",
    ],
    copts = ["-std=c++20"],
)
```

**Step 8: Run tests to verify they pass**

```bash
bazel test //tests:newton_solver_test --test_output=all
```

Expected: All tests PASS

**Step 9: Commit Task 3**

```bash
git add src/cpp/newton_solver.hpp src/cpp/BUILD.bazel tests/newton_solver_test.cc tests/BUILD.bazel
git commit -m "Add NewtonSolver for implicit PDE stages

Implements Newton-Raphson solver with:
- Quasi-Newton: Build Jacobian once per solve via finite differences
- Convergence via relative step delta
- Compile-time BC dispatch (Dirichlet, Neumann)
- Designed for reuse (no allocation during solve)

Critical fixes from design review:
- Pass u explicitly to apply_bc_to_residual (avoid stale reads)
- Use 2n tridiagonal workspace from NewtonWorkspace

Part of unified root-finding API design (Phase 1, Step 3).

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Integrate NewtonSolver into PDESolver

**Files:**
- Modify: `src/cpp/pde_solver.hpp`
- Test: `tests/pde_solver_test.cc` (update existing tests)

**Step 1: Write test for PDESolver with Newton**

Add to `tests/pde_solver_test.cc`:

```cpp
TEST(PDESolverTest, UsesNewtonSolverForStages) {
    // Setup PDE solver with Newton integration
    const size_t n = 101;
    auto grid = mango::make_uniform_grid(0.0, 1.0, n);

    mango::TimeDomain time{.t_start = 0.0, .t_end = 0.1, .dt = 0.01};
    mango::TRBDF2Config trbdf2_config;
    mango::RootFindingConfig root_config{.max_iter = 20, .tolerance = 1e-6};

    mango::bc::Dirichlet left_bc{[](double, double) { return 0.0; }};
    mango::bc::Dirichlet right_bc{[](double, double) { return 0.0; }};
    mango::op::Laplacian spatial_op{1.0};

    mango::PDESolver solver(grid.data(), time, trbdf2_config, root_config,
                           left_bc, right_bc, spatial_op);

    // Initial condition: u(x, 0) = sin(πx)
    auto ic = [](double x) { return std::sin(M_PI * x); };
    solver.initialize(ic);

    bool converged = solver.solve();

    EXPECT_TRUE(converged);

    // Verify solution decayed (heat equation with zero BCs)
    auto solution = solver.solution();
    for (size_t i = 1; i < n - 1; ++i) {
        EXPECT_LT(std::abs(solution[i]), std::abs(std::sin(M_PI * grid[i])));
    }
}

TEST(PDESolverTest, NewtonConvergenceReported) {
    // Test that Newton convergence failures propagate
    const size_t n = 51;
    auto grid = mango::make_uniform_grid(0.0, 1.0, n);

    mango::TimeDomain time{.t_start = 0.0, .t_end = 1.0, .dt = 0.5};  // Large dt
    mango::TRBDF2Config trbdf2_config;
    mango::RootFindingConfig root_config{.max_iter = 2, .tolerance = 1e-12};  // Hard to converge

    mango::bc::Dirichlet left_bc{[](double, double) { return 0.0; }};
    mango::bc::Dirichlet right_bc{[](double, double) { return 0.0; }};
    mango::op::Laplacian spatial_op{1.0};

    mango::PDESolver solver(grid.data(), time, trbdf2_config, root_config,
                           left_bc, right_bc, spatial_op);

    auto ic = [](double x) { return std::sin(M_PI * x); };
    solver.initialize(ic);

    bool converged = solver.solve();

    // With harsh convergence requirements, should fail
    EXPECT_FALSE(converged);
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:pde_solver_test --test_output=all
```

Expected: Build fails because PDESolver doesn't have RootFindingConfig parameter

**Step 3: Modify PDESolver to add Newton integration**

Read current PDESolver to understand structure:

```bash
# (This step would use Read tool, but shown as command for brevity)
```

Modify `src/cpp/pde_solver.hpp`:

1. Add `#include "newton_solver.hpp"` and `#include "root_finding.hpp"` to includes
2. Add `RootFindingConfig` parameter to constructor
3. Add `NewtonSolver` member variable
4. Update `solve_stage1()` and `solve_stage2()` to use `newton_solver_.solve()`

Changes:
```cpp
// Add to includes (after workspace.hpp)
#include "newton_solver.hpp"
#include "root_finding.hpp"

// In PDESolver class declaration:
template<typename BoundaryL, typename BoundaryR, typename SpatialOp>
class PDESolver {
public:
    PDESolver(std::span<const double> grid,
              const TimeDomain& time,
              const TRBDF2Config& trbdf2_config,
              const RootFindingConfig& root_config,  // NEW
              const BoundaryL& left_bc,
              const BoundaryR& right_bc,
              const SpatialOp& spatial_op)
        : grid_(grid)
        , time_(time)
        , trbdf2_config_(trbdf2_config)
        , root_config_(root_config)  // NEW
        , left_bc_(left_bc)
        , right_bc_(right_bc)
        , spatial_op_(spatial_op)
        , n_(grid.size())
        , workspace_(n_, grid)
        , u_current_(n_)
        , u_old_(n_)
        , rhs_(n_)
        , newton_solver_(n_, root_config_, workspace_,  // NEW
                        left_bc_, right_bc_, spatial_op_, grid)
    {}

private:
    // Add member:
    RootFindingConfig root_config_;
    NewtonSolver<BoundaryL, BoundaryR, SpatialOp> newton_solver_;  // NEW

    // Update solve_stage1:
    bool solve_stage1(double t_n, double t_stage, double dt) {
        const double w1 = trbdf2_config_.stage1_weight(dt);

        // Compute RHS = u^n + w1·L(u^n)
        spatial_op_(t_n, grid_, std::span{u_old_}, workspace_.lu(), workspace_.dx());
        for (size_t i = 0; i < n_; ++i) {
            rhs_[i] = u_old_[i] + w1 * workspace_.lu()[i];
        }

        // Initial guess: u* = u^n
        std::copy(u_old_.begin(), u_old_.end(), u_current_.begin());

        // Use Newton solver
        auto result = newton_solver_.solve(t_stage, w1,
                                          std::span{u_current_},
                                          std::span{rhs_});

        return result.converged;
    }

    // Update solve_stage2:
    bool solve_stage2(double t_stage, double t_next, double dt) {
        const double gamma = trbdf2_config_.gamma;
        const double alpha = 1.0 / (gamma * (2.0 - gamma));
        const double beta = -(1.0 - gamma) * (1.0 - gamma) / (gamma * (2.0 - gamma));
        const double w2 = trbdf2_config_.stage2_weight(dt);

        // RHS = alpha·u^{n+γ} + beta·u^n
        for (size_t i = 0; i < n_; ++i) {
            rhs_[i] = alpha * u_current_[i] + beta * u_old_[i];
        }

        // Use Newton solver
        auto result = newton_solver_.solve(t_next, w2,
                                          std::span{u_current_},
                                          std::span{rhs_});

        return result.converged;
    }
};
```

**Step 4: Update BUILD.bazel dependencies**

Edit `src/cpp/BUILD.bazel`, update pde_solver library:

```python
cc_library(
    name = "pde_solver",
    hdrs = ["pde_solver.hpp"],
    deps = [
        ":workspace",
        ":trbdf2_config",
        ":time_domain",
        ":boundary_conditions",
        ":spatial_operators",
        ":newton_solver",  # NEW
        ":root_finding",   # NEW
    ],
    copts = [
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-O3",
    ],
    visibility = ["//visibility:public"],
)
```

**Step 5: Run tests to verify they pass**

```bash
bazel test //tests:pde_solver_test --test_output=all
```

Expected: All tests PASS

**Step 6: Run all tests to ensure no regressions**

```bash
bazel test //tests:all --test_output=errors
```

Expected: All tests PASS

**Step 7: Commit Task 4**

```bash
git add src/cpp/pde_solver.hpp src/cpp/BUILD.bazel tests/pde_solver_test.cc
git commit -m "Integrate NewtonSolver into PDESolver

Refactors PDESolver to use persistent NewtonSolver:
- Add RootFindingConfig parameter to constructor
- Create NewtonSolver once as member (avoids repeated 8n allocation)
- Update solve_stage1/2 to call newton_solver_.solve()
- Remove old fixed-point iteration code (replaced by Newton)

Memory efficiency:
- Before: 15n doubles (PDESolver arrays + Newton arrays)
- After: 13n doubles (PDESolver arrays + NewtonWorkspace + borrowed)
- Reduction: 13% memory savings

Part of unified root-finding API design (Phase 2 complete).

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Remove Old Newton Arrays from PDESolver

**Files:**
- Modify: `src/cpp/pde_solver.hpp`

**Step 1: Identify old Newton member arrays**

The following arrays should be removed (no longer needed):
- `jacobian_lower_`
- `jacobian_diag_`
- `jacobian_upper_`
- `residual_`
- `delta_u_`
- `u_perturb_`
- `Lu_perturb_`
- `tridiag_workspace_`
- `u_old_newton_`

Keep:
- `u_current_` (PDE state)
- `u_old_` (PDE state)
- `rhs_` (PDE computation)

**Step 2: Write test to verify removal doesn't break anything**

```bash
bazel test //tests:pde_solver_test --test_output=all
```

Expected: All tests still PASS (Newton arrays not used anymore)

**Step 3: Remove old Newton arrays from PDESolver**

Edit `src/cpp/pde_solver.hpp`, remove member variables:

```cpp
// REMOVE these lines:
// std::vector<double> jacobian_lower_;
// std::vector<double> jacobian_diag_;
// std::vector<double> jacobian_upper_;
// std::vector<double> residual_;
// std::vector<double> delta_u_;
// std::vector<double> u_perturb_;
// std::vector<double> Lu_perturb_;
// std::vector<double> tridiag_workspace_;
// std::vector<double> u_old_newton_;
```

**Step 4: Remove initialization in constructor**

Remove corresponding initialization code from constructor.

**Step 5: Run tests to verify removal is clean**

```bash
bazel test //tests:all --test_output=errors
```

Expected: All tests PASS

**Step 6: Commit Task 5**

```bash
git add src/cpp/pde_solver.hpp
git commit -m "Remove old Newton member arrays from PDESolver

Cleanup: Remove 9 Newton member arrays (~11n doubles) now managed
by NewtonSolver and NewtonWorkspace.

Memory freed: ~88 KB for n=10,000 grid points.

Arrays removed: jacobian_{lower,diag,upper}, residual, delta_u,
u_perturb, Lu_perturb, tridiag_workspace, u_old_newton.

Part of unified root-finding API design (Phase 2 cleanup).

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Documentation and Examples

**Files:**
- Update: `CLAUDE.md` (project instructions)
- Create: `examples/example_newton_solver.cc` (standalone example)
- Modify: `examples/BUILD.bazel`

**Step 1: Add documentation to CLAUDE.md**

Add section after PDE solver documentation:

```markdown
## Unified Root-Finding API

The library provides a unified configuration and result interface for all root-finding methods.

### Configuration

```cpp
#include "mango/cpp/root_finding.hpp"

mango::RootFindingConfig config{
    .max_iter = 100,
    .tolerance = 1e-6,
    .jacobian_fd_epsilon = 1e-7,  // Newton-specific
    .brent_tol_abs = 1e-6          // Brent-specific
};
```

### Newton-Raphson Solver

Integrated into PDESolver for implicit time-stepping:

```cpp
mango::PDESolver solver(grid, time, trbdf2_config, root_config,
                       left_bc, right_bc, spatial_op);

solver.initialize(initial_condition);
bool converged = solver.solve();  // Uses Newton for each stage
```

**Memory efficiency:**
- NewtonWorkspace allocates 8n doubles (Jacobian, residual, delta, workspace)
- Borrows 2n doubles from WorkspaceStorage (u_stage, rhs as scratch)
- Total: 13n doubles for entire solver (vs. 15n before)

**Design:**
- Persistent solver instance (created once, reused)
- Quasi-Newton: Jacobian built once per stage
- Compile-time BC dispatch (Dirichlet, Neumann)
- Zero allocation during solve() after construction

### Workspace Management

NewtonWorkspace implements hybrid allocation:
- Owns: Jacobian matrices, residual, delta_u, u_old, tridiag_workspace
- Borrows: Lu (read-only), u_perturb (from u_stage), Lu_perturb (from rhs)

Safe borrowing: u_stage and rhs are unused during Newton iteration.
```

**Step 2: Create standalone example**

Create `examples/example_newton_solver.cc`:

```cpp
#include "mango/cpp/pde_solver.hpp"
#include "mango/cpp/boundary_conditions.hpp"
#include "mango/cpp/spatial_operators.hpp"
#include "mango/cpp/root_finding.hpp"
#include "mango/cpp/grid.hpp"
#include "mango/cpp/time_domain.hpp"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    // Heat equation: ∂u/∂t = ∂²u/∂x²
    // Domain: x ∈ [0, 1], t ∈ [0, 0.1]
    // BCs: u(0, t) = 0, u(1, t) = 0
    // IC: u(x, 0) = sin(πx)

    const size_t n = 101;
    auto grid = mango::make_uniform_grid(0.0, 1.0, n);

    mango::TimeDomain time{
        .t_start = 0.0,
        .t_end = 0.1,
        .dt = 0.001
    };

    mango::TRBDF2Config trbdf2_config;
    mango::RootFindingConfig root_config{
        .max_iter = 20,
        .tolerance = 1e-6,
        .jacobian_fd_epsilon = 1e-7
    };

    // Boundary conditions
    mango::bc::Dirichlet left_bc{[](double, double) { return 0.0; }};
    mango::bc::Dirichlet right_bc{[](double, double) { return 0.0; }};

    // Spatial operator: L(u) = ∂²u/∂x²
    mango::op::Laplacian spatial_op{1.0};  // Diffusion coefficient D = 1.0

    // Create solver with Newton integration
    mango::PDESolver solver(grid.data(), time, trbdf2_config, root_config,
                           left_bc, right_bc, spatial_op);

    // Initial condition: u(x, 0) = sin(πx)
    auto initial_condition = [](double x) {
        return std::sin(M_PI * x);
    };
    solver.initialize(initial_condition);

    std::cout << "Solving heat equation with Newton-Raphson...\n";
    std::cout << "Grid size: " << n << "\n";
    std::cout << "Time steps: " << time.n_steps() << "\n";
    std::cout << "Newton config: max_iter=" << root_config.max_iter
              << ", tol=" << root_config.tolerance << "\n\n";

    bool converged = solver.solve();

    if (converged) {
        std::cout << "Solver converged successfully!\n\n";

        auto solution = solver.solution();

        // Print solution at a few points
        std::cout << "Solution at t=" << time.t_end << ":\n";
        for (size_t i = 0; i < n; i += 20) {
            std::cout << "  u(" << grid[i] << ") = " << solution[i] << "\n";
        }
    } else {
        std::cout << "Solver failed to converge.\n";
        return 1;
    }

    return 0;
}
```

**Step 3: Add example to BUILD.bazel**

Edit `examples/BUILD.bazel`, add:

```python
cc_binary(
    name = "example_newton_solver",
    srcs = ["example_newton_solver.cc"],
    deps = [
        "//src/cpp:pde_solver",
        "//src/cpp:boundary_conditions",
        "//src/cpp:spatial_operators",
        "//src/cpp:root_finding",
        "//src/cpp:grid",
        "//src/cpp:time_domain",
    ],
    copts = ["-std=c++20"],
)
```

**Step 4: Build and run example**

```bash
bazel build //examples:example_newton_solver
./bazel-bin/examples/example_newton_solver
```

Expected: Program runs, prints solution, exits with code 0

**Step 5: Commit Task 6**

```bash
git add CLAUDE.md examples/example_newton_solver.cc examples/BUILD.bazel
git commit -m "Add documentation and example for unified root-finding API

Documentation:
- Add unified root-finding API section to CLAUDE.md
- Document memory efficiency (13n vs 15n)
- Document Newton integration into PDESolver

Example:
- Standalone heat equation solver using Newton-Raphson
- Demonstrates RootFindingConfig usage
- Shows proper solver initialization and usage

Part of unified root-finding API design (Phase 1 complete).

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Final Validation and Testing

**Files:**
- Run: All tests
- Check: Memory usage, performance

**Step 1: Run complete test suite**

```bash
bazel test //tests:all --test_output=errors
```

Expected: All tests PASS with no errors

**Step 2: Run tests with sanitizers (optional but recommended)**

```bash
bazel test //tests:all --config=asan --test_output=errors
```

Expected: No memory leaks, no undefined behavior

**Step 3: Build all targets**

```bash
bazel build //...
```

Expected: All targets build successfully

**Step 4: Run example programs**

```bash
./bazel-bin/examples/example_newton_solver
```

Expected: Program completes successfully

**Step 5: Verify no regressions in existing tests**

Compare test results with baseline (before changes). All existing tests should still pass.

**Step 6: Create final summary commit**

```bash
git add -A
git commit -m "Complete unified root-finding API implementation

Summary of changes:
- RootFindingConfig and RootFindingResult types (Task 1)
- NewtonWorkspace with hybrid allocation (Task 2)
- NewtonSolver with quasi-Newton iteration (Task 3)
- PDESolver integration (Task 4)
- Cleanup old Newton arrays (Task 5)
- Documentation and examples (Task 6)

Memory efficiency: 15n → 13n (13% reduction)
Performance: Zero allocation during solve() after construction

All tests pass. No regressions in existing functionality.

Design approved and implemented per:
docs/plans/2025-11-04-unified-root-finding-api.md

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Execution Notes

### Estimated Time per Task

- Task 1: 10-15 minutes (types are simple structs)
- Task 2: 15-20 minutes (workspace allocation logic)
- Task 3: 30-40 minutes (Newton solver core, multiple methods)
- Task 4: 20-25 minutes (PDESolver integration)
- Task 5: 5-10 minutes (cleanup)
- Task 6: 15-20 minutes (documentation, example)
- Task 7: 10-15 minutes (final validation)

**Total estimated time: 2-2.5 hours**

### Testing Strategy

- **Unit tests**: Each component tested independently
- **Integration tests**: PDESolver with Newton tested end-to-end
- **Regression tests**: Existing PDE solver tests must still pass
- **Example validation**: Standalone example must run successfully

### Success Criteria

✅ All unit tests pass
✅ All integration tests pass
✅ No regressions in existing tests
✅ Memory usage reduced by ~13%
✅ Example program runs without errors
✅ Documentation updated

### Rollback Plan

If any task fails:
1. Revert the specific commit
2. Investigate failure
3. Fix and retry

All tasks are independent up to Task 4, so early tasks can be completed and committed before later tasks.

---

## References

- Design document: `docs/plans/2025-11-04-unified-root-finding-api.md`
- Current Newton implementation: `src/cpp/pde_solver.hpp`
- Workspace design: `src/cpp/workspace.hpp`
- Tridiagonal solver: `src/cpp/tridiagonal_solver.hpp`
- Boundary conditions: `src/cpp/boundary_conditions.hpp`
- Spatial operators: `src/cpp/spatial_operators.hpp`
