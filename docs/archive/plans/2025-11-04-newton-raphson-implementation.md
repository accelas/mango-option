# Newton-Raphson PDE Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace fixed-point iteration with Newton-Raphson method in PDESolver to fix convergence failures for stiff problems (200+ grid points, 50+ time steps).

**Architecture:** Quasi-Newton method with tridiagonal solver. Compute Jacobian once per time step via finite differences, solve J·δu = r at each iteration, check convergence via step-to-step delta (not residual). Uses compile-time boundary condition dispatch.

**Tech Stack:** C++20, Bazel, GoogleTest, existing boundary_conditions.hpp with tag-based dispatch

**Design Document:** `/home/kai/work/iv_calc/docs/designs/2025-11-04-newton-raphson-pde-solver.md` (approved after 3 reviews, 6 critical issues fixed)

---

## Task 1: Add Tridiagonal Solver Tests

**Context:** Tridiagonal solver already implemented at `src/cpp/tridiagonal_solver.hpp`. Need comprehensive tests before integrating with Newton solver.

**Files:**
- Create: `tests/tridiagonal_solver_test.cc`
- Modify: `tests/BUILD.bazel` (add new test target)

**Step 1: Create test file with simple 3×3 system test**

Create `tests/tridiagonal_solver_test.cc`:

```cpp
#include "mango/cpp/tridiagonal_solver.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

TEST(TridiagonalSolverTest, Simple3x3System) {
    // System:
    // 2x + 1y       = 1
    // 1x + 2y + 1z  = 0
    //      1y + 2z  = 1
    // Solution: x=1, y=-1, z=1

    std::vector<double> lower = {1.0, 1.0};      // size n-1
    std::vector<double> diag = {2.0, 2.0, 2.0};  // size n
    std::vector<double> upper = {1.0, 1.0};      // size n-1
    std::vector<double> rhs = {1.0, 0.0, 1.0};
    std::vector<double> solution(3);
    std::vector<double> workspace(6);  // 2n

    bool success = mango::solve_tridiagonal(
        std::span{lower}, std::span{diag}, std::span{upper},
        std::span{rhs}, std::span{solution}, std::span{workspace}
    );

    EXPECT_TRUE(success);
    EXPECT_NEAR(solution[0], 1.0, 1e-10);
    EXPECT_NEAR(solution[1], -1.0, 1e-10);
    EXPECT_NEAR(solution[2], 1.0, 1e-10);
}
```

**Step 2: Add singular matrix test**

Add to same file:

```cpp
TEST(TridiagonalSolverTest, SingularMatrix) {
    // All zeros diagonal - should detect singularity
    std::vector<double> lower = {1.0};
    std::vector<double> diag = {0.0, 0.0};  // Singular!
    std::vector<double> upper = {1.0};
    std::vector<double> rhs = {1.0, 1.0};
    std::vector<double> solution(2);
    std::vector<double> workspace(4);

    bool success = mango::solve_tridiagonal(
        std::span{lower}, std::span{diag}, std::span{upper},
        std::span{rhs}, std::span{solution}, std::span{workspace}
    );

    EXPECT_FALSE(success);  // Should fail
}
```

**Step 3: Add heat equation discretization test**

Add to same file:

```cpp
TEST(TridiagonalSolverTest, HeatEquationDiscretization) {
    // Heat equation: ∂u/∂t = D·∂²u/∂x²
    // Implicit Euler: u^{n+1} - dt·D·∂²u^{n+1}/∂x² = u^n
    // With D=1, dt=0.01, dx=0.1, central difference:
    // u_i - 0.01·(u_{i-1} - 2u_i + u_{i+1})/(0.1)² = rhs_i
    // (1 + 2·0.01/0.01)·u_i - (0.01/0.01)·u_{i±1} = rhs_i
    // 3u_i - u_{i-1} - u_{i+1} = rhs_i

    const size_t n = 5;
    std::vector<double> lower(n-1, -1.0);
    std::vector<double> diag(n, 3.0);
    std::vector<double> upper(n-1, -1.0);
    std::vector<double> rhs = {1.0, 2.0, 3.0, 2.0, 1.0};
    std::vector<double> solution(n);
    std::vector<double> workspace(2*n);

    bool success = mango::solve_tridiagonal(
        std::span{lower}, std::span{diag}, std::span{upper},
        std::span{rhs}, std::span{solution}, std::span{workspace}
    );

    EXPECT_TRUE(success);
    // Verify solution satisfies the system (spot check middle point)
    double check = -lower[1] * solution[1] + diag[2] * solution[2]
                   - upper[2] * solution[3];
    EXPECT_NEAR(check, rhs[2], 1e-9);
}
```

**Step 4: Add diagonally dominant matrix test**

Add to same file:

```cpp
TEST(TridiagonalSolverTest, DiagonallyDominant) {
    // Diagonally dominant matrix (guaranteed stable)
    // |a_ii| >= sum(|a_ij|) for all i
    const size_t n = 10;
    std::vector<double> lower(n-1, -1.0);
    std::vector<double> diag(n, 10.0);  // >> 2 (sum of off-diag)
    std::vector<double> upper(n-1, -1.0);
    std::vector<double> rhs(n, 1.0);
    std::vector<double> solution(n);
    std::vector<double> workspace(2*n);

    bool success = mango::solve_tridiagonal(
        std::span{lower}, std::span{diag}, std::span{upper},
        std::span{rhs}, std::span{solution}, std::span{workspace}
    );

    EXPECT_TRUE(success);
    // Should converge without issue
    for (size_t i = 0; i < n; ++i) {
        EXPECT_FALSE(std::isnan(solution[i]));
        EXPECT_FALSE(std::isinf(solution[i]));
    }
}
```

**Step 5: Add test to BUILD.bazel**

Modify `tests/BUILD.bazel`, add after line 338:

```python
cc_test(
    name = "tridiagonal_solver_test",
    srcs = ["tridiagonal_solver_test.cc"],
    deps = [
        "//src/cpp:tridiagonal_solver",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
    copts = ["-std=c++20", "-Wall", "-Wextra"],
)
```

**Step 6: Run tests to verify**

```bash
bazel test //tests:tridiagonal_solver_test --test_output=all
```

Expected: All 4 tests PASS

**Step 7: Commit**

```bash
git add tests/tridiagonal_solver_test.cc tests/BUILD.bazel
git commit -m "test: add comprehensive tridiagonal solver tests

- Simple 3×3 system with known solution
- Singular matrix detection
- Heat equation discretization
- Diagonally dominant matrix stability

All tests pass. Tridiagonal solver ready for Newton integration.
"
```

---

## Task 2: Update TRBDF2Config for Newton

**Context:** Add jacobian_fd_epsilon parameter, reduce max_iter from 100 to 20, remove omega (no under-relaxation needed).

**Files:**
- Modify: `src/cpp/trbdf2_config.hpp:1-30`
- Modify: `tests/trbdf2_config_test.cc:5-13`

**Step 1: Update TRBDF2Config struct**

In `src/cpp/trbdf2_config.hpp`, replace the struct (lines ~8-20):

```cpp
struct TRBDF2Config {
    size_t max_iter = 20;         // Was 100, Newton converges faster
    double tolerance = 1e-6;      // Keep at 1e-6 (matches C implementation)
    double gamma = 2.0 - std::sqrt(2.0);  // L-stability parameter
    size_t cache_blocking_threshold = 5000;
    double jacobian_fd_epsilon = 1e-7;  // NEW: FD epsilon for Jacobian
    // omega removed - no under-relaxation needed for Newton

    double stage1_weight(double dt) const {
        return gamma * dt / 2.0;
    }

    double stage2_weight(double dt) const {
        return (1.0 - gamma) * dt / (2.0 - gamma);
    }
};
```

**Step 2: Update test expectations**

In `tests/trbdf2_config_test.cc`, update DefaultValues test:

```cpp
TEST(TRBDF2ConfigTest, DefaultValues) {
    mango::TRBDF2Config config;

    EXPECT_EQ(config.max_iter, 20);  // Changed from 100
    EXPECT_DOUBLE_EQ(config.tolerance, 1e-6);
    EXPECT_DOUBLE_EQ(config.jacobian_fd_epsilon, 1e-7);  // NEW

    // γ = 2 - √2 ≈ 0.5857864376269049
    EXPECT_NEAR(config.gamma, 2.0 - std::sqrt(2.0), 1e-10);
}
```

**Step 3: Run test to verify**

```bash
bazel test //tests:trbdf2_config_test --test_output=all
```

Expected: PASS (both tests)

**Step 4: Commit**

```bash
git add src/cpp/trbdf2_config.hpp tests/trbdf2_config_test.cc
git commit -m "refactor: update TRBDF2Config for Newton iteration

- Reduce max_iter: 100 → 20 (Newton converges faster)
- Add jacobian_fd_epsilon: 1e-7 (configurable FD step)
- Remove omega (no under-relaxation for Newton)
- Update test expectations

Matches C implementation defaults.
"
```

---

## Task 3: Add Newton Arrays to PDESolver

**Context:** Add 9 new member arrays to PDESolver for Newton-Raphson. Don't implement methods yet, just add storage and initialization.

**Files:**
- Modify: `src/cpp/pde_solver.hpp:133-141` (add members)
- Modify: `src/cpp/pde_solver.hpp:56-65` (update constructor)

**Step 1: Add new member variables**

In `src/cpp/pde_solver.hpp`, after line 137 (after `temp_`), add:

```cpp
    // Newton-Raphson arrays
    std::vector<double> jacobian_lower_;      // n-1: Lower diagonal of Jacobian
    std::vector<double> jacobian_diag_;       // n: Main diagonal of Jacobian
    std::vector<double> jacobian_upper_;      // n-1: Upper diagonal of Jacobian
    std::vector<double> residual_;            // n: Residual vector r(u)
    std::vector<double> delta_u_;             // n: Newton step δu
    std::vector<double> u_perturb_;           // n: Perturbed u for finite differences
    std::vector<double> Lu_perturb_;          // n: L(u_perturb) for finite differences
    std::vector<double> tridiag_workspace_;   // 2n: Workspace for tridiagonal solver
    std::vector<double> rhs_;                 // n: RHS vector (persistent, not local)
    std::vector<double> u_old_newton_;        // n: Previous u for step delta check
```

**Step 2: Initialize arrays in constructor**

In `src/cpp/pde_solver.hpp`, in the constructor initializer list (after line 62), add:

```cpp
        , jacobian_lower_(n_ - 1)
        , jacobian_diag_(n_)
        , jacobian_upper_(n_ - 1)
        , residual_(n_)
        , delta_u_(n_)
        , u_perturb_(n_)
        , Lu_perturb_(n_)
        , tridiag_workspace_(2 * n_)
        , rhs_(n_)
        , u_old_newton_(n_)
```

**Step 3: Verify it compiles**

```bash
bazel build //src/cpp:pde_solver
```

Expected: SUCCESS (no errors)

**Step 4: Run existing tests to ensure no regression**

```bash
bazel test //tests:time_domain_test //tests:trbdf2_config_test //tests:fixed_point_solver_test
```

Expected: All PASS

**Step 5: Commit**

```bash
git add src/cpp/pde_solver.hpp
git commit -m "refactor: add Newton-Raphson member arrays to PDESolver

Add 10 new member arrays for Newton iteration:
- Jacobian storage (tridiagonal: lower, diag, upper)
- Working arrays (residual, delta_u, u_perturb, Lu_perturb)
- Tridiagonal solver workspace
- RHS vector (persistent member, not local)
- u_old_newton for step delta convergence check

Total: 10n doubles (~80 KB for n=1000)
No functionality changes yet, just storage.
"
```

---

## Task 4: Implement Helper Methods

**Context:** Implement compute_residual and compute_step_delta_error helper methods. These are small, self-contained functions.

**Files:**
- Modify: `src/cpp/pde_solver.hpp:260-261` (add private methods after solve_stage2)

**Step 1: Add compute_residual method**

In `src/cpp/pde_solver.hpp`, after the `solve_stage2()` method (around line 260), add:

```cpp
    /// Compute residual: r = rhs - u + coeff_dt·L(u)
    /// ALL points use PDE formula (Dirichlet will be overwritten later)
    void compute_residual(std::span<const double> u, double coeff_dt,
                          std::span<const double> Lu, std::span<const double> rhs,
                          std::span<double> residual) {
        for (size_t i = 0; i < n_; ++i) {
            residual[i] = rhs[i] - u[i] + coeff_dt * Lu[i];
        }
    }
```

**Step 2: Add compute_step_delta_error method**

Add after compute_residual:

```cpp
    /// Compute step-to-step delta error (RMS norm)
    /// This matches C implementation convergence criterion
    double compute_step_delta_error(std::span<const double> u_new,
                                     std::span<const double> u_old) {
        double sum_sq_error = 0.0;
        double sum_sq_norm = 0.0;

        for (size_t i = 0; i < n_; ++i) {
            double diff = u_new[i] - u_old[i];
            sum_sq_error += diff * diff;
            sum_sq_norm += u_new[i] * u_new[i];
        }

        double rms_error = std::sqrt(sum_sq_error / n_);
        double rms_norm = std::sqrt(sum_sq_norm / n_);

        // Relative error with safeguard against division by zero
        const double epsilon = 1e-12;
        return (rms_norm > epsilon) ? rms_error / (rms_norm + epsilon) : rms_error;
    }
```

**Step 3: Verify it compiles**

```bash
bazel build //src/cpp:pde_solver
```

Expected: SUCCESS

**Step 4: Run existing tests**

```bash
bazel test //tests:time_domain_test //tests:trbdf2_config_test
```

Expected: All PASS

**Step 5: Commit**

```bash
git add src/cpp/pde_solver.hpp
git commit -m "feat: add Newton helper methods (residual, convergence)

- compute_residual(): r = rhs - u + coeff_dt·L(u) for all points
- compute_step_delta_error(): RMS of u_new - u_old (matches C impl)

Convergence uses step delta, not residual (critical fix from review).
"
```

---

## Task 5: Implement apply_bc_to_residual

**Context:** Apply boundary conditions to residual using compile-time dispatch. Dirichlet overwrites with constraint equation, Neumann keeps PDE residual.

**Files:**
- Modify: `src/cpp/pde_solver.hpp:260+` (add after helper methods)

**Step 1: Add apply_bc_to_residual method**

Add after compute_step_delta_error:

```cpp
    /// Apply boundary conditions to residual
    /// Dirichlet: r = g(t) - u (constraint)
    /// Neumann: keep PDE residual (already computed)
    void apply_bc_to_residual(std::span<double> residual, double t) {
        // Left boundary - compile-time dispatch on tag type
        if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::dirichlet_tag>) {
            // Dirichlet: Constraint equation r[0] = g(t) - u[0]
            // Sign matches C implementation (src/pde_solver.c:285)
            double g = left_bc_.value(t, grid_[0]);
            residual[0] = g - u_current_[0];
        } else if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::neumann_tag>) {
            // Neumann: Use PDE residual (already computed, no modification)
        } else {
            // Robin: Use PDE residual (boundary handled via apply_boundary_conditions)
        }

        // Right boundary - compile-time dispatch on tag type
        if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::dirichlet_tag>) {
            // Dirichlet: Constraint equation
            double g = right_bc_.value(t, grid_[n_ - 1]);
            residual[n_ - 1] = g - u_current_[n_ - 1];
        } else if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::neumann_tag>) {
            // Neumann: Use PDE residual (already computed)
        } else {
            // Robin: Use PDE residual
        }
    }
```

**Step 2: Verify it compiles**

```bash
bazel build //src/cpp:pde_solver
```

Expected: SUCCESS

**Step 3: Run existing tests**

```bash
bazel test //tests:time_domain_test
```

Expected: PASS

**Step 4: Commit**

```bash
git add src/cpp/pde_solver.hpp
git commit -m "feat: add apply_bc_to_residual with compile-time dispatch

Uses if constexpr with boundary tag types:
- Dirichlet: r = g(t) - u (constraint equation)
- Neumann: keep PDE residual (no modification)
- Robin: keep PDE residual

Critical fix: compile-time dispatch, not runtime .type() method.
"
```

---

## Task 6: Implement build_jacobian_boundaries

**Context:** Compute Jacobian at boundaries using compile-time dispatch. Dirichlet = identity row, Neumann = finite differences.

**Files:**
- Modify: `src/cpp/pde_solver.hpp` (add after apply_bc_to_residual)

**Step 1: Add build_jacobian_boundaries method**

Add after apply_bc_to_residual:

```cpp
    /// Build Jacobian at boundaries (compile-time dispatch)
    void build_jacobian_boundaries(double t, double coeff_dt,
                                    std::span<const double> u, double eps) {
        // Left boundary - compile-time dispatch
        if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::dirichlet_tag>) {
            // Dirichlet: Identity row J[0,0] = 1, J[0,1] = 0
            jacobian_diag_[0] = 1.0;
            jacobian_upper_[0] = 0.0;
        } else if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::neumann_tag>) {
            // Neumann: Compute Jacobian for PDE at boundary
            // Perturb u[0] and evaluate effect on L[0]
            u_perturb_[0] = u[0] + eps;
            spatial_op_(t, grid_, std::span{u_perturb_}, std::span{Lu_perturb_}, workspace_.dx());
            double dL0_du0 = (Lu_perturb_[0] - Lu_[0]) / eps;
            jacobian_diag_[0] = 1.0 - coeff_dt * dL0_du0;
            u_perturb_[0] = u[0];  // Restore

            // Perturb u[1] (affects L[0] via stencil)
            u_perturb_[1] = u[1] + eps;
            spatial_op_(t, grid_, std::span{u_perturb_}, std::span{Lu_perturb_}, workspace_.dx());
            double dL0_du1 = (Lu_perturb_[0] - Lu_[0]) / eps;
            jacobian_upper_[0] = -coeff_dt * dL0_du1;
            u_perturb_[1] = u[1];  // Restore
        } else {
            // Robin: Similar to Neumann (use PDE discretization)
            // Simplified: treat as Neumann for now
            u_perturb_[0] = u[0] + eps;
            spatial_op_(t, grid_, std::span{u_perturb_}, std::span{Lu_perturb_}, workspace_.dx());
            double dL0_du0 = (Lu_perturb_[0] - Lu_[0]) / eps;
            jacobian_diag_[0] = 1.0 - coeff_dt * dL0_du0;
            u_perturb_[0] = u[0];

            u_perturb_[1] = u[1] + eps;
            spatial_op_(t, grid_, std::span{u_perturb_}, std::span{Lu_perturb_}, workspace_.dx());
            double dL0_du1 = (Lu_perturb_[0] - Lu_[0]) / eps;
            jacobian_upper_[0] = -coeff_dt * dL0_du1;
            u_perturb_[1] = u[1];
        }

        // Right boundary - compile-time dispatch
        if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::dirichlet_tag>) {
            // Dirichlet: Identity row
            jacobian_diag_[n_ - 1] = 1.0;
            jacobian_lower_[n_ - 2] = 0.0;
        } else if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::neumann_tag>) {
            // Neumann: FD computation for right boundary
            size_t i = n_ - 1;

            u_perturb_[i] = u[i] + eps;
            spatial_op_(t, grid_, std::span{u_perturb_}, std::span{Lu_perturb_}, workspace_.dx());
            double dLi_dui = (Lu_perturb_[i] - Lu_[i]) / eps;
            jacobian_diag_[i] = 1.0 - coeff_dt * dLi_dui;
            u_perturb_[i] = u[i];

            u_perturb_[i-1] = u[i-1] + eps;
            spatial_op_(t, grid_, std::span{u_perturb_}, std::span{Lu_perturb_}, workspace_.dx());
            double dLi_duim1 = (Lu_perturb_[i] - Lu_[i]) / eps;
            jacobian_lower_[i-1] = -coeff_dt * dLi_duim1;
            u_perturb_[i-1] = u[i-1];
        } else {
            // Robin: Similar to Neumann
            size_t i = n_ - 1;

            u_perturb_[i] = u[i] + eps;
            spatial_op_(t, grid_, std::span{u_perturb_}, std::span{Lu_perturb_}, workspace_.dx());
            double dLi_dui = (Lu_perturb_[i] - Lu_[i]) / eps;
            jacobian_diag_[i] = 1.0 - coeff_dt * dLi_dui;
            u_perturb_[i] = u[i];

            u_perturb_[i-1] = u[i-1] + eps;
            spatial_op_(t, grid_, std::span{u_perturb_}, std::span{Lu_perturb_}, workspace_.dx());
            double dLi_duim1 = (Lu_perturb_[i] - Lu_[i]) / eps;
            jacobian_lower_[i-1] = -coeff_dt * dLi_duim1;
            u_perturb_[i-1] = u[i-1];
        }
    }
```

**Step 2: Verify it compiles**

```bash
bazel build //src/cpp:pde_solver
```

Expected: SUCCESS

**Step 3: Commit**

```bash
git add src/cpp/pde_solver.hpp
git commit -m "feat: add build_jacobian_boundaries with compile-time dispatch

Uses if constexpr for boundary types:
- Dirichlet: Identity row (enforces u = g)
- Neumann: Finite difference Jacobian (PDE at boundary)
- Robin: Treated like Neumann for now

Critical: requires u_perturb_ pre-initialized (done in build_jacobian).
"
```

---

## Task 7: Implement build_jacobian

**Context:** Build full Jacobian matrix via finite differences. CRITICAL: Initialize u_perturb_ first!

**Files:**
- Modify: `src/cpp/pde_solver.hpp` (add after build_jacobian_boundaries)
- Add: `#include "tridiagonal_solver.hpp"` at top

**Step 1: Add include for tridiagonal solver**

At top of `src/cpp/pde_solver.hpp` (after other includes, around line 8):

```cpp
#include "tridiagonal_solver.hpp"
```

**Step 2: Add build_jacobian method**

Add after build_jacobian_boundaries:

```cpp
    /// Build Jacobian matrix via finite differences
    /// CRITICAL: Initializes u_perturb_ to avoid undefined behavior
    void build_jacobian(double t, double coeff_dt,
                        std::span<const double> u, double eps) {
        // CRITICAL: Initialize u_perturb_ with current u before perturbations
        // Without this, finite differences work off undefined data!
        std::copy(u.begin(), u.end(), u_perturb_.begin());

        // Evaluate L(u) as baseline
        spatial_op_(t, grid_, u, std::span{Lu_}, workspace_.dx());

        // Interior points: tridiagonal structure
        for (size_t i = 1; i < n_ - 1; ++i) {
            // ∂L_i/∂u_i (diagonal)
            u_perturb_[i] = u[i] + eps;
            spatial_op_(t, grid_, std::span{u_perturb_},
                        std::span{Lu_perturb_}, workspace_.dx());
            double dLi_dui = (Lu_perturb_[i] - Lu_[i]) / eps;
            jacobian_diag_[i] = 1.0 - coeff_dt * dLi_dui;
            u_perturb_[i] = u[i];  // Restore

            // ∂L_i/∂u_{i-1} (lower diagonal)
            u_perturb_[i-1] = u[i-1] + eps;
            spatial_op_(t, grid_, std::span{u_perturb_},
                        std::span{Lu_perturb_}, workspace_.dx());
            double dLi_duim1 = (Lu_perturb_[i] - Lu_[i]) / eps;
            jacobian_lower_[i-1] = -coeff_dt * dLi_duim1;
            u_perturb_[i-1] = u[i-1];  // Restore

            // ∂L_i/∂u_{i+1} (upper diagonal)
            u_perturb_[i+1] = u[i+1] + eps;
            spatial_op_(t, grid_, std::span{u_perturb_},
                        std::span{Lu_perturb_}, workspace_.dx());
            double dLi_duip1 = (Lu_perturb_[i] - Lu_[i]) / eps;
            jacobian_upper_[i] = -coeff_dt * dLi_duip1;
            u_perturb_[i+1] = u[i+1];  // Restore
        }

        // Boundary rows - call helper (uses compile-time dispatch)
        build_jacobian_boundaries(t, coeff_dt, u, eps);
    }
```

**Step 3: Verify it compiles**

```bash
bazel build //src/cpp:pde_solver
```

Expected: SUCCESS

**Step 4: Commit**

```bash
git add src/cpp/pde_solver.hpp
git commit -m "feat: add build_jacobian with finite differences

Computes Jacobian J = I - coeff_dt·∂L/∂u via FD:
- Interior: 3-point stencil (lower, diag, upper)
- Boundaries: delegate to build_jacobian_boundaries

CRITICAL FIX: std::copy(u, u_perturb_) at start to avoid undefined data.
Matches C implementation (src/pde_solver.c:222-269).
"
```

---

## Task 8: Implement newton_solve

**Context:** Core Newton iteration loop. Quasi-Newton: build Jacobian once, reuse for all iterations. Check convergence via step delta.

**Files:**
- Modify: `src/cpp/pde_solver.hpp` (add after build_jacobian)

**Step 1: Add newton_solve method**

Add after build_jacobian:

```cpp
    /// Newton-Raphson solver for implicit system
    /// Quasi-Newton: Jacobian built once and reused
    bool newton_solve(double t, double coeff_dt,
                      std::span<double> u, std::span<const double> rhs) {
        const double eps = config_.jacobian_fd_epsilon;

        // Initialize boundary conditions before Jacobian computation
        // (Required for valid finite difference perturbations)
        apply_boundary_conditions(u, t);

        // Quasi-Newton: Build Jacobian once and reuse for all iterations
        // Trade-off: Slightly slower convergence vs. lower per-iteration cost
        // For mildly nonlinear problems (typical in PDEs), this achieves
        // superlinear convergence while avoiding repeated FD evaluations
        build_jacobian(t, coeff_dt, u, eps);

        // Save u_old for step delta convergence check
        std::copy(u.begin(), u.end(), u_old_newton_.begin());

        for (size_t iter = 0; iter < config_.max_iter; ++iter) {
            // Evaluate L(u)
            spatial_op_(t, grid_, u, std::span{Lu_}, workspace_.dx());

            // Compute residual: r = rhs - u + coeff_dt·L(u)
            compute_residual(u, coeff_dt, std::span{Lu_}, rhs, std::span{residual_});

            // Apply boundary conditions to residual
            apply_bc_to_residual(std::span{residual_}, t);

            // Solve J·δu = r (NOTE: no negation! residual already has correct sign)
            bool success = solve_tridiagonal(
                std::span{jacobian_lower_}, std::span{jacobian_diag_},
                std::span{jacobian_upper_}, std::span{residual_},
                std::span{delta_u_}, std::span{tridiag_workspace_}
            );

            if (!success) {
                return false;  // Jacobian singular
            }

            // Update: u ← u + δu
            // Note: This is the critical Newton step. The residual computation
            // returned r = rhs - u + coeff_dt·L(u), so solving J·δu = r and
            // updating u ← u + δu moves toward the solution where r = 0.
            for (size_t i = 0; i < n_; ++i) {
                u[i] += delta_u_[i];
            }

            // Apply boundary conditions
            apply_boundary_conditions(u, t);

            // Check convergence: step-to-step delta (NOT residual!)
            double error = compute_step_delta_error(u, std::span{u_old_newton_});
            if (error < config_.tolerance) {
                return true;  // Converged
            }

            // Save current u for next iteration's delta check
            std::copy(u.begin(), u.end(), u_old_newton_.begin());
        }

        return false;  // Max iterations
    }
```

**Step 2: Verify it compiles**

```bash
bazel build //src/cpp:pde_solver
```

Expected: SUCCESS

**Step 3: Commit**

```bash
git add src/cpp/pde_solver.hpp
git commit -m "feat: add newton_solve core iteration loop

Quasi-Newton method:
- Build Jacobian once, reuse for all iterations
- Solve J·δu = r at each iteration (no residual negation!)
- Update u ← u + δu
- Check convergence via step delta (u_new - u_old), not residual

Critical fixes from codex review:
- No residual negation (sign was wrong)
- Step delta convergence (not residual norm)
- Boundary initialization before Jacobian build

Matches C implementation (src/pde_solver.c:155-351).
"
```

---

## Task 9: Update solve_stage1 and solve_stage2

**Context:** Replace fixed-point iteration with newton_solve. Update both stages to use new Newton method.

**Files:**
- Modify: `src/cpp/pde_solver.hpp:165-203,211-259` (replace solve_stage1 and solve_stage2 bodies)

**Step 1: Replace solve_stage1 body**

In `src/cpp/pde_solver.hpp`, replace the solve_stage1 method body (keep signature):

```cpp
    bool solve_stage1(double t_n, double t_stage, double dt) {
        const double w1 = config_.stage1_weight(dt);  // γ·dt/2

        // Compute L(u^n)
        spatial_op_(t_n, grid_, std::span{u_old_}, std::span{Lu_}, workspace_.dx());

        // RHS = u^n + w1·L(u^n)
        for (size_t i = 0; i < n_; ++i) {
            rhs_[i] = u_old_[i] + w1 * Lu_[i];
        }

        // Initial guess: u* = u^n
        std::copy(u_old_.begin(), u_old_.end(), u_current_.begin());

        // Newton iteration
        return newton_solve(t_stage, w1, std::span{u_current_}, std::span{rhs_});
    }
```

**Step 2: Replace solve_stage2 body**

In `src/cpp/pde_solver.hpp`, replace the solve_stage2 method body (keep signature):

```cpp
    bool solve_stage2(double t_stage, double t_next, double dt) {
        const double gamma = config_.gamma;
        const double one_minus_gamma = 1.0 - gamma;
        const double two_minus_gamma = 2.0 - gamma;
        const double denom = gamma * two_minus_gamma;

        // Correct BDF2 coefficients (Ascher, Ruuth, Wetton 1995)
        const double alpha = 1.0 / denom;  // Coefficient for u^{n+γ}
        const double beta = -(one_minus_gamma * one_minus_gamma) / denom;  // Coefficient for u^n
        const double w2 = config_.stage2_weight(dt);  // (1-γ)·dt/(2-γ)

        // RHS = alpha·u^{n+γ} + beta·u^n (u_current_ currently holds u^{n+γ})
        for (size_t i = 0; i < n_; ++i) {
            rhs_[i] = alpha * u_current_[i] + beta * u_old_[i];
        }

        // Initial guess: u^{n+1} = u* (already in u_current_)
        // (No need to copy, u_current_ already has u^{n+γ})

        // Newton iteration
        return newton_solve(t_next, w2, std::span{u_current_}, std::span{rhs_});
    }
```

**Step 3: Verify it compiles**

```bash
bazel build //src/cpp:pde_solver
```

Expected: SUCCESS

**Step 4: Run existing tests (expect some to still work)**

```bash
bazel test //tests:time_domain_test
```

Expected: PASS (this test doesn't use PDESolver)

**Step 5: Commit**

```bash
git add src/cpp/pde_solver.hpp
git commit -m "refactor: replace fixed-point with Newton in solve_stage1/2

Replace fixed_point_solve_vector with newton_solve:
- Stage 1 (Trapezoidal): RHS = u^n + w1·L(u^n)
- Stage 2 (BDF2): RHS = α·u^{n+γ} + β·u^n

Fixed-point code removed. Newton-Raphson now active.
All TR-BDF2 stages use quasi-Newton iteration.
"
```

---

## Task 10: Add Newton Convergence Test

**Context:** Add test to verify Newton converges in < 20 iterations for simple heat equation.

**Files:**
- Modify: `tests/pde_solver_test.cc` (add new test after HeatEquationDirichletBC)

**Step 1: Add NewtonConvergence test**

In `tests/pde_solver_test.cc`, add after line 76 (after HeatEquationDirichletBC test):

```cpp
TEST(PDESolverTest, NewtonConvergence) {
    // Test that Newton converges in < 20 iterations for simple heat equation
    // This verifies the quasi-Newton implementation is working

    const size_t n = 101;
    std::vector<double> grid_storage(n);
    mango::Grid grid = mango::create_uniform_grid(0.0, 1.0, n, grid_storage);

    mango::TimeDomain time(0.0, 0.1, 0.01);  // 10 steps

    mango::TRBDF2Config config;
    config.max_iter = 20;  // Newton should converge well within this

    // Heat equation: ∂u/∂t = D·∂²u/∂x²
    double D = 0.1;
    auto spatial_op = [D](double /*t*/, std::span<const double> grid_pts,
                          std::span<const double> u,
                          std::span<double> Lu,
                          std::span<const double> dx) {
        const size_t n = u.size();
        Lu[0] = Lu[n-1] = 0.0;  // Boundaries

        for (size_t i = 1; i < n - 1; ++i) {
            double dx_avg = (dx[i-1] + dx[i]) / 2.0;
            Lu[i] = D * (u[i-1] - 2.0*u[i] + u[i+1]) / (dx_avg * dx_avg);
        }
    };

    // Dirichlet BCs: u(0,t) = 0, u(1,t) = 0
    auto left_bc = mango::DirichletBC([](double /*t*/, double /*x*/) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double /*t*/, double /*x*/) { return 0.0; });

    // Create solver
    mango::PDESolver solver(grid, time, config, left_bc, right_bc, spatial_op);

    // Initial condition: u(x,0) = sin(πx)
    solver.initialize([](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < u.size(); ++i) {
            u[i] = std::sin(M_PI * x[i]);
        }
    });

    // Solve - should converge (Newton is robust)
    bool converged = solver.solve();
    EXPECT_TRUE(converged);

    // Solution should decay exponentially: u(x,t) ≈ exp(-π²Dt)sin(πx)
    auto solution = solver.solution();
    double expected_decay = std::exp(-M_PI * M_PI * D * 0.1);

    // Check middle point
    size_t mid = n / 2;
    double expected = expected_decay * std::sin(M_PI * grid[mid]);
    EXPECT_NEAR(solution[mid], expected, 0.01);  // 1% tolerance
}
```

**Step 2: Run test**

```bash
bazel test //tests:pde_solver_test --test_filter=NewtonConvergence --test_output=all
```

Expected: PASS (Newton should converge for this simple problem)

If test fails with convergence error, investigate Jacobian computation or boundary handling.

**Step 3: Run HeatEquationDirichletBC test**

```bash
bazel test //tests:pde_solver_test --test_filter=HeatEquationDirichletBC --test_output=all
```

Expected: PASS (should still work with Newton)

**Step 4: Commit**

```bash
git add tests/pde_solver_test.cc
git commit -m "test: add NewtonConvergence test

Verifies Newton-Raphson converges in < 20 iterations for heat equation.
Simple problem: ∂u/∂t = 0.1·∂²u/∂x², Dirichlet BCs, 10 time steps.

Solution should decay exponentially (analytical verification).
"
```

---

## Task 11: Re-enable CacheBlockingCorrectness Test

**Context:** This test was disabled because fixed-point failed for 200 points × 50 steps. Newton should handle this.

**Files:**
- Modify: `tests/BUILD.bazel:77` (remove manual tag)

**Step 1: Remove manual tag**

In `tests/BUILD.bazel`, find the pde_solver_test target (around line 66-78), and remove the `tags = ["manual"]` line:

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
    # tags = ["manual"] removed - Newton should handle this now
)
```

**Step 2: Run CacheBlockingCorrectness test**

```bash
bazel test //tests:pde_solver_test --test_filter=CacheBlockingCorrectness --test_output=all
```

Expected: PASS (Newton handles 200×50 problem)

If test fails:
- Check max_iter (should be 20)
- Check tolerance (should be 1e-6)
- Verify Jacobian computation is correct

**Step 3: Run all pde_solver tests**

```bash
bazel test //tests:pde_solver_test --test_output=all
```

Expected: All 3 tests PASS (HeatEquationDirichletBC, NewtonConvergence, CacheBlockingCorrectness)

**Step 4: Commit**

```bash
git add tests/BUILD.bazel
git commit -m "test: re-enable CacheBlockingCorrectness test

Fixed-point failed for 200 points × 50 steps (convergence failure).
Newton-Raphson handles this robustly.

Test should now pass with Newton iteration.
"
```

---

## Task 12: Run Full Test Suite

**Context:** Verify all tests pass, including regression tests.

**Files:** None (verification step)

**Step 1: Run all tests**

```bash
bazel test //tests/... --test_tag_filters=-manual,-slow --test_output=errors
```

Expected: All tests PASS

**Step 2: Specifically verify key tests**

```bash
bazel test //tests:tridiagonal_solver_test --test_output=all
bazel test //tests:trbdf2_config_test --test_output=all
bazel test //tests:pde_solver_test --test_output=all
bazel test //tests:time_domain_test --test_output=all
```

Expected: All PASS

**Step 3: Check for any warnings or errors**

```bash
bazel build //src/cpp:pde_solver --verbose_failures
```

Expected: SUCCESS with no warnings

**Step 4: Document verification**

No commit needed - this is verification. Ready to proceed to cleanup if all tests pass.

---

## Task 13: Remove Fixed-Point Code (Cleanup)

**Context:** Fixed-point iteration replaced by Newton. Remove unused code and omega parameter.

**Files:**
- Delete: `src/cpp/fixed_point_solver.hpp`
- Delete: `tests/fixed_point_solver_test.cc`
- Modify: `src/cpp/BUILD.bazel` (remove fixed_point_solver target)
- Modify: `tests/BUILD.bazel` (remove fixed_point_solver_test)
- Modify: `src/cpp/pde_solver.hpp` (remove include)

**Step 1: Remove fixed_point_solver.hpp include**

In `src/cpp/pde_solver.hpp`, remove line ~8:

```cpp
#include "fixed_point_solver.hpp"  // DELETE THIS LINE
```

**Step 2: Remove u_stage_ member**

In `src/cpp/pde_solver.hpp`, remove the u_stage_ member (no longer needed):

```cpp
    std::vector<double> u_stage_;    // DELETE THIS LINE
```

And remove from constructor:

```cpp
        , u_stage_(n_)  // DELETE THIS LINE
```

**Step 3: Build to verify nothing uses fixed_point_solver**

```bash
bazel build //src/cpp:pde_solver
```

Expected: SUCCESS (no references to fixed_point_solver remain)

**Step 4: Delete fixed_point_solver files**

```bash
rm src/cpp/fixed_point_solver.hpp
rm tests/fixed_point_solver_test.cc
```

**Step 5: Remove from BUILD files**

In `src/cpp/BUILD.bazel`, remove fixed_point_solver target (if exists).

In `tests/BUILD.bazel`, remove fixed_point_solver_test target (lines ~330-338):

```python
# DELETE THIS ENTIRE cc_test BLOCK:
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

**Step 6: Run tests to verify**

```bash
bazel test //tests/... --test_tag_filters=-manual,-slow
```

Expected: All tests PASS (no fixed_point tests)

**Step 7: Commit**

```bash
git add -u  # Stage deletions
git add src/cpp/pde_solver.hpp src/cpp/BUILD.bazel tests/BUILD.bazel
git commit -m "refactor: remove fixed-point iteration code

Newton-Raphson fully replaces fixed-point (Picard) iteration.
Removed:
- src/cpp/fixed_point_solver.hpp
- tests/fixed_point_solver_test.cc
- u_stage_ member (no longer needed)
- omega parameter (no under-relaxation in Newton)

All functionality now uses Newton iteration.
Code preserved in git history for reference.
"
```

---

## Verification & Completion

After completing all tasks, verify the implementation:

1. **All tests pass:**
   ```bash
   bazel test //tests/... --test_tag_filters=-manual,-slow
   ```

2. **Specific Newton tests pass:**
   ```bash
   bazel test //tests:tridiagonal_solver_test
   bazel test //tests:pde_solver_test
   ```

3. **CacheBlockingCorrectness passes** (the key test that was failing):
   ```bash
   bazel test //tests:pde_solver_test --test_filter=CacheBlockingCorrectness
   ```

4. **No build warnings:**
   ```bash
   bazel build //src/cpp:pde_solver --verbose_failures
   ```

**Success criteria:**
- ✅ All tests pass
- ✅ CacheBlockingCorrectness test passes (was failing with fixed-point)
- ✅ Newton converges in < 20 iterations
- ✅ No compilation warnings
- ✅ Fixed-point code removed

**Next step:** @superpowers:finishing-a-development-branch to create PR

---

## Notes for Engineer

**Critical implementation details:**

1. **Sign in Newton step**: Do NOT negate residual before solving. Solve `J·δu = r` directly.

2. **Convergence criterion**: Use step-to-step delta (`u_new - u_old`), NOT residual norm.

3. **Boundary residuals**: Compute ALL points with PDE formula first. Only Dirichlet overwrites in `apply_bc_to_residual`.

4. **Jacobian initialization**: MUST call `std::copy(u, u_perturb_)` before finite difference loop.

5. **Compile-time dispatch**: Use `if constexpr` with boundary tag types, not runtime `.type()` method.

6. **Memory**: `rhs_` is persistent member, not local variable in solve_stage methods.

**Design document reference:** All algorithms match `/home/kai/work/iv_calc/docs/designs/2025-11-04-newton-raphson-pde-solver.md` (approved after 3 reviews, 6 critical bugs fixed).

**C implementation reference:** `src/pde_solver.c:155-351` for comparison/verification.
