<!-- SPDX-License-Identifier: MIT -->
# Unified Root-Finding API with Workspace Management

**Date:** 2025-11-04
**Status:** Design Phase (Revision 2 - Critical bugs fixed)
**Target:** C++20 integration with existing WorkspaceStorage architecture
**Context:** Part of C++20 migration (see 2025-11-03-cpp-migration-design.md)

---

## Problem Statement

### Current Issues

1. **Newton-Raphson has inconsistent memory management**
   - PDESolver allocates 10 separate member arrays (~11n doubles)
   - WorkspaceStorage manages 5n doubles in unified buffer
   - Duplication: `rhs_`, `Lu_`, and other arrays exist in both places
   - No buffer reuse between PDE stages and Newton iteration

2. **No unified root-finding API**
   - Newton-Raphson: Embedded in PDESolver, implicit system solver
   - Brent's method: Separate C-based API in `src/brent.h`, scalar root-finding
   - Inconsistent interfaces make it hard to swap methods or extend

3. **Memory allocation overhead**
   ```
   Current Newton arrays (10 member variables):
   - jacobian_lower_ (n-1)
   - jacobian_diag_ (n)
   - jacobian_upper_ (n-1)
   - residual_ (n)
   - delta_u_ (n)
   - u_perturb_ (n)
   - Lu_perturb_ (n)
   - tridiag_workspace_ (2n)
   - rhs_ (n)
   - u_old_newton_ (n)

   Total: ~11n doubles = 88 KB for n=10,000
   ```

4. **Future extensibility concerns**
   - Adding new root-finding methods (secant, hybrid, quasi-Newton variants) requires duplicating infrastructure
   - No clear pattern for where workspace belongs

---

## Design Goals

1. **Unified configuration**: Common result types and config structure for all methods
2. **Zero-copy workspace reuse**: Newton reuses WorkspaceStorage arrays as scratch space
3. **Clean separation**: Root-finding workspace separate from PDE workspace
4. **Minimal allocation**: Only allocate what each method truly needs
5. **Extensibility**: Easy to add new root-finding methods

---

## Architecture Overview

### Three-Layer Design

```
┌─────────────────────────────────────┐
│      PDESolver (orchestrator)       │
│  - Owns WorkspaceStorage (5n)       │
│  - Creates NewtonSolver once        │
│  - Passes workspace references      │
└─────────────────────────────────────┘
                 │
                 ├──────────────────────┐
                 ▼                      ▼
┌──────────────────────────┐  ┌────────────────────┐
│   WorkspaceStorage       │  │  NewtonSolver      │
│   (PDE state: 5n)        │  │  (root finding)    │
│                          │  │                    │
│  Arrays:                 │  │  Owns:             │
│  - u_current             │  │  - NewtonWorkspace │
│  - u_next                │  │    (8n dedicated)  │
│  - u_stage  (scratch)    │  │                    │
│  - rhs      (scratch)    │  │  Borrows (spans):  │
│  - Lu       (read-only)  │  │  - Lu (read)       │
│  - dx       (read-only)  │  │  - u_stage→u_perturb│
└──────────────────────────┘  │  - rhs→Lu_perturb  │
                              └────────────────────┘
```

**Key Insight**: WorkspaceStorage's `u_stage` and `rhs` are unused during Newton iteration (which operates on `u_current`). Newton borrows them as scratch space via spans.

---

## Detailed Design

### 1. Root-Finding Configuration (Unified)

```cpp
namespace mango {

/// Configuration for all root-finding methods
struct RootFindingConfig {
    size_t max_iter = 100;
    double tolerance = 1e-6;

    // Method-specific parameters (coexist peacefully)
    double jacobian_fd_epsilon = 1e-7;  // Newton: finite difference step
    double brent_tol_abs = 1e-6;        // Brent: absolute tolerance

    // Future methods can add their parameters here
    // double secant_damping = 0.8;  // Example
};

/// Result from any root-finding method
struct RootFindingResult {
    bool converged;
    size_t iterations;
    double final_error;

    // Optional diagnostics
    std::optional<std::string> failure_reason;
};

// NOTE: No RootFindingMethod concept - different methods have fundamentally
// different call signatures (Newton: implicit system, Brent: scalar bracketing).
// Unified API is about consistent return types and configuration, not identical
// call signatures. Use duck typing.

}  // namespace mango
```

### 2. NewtonWorkspace (Hybrid Allocation Strategy)

**Philosophy**: Allocate only what you own, borrow what you can reuse.

**CRITICAL FIX**: Allocate 8n doubles (not 6n) to include 2n for tridiagonal solver workspace.

```cpp
namespace mango {

/// Workspace for Newton-Raphson iteration
///
/// Memory strategy:
/// - Allocates: 8n doubles (Jacobian: 3n-2, residual: n, delta_u: n, u_old: n, tridiag: 2n)
/// - Borrows: 2n doubles from WorkspaceStorage as scratch space
/// - Total: 8n allocated + 2n borrowed (vs. 11n if everything owned)
///
/// Borrowed arrays are safe because:
/// - u_stage: Not used during Newton (operates on u_current)
/// - rhs: Passed as const to Newton, can reuse for Lu_perturb scratch
/// - Lu: Read-only during Jacobian build
class NewtonWorkspace {
public:
    /// Construct workspace borrowing scratch arrays from PDE workspace
    ///
    /// @param n Grid size
    /// @param pde_ws PDE workspace to borrow scratch space from
    NewtonWorkspace(size_t n, WorkspaceStorage& pde_ws)
        : n_(n)
        , buffer_(compute_buffer_size(n))  // Allocate 8n doubles
        , Lu_(pde_ws.lu())                 // Borrow (read-only)
        , u_perturb_(pde_ws.u_stage())     // Borrow (scratch)
        , Lu_perturb_(pde_ws.rhs())        // Borrow (scratch)
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
    std::span<double> tridiag_workspace() { return tridiag_workspace_; }  // NEW: owned 2n

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
    std::span<double> u_perturb_;   // n (scratch)
    std::span<double> Lu_perturb_;  // n (scratch)

    static constexpr size_t compute_buffer_size(size_t n) {
        // jacobian: (n-1) + n + (n-1) = 3n - 2
        // residual: n
        // delta_u: n
        // u_old: n
        // tridiag_workspace: 2n (CRITICAL FIX)
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

**Memory impact:**
```
Before: 11n doubles allocated (88 KB for n=10,000)
After:  8n doubles allocated + 2n borrowed (64 KB for n=10,000, plus zero-copy borrows)
Reduction: 27% memory allocation
```

### 3. NewtonSolver (Unified Interface)

**CRITICAL FIXES**:
1. Pass `u` explicitly to `apply_bc_to_residual` to avoid reading stale workspace data
2. Designed for reuse - no per-call allocations after construction

```cpp
namespace mango {

/// Newton-Raphson solver for implicit PDE stages
///
/// Solves nonlinear system: F(u) = rhs - u + coeff_dt·L(u) = 0
/// where L is the spatial operator.
///
/// Algorithm:
/// 1. Build Jacobian J = ∂F/∂u via finite differences (quasi-Newton: once per solve)
/// 2. Iterate: Solve J·δu = F(u), update u ← u + δu
/// 3. Check convergence: ||u_new - u_old|| / ||u_new|| < tolerance
///
/// **Designed for reuse**: Create once, call solve() multiple times.
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
        , newton_ws_(n, workspace)  // Allocate 8n once
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
                newton_ws_.tridiag_workspace()  // Now has correct 2n size
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

    // Implementation of helper methods

    void compute_residual(std::span<const double> u, double coeff_dt,
                         std::span<const double> Lu,
                         std::span<const double> rhs,
                         std::span<double> residual)
    {
        for (size_t i = 0; i < n_; ++i) {
            residual[i] = rhs[i] - u[i] + coeff_dt * Lu[i];
        }
    }

    double compute_step_delta_error(std::span<const double> u_new,
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

    /// CRITICAL FIX: Pass u explicitly to avoid reading stale workspace
    void apply_bc_to_residual(std::span<double> residual,
                              std::span<const double> u,  // NEW: explicit parameter
                              double t)
    {
        // Compile-time dispatch based on BC tag
        if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::dirichlet_tag>) {
            double g = left_bc_.value(t, grid_[0]);
            residual[0] = g - u[0];  // Read from passed u, not workspace
        } else if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::neumann_tag>) {
            // Neumann: Use PDE residual (already computed)
        }

        // Similar for right boundary
        if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::dirichlet_tag>) {
            double g = right_bc_.value(t, grid_[n_ - 1]);
            residual[n_ - 1] = g - u[n_ - 1];  // Read from passed u
        } else if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::neumann_tag>) {
            // Neumann: Use PDE residual
        }
    }

    void apply_boundary_conditions(std::span<double> u, double t)
    {
        // Apply left BC
        if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::dirichlet_tag>) {
            u[0] = left_bc_.value(t, grid_[0]);
        } else if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::neumann_tag>) {
            double dx = workspace_.dx()[0];
            double g = left_bc_.gradient(t, grid_[0]);
            u[0] = u[1] - g * dx;
        }

        // Apply right BC
        if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::dirichlet_tag>) {
            u[n_ - 1] = right_bc_.value(t, grid_[n_ - 1]);
        } else if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::neumann_tag>) {
            double dx = workspace_.dx()[n_ - 2];
            double g = right_bc_.gradient(t, grid_[n_ - 1]);
            u[n_ - 1] = u[n_ - 2] + g * dx;
        }
    }

    void build_jacobian(double t, double coeff_dt,
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

        // Boundary rows (compile-time dispatch)
        build_jacobian_boundaries(t, coeff_dt, u, eps);
    }

    void build_jacobian_boundaries(double t, double coeff_dt,
                                   std::span<const double> u, double eps)
    {
        // Left boundary
        if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::dirichlet_tag>) {
            newton_ws_.jacobian_diag()[0] = 1.0;
            newton_ws_.jacobian_upper()[0] = 0.0;
        } else if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::neumann_tag>) {
            // Finite difference for Neumann
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

        // Right boundary (similar logic)
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
};

}  // namespace mango
```

### 4. BrentSolver (For Comparison)

```cpp
namespace mango {

/// Brent's method for scalar root-finding
///
/// Combines bisection, secant, and inverse quadratic interpolation.
/// More robust than Newton for 1D problems, doesn't need derivatives.
///
/// Use cases:
/// - Implied volatility calculation
/// - American option critical price
/// - Any scalar equation f(x) = 0 with bracketed root
class BrentSolver {
public:
    explicit BrentSolver(const RootFindingConfig& config)
        : config_(config)
    {}

    /// Find root of f(x) = 0 in interval [a, b]
    ///
    /// Precondition: f(a) and f(b) must have opposite signs
    ///
    /// @param f Function to find root of
    /// @param a Left bracket
    /// @param b Right bracket
    /// @return Result with root (if converged)
    template<typename Func>
    RootFindingResult solve(Func&& f, double a, double b)
    {
        double fa = f(a);
        double fb = f(b);

        // Check bracketing
        if (fa * fb > 0.0) {
            return {false, 0, std::abs(b - a), "Root not bracketed"};
        }

        // Check if endpoints are roots
        if (std::abs(fa) < config_.brent_tol_abs) {
            return {true, 1, std::abs(fa), std::nullopt};
        }
        if (std::abs(fb) < config_.brent_tol_abs) {
            return {true, 1, std::abs(fb), std::nullopt};
        }

        // Brent iteration (implementation from src/brent.h)
        // ... (existing Brent algorithm)

        // Return result
        return {true, iterations, std::abs(fb), std::nullopt};
    }

    const RootFindingConfig& config() const { return config_; }

private:
    RootFindingConfig config_;

    // No workspace needed - scalar problem uses local variables
};

}  // namespace mango
```

### 5. Integration with PDESolver

**KEY CHANGE**: Create NewtonSolver once and reuse to avoid repeated 8n allocations.

```cpp
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
        , Lu_(n_)
        , temp_(n_)
        , rhs_(n_)
        , newton_solver_(n_, root_config_, workspace_,  // NEW: Create once
                        left_bc_, right_bc_, spatial_op_, grid)
    {}

    bool solve() {
        double t = time_.t_start();
        const double dt = time_.dt();

        for (size_t step = 0; step < time_.n_steps(); ++step) {
            std::copy(u_current_.begin(), u_current_.end(), u_old_.begin());

            // Stage 1: Trapezoidal rule
            const double t_stage1 = t + trbdf2_config_.gamma * dt;
            bool stage1_ok = solve_stage1(t, t_stage1, dt);
            if (!stage1_ok) return false;

            // Stage 2: BDF2
            const double t_next = t + dt;
            bool stage2_ok = solve_stage2(t_stage1, t_next, dt);
            if (!stage2_ok) return false;

            t = t_next;
        }

        return true;
    }

private:
    // Stage solving using Newton (reuses solver)
    bool solve_stage1(double t_n, double t_stage, double dt) {
        const double w1 = trbdf2_config_.stage1_weight(dt);

        // Compute RHS = u^n + w1·L(u^n)
        spatial_op_(t_n, grid_, std::span{u_old_}, std::span{Lu_}, workspace_.dx());
        for (size_t i = 0; i < n_; ++i) {
            rhs_[i] = u_old_[i] + w1 * Lu_[i];
        }

        // Initial guess: u* = u^n
        std::copy(u_old_.begin(), u_old_.end(), u_current_.begin());

        // Reuse persistent Newton solver (no allocation)
        auto result = newton_solver_.solve(t_stage, w1,
                                          std::span{u_current_},
                                          std::span{rhs_});

        return result.converged;
    }

    bool solve_stage2(double t_stage, double t_next, double dt) {
        const double gamma = trbdf2_config_.gamma;
        const double alpha = 1.0 / (gamma * (2.0 - gamma));
        const double beta = -(1.0 - gamma) * (1.0 - gamma) / (gamma * (2.0 - gamma));
        const double w2 = trbdf2_config_.stage2_weight(dt);

        // RHS = alpha·u^{n+γ} + beta·u^n
        for (size_t i = 0; i < n_; ++i) {
            rhs_[i] = alpha * u_current_[i] + beta * u_old_[i];
        }

        // Reuse persistent Newton solver
        auto result = newton_solver_.solve(t_next, w2,
                                          std::span{u_current_},
                                          std::span{rhs_});

        return result.converged;
    }

    // Member variables
    std::span<const double> grid_;
    TimeDomain time_;
    TRBDF2Config trbdf2_config_;
    RootFindingConfig root_config_;
    const BoundaryL& left_bc_;
    const BoundaryR& right_bc_;
    const SpatialOp& spatial_op_;
    size_t n_;

    WorkspaceStorage workspace_;  // 5n for PDE state
    std::vector<double> u_current_, u_old_, Lu_, temp_, rhs_;

    // NEW: Persistent Newton solver (created once, reused)
    NewtonSolver<BoundaryL, BoundaryR, SpatialOp> newton_solver_;
};
```

---

## Benefits

### 1. Memory Efficiency

```
Before (current):
PDESolver members:
- u_current, u_old, Lu, temp: 4n
- Newton arrays: 11n
Total: 15n doubles = 120 KB for n=10,000

After (proposed):
PDESolver members: 5n + rhs (6n)
WorkspaceStorage: 5n
NewtonSolver (persistent): 8n
Total allocated: 13n doubles = 104 KB for n=10,000
Borrowed: 2n (zero allocation cost)

Reduction: 13% fewer allocations (15n → 13n)
```

### 2. Unified Configuration

```cpp
// Newton and Brent share same config and result types
RootFindingConfig config{.max_iter = 50, .tolerance = 1e-8};

NewtonSolver newton(..., config, ...);
BrentSolver brent(config);

RootFindingResult r1 = newton.solve(...);
RootFindingResult r2 = brent.solve(...);

// Same result type, different call signatures (no forced abstraction)
```

### 3. No Repeated Allocation

- NewtonSolver created once in PDESolver constructor
- Reused for stage1 and stage2 across all time steps
- 8n allocation happens once, not 2×n_steps times

### 4. Clean Separation

- WorkspaceStorage: Knows about PDE state, not root-finding
- NewtonWorkspace: Knows about Newton, borrows what it can
- PDESolver: Orchestrates, creates solver once

---

## Testing Strategy

### Unit Tests

**NewtonWorkspace allocation:**
```cpp
TEST(NewtonWorkspace, CorrectAllocation) {
    WorkspaceStorage pde_ws(101, grid);
    NewtonWorkspace newton_ws(101, pde_ws);

    // Check owned arrays have correct size
    EXPECT_EQ(newton_ws.jacobian_diag().size(), 101);
    EXPECT_EQ(newton_ws.jacobian_lower().size(), 100);
    EXPECT_EQ(newton_ws.residual().size(), 101);
    EXPECT_EQ(newton_ws.tridiag_workspace().size(), 202);  // CRITICAL: 2n

    // Check borrowed arrays point to PDE workspace
    EXPECT_EQ(newton_ws.Lu().data(), pde_ws.lu().data());
    EXPECT_EQ(newton_ws.u_perturb().data(), pde_ws.u_stage().data());
}
```

**Solver reuse:**
```cpp
TEST(NewtonSolver, ReuseAcrossStages) {
    NewtonSolver solver(n, config, workspace, left_bc, right_bc, op, grid);

    // Solve multiple times with same solver
    auto r1 = solver.solve(t1, w1, u, rhs1);
    auto r2 = solver.solve(t2, w2, u, rhs2);

    EXPECT_TRUE(r1.converged);
    EXPECT_TRUE(r2.converged);
}
```

### Integration Tests

**PDE solver with Newton:**
```cpp
TEST(PDESolver, NewtonIntegration) {
    PDESolver solver(grid, time, trbdf2_config, root_config,
                    left_bc, right_bc, spatial_op);

    solver.initialize(initial_condition);
    bool converged = solver.solve();

    EXPECT_TRUE(converged);

    // Verify solution accuracy
    auto solution = solver.solution();
    EXPECT_NEAR(solution[n/2], expected_value, 1e-6);
}
```

**Boundary residual correctness:**
```cpp
TEST(NewtonSolver, BoundaryResidualReadsCorrectData) {
    // Test that apply_bc_to_residual reads from passed u, not workspace
    NewtonSolver solver(...);

    std::vector<double> u(n, 1.0);
    std::vector<double> rhs(n, 0.0);

    auto result = solver.solve(0.0, 0.01, std::span{u}, std::span{rhs});

    // Should converge without reading stale data
    EXPECT_TRUE(result.converged);
}
```

---

## Migration Path

### Phase 1: Add Unified API (No Breaking Changes)

1. Add `RootFindingConfig` and `RootFindingResult` to `src/cpp/root_finding.hpp`
2. Add `NewtonWorkspace` to `src/cpp/newton_workspace.hpp`
3. Add `NewtonSolver` to `src/cpp/newton_solver.hpp`
4. Keep existing Newton code in PDESolver (temporarily)

### Phase 2: Refactor PDESolver

1. Add `RootFindingConfig` parameter to PDESolver constructor
2. Add `NewtonSolver` member (persistent)
3. Replace `solve_stage1/2` to use `newton_solver_.solve()`
4. Remove Newton member arrays from PDESolver
5. Update tests

### Phase 3: Refactor Brent (Optional)

1. Wrap existing `src/brent.h` in `BrentSolver` class
2. Use same `RootFindingConfig` and `RootFindingResult`
3. Keep C API as thin wrapper for compatibility

---

## Fixes from Code Review

### Critical Issues Fixed

1. **Tridiagonal workspace size**: Allocated 2n (was incorrectly borrowing n)
2. **RootFindingMethod concept removed**: Different call signatures are intentional
3. **Boundary residual data source**: Pass u explicitly to avoid stale workspace reads
4. **Repeated allocation**: Create NewtonSolver once, reuse across time steps

### Design Concerns Addressed

1. **Memory allocation**: 8n + 2n borrowed = 10n (vs. 11n before), 13n total including PDE state
2. **Solver reuse**: Persistent member in PDESolver, not recreated per stage
3. **Config structure**: Single struct is acceptable for now, can refactor later if needed

---

## Open Questions

1. **Should we eventually migrate PDESolver vectors to WorkspaceStorage?**
   - Current: PDESolver has separate u_current_, u_old_, etc.
   - Future: Could use WorkspaceStorage arrays exclusively
   - Recommendation: Defer to future refactor (out of scope for this design)

2. **Should Jacobian be stored between time steps?**
   - Current design: Rebuild each time step (quasi-Newton within step only)
   - Alternative: Store and update incrementally
   - Recommendation: Rebuild per step (simpler, more robust for nonlinear problems)

---

## References

- C++20 Migration Design: `docs/plans/2025-11-03-cpp-migration-design.md`
- Current Newton Implementation: `src/cpp/pde_solver.hpp` (lines 62-71, newton arrays)
- Current Brent Implementation: `src/brent.h`
- Tridiagonal Solver: `src/cpp/tridiagonal_solver.hpp`
- Workspace Design: `src/cpp/workspace.hpp`
