#pragma once

#include "grid.hpp"
#include "workspace.hpp"
#include "boundary_conditions.hpp"
#include "time_domain.hpp"
#include "trbdf2_config.hpp"
#include "fixed_point_solver.hpp"
#include "tridiagonal_solver.hpp"
#include <span>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>

namespace mango {

/// PDE Solver with TR-BDF2 time stepping and cache blocking
///
/// Solves PDEs of the form: ∂u/∂t = L(u, x, t)
/// where L is a spatial operator (e.g., diffusion, advection, reaction)
///
/// Time stepping uses TR-BDF2 (Two-stage Runge-Kutta with BDF2):
/// - Stage 1: Trapezoidal rule to t_n + γ·dt
/// - Stage 2: BDF2 from t_n to t_n+1
/// where γ = 2 - √2 for L-stability
///
/// Cache blocking is automatically applied for large grids (n ≥ threshold)
/// to improve cache locality and reduce memory bandwidth.
///
/// @tparam BoundaryL Left boundary condition type
/// @tparam BoundaryR Right boundary condition type
/// @tparam SpatialOp Spatial operator type
template<typename BoundaryL, typename BoundaryR, typename SpatialOp>
class PDESolver {
public:
    /// Constructor
    ///
    /// @param grid Spatial grid (x coordinates)
    /// @param time Time domain configuration
    /// @param config TR-BDF2 configuration
    /// @param left_bc Left boundary condition
    /// @param right_bc Right boundary condition
    /// @param spatial_op Spatial operator L(u)
    PDESolver(std::span<const double> grid,
              const TimeDomain& time,
              const TRBDF2Config& config,
              const BoundaryL& left_bc,
              const BoundaryR& right_bc,
              const SpatialOp& spatial_op)
        : grid_(grid)
        , time_(time)
        , config_(config)
        , left_bc_(left_bc)
        , right_bc_(right_bc)
        , spatial_op_(spatial_op)
        , n_(grid.size())
        , workspace_(n_, grid)
        , u_current_(n_)
        , u_stage_(n_)
        , u_old_(n_)
        , Lu_(n_)
        , temp_(n_)
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
    {
        // Determine cache blocking strategy
        use_cache_blocking_ = (n_ >= config_.cache_blocking_threshold);
    }

    /// Initialize with initial condition
    ///
    /// @param ic Initial condition function: ic(x, u)
    template<typename IC>
    void initialize(IC&& ic) {
        ic(grid_, std::span{u_current_});

        // Apply boundary conditions at t=0
        double t = time_.t_start();
        apply_boundary_conditions(std::span{u_current_}, t);
    }

    /// Solve PDE from t_start to t_end
    ///
    /// @return true if converged at all time steps, false otherwise
    bool solve() {
        double t = time_.t_start();
        const double dt = time_.dt();

        for (size_t step = 0; step < time_.n_steps(); ++step) {
            // Store u^n for TR-BDF2
            std::copy(u_current_.begin(), u_current_.end(), u_old_.begin());

            // Stage 1: Trapezoidal rule to t_n + γ·dt
            double t_stage1 = t + config_.gamma * dt;
            bool stage1_ok = solve_stage1(t, t_stage1, dt);
            if (!stage1_ok) {
                return false;  // Convergence failure
            }

            // Stage 2: BDF2 from t_n to t_n+1
            double t_next = t + dt;
            bool stage2_ok = solve_stage2(t_stage1, t_next, dt);
            if (!stage2_ok) {
                return false;  // Convergence failure
            }

            // Update time
            t = t_next;
        }

        return true;
    }

    /// Get current solution
    std::span<const double> solution() const {
        return std::span{u_current_};
    }

private:
    // Grid and configuration
    std::span<const double> grid_;
    TimeDomain time_;
    TRBDF2Config config_;
    BoundaryL left_bc_;
    BoundaryR right_bc_;
    SpatialOp spatial_op_;

    // Grid size
    size_t n_;

    // Workspace for cache blocking
    WorkspaceStorage workspace_;

    // Solution storage
    std::vector<double> u_current_;  // u^{n+1} or u^{n+γ}
    std::vector<double> u_stage_;    // Staging buffer for fixed-point
    std::vector<double> u_old_;      // u^n
    std::vector<double> Lu_;         // L(u) temporary
    std::vector<double> temp_;       // Fixed-point iteration temp

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

    // Cache blocking flag
    bool use_cache_blocking_;

    /// Apply boundary conditions
    void apply_boundary_conditions(std::span<double> u, double t) {
        auto dx_span = workspace_.dx();

        // Left boundary
        double x_left = grid_[0];
        double dx_left = (n_ > 1) ? dx_span[0] : 1.0;
        double u_interior_left = (n_ > 1) ? u[1] : 0.0;
        left_bc_.apply(u[0], x_left, t, dx_left, u_interior_left, 0.0, bc::BoundarySide::Left);

        // Right boundary
        double x_right = grid_[n_ - 1];
        double dx_right = (n_ > 1) ? dx_span[n_ - 2] : 1.0;
        double u_interior_right = (n_ > 1) ? u[n_ - 2] : 0.0;
        right_bc_.apply(u[n_ - 1], x_right, t, dx_right, u_interior_right, 0.0, bc::BoundarySide::Right);
    }

    /// TR-BDF2 Stage 1: Trapezoidal rule
    ///
    /// u^{n+γ} = u^n + (γ·dt/2) · [L(u^n) + L(u^{n+γ})]
    ///
    /// Solved via fixed-point iteration:
    /// u^{n+γ} = G(u^{n+γ}) where G(u) = u^n + (γ·dt/2) · [L(u^n) + L(u)]
    bool solve_stage1(double t_n, double t_stage, double dt) {
        const double w1 = config_.stage1_weight(dt);

        // Compute L(u^n) → Lu_
        spatial_op_(t_n, grid_, std::span{u_old_}, std::span{Lu_}, workspace_.dx());

        // Fixed-point iteration: u^{n+γ} = u^n + w1 · [L(u^n) + L(u^{n+γ})]
        auto iterate = [&](std::span<const double> u, std::span<double> G_u) {
            // Compute L(u) → temp_
            spatial_op_(t_stage, grid_, u, std::span{temp_}, workspace_.dx());

            // G(u) = u^n + w1 · [L(u^n) + L(u)]
            for (size_t i = 0; i < n_; ++i) {
                G_u[i] = u_old_[i] + w1 * (Lu_[i] + temp_[i]);
            }

            // Apply boundary conditions to the iterate
            apply_boundary_conditions(G_u, t_stage);
        };

        // Initial guess: u^{n+γ} = u^n
        std::copy(u_old_.begin(), u_old_.end(), u_current_.begin());

        // Solve fixed-point problem
        size_t iterations = 0;
        bool converged = fixed_point_solve_vector(
            std::span{u_current_},
            iterate,
            std::span{u_stage_},
            config_.max_iter,
            config_.tolerance,
            config_.omega,
            iterations
        );

        // Boundary conditions applied during iteration

        return converged;
    }

    /// TR-BDF2 Stage 2: BDF2
    ///
    /// Standard TR-BDF2 formulation (Ascher, Ruuth, Wetton 1995):
    /// u^{n+1} - [(1-γ)·dt/(2-γ)]·L(u^{n+1}) = [1/(γ(2-γ))]·u^{n+γ} - [(1-γ)²/(γ(2-γ))]·u^n
    ///
    /// Solved via fixed-point iteration
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
        std::vector<double> rhs(n_);
        for (size_t i = 0; i < n_; ++i) {
            rhs[i] = alpha * u_current_[i] + beta * u_old_[i];
        }

        // Fixed-point iteration: u^{n+1} = rhs + w2·L(u^{n+1})
        auto iterate = [&](std::span<const double> u, std::span<double> G_u) {
            // Compute L(u) → temp_
            spatial_op_(t_next, grid_, u, std::span{temp_}, workspace_.dx());

            // G(u) = rhs + w2·L(u)
            for (size_t i = 0; i < n_; ++i) {
                G_u[i] = rhs[i] + w2 * temp_[i];
            }

            // Apply boundary conditions to the iterate
            apply_boundary_conditions(G_u, t_next);
        };

        // Initial guess: u^{n+1} = u^{n+γ} (already in u_current_)
        // (No need to copy, u_current_ already has u^{n+γ})

        // Solve fixed-point problem
        size_t iterations = 0;
        bool converged = fixed_point_solve_vector(
            std::span{u_current_},
            iterate,
            std::span{u_stage_},
            config_.max_iter,
            config_.tolerance,
            config_.omega,
            iterations
        );

        // Boundary conditions applied during iteration

        return converged;
    }

    /// Compute residual: r = rhs - u + coeff_dt·L(u)
    /// ALL points use PDE formula (Dirichlet will be overwritten later)
    void compute_residual(std::span<const double> u, double coeff_dt,
                          std::span<const double> Lu, std::span<const double> rhs,
                          std::span<double> residual) {
        for (size_t i = 0; i < n_; ++i) {
            residual[i] = rhs[i] - u[i] + coeff_dt * Lu[i];
        }
    }

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

    /// Build Jacobian at boundaries (compile-time dispatch)
    void build_jacobian_boundaries(double t, double coeff_dt,
                                    std::span<const double> u, double eps) {
        // Initialize u_perturb_ and compute baseline L(u)
        // CRITICAL: Lu_ must be computed before finite differences
        std::copy(u.begin(), u.end(), u_perturb_.begin());
        spatial_op_(t, grid_, u, std::span{Lu_}, workspace_.dx());

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
};

}  // namespace mango
