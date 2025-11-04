#pragma once

#include "grid.hpp"
#include "workspace.hpp"
#include "boundary_conditions.hpp"
#include "time_domain.hpp"
#include "trbdf2_config.hpp"
#include "fixed_point_solver.hpp"
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
};

}  // namespace mango
