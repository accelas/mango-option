#pragma once

#include "grid.hpp"
#include "workspace.hpp"
#include "boundary_conditions.hpp"
#include "time_domain.hpp"
#include "trbdf2_config.hpp"
#include "tridiagonal_solver.hpp"
#include "newton_solver.hpp"
#include "root_finding.hpp"
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
    /// @param root_config Root-finding configuration for Newton solver
    /// @param left_bc Left boundary condition
    /// @param right_bc Right boundary condition
    /// @param spatial_op Spatial operator L(u)
    PDESolver(std::span<const double> grid,
              const TimeDomain& time,
              const TRBDF2Config& config,
              const RootFindingConfig& root_config,
              const BoundaryL& left_bc,
              const BoundaryR& right_bc,
              const SpatialOp& spatial_op)
        : grid_(grid)
        , time_(time)
        , config_(config)
        , root_config_(root_config)
        , left_bc_(left_bc)
        , right_bc_(right_bc)
        , spatial_op_(spatial_op)
        , n_(grid.size())
        , workspace_(n_, grid, config_.cache_blocking_threshold)
        , u_current_(n_)
        , u_old_(n_)
        , rhs_(n_)
        , newton_solver_(n_, root_config_, workspace_, left_bc_, right_bc_, spatial_op_, grid)
    {
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
    RootFindingConfig root_config_;
    BoundaryL left_bc_;
    BoundaryR right_bc_;
    SpatialOp spatial_op_;

    // Grid size
    size_t n_;

    // Workspace for cache blocking
    WorkspaceStorage workspace_;

    // Solution storage
    std::vector<double> u_current_;  // u^{n+1} or u^{n+γ}
    std::vector<double> u_old_;      // u^n
    std::vector<double> rhs_;        // n: RHS vector for stages

    // Newton solver (persistent, created once)
    NewtonSolver<BoundaryL, BoundaryR, SpatialOp> newton_solver_;

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

    /// Apply spatial operator with cache-blocking for large grids
    void apply_operator_with_blocking(double t,
                                      std::span<const double> u,
                                      std::span<double> Lu) {
        const size_t n = grid_.size();

        // Small grid: use full-array path (no blocking overhead)
        if (workspace_.cache_config().n_blocks == 1) {
            spatial_op_(t, grid_, u, Lu, workspace_.dx());
            return;
        }

        // Large grid: blocked evaluation
        for (size_t block = 0; block < workspace_.cache_config().n_blocks; ++block) {
            auto [interior_start, interior_end] =
                workspace_.get_block_interior_range(block);

            // Skip boundary-only blocks
            if (interior_start >= interior_end) continue;

            // Compute halo sizes (clamped at global boundaries)
            const size_t halo_left = std::min(workspace_.cache_config().overlap,
                                             interior_start);
            const size_t halo_right = std::min(workspace_.cache_config().overlap,
                                              n - interior_end);
            const size_t interior_count = interior_end - interior_start;

            // Build spans with halos
            auto x_halo = std::span{grid_.data() + interior_start - halo_left,
                                   interior_count + halo_left + halo_right};
            auto u_halo = std::span{u.data() + interior_start - halo_left,
                                   interior_count + halo_left + halo_right};
            auto lu_out = std::span{Lu.data() + interior_start, interior_count};

            // Call block-aware operator
            spatial_op_.apply_block(t, interior_start, halo_left, halo_right,
                                   x_halo, u_halo, lu_out, workspace_.dx());
        }

        // Zero boundary values (BCs will override after)
        Lu[0] = Lu[n-1] = 0.0;
    }

    /// TR-BDF2 Stage 1: Trapezoidal rule
    ///
    /// u^{n+γ} = u^n + (γ·dt/2) · [L(u^n) + L(u^{n+γ})]
    ///
    /// Solved via Newton-Raphson iteration
    bool solve_stage1(double t_n, double t_stage, double dt) {
        const double w1 = config_.stage1_weight(dt);  // γ·dt/2

        // Compute L(u^n)
        apply_operator_with_blocking(t_n, std::span{u_old_}, workspace_.lu());

        // RHS = u^n + w1·L(u^n)
        for (size_t i = 0; i < n_; ++i) {
            rhs_[i] = u_old_[i] + w1 * workspace_.lu()[i];
        }

        // Initial guess: u* = u^n
        std::copy(u_old_.begin(), u_old_.end(), u_current_.begin());

        // Use NewtonSolver
        auto result = newton_solver_.solve(t_stage, w1,
                                          std::span{u_current_},
                                          std::span{rhs_});

        return result.converged;
    }

    /// TR-BDF2 Stage 2: BDF2
    ///
    /// Standard TR-BDF2 formulation (Ascher, Ruuth, Wetton 1995):
    /// u^{n+1} - [(1-γ)·dt/(2-γ)]·L(u^{n+1}) = [1/(γ(2-γ))]·u^{n+γ} - [(1-γ)²/(γ(2-γ))]·u^n
    ///
    /// Solved via Newton-Raphson iteration
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

        // Use NewtonSolver
        auto result = newton_solver_.solve(t_next, w2,
                                          std::span{u_current_},
                                          std::span{rhs_});

        return result.converged;
    }
};

}  // namespace mango
