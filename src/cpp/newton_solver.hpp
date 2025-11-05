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
/// Solves nonlinear system: F(u) = u - rhs - coeff_dt·L(u) = 0
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
    /// Equivalently: F(u) = u - rhs - coeff_dt·L(u) = 0
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

    // Helper methods
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

    /// Apply spatial operator with blocking awareness
    void apply_spatial_operator_blocked(double t,
                                       std::span<const double> u,
                                       std::span<double> Lu);
};

// ============================================================================
// Implementation
// ============================================================================

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
        apply_spatial_operator_blocked(t, u, workspace_.lu());

        // Compute residual: F(u) = u - rhs - coeff_dt·L(u)
        compute_residual(u, coeff_dt, workspace_.lu(), rhs,
                       newton_ws_.residual());

        // CRITICAL FIX: Pass u explicitly to avoid reading stale workspace
        apply_bc_to_residual(newton_ws_.residual(), u, t);

        // Newton method: Solve J·δu = -F(u), then update u ← u + δu
        // Negate residual for RHS
        for (size_t i = 0; i < n_; ++i) {
            newton_ws_.residual()[i] = -newton_ws_.residual()[i];
        }

        // Solve J·δu = -F(u) using Thomas algorithm
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

template<typename BoundaryL, typename BoundaryR, typename SpatialOp>
void NewtonSolver<BoundaryL, BoundaryR, SpatialOp>::compute_residual(
    std::span<const double> u, double coeff_dt,
    std::span<const double> Lu,
    std::span<const double> rhs,
    std::span<double> residual)
{
    // F(u) = u - rhs - coeff_dt·L(u) = 0
    // We want to solve u = rhs + coeff_dt·L(u)
    for (size_t i = 0; i < n_; ++i) {
        residual[i] = u[i] - rhs[i] - coeff_dt * Lu[i];
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
    // For Dirichlet BC: F(u) = u - g, so residual = u - g
    // Left boundary
    if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::dirichlet_tag>) {
        double g = left_bc_.value(t, grid_[0]);
        residual[0] = u[0] - g;  // u - g (we want u = g, so F = u - g = 0)
    }

    // Right boundary
    if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::dirichlet_tag>) {
        double g = right_bc_.value(t, grid_[n_ - 1]);
        residual[n_ - 1] = u[n_ - 1] - g;  // u - g
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
    apply_spatial_operator_blocked(t, u, workspace_.lu());

    // Interior points: tridiagonal structure via finite differences
    // J = ∂F/∂u where F(u) = u - rhs - coeff_dt·L(u)
    // So ∂F/∂u = I - coeff_dt·∂L/∂u
    for (size_t i = 1; i < n_ - 1; ++i) {
        // Diagonal: ∂F/∂u_i = 1 - coeff_dt·∂L_i/∂u_i
        newton_ws_.u_perturb()[i] = u[i] + eps;
        apply_spatial_operator_blocked(t, newton_ws_.u_perturb(), newton_ws_.Lu_perturb());
        double dLi_dui = (newton_ws_.Lu_perturb()[i] - workspace_.lu()[i]) / eps;
        newton_ws_.jacobian_diag()[i] = 1.0 - coeff_dt * dLi_dui;
        newton_ws_.u_perturb()[i] = u[i];

        // Lower diagonal: ∂F_i/∂u_{i-1} = -coeff_dt·∂L_i/∂u_{i-1}
        newton_ws_.u_perturb()[i - 1] = u[i - 1] + eps;
        apply_spatial_operator_blocked(t, newton_ws_.u_perturb(), newton_ws_.Lu_perturb());
        double dLi_duim1 = (newton_ws_.Lu_perturb()[i] - workspace_.lu()[i]) / eps;
        newton_ws_.jacobian_lower()[i - 1] = -coeff_dt * dLi_duim1;
        newton_ws_.u_perturb()[i - 1] = u[i - 1];

        // Upper diagonal: ∂F_i/∂u_{i+1} = -coeff_dt·∂L_i/∂u_{i+1}
        newton_ws_.u_perturb()[i + 1] = u[i + 1] + eps;
        apply_spatial_operator_blocked(t, newton_ws_.u_perturb(), newton_ws_.Lu_perturb());
        double dLi_duip1 = (newton_ws_.Lu_perturb()[i] - workspace_.lu()[i]) / eps;
        newton_ws_.jacobian_upper()[i] = -coeff_dt * dLi_duip1;
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
        // For Dirichlet: F(u) = u - g, so ∂F/∂u = 1
        newton_ws_.jacobian_diag()[0] = 1.0;
        newton_ws_.jacobian_upper()[0] = 0.0;
    } else if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryL>, bc::neumann_tag>) {
        // For Neumann: F(u) = u - rhs - coeff_dt·L(u)
        newton_ws_.u_perturb()[0] = u[0] + eps;
        apply_spatial_operator_blocked(t, newton_ws_.u_perturb(), newton_ws_.Lu_perturb());
        double dL0_du0 = (newton_ws_.Lu_perturb()[0] - workspace_.lu()[0]) / eps;
        newton_ws_.jacobian_diag()[0] = 1.0 - coeff_dt * dL0_du0;
        newton_ws_.u_perturb()[0] = u[0];

        newton_ws_.u_perturb()[1] = u[1] + eps;
        apply_spatial_operator_blocked(t, newton_ws_.u_perturb(), newton_ws_.Lu_perturb());
        double dL0_du1 = (newton_ws_.Lu_perturb()[0] - workspace_.lu()[0]) / eps;
        newton_ws_.jacobian_upper()[0] = -coeff_dt * dL0_du1;
        newton_ws_.u_perturb()[1] = u[1];
    }

    // Right boundary
    if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::dirichlet_tag>) {
        // For Dirichlet: F(u) = u - g, so ∂F/∂u = 1
        newton_ws_.jacobian_diag()[n_ - 1] = 1.0;
    } else if constexpr (std::is_same_v<bc::boundary_tag_t<BoundaryR>, bc::neumann_tag>) {
        // For Neumann: F(u) = u - rhs - coeff_dt·L(u)
        size_t i = n_ - 1;
        newton_ws_.u_perturb()[i] = u[i] + eps;
        apply_spatial_operator_blocked(t, newton_ws_.u_perturb(), newton_ws_.Lu_perturb());
        double dLi_dui = (newton_ws_.Lu_perturb()[i] - workspace_.lu()[i]) / eps;
        newton_ws_.jacobian_diag()[i] = 1.0 - coeff_dt * dLi_dui;
        newton_ws_.u_perturb()[i] = u[i];
    }
}

template<typename BoundaryL, typename BoundaryR, typename SpatialOp>
void NewtonSolver<BoundaryL, BoundaryR, SpatialOp>::apply_spatial_operator_blocked(
    double t,
    std::span<const double> u,
    std::span<double> Lu)
{
    const size_t n = grid_.size();

    // Check if blocking is enabled
    if (workspace_.cache_config().n_blocks == 1) {
        // Full-array path
        spatial_op_(t, grid_, u, Lu, workspace_.dx());
        return;
    }

    // Blocked path
    for (size_t block = 0; block < workspace_.cache_config().n_blocks; ++block) {
        auto [interior_start, interior_end] =
            workspace_.get_block_interior_range(block);

        if (interior_start >= interior_end) continue;

        const size_t halo_left = std::min(workspace_.cache_config().overlap,
                                         interior_start);
        const size_t halo_right = std::min(workspace_.cache_config().overlap,
                                          n - interior_end);
        const size_t interior_count = interior_end - interior_start;

        auto x_halo = std::span{grid_.data() + interior_start - halo_left,
                               interior_count + halo_left + halo_right};
        auto u_halo = std::span{u.data() + interior_start - halo_left,
                               interior_count + halo_left + halo_right};
        auto lu_out = std::span{Lu.data() + interior_start, interior_count};

        spatial_op_.apply_block(t, interior_start, halo_left, halo_right,
                               x_halo, u_halo, lu_out, workspace_.dx());
    }

    Lu[0] = Lu[n-1] = 0.0;
}

}  // namespace mango
