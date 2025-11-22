/**
 * @file american_option.cpp
 * @brief American option pricing solver implementation
 */

#include "src/option/american_option.hpp"
#include "src/option/american_pde_solver.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <format>

namespace mango {

// Note: estimate_grid_for_option() is now defined in american_option.hpp

// Constructor with PDEWorkspace
AmericanOptionSolver::AmericanOptionSolver(
    const PricingParams& params,
    PDEWorkspace workspace,
    std::optional<std::span<const double>> snapshot_times,
    std::optional<GridSpec<double>> custom_grid,
    std::optional<size_t> custom_n_time)
    : params_(params)
    , workspace_(workspace)
    , custom_grid_(custom_grid)
    , custom_n_time_(custom_n_time)
{
    // Store snapshot times if provided
    if (snapshot_times.has_value()) {
        snapshot_times_.assign(snapshot_times->begin(), snapshot_times->end());
    }

    // Validate parameters
    auto validation = validate_pricing_params(params_);
    if (!validation) {
        throw std::invalid_argument(validation.error());
    }
}

// ============================================================================
// Public API
// ============================================================================

std::expected<AmericanOptionResult, SolverError> AmericanOptionSolver::solve() {
    // Use custom grid if provided, otherwise estimate from params
    auto [grid_spec, n_time] = [this]() -> std::tuple<GridSpec<double>, size_t> {
        if (custom_grid_.has_value() && custom_n_time_.has_value()) {
            // Use custom grid configuration (for benchmarking / manual grid control)
            return {custom_grid_.value(), custom_n_time_.value()};
        } else {
            // Auto-estimate grid from option parameters
            return estimate_grid_for_option(params_);
        }
    }();

    TimeDomain time_domain = TimeDomain::from_n_steps(0.0, params_.maturity, n_time);

    // Validate workspace size matches grid
    if (workspace_.size() != grid_spec.n_points()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .message = std::format(
                "Workspace size mismatch: workspace has {} points, grid needs {}",
                workspace_.size(), grid_spec.n_points()),
            .iterations = 0
        });
    }

    // Create Grid with optional snapshots
    auto grid_result = Grid<double>::create(
        grid_spec, time_domain,
        snapshot_times_.empty() ? std::span<const double>() : std::span<const double>(snapshot_times_));

    if (!grid_result.has_value()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .message = std::format("Failed to create Grid: {}", grid_result.error()),
            .iterations = 0
        });
    }
    auto grid = grid_result.value();

    // Initialize dx in workspace from grid spacing
    auto dx_span = workspace_.dx();
    auto grid_points = grid->x();
    for (size_t i = 0; i < grid_points.size() - 1; ++i) {
        dx_span[i] = grid_points[i + 1] - grid_points[i];
    }

    // Create appropriate PDE solver (put vs call)
    std::expected<void, SolverError> solve_result;

    if (params_.type == OptionType::PUT) {
        AmericanPutSolver pde_solver(params_, grid, workspace_);
        pde_solver.initialize(AmericanPutSolver::payoff);
        solve_result = pde_solver.solve();
    } else {
        AmericanCallSolver pde_solver(params_, grid, workspace_);
        pde_solver.initialize(AmericanCallSolver::payoff);
        solve_result = pde_solver.solve();
    }

    if (!solve_result.has_value()) {
        return std::unexpected(solve_result.error());
    }

    // Return wrapper (Grid + PricingParams)
    return AmericanOptionResult(grid, params_);
}


}  // namespace mango
