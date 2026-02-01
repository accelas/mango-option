// SPDX-License-Identifier: MIT
/**
 * @file american_option.cpp
 * @brief American option pricing solver implementation
 */

#include "src/option/american_option.hpp"
#include "src/option/american_pde_solver.hpp"
#include "src/option/discrete_dividend_event.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <format>

namespace mango {

// Note: estimate_grid_for_option() is now defined in american_option.hpp

// Constructor with PDEWorkspace
// Factory method (noexcept, returns std::expected)
std::expected<AmericanOptionSolver, ValidationError>
AmericanOptionSolver::create(
    const PricingParams& params,
    PDEWorkspace workspace,
    std::optional<std::span<const double>> snapshot_times,
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid_config) noexcept
{
    // Validate parameters first
    auto validation = validate_pricing_params(params);
    if (!validation) {
        return std::unexpected(validation.error());
    }

    // Construct (all validation done, cannot fail)
    return AmericanOptionSolver(params, workspace, snapshot_times, custom_grid_config);
}

// Constructor (throws for backward compatibility)
AmericanOptionSolver::AmericanOptionSolver(
    const PricingParams& params,
    PDEWorkspace workspace,
    std::optional<std::span<const double>> snapshot_times,
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid_config)
    : params_(params)
    , workspace_(workspace)
    , custom_grid_config_(custom_grid_config)
{
    trbdf2_config_.rannacher_startup = true;

    // Store snapshot times if provided
    if (snapshot_times.has_value()) {
        snapshot_times_.assign(snapshot_times->begin(), snapshot_times->end());
    }

    // Validate parameters
    auto validation = validate_pricing_params(params_);
    if (!validation) {
        auto err = validation.error();
        throw std::invalid_argument(
            "Validation error code " + std::to_string(static_cast<int>(err.code)) +
            " (value=" + std::to_string(err.value) + ")");
    }
}

// ============================================================================
// Public API
// ============================================================================

std::expected<AmericanOptionResult, SolverError> AmericanOptionSolver::solve() {
    // Use custom grid config if provided, otherwise estimate from params
    auto [grid_spec, time_domain] = custom_grid_config_.has_value()
        ? custom_grid_config_.value()
        : estimate_grid_for_option(params_);

    // Validate workspace size matches grid
    if (workspace_.size() != grid_spec.n_points()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .iterations = 0,
            .residual = static_cast<double>(workspace_.size())  // Store actual workspace size
        });
    }

    // Create Grid with optional snapshots
    auto grid_result = Grid<double>::create(
        grid_spec, time_domain,
        snapshot_times_.empty() ? std::span<const double>() : std::span<const double>(snapshot_times_));

    if (!grid_result.has_value()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .iterations = 0,
            .residual = grid_result.error().value  // Store the error value from ValidationError
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
        pde_solver.set_config(trbdf2_config_);

        // Register discrete dividend events
        for (const auto& [t_cal, amount] : params_.discrete_dividends) {
            double tau = params_.maturity - t_cal;
            if (tau > 0.0 && tau < params_.maturity) {
                pde_solver.add_temporal_event(tau,
                    make_dividend_event(amount, params_.strike, params_.type));
            }
        }

        solve_result = pde_solver.solve();
    } else {
        AmericanCallSolver pde_solver(params_, grid, workspace_);
        pde_solver.initialize(AmericanCallSolver::payoff);
        pde_solver.set_config(trbdf2_config_);

        for (const auto& [t_cal, amount] : params_.discrete_dividends) {
            double tau = params_.maturity - t_cal;
            if (tau > 0.0 && tau < params_.maturity) {
                pde_solver.add_temporal_event(tau,
                    make_dividend_event(amount, params_.strike, params_.type));
            }
        }

        solve_result = pde_solver.solve();
    }

    if (!solve_result.has_value()) {
        return std::unexpected(solve_result.error());
    }

    // Return wrapper (Grid + PricingParams)
    return AmericanOptionResult(grid, params_);
}


}  // namespace mango
