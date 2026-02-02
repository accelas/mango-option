// SPDX-License-Identifier: MIT
/**
 * @file american_option.cpp
 * @brief American option pricing solver implementation
 */

#include "src/option/american_option.hpp"
#include "src/option/american_pde_solver.hpp"
#include "src/option/discrete_dividend_event.hpp"
#include "src/math/cubic_spline_solver.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <format>

namespace mango {

// Note: estimate_grid_for_option() is now defined in american_option.hpp

// Helper for std::visit with multiple lambdas
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };

// Resolve grid specification to concrete GridSpec + TimeDomain
static std::pair<GridSpec<double>, TimeDomain> resolve_grid(
    const PricingParams& params,
    const std::optional<PDEGridSpec>& grid)
{
    if (!grid.has_value()) {
        return estimate_grid_for_option(params);
    }
    return std::visit(overloaded{
        [&](const GridAccuracyParams& acc) {
            return estimate_grid_for_option(params, acc);
        },
        [&](const PDEGridConfig& eg) {
            auto td = eg.mandatory_times.empty()
                ? TimeDomain::from_n_steps(0.0, params.maturity, eg.n_time)
                : TimeDomain::with_mandatory_points(0.0, params.maturity,
                    params.maturity / static_cast<double>(eg.n_time), eg.mandatory_times);
            return std::make_pair(eg.grid_spec, td);
        }
    }, *grid);
}

// Factory method (returns std::expected)
std::expected<AmericanOptionSolver, ValidationError>
AmericanOptionSolver::create(
    const PricingParams& params,
    PDEWorkspace workspace,
    std::optional<PDEGridSpec> grid,
    std::optional<std::span<const double>> snapshot_times)
{
    // Validate parameters first
    auto validation = validate_pricing_params(params);
    if (!validation) {
        return std::unexpected(validation.error());
    }

    // Resolve grid specification to concrete GridSpec + TimeDomain
    auto grid_config = resolve_grid(params, grid);

    // Validate workspace size matches resolved grid
    if (workspace.size() != grid_config.first.n_points()) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidGridSize,
            static_cast<double>(workspace.size()),
            grid_config.first.n_points()));
    }

    // Construct (all validation done, cannot fail)
    return AmericanOptionSolver(params, workspace, std::move(grid_config), snapshot_times);
}

AmericanOptionSolver::AmericanOptionSolver(
    const PricingParams& params,
    PDEWorkspace workspace,
    std::pair<GridSpec<double>, TimeDomain> grid_config,
    std::optional<std::span<const double>> snapshot_times)
    : params_(params)
    , workspace_(workspace)
    , grid_config_(std::move(grid_config))
{
    trbdf2_config_.rannacher_startup = true;

    // Store snapshot times if provided
    if (snapshot_times.has_value()) {
        snapshot_times_.assign(snapshot_times->begin(), snapshot_times->end());
    }
}

// ============================================================================
// Public API
// ============================================================================

std::expected<AmericanOptionResult, SolverError> AmericanOptionSolver::solve() {
    // Grid is always resolved and validated at create() time
    auto& [grid_spec, time_domain] = grid_config_;

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

    // Pre-allocate spline for dividend events (zero-alloc after first build).
    // Use a workspace scratch buffer as dummy y-data to avoid a heap allocation.
    CubicSpline<double> dividend_spline;
    if (!params_.discrete_dividends.empty()) {
        auto x = grid->x();
        auto scratch = workspace_.reserved1();
        std::fill(scratch.begin(), scratch.end(), 0.0);
        [[maybe_unused]] auto err = dividend_spline.build(
            x, std::span<const double>(scratch.data(), x.size()));
    }

    // Create appropriate PDE solver (put vs call)
    std::expected<void, SolverError> solve_result;

    if (params_.option_type == OptionType::PUT) {
        AmericanPutSolver pde_solver(params_, grid, workspace_);
        if (custom_ic_) {
            pde_solver.initialize(*custom_ic_);
        } else {
            pde_solver.initialize(AmericanPutSolver::payoff);
        }
        pde_solver.set_config(trbdf2_config_);

        // Register discrete dividend events
        for (const auto& div : params_.discrete_dividends) {
            double tau = params_.maturity - div.calendar_time;
            if (tau > 0.0 && tau < params_.maturity) {
                pde_solver.add_temporal_event(tau,
                    make_put_dividend_event(div.amount, params_.strike, &dividend_spline));
            }
        }

        solve_result = pde_solver.solve();
    } else {
        AmericanCallSolver pde_solver(params_, grid, workspace_);
        if (custom_ic_) {
            pde_solver.initialize(*custom_ic_);
        } else {
            pde_solver.initialize(AmericanCallSolver::payoff);
        }
        pde_solver.set_config(trbdf2_config_);

        for (const auto& div : params_.discrete_dividends) {
            double tau = params_.maturity - div.calendar_time;
            if (tau > 0.0 && tau < params_.maturity) {
                pde_solver.add_temporal_event(tau,
                    make_call_dividend_event(div.amount, params_.strike, &dividend_spline));
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
