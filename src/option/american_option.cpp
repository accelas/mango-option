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

// Constructor with PDEWorkspace
AmericanOptionSolver::AmericanOptionSolver(
    const PricingParams& params,
    PDEWorkspace workspace,
    std::optional<std::span<const double>> snapshot_times)
    : params_(params)
    , workspace_(workspace)
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

// DEPRECATED: Constructor with AmericanSolverWorkspace (for backward compatibility)
AmericanOptionSolver::AmericanOptionSolver(
    const AmericanOptionParams& params,
    std::shared_ptr<AmericanSolverWorkspace> workspace)
    : params_(params)
    , legacy_workspace_(std::move(workspace))
{
    // Validate parameters
    auto validation = validate_pricing_params(params_);
    if (!validation) {
        throw std::invalid_argument(validation.error());
    }

    // Validate workspace is not null
    if (!legacy_workspace_) {
        throw std::invalid_argument("Workspace cannot be null");
    }
}

// DEPRECATED: Factory method (for backward compatibility)
std::expected<AmericanOptionSolver, std::string> AmericanOptionSolver::create(
    const AmericanOptionParams& params,
    std::shared_ptr<AmericanSolverWorkspace> workspace) {

    // Validate workspace first
    if (!workspace) {
        return std::unexpected("Workspace cannot be null");
    }

    // Chain validation and construction using monadic operations
    return validate_pricing_params(params)
        .and_then([&]() -> std::expected<AmericanOptionSolver, std::string> {
            try {
                // Suppress deprecation warning - this is the deprecated factory method
                #pragma GCC diagnostic push
                #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
                return AmericanOptionSolver(params, workspace);
                #pragma GCC diagnostic pop
            } catch (const std::exception& e) {
                return std::unexpected(std::string("Failed to create solver: ") + e.what());
            }
        });
}

// ============================================================================
// Public API
// ============================================================================

std::expected<AmericanOptionResult, SolverError> AmericanOptionSolver::solve() {
    // Branch based on which API was used
    if (using_new_api()) {
        // NEW API: Create Grid, initialize dx, solve PDE, return wrapper

        // Estimate grid configuration from params
        auto [grid_spec, n_time] = estimate_grid_for_option(params_);
        TimeDomain time_domain = TimeDomain::from_n_steps(0.0, params_.maturity, n_time);

        // Validate workspace size matches estimated grid
        if (workspace_->size() != grid_spec.n_points()) {
            return std::unexpected(SolverError{
                .code = SolverErrorCode::InvalidConfiguration,
                .message = std::format(
                    "Workspace size mismatch: workspace has {} points, grid needs {}",
                    workspace_->size(), grid_spec.n_points()),
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
        auto dx_span = workspace_->dx();
        auto grid_points = grid->x();
        for (size_t i = 0; i < grid_points.size() - 1; ++i) {
            dx_span[i] = grid_points[i + 1] - grid_points[i];
        }

        // Create appropriate PDE solver (put vs call)
        std::expected<void, SolverError> solve_result;

        if (params_.type == OptionType::PUT) {
            AmericanPutSolver pde_solver(params_, grid, workspace_.value());
            pde_solver.initialize(AmericanPutSolver::payoff);
            solve_result = pde_solver.solve();
        } else {
            AmericanCallSolver pde_solver(params_, grid, workspace_.value());
            pde_solver.initialize(AmericanCallSolver::payoff);
            solve_result = pde_solver.solve();
        }

        if (!solve_result.has_value()) {
            return std::unexpected(solve_result.error());
        }

        // Return wrapper (Grid + PricingParams)
        return AmericanOptionResult(grid, params_);

    } else {
        // DEPRECATED API: Use legacy workspace

        // Create appropriate solver based on option type using legacy API
        AmericanSolverVariant solver = [&]() -> AmericanSolverVariant {
            switch (params_.type) {
                case OptionType::CALL:
                    return AmericanCallSolver(params_,
                                             legacy_workspace_->grid_with_solution(),
                                             legacy_workspace_->workspace_spans());
                case OptionType::PUT:
                    return AmericanPutSolver(params_,
                                            legacy_workspace_->grid_with_solution(),
                                            legacy_workspace_->workspace_spans());
                default:
                    throw std::runtime_error("Unknown option type");
            }
        }();

        // Initialize with payoff at maturity (t=0 in PDE time)
        std::visit([&](auto& s) {
            using SolverType = std::decay_t<decltype(s)>;
            s.initialize(SolverType::payoff);
        }, solver);

        // Solve using variant dispatch (static, zero-cost)
        auto solve_result = std::visit([](auto& s) { return s.solve(); }, solver);
        if (!solve_result) {
            return std::unexpected(solve_result.error());
        }

        // Return wrapper (convert from legacy workspace)
        return AmericanOptionResult(legacy_workspace_->grid_with_solution(), params_);
    }
}


}  // namespace mango
