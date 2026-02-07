// SPDX-License-Identifier: MIT
/**
 * @file american_option.hpp
 * @brief American option pricing solver using finite difference method
 */

#ifndef MANGO_AMERICAN_OPTION_HPP
#define MANGO_AMERICAN_OPTION_HPP

#include "mango/pde/core/pde_solver.hpp"
#include "mango/pde/operators/black_scholes_pde.hpp"
#include "mango/option/option_concepts.hpp"
#include <expected>
#include "mango/support/error_types.hpp"
#include "mango/support/parallel.hpp"
#include "mango/option/american_option_result.hpp"
#include "mango/option/option_spec.hpp"  // For OptionType enum
#include "mango/option/grid_spec_types.hpp"
#include "mango/pde/core/pde_workspace.hpp"
#include <vector>
#include <memory>
#include <memory_resource>
#include <cmath>
#include <functional>
#include <optional>

namespace mango {

/**
 * American option pricing solver using finite difference method.
 *
 * Solves the Black-Scholes PDE with obstacle constraints in log-moneyness
 * coordinates using TR-BDF2 time stepping and projection method for
 * early exercise boundary.
 */
class AmericanOptionSolver {
public:
    /**
     * Factory method to create AmericanOptionSolver with validation.
     *
     * Validates option parameters before construction, providing type-safe
     * error handling via std::expected. Use this instead of the constructor.
     *
     * @param params Option pricing parameters
     * @param workspace PDEWorkspace with pre-allocated buffers
     * @param grid Optional grid specification (GridAccuracyParams or PDEGridConfig).
     *             When nullopt, auto-estimates from option parameters.
     * @param snapshot_times Optional times to record solution snapshots
     * @return AmericanOptionSolver on success, ValidationError on failure
     */
    static std::expected<AmericanOptionSolver, ValidationError>
    create(const PricingParams& params,
           PDEWorkspace workspace,
           std::optional<PDEGridSpec> grid = std::nullopt,
           std::optional<std::span<const double>> snapshot_times = std::nullopt);

    /**
     * Set snapshot times for solution recording.
     *
     * Must be called before solve(). Allows setup callbacks to register
     * snapshots for price table construction.
     *
     * @param times Snapshot times (must be in [0, maturity])
     */
    void set_snapshot_times(std::span<const double> times) {
        snapshot_times_.assign(times.begin(), times.end());
    }

    /// Set TR-BDF2 configuration (e.g., enable Rannacher startup)
    void set_trbdf2_config(const TRBDF2Config& config) {
        trbdf2_config_ = config;
    }

    /**
     * Solve for option value.
     *
     * Returns AmericanOptionResult wrapper containing Grid and PricingParams.
     * If snapshot_times were provided at construction, the Grid will contain
     * recorded solution snapshots.
     *
     * @return Result wrapper with value(), Greeks, and snapshot access
     */
    std::expected<AmericanOptionResult, SolverError> solve();

private:
    AmericanOptionSolver(const PricingParams& params,
                        PDEWorkspace workspace,
                        std::pair<GridSpec<double>, TimeDomain> grid_config,
                        std::optional<std::span<const double>> snapshot_times = std::nullopt);

    // Parameters
    PricingParams params_;

    // PDEWorkspace (owns spans to external buffer)
    PDEWorkspace workspace_;

    // Snapshot times for Grid creation
    std::vector<double> snapshot_times_;

    // Resolved grid configuration (GridSpec + TimeDomain)
    // Always resolved at create() time — never empty
    std::pair<GridSpec<double>, TimeDomain> grid_config_;

    // TR-BDF2 configuration for the PDE solver
    TRBDF2Config trbdf2_config_;

public:
    /// Callable type for custom initial conditions: f(x, u) fills u given grid points x
    using InitialCondition = std::function<void(std::span<const double>, std::span<double>)>;

    /// Set a custom initial condition (overrides the standard payoff)
    void set_initial_condition(InitialCondition ic) { custom_ic_ = std::move(ic); }

    /// Disable obstacle projection for European PDE solve.
    void set_projection_enabled(bool enabled) { projection_enabled_ = enabled; }
    bool projection_enabled() const { return projection_enabled_; }

private:
    /// Optional custom initial condition (replaces default payoff when set)
    std::optional<InitialCondition> custom_ic_;

    /// Whether obstacle projection is enabled (true = American, false = European)
    bool projection_enabled_ = true;
};

static_assert(OptionSolver<AmericanOptionSolver>);

/// Solve a single American option with automatic grid determination
///
/// Convenience API that automatically determines optimal grid parameters
/// based on option characteristics, eliminating need for manual grid specification.
///
/// Note: Allocates temporary workspace buffer (discarded after solve).
/// For reusable workspaces, caller should manage buffer and use PDEWorkspace directly.
///
/// @param params Option parameters
/// @return Expected containing result on success, error on failure
inline std::expected<AmericanOptionResult, SolverError> solve_american_option(
    const PricingParams& params)
{
    // Estimate grid for this option
    auto [grid_spec, time_domain] = estimate_pde_grid(params);

    // Allocate workspace buffer (local, temporary)
    size_t n = grid_spec.n_points();
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), std::pmr::get_default_resource());

    // Create workspace spans from buffer
    auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
    if (!workspace_result.has_value()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            // error code set above + workspace_result.error(),
            .iterations = 0
        });
    }

    // Collect mandatory tau values for discrete dividends
    std::vector<double> mandatory_tau;
    for (const auto& div : params.discrete_dividends) {
        double tau = params.maturity - div.calendar_time;
        if (tau > 0.0 && tau < params.maturity) {
            mandatory_tau.push_back(tau);
        }
    }

    // Create and solve using PDEWorkspace API
    // Buffer stays alive during solve(), result contains Grid with solution
    auto solver_result = AmericanOptionSolver::create(
        params, workspace_result.value(),
        PDEGridConfig{.grid_spec = grid_spec, .n_time = time_domain.n_steps(),
                        .mandatory_times = std::move(mandatory_tau)});
    if (!solver_result) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .iterations = 0
        });
    }
    return solver_result.value().solve();
}

}  // namespace mango

#endif  // MANGO_AMERICAN_OPTION_HPP
