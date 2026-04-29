// SPDX-License-Identifier: MIT
/**
 * @file american_option.hpp
 * @brief American option pricing solver using finite difference method
 */

#ifndef MANGO_AMERICAN_OPTION_HPP
#define MANGO_AMERICAN_OPTION_HPP

#include "mango/pde/core/trbdf2_config.hpp"
#include "mango/option/option_concepts.hpp"
#include <expected>
#include "mango/support/error_types.hpp"
#include "mango/support/parallel.hpp"
#include "mango/option/american_option_result.hpp"
#include "mango/option/option_spec.hpp"  // For OptionType enum
#include "mango/option/grid_spec_types.hpp"
#include <vector>
#include <memory>
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
     * Create solver with auto-managed scratch buffer.
     *
     * Internally uses a thread-local PMR arena (~270 KB per thread)
     * for typical grid sizes; falls back to heap for n > 2048.
     *
     * @param params Option pricing parameters
     * @param grid Optional grid specification. When nullopt, auto-estimates
     *             from option parameters.
     * @param snapshot_times Optional times to record solution snapshots
     * @return AmericanOptionSolver on success, ValidationError on failure
     */
    static std::expected<AmericanOptionSolver, ValidationError>
    create(const PricingParams& params,
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
                        std::pair<GridSpec<double>, TimeDomain> grid_config,
                        std::optional<std::span<const double>> snapshot_times = std::nullopt);

    // Parameters
    PricingParams params_;

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

private:
    /// Optional custom initial condition (replaces default payoff when set)
    std::optional<InitialCondition> custom_ic_;
};

static_assert(OptionSolver<AmericanOptionSolver>);

/// Solve a single American option with automatic grid determination.
/// Convenience wrapper around AmericanOptionSolver::create + solve.
std::expected<AmericanOptionResult, SolverError>
solve_american_option(const PricingParams& params);

}  // namespace mango

#endif  // MANGO_AMERICAN_OPTION_HPP
