// SPDX-License-Identifier: MIT
/**
 * @file american_option.hpp
 * @brief American option pricing solver using finite difference method
 */

#ifndef MANGO_AMERICAN_OPTION_HPP
#define MANGO_AMERICAN_OPTION_HPP

#include "src/pde/core/pde_solver.hpp"
#include "src/pde/operators/black_scholes_pde.hpp"
#include "src/pde/operators/centered_difference_facade.hpp"
#include "src/option/option_concepts.hpp"
#include <expected>
#include "src/support/error_types.hpp"
#include "src/support/parallel.hpp"
#include "src/option/american_option_result.hpp"
#include "src/option/option_spec.hpp"  // For OptionType enum
#include "src/option/grid_spec_types.hpp"
#include "src/pde/core/pde_workspace.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <functional>
#include <optional>

namespace mango {

/**
 * Estimate grid specification from option parameters.
 *
 * Automatically determines appropriate spatial/temporal discretization
 * based on option characteristics (volatility, maturity, moneyness).
 *
 * @param params Option pricing parameters
 * @param accuracy Grid accuracy parameters (optional)
 * @return Pair of (GridSpec, TimeDomain) for consistent discretization
 */
inline std::pair<GridSpec<double>, TimeDomain> estimate_grid_for_option(
    const PricingParams& params,
    const GridAccuracyParams& accuracy = GridAccuracyParams{})
{
    // Domain bounds (centered on current moneyness)
    double sigma_sqrt_T = params.volatility * std::sqrt(params.maturity);
    double x0 = std::log(params.spot / params.strike);

    double x_min = x0 - accuracy.n_sigma * sigma_sqrt_T;
    double x_max = x0 + accuracy.n_sigma * sigma_sqrt_T;

    // Spatial resolution (target truncation error)
    double dx_target = params.volatility * std::sqrt(accuracy.tol);
    size_t Nx = static_cast<size_t>(std::ceil((x_max - x_min) / dx_target));
    Nx = std::clamp(Nx, accuracy.min_spatial_points, accuracy.max_spatial_points);

    // Ensure odd number of points (for centered stencils)
    if (Nx % 2 == 0) Nx++;

    // Temporal resolution (coupled to smallest spatial spacing)
    // For sinh grid with clustering α, dx_min ≈ dx_avg · exp(-α)
    double dx_avg = (x_max - x_min) / static_cast<double>(Nx);
    double dx_min = dx_avg * std::exp(-accuracy.alpha);  // Sinh clustering factor

    double dt = accuracy.c_t * dx_min;
    size_t Nt = static_cast<size_t>(std::ceil(params.maturity / dt));
    Nt = std::min(Nt, accuracy.max_time_steps);  // Upper bound for stability

    // Widen grid for dividend shift: spline evaluates at x'=ln(exp(x)-D/K)
    // Only consider dividends strictly within (0, T) — same filter as mandatory tau
    double max_d_over_k = 0.0;
    for (const auto& [t_cal, amount] : params.discrete_dividends) {
        double tau = params.maturity - t_cal;
        if (tau > 0.0 && tau < params.maturity) {
            max_d_over_k = std::max(max_d_over_k, amount / params.strike);
        }
    }
    if (max_d_over_k > 0.0 && std::exp(x_min) > max_d_over_k) {
        x_min = std::log(std::exp(x_min) - max_d_over_k);
    } else if (max_d_over_k > 0.0) {
        x_min -= 1.0;  // conservative extension
    }

    // Create sinh-spaced GridSpec for better resolution near strike (x=0 in log-moneyness)
    auto grid_spec = GridSpec<double>::sinh_spaced(x_min, x_max, Nx, accuracy.alpha);

    // Convert discrete dividend calendar times to time-to-expiry (tau)
    std::vector<double> mandatory_tau;
    for (const auto& [t_cal, amount] : params.discrete_dividends) {
        double tau = params.maturity - t_cal;
        if (tau > 0.0 && tau < params.maturity) {
            mandatory_tau.push_back(tau);
        }
    }

    // Apply max_time_steps cap to both uniform and non-uniform paths
    double dt_capped = std::max(dt, params.maturity / static_cast<double>(accuracy.max_time_steps));

    TimeDomain time_domain = mandatory_tau.empty()
        ? TimeDomain::from_n_steps(0.0, params.maturity, Nt)
        : TimeDomain::with_mandatory_points(0.0, params.maturity, dt_capped, mandatory_tau);

    return {grid_spec.value(), time_domain};
}

/**
 * Compute global grid for batch processing.
 *
 * Takes union of individual grid requirements to create a single
 * grid suitable for all options in the batch.
 *
 * @param params Span of option parameters for the batch
 * @param accuracy Grid accuracy parameters (optional)
 * @return Pair of (GridSpec, TimeDomain) covering all options
 */
inline std::pair<GridSpec<double>, TimeDomain> compute_global_grid_for_batch(
    std::span<const PricingParams> params,
    const GridAccuracyParams& accuracy = GridAccuracyParams{})
{
    if (params.empty()) {
        // Return minimal valid sinh grid for empty batch
        auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, accuracy.alpha);
        TimeDomain time_domain = TimeDomain::from_n_steps(0.0, 1.0, 100);
        return {grid_spec.value(), time_domain};
    }

    double global_x_min = std::numeric_limits<double>::max();
    double global_x_max = std::numeric_limits<double>::lowest();
    size_t global_Nx = 0;
    size_t global_Nt = 0;
    double max_maturity = 0.0;

    // Estimate grid for each option and take union/maximum
    for (const auto& p : params) {
        auto [grid_spec, time_domain] = estimate_grid_for_option(p, accuracy);
        global_x_min = std::min(global_x_min, grid_spec.x_min());
        global_x_max = std::max(global_x_max, grid_spec.x_max());
        global_Nx = std::max(global_Nx, grid_spec.n_points());
        global_Nt = std::max(global_Nt, time_domain.n_steps());
        max_maturity = std::max(max_maturity, p.maturity);
    }

    // Create sinh-spaced grid with same concentration parameter
    auto grid_spec = GridSpec<double>::sinh_spaced(global_x_min, global_x_max, global_Nx, accuracy.alpha);

    // Collect union of all dividend tau values across the batch
    std::vector<double> all_mandatory_tau;
    for (const auto& p : params) {
        for (const auto& [t_cal, amount] : p.discrete_dividends) {
            double tau = p.maturity - t_cal;
            if (tau > 0.0 && tau < max_maturity) {
                all_mandatory_tau.push_back(tau);
            }
        }
    }

    double global_dt = max_maturity / static_cast<double>(global_Nt);
    double dt_capped = std::max(global_dt,
        max_maturity / static_cast<double>(accuracy.max_time_steps));

    TimeDomain time_domain = all_mandatory_tau.empty()
        ? TimeDomain::from_n_steps(0.0, max_maturity, global_Nt)
        : TimeDomain::with_mandatory_points(0.0, max_maturity, dt_capped, all_mandatory_tau);

    return {grid_spec.value(), time_domain};
}

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
     * @param grid Optional grid specification (GridAccuracyParams or ExplicitPDEGrid).
     *             When nullopt, auto-estimates from option parameters.
     * @param snapshot_times Optional times to record solution snapshots
     * @return AmericanOptionSolver on success, ValidationError on failure
     */
    static std::expected<AmericanOptionSolver, ValidationError>
    create(const PricingParams& params,
           PDEWorkspace workspace,
           std::optional<PDEGridSpec> grid = std::nullopt,
           std::optional<std::span<const double>> snapshot_times = std::nullopt) noexcept;

    /**
     * Direct PDEWorkspace constructor (deprecated - use create() instead).
     *
     * @deprecated Use AmericanOptionSolver::create() for type-safe error handling
     * @throws std::invalid_argument if parameters are invalid
     *
     * @param params Option pricing parameters
     * @param workspace PDEWorkspace with pre-allocated buffers
     * @param grid Optional grid specification (GridAccuracyParams or ExplicitPDEGrid).
     *             When nullopt, auto-estimates from option parameters.
     * @param snapshot_times Optional times to record solution snapshots
     */
    AmericanOptionSolver(const PricingParams& params,
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
    // Parameters
    PricingParams params_;

    // PDEWorkspace (owns spans to external buffer)
    PDEWorkspace workspace_;

    // Snapshot times for Grid creation
    std::vector<double> snapshot_times_;

    // Resolved grid configuration (GridSpec + TimeDomain)
    // Resolved from PDEGridSpec at construction time
    std::optional<std::pair<GridSpec<double>, TimeDomain>> grid_config_;

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

}  // namespace mango

#endif  // MANGO_AMERICAN_OPTION_HPP
