// SPDX-License-Identifier: MIT
/**
 * @file american_option.hpp
 * @brief American option pricing solver using finite difference method
 */

#ifndef MANGO_AMERICAN_OPTION_HPP
#define MANGO_AMERICAN_OPTION_HPP

#include "mango/pde/core/pde_solver.hpp"
#include "mango/pde/operators/black_scholes_pde.hpp"
#include "mango/pde/operators/centered_difference_facade.hpp"
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
 * Estimate grid specification from option parameters.
 *
 * Automatically determines appropriate spatial/temporal discretization
 * based on option characteristics (volatility, maturity, moneyness).
 *
 * @param params Option pricing parameters
 * @param accuracy Grid accuracy parameters (optional)
 * @return Pair of (GridSpec, TimeDomain) for consistent discretization
 */
inline std::pair<GridSpec<double>, TimeDomain> estimate_pde_grid(
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

    // Widen grid for dividend shift: spline evaluates at x'=ln(exp(x)-D/K)
    // Only consider dividends strictly within (0, T) — same filter as mandatory tau
    double max_d_over_k = 0.0;
    for (const auto& div : params.discrete_dividends) {
        double tau = params.maturity - div.calendar_time;
        if (tau > 0.0 && tau < params.maturity) {
            max_d_over_k = std::max(max_d_over_k, div.amount / params.strike);
        }
    }
    if (max_d_over_k > 0.0 && std::exp(x_min) > max_d_over_k) {
        x_min = std::log(std::exp(x_min) - max_d_over_k);
    } else if (max_d_over_k > 0.0) {
        x_min -= 1.0;  // conservative extension
    }

    // Build cluster list for multi-sinh grid.
    // Strike cluster at x=0 (payoff kink) is the dominant coarse-grid error source.
    // Spot cluster at x0 (query point) is secondary for value_at()/delta().
    // When S≈K, auto-merge combines them. Skip strike cluster if outside domain.
    std::vector<MultiSinhCluster<double>> clusters;
    clusters.push_back({.center_x = x0, .alpha = accuracy.alpha, .weight = 0.5});
    if (0.0 >= x_min && 0.0 <= x_max) {
        clusters.push_back({.center_x = 0.0, .alpha = accuracy.alpha, .weight = 1.0});
    }
    auto grid_spec = GridSpec<double>::multi_sinh_spaced(x_min, x_max, Nx, std::move(clusters));

    // Temporal resolution: compute actual dx_min from generated grid.
    // The old formula dx_min = dx_avg·exp(-α) was an approximation that became
    // wildly pessimistic at high α (e.g., α≈4 gave ~7x more time steps than needed).
    // TR-BDF2 is L-stable so CFL is for accuracy, not stability.
    auto grid_buf = grid_spec.value().generate();
    auto pts = grid_buf.view().span();
    double dx_min = pts[1] - pts[0];
    for (size_t i = 2; i < pts.size(); ++i) {
        dx_min = std::min(dx_min, pts[i] - pts[i - 1]);
    }

    double dt = accuracy.c_t * dx_min;
    size_t Nt = static_cast<size_t>(std::ceil(params.maturity / dt));
    Nt = std::min(Nt, accuracy.max_time_steps);  // Upper bound

    // Convert discrete dividend calendar times to time-to-expiry (tau)
    std::vector<double> mandatory_tau;
    for (const auto& div : params.discrete_dividends) {
        double tau = params.maturity - div.calendar_time;
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
inline std::pair<GridSpec<double>, TimeDomain> estimate_batch_pde_grid(
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
        auto [grid_spec, time_domain] = estimate_pde_grid(p, accuracy);
        global_x_min = std::min(global_x_min, grid_spec.x_min());
        global_x_max = std::max(global_x_max, grid_spec.x_max());
        global_Nx = std::max(global_Nx, grid_spec.n_points());
        global_Nt = std::max(global_Nt, time_domain.n_steps());
        max_maturity = std::max(max_maturity, p.maturity);
    }

    // Create multi-sinh grid with strike cluster at x=0
    // Batch uses shared strike, so x=0 is the common payoff kink
    auto grid_spec = GridSpec<double>::multi_sinh_spaced(global_x_min, global_x_max, global_Nx, {
        {.center_x = 0.0, .alpha = accuracy.alpha, .weight = 1.0},
    });

    // Collect union of all dividend tau values across the batch
    std::vector<double> all_mandatory_tau;
    for (const auto& p : params) {
        for (const auto& div : p.discrete_dividends) {
            double tau = p.maturity - div.calendar_time;
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

private:
    /// Optional custom initial condition (replaces default payoff when set)
    std::optional<InitialCondition> custom_ic_;
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
