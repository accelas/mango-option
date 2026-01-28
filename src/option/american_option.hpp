/**
 * @file american_option.hpp
 * @brief American option pricing solver using finite difference method
 */

#ifndef MANGO_AMERICAN_OPTION_HPP
#define MANGO_AMERICAN_OPTION_HPP

#include "src/pde/core/pde_solver.hpp"
#include "src/pde/operators/black_scholes_pde.hpp"
#include "src/pde/operators/centered_difference_facade.hpp"
#include <expected>
#include "src/support/error_types.hpp"
#include "src/support/parallel.hpp"
#include "src/option/american_option_result.hpp"
#include "src/option/option_spec.hpp"  // For OptionType enum
#include "src/pde/core/pde_workspace.hpp"
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <functional>
#include <optional>

namespace mango {

/**
 * @brief Backward compatibility alias for PricingParams
 * @deprecated Use PricingParams from option_spec.hpp instead
 */
using AmericanOptionParams = PricingParams;

/**
 * Grid estimation accuracy parameters.
 *
 * Controls spatial/temporal resolution tradeoffs for American option PDE solver.
 */
struct GridAccuracyParams {
    /// Domain half-width in units of σ√T (default: 5.0 covers ±5 std devs)
    double n_sigma = 5.0;

    /// Sinh clustering strength (default: 2.0 concentrates points near strike)
    double alpha = 2.0;

    /// Target spatial truncation error (default: 1e-2 for ~1e-3 price accuracy)
    /// - 1e-2: Fast mode (~100-150 points, ~5ms per option)
    /// - 1e-3: Medium accuracy (~300-400 points, ~50ms per option)
    /// - 1e-6: High accuracy mode (~1200 points, ~300ms per option)
    double tol = 1e-2;

    /// CFL safety factor for time step (default: 0.75)
    double c_t = 0.75;

    /// Minimum spatial grid points (default: 100)
    size_t min_spatial_points = 100;

    /// Maximum spatial grid points (default: 1200)
    size_t max_spatial_points = 1200;

    /// Maximum time steps (default: 5000)
    size_t max_time_steps = 5000;
};

enum class GridAccuracyProfile {
    Fast,
    Medium,
    Accurate
};

inline GridAccuracyParams grid_accuracy_profile(GridAccuracyProfile profile) {
    GridAccuracyParams params;
    switch (profile) {
        case GridAccuracyProfile::Fast:
            params.tol = 1e-2;
            params.min_spatial_points = 100;
            params.max_spatial_points = 800;
            params.max_time_steps = 3000;
            break;
        case GridAccuracyProfile::Medium:
            params.tol = 5e-3;
            params.min_spatial_points = 150;
            params.max_spatial_points = 1500;
            params.max_time_steps = 6000;
            break;
        case GridAccuracyProfile::Accurate:
            params.tol = 5e-5;
            params.min_spatial_points = 201;
            params.max_spatial_points = 2500;
            params.max_time_steps = 12000;
            break;
    }
    return params;
}

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

    // Create sinh-spaced GridSpec for better resolution near strike (x=0 in log-moneyness)
    auto grid_spec = GridSpec<double>::sinh_spaced(x_min, x_max, Nx, accuracy.alpha);
    TimeDomain time_domain = TimeDomain::from_n_steps(0.0, params.maturity, Nt);
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
    TimeDomain time_domain = TimeDomain::from_n_steps(0.0, max_maturity, global_Nt);
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
     * @param snapshot_times Optional times to record solution snapshots
     * @param custom_grid_config Optional custom grid config (GridSpec + TimeDomain)
     *                           When provided, bypasses auto-estimation entirely.
     *                           Both must be provided together to ensure consistent discretization.
     * @return AmericanOptionSolver on success, ValidationError on failure
     */
    static std::expected<AmericanOptionSolver, ValidationError>
    create(const PricingParams& params,
           PDEWorkspace workspace,
           std::optional<std::span<const double>> snapshot_times = std::nullopt,
           std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid_config = std::nullopt) noexcept;

    /**
     * Direct PDEWorkspace constructor (deprecated - use create() instead).
     *
     * @deprecated Use AmericanOptionSolver::create() for type-safe error handling
     * @throws std::invalid_argument if parameters are invalid
     *
     * @param params Option pricing parameters
     * @param workspace PDEWorkspace with pre-allocated buffers
     * @param snapshot_times Optional times to record solution snapshots
     * @param custom_grid_config Optional custom grid config (GridSpec + TimeDomain)
     */
    AmericanOptionSolver(const PricingParams& params,
                        PDEWorkspace workspace,
                        std::optional<std::span<const double>> snapshot_times = std::nullopt,
                        std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid_config = std::nullopt);

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

    // Optional custom grid configuration (bypasses auto-estimation)
    // Both GridSpec and TimeDomain must be provided together for consistent discretization
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid_config_;
};

}  // namespace mango

#endif  // MANGO_AMERICAN_OPTION_HPP
