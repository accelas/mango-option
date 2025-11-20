/**
 * @file grid_estimation.hpp
 * @brief Shared utilities for estimating PDE grid parameters from option characteristics
 */

#ifndef MANGO_GRID_ESTIMATION_HPP
#define MANGO_GRID_ESTIMATION_HPP

#include "src/option/option_spec.hpp"
#include "src/pde/core/grid.hpp"
#include <tuple>
#include <cmath>
#include <algorithm>
#include <span>
#include <limits>

namespace mango {

/**
 * Grid accuracy parameters controlling grid size/resolution tradeoff.
 * Used by batch solvers and automatic grid estimation.
 */
struct GridAccuracyParams {
    double n_sigma = 5.0;  ///< Number of standard deviations for domain bounds
    double tol = 1e-6;     ///< Target truncation error tolerance
};

/**
 * Estimate optimal grid parameters from option characteristics.
 *
 * Automatically determines spatial and temporal grid resolution based on:
 * - Domain bounds (n_sigma standard deviations centered on log-moneyness)
 * - Spatial resolution (target truncation error from volatility)
 * - Temporal resolution (CFL-like stability condition)
 *
 * @param params Option pricing parameters
 * @param n_sigma Number of standard deviations for domain (default: 5.0)
 * @param tol Target truncation error tolerance (default: 1e-6)
 * @return Tuple of (GridSpec, n_time) for use with AmericanSolverWorkspace
 */
inline std::tuple<GridSpec<double>, size_t> estimate_grid_for_option(
    const PricingParams& params,
    double n_sigma = 5.0,
    double tol = 1e-6)
{
    // Domain bounds (centered on current moneyness)
    double sigma_sqrt_T = params.volatility * std::sqrt(params.maturity);
    double x0 = std::log(params.spot / params.strike);

    double x_min = x0 - n_sigma * sigma_sqrt_T;
    double x_max = x0 + n_sigma * sigma_sqrt_T;

    // Spatial resolution (target truncation error)
    double dx_target = params.volatility * std::sqrt(tol);
    size_t Nx = static_cast<size_t>(std::ceil((x_max - x_min) / dx_target));
    Nx = std::clamp(Nx, size_t{200}, size_t{1200});

    // Ensure odd number of points (for centered stencils)
    if (Nx % 2 == 0) Nx++;

    // Temporal resolution (CFL-like condition for stability)
    double dx_actual = (x_max - x_min) / (Nx - 1);
    double dt_target = 0.75 * dx_actual * dx_actual / (params.volatility * params.volatility);
    size_t Nt = static_cast<size_t>(std::ceil(params.maturity / dt_target));
    Nt = std::clamp(Nt, size_t{200}, size_t{4000});

    auto grid_spec = GridSpec<double>::uniform(x_min, x_max, Nx);
    return {grid_spec.value(), Nt};
}

/**
 * Overload accepting GridAccuracyParams struct.
 */
inline std::tuple<GridSpec<double>, size_t> estimate_grid_for_option(
    const PricingParams& params,
    const GridAccuracyParams& accuracy)
{
    return estimate_grid_for_option(params, accuracy.n_sigma, accuracy.tol);
}

/**
 * Compute global grid parameters for a batch of options.
 *
 * Determines a single grid that accommodates all options in the batch.
 * Used by price table construction and batch solvers with shared grids.
 *
 * @param params Span of option parameters
 * @param accuracy Grid accuracy parameters
 * @return Tuple of (GridSpec, n_time) suitable for all options
 */
inline std::tuple<GridSpec<double>, size_t> compute_global_grid_for_batch(
    std::span<const PricingParams> params,
    const GridAccuracyParams& accuracy = GridAccuracyParams{})
{
    if (params.empty()) {
        // Return default grid if no options provided
        auto grid_spec = GridSpec<double>::uniform(-3.0, 3.0, 201);
        return {grid_spec.value(), 1000};
    }

    // Find extreme values across all options
    double x_min_global = std::numeric_limits<double>::max();
    double x_max_global = std::numeric_limits<double>::lowest();
    size_t Nx_max = 0;
    size_t Nt_max = 0;

    for (const auto& p : params) {
        auto [grid_spec, n_time] = estimate_grid_for_option(p, accuracy);
        x_min_global = std::min(x_min_global, grid_spec.x_min());
        x_max_global = std::max(x_max_global, grid_spec.x_max());
        Nx_max = std::max(Nx_max, grid_spec.n_points());
        Nt_max = std::max(Nt_max, n_time);
    }

    // Create unified grid
    auto global_grid = GridSpec<double>::uniform(x_min_global, x_max_global, Nx_max);
    return {global_grid.value(), Nt_max};
}

}  // namespace mango

#endif  // MANGO_GRID_ESTIMATION_HPP
