/**
 * @file american_solver_workspace.hpp
 * @brief Reusable workspace for American option solving with PMR memory allocation
 */

#pragma once

#include "src/pde/core/pde_workspace.hpp"
#include "src/pde/core/grid.hpp"
#include "src/option/option_spec.hpp"
#include <memory>
#include <expected>
#include <string>
#include <memory_resource>
#include <tuple>
#include <cmath>
#include <algorithm>

namespace mango {

/**
 * Estimate grid specification for a single option using sinh-grid heuristics.
 *
 * Implements single-pass grid determination from sinh-grid specification:
 * - Spatial domain: x₀ ± n_sigma·σ√T (covers probability distribution)
 * - Spatial resolution: Δx ~ σ√tol (target truncation error)
 * - Temporal resolution: Δt ~ c_t·Δx_min (couples time/space errors)
 *
 * Grid is centered on current log-moneyness x₀ = ln(S/K), not at x=0.
 * This is appropriate for independent options (vs option chains).
 *
 * @param params Option parameters (spot, strike, maturity, volatility, etc.)
 * @param n_sigma Domain half-width in units of σ√T (default: 5.0)
 * @param alpha Sinh clustering strength (default: 2.0 for Europeans)
 * @param tol Target price tolerance (default: 1e-6)
 * @param c_t Time step safety factor (default: 0.75)
 * @return Tuple of (GridSpec, n_time)
 */
inline std::tuple<GridSpec<double>, size_t> estimate_grid_for_option(
    const PricingParams& params,
    double n_sigma = 5.0,
    double alpha = 2.0,
    double tol = 1e-6,
    double c_t = 0.75)
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

    // Temporal resolution (coupled to smallest spatial spacing)
    // For sinh grid with clustering α, dx_min ≈ dx_avg · exp(-α)
    double dx_avg = (x_max - x_min) / static_cast<double>(Nx);
    double dx_min = dx_avg * std::exp(-alpha);  // Sinh clustering factor

    double dt = c_t * dx_min;
    size_t Nt = static_cast<size_t>(std::ceil(params.maturity / dt));
    Nt = std::min(Nt, size_t{5000});  // Upper bound for stability

    // Create uniform GridSpec (workspace creation uses uniform grids)
    auto grid_spec = GridSpec<double>::uniform(x_min, x_max, Nx);
    // GridSpec factory returns expected, but params are validated above so should never fail
    return {grid_spec.value(), Nt};
}

/**
 * Workspace for American option solving with PMR-based memory allocation.
 *
 * Provides unified workspace for American option pricing with:
 * - PDEWorkspace allocated from provided memory resource
 * - GridSpacing<double> for spatial operators
 * - Grid configuration (spatial + temporal parameters)
 *
 * Example usage:
 * ```cpp
 * std::pmr::synchronized_pool_resource pool;
 * auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0);
 *
 * auto workspace = AmericanSolverWorkspace::create(
 *     grid_spec.value(), 1000, &pool);
 *
 * if (!workspace.has_value()) {
 *     std::cerr << "Failed: " << workspace.error() << "\n";
 *     return;
 *     }
 *
 * // Use workspace with solver
 * auto solver = AmericanPutSolver(params, workspace.value());
 * ```
 *
 * Thread safety: **NOT thread-safe for concurrent solving**.
 * Use BatchAmericanOptionSolver for parallel option pricing.
 */
class AmericanSolverWorkspace {
public:
    /**
     * Factory method creates workspace from GridSpec.
     *
     * @param grid_spec Grid specification for spatial domain
     * @param n_time Number of time steps
     * @param resource PMR memory resource for workspace allocation
     * @return Expected containing shared workspace on success, error message on failure
     */
    static std::expected<std::shared_ptr<AmericanSolverWorkspace>, std::string>
    create(const GridSpec<double>& grid_spec,
           size_t n_time,
           std::pmr::memory_resource* resource);

    PDEWorkspace* pde_workspace() const { return pde_workspace_.get(); }
    GridSpacing<double> grid_spacing() const { return *grid_spacing_; }

    std::span<const double> grid() const {
        return grid_buffer_.span();
    }

    std::span<const double> grid_span() const {
        return grid_buffer_.span();
    }

    size_t n_space() const { return grid_buffer_.size(); }
    size_t n_time() const { return n_time_; }

    double x_min() const {
        auto g = grid();
        return g.empty() ? 0.0 : g[0];
    }

    double x_max() const {
        auto g = grid();
        return g.empty() ? 0.0 : g[g.size() - 1];
    }

private:
    AmericanSolverWorkspace(GridBuffer<double> grid_buf,
                           std::shared_ptr<PDEWorkspace> pde_ws,
                           std::shared_ptr<GridSpacing<double>> spacing,
                           size_t n_time)
        : grid_buffer_(std::move(grid_buf))
        , pde_workspace_(std::move(pde_ws))
        , grid_spacing_(std::move(spacing))
        , n_time_(n_time)
    {}

    GridBuffer<double> grid_buffer_;  // Must come before pde_workspace_ (owns grid data)
    std::shared_ptr<PDEWorkspace> pde_workspace_;
    std::shared_ptr<GridSpacing<double>> grid_spacing_;
    size_t n_time_;
};

}  // namespace mango
