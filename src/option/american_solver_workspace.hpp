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
#include <limits>
#include <span>

namespace mango {

/**
 * Grid accuracy parameters for automatic grid estimation.
 *
 * Controls the tradeoff between numerical accuracy and computational cost:
 * - Lower tol → finer spatial grid → higher accuracy, slower computation
 * - Higher c_t → smaller time steps → more stable, slower computation
 *
 * Default values (tol=1e-2) produce ~100-150 spatial points and ~500-1000 time steps,
 * balancing accuracy (~1e-3 price error) with performance (~5ms per option).
 *
 * For higher accuracy (tol=1e-6), grids grow to 1200×5000, achieving ~1e-6 price error
 * but at 60× computational cost relative to default settings.
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
 * @param accuracy Grid accuracy parameters controlling size/resolution tradeoff
 * @return Tuple of (GridSpec, n_time)
 */
inline std::tuple<GridSpec<double>, size_t> estimate_grid_for_option(
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
    // Grid is centered on current moneyness x0, with concentration parameter alpha
    auto grid_spec = GridSpec<double>::sinh_spaced(x_min, x_max, Nx, accuracy.alpha);
    // GridSpec factory returns expected, but params are validated above so should never fail
    return {grid_spec.value(), Nt};
}

/**
 * Compute global grid for a batch of options requiring shared grid.
 *
 * Finds a single grid that accommodates all options in the batch by:
 * - Taking union of spatial domains (max extent across all options)
 * - Using maximum resolution (finest grid needed by any option)
 * - Using maximum time steps (finest temporal resolution)
 *
 * This ensures the grid is large enough and fine enough for every option,
 * enabling consistent interpolation across the batch.
 *
 * Use case: Price table construction where all (σ,r) combinations must
 * share the same grid for 4D interpolation.
 *
 * @param params Span of option parameters
 * @param accuracy Grid accuracy parameters controlling size/resolution tradeoff
 * @return Tuple of (GridSpec, n_time) that works for all options
 */
inline std::tuple<GridSpec<double>, size_t> compute_global_grid_for_batch(
    std::span<const PricingParams> params,
    const GridAccuracyParams& accuracy = GridAccuracyParams{})
{
    if (params.empty()) {
        // Return minimal valid sinh grid for empty batch
        auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, accuracy.alpha);
        return {grid_spec.value(), 100};
    }

    double global_x_min = std::numeric_limits<double>::max();
    double global_x_max = std::numeric_limits<double>::lowest();
    size_t global_Nx = 0;
    size_t global_Nt = 0;

    // Estimate grid for each option and take union/maximum
    for (const auto& p : params) {
        auto [grid_spec, Nt] = estimate_grid_for_option(p, accuracy);
        global_x_min = std::min(global_x_min, grid_spec.x_min());
        global_x_max = std::max(global_x_max, grid_spec.x_max());
        global_Nx = std::max(global_Nx, grid_spec.n_points());
        global_Nt = std::max(global_Nt, Nt);
    }

    // Create sinh-spaced grid with same concentration parameter for consistent resolution
    // Sinh grid provides better resolution near the center (strike region) for all options
    auto grid_spec = GridSpec<double>::sinh_spaced(global_x_min, global_x_max, global_Nx, accuracy.alpha);
    return {grid_spec.value(), global_Nt};
}

/**
 * Workspace for American option solving with PMR-based memory allocation.
 *
 * Provides unified workspace for American option pricing with:
 * - Grid for grid + solution storage
 * - PDEWorkspace allocated from provided memory resource
 * - PDEWorkspace for non-owning spans to workspace buffers
 * - GridSpacing<double> for spatial operators
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
 * auto solver = AmericanPutSolver(params, workspace->grid_with_solution(), workspace->workspace_spans());
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
           std::pmr::memory_resource* resource,
           double maturity = 1.0);

    // New API: Grid + PDEWorkspace
    std::shared_ptr<Grid<double>> grid_with_solution() const { return grid_with_solution_; }
    PDEWorkspace workspace_spans() const { return workspace_spans_; }

    size_t n_space() const { return grid_with_solution_->n_space(); }
    size_t n_time() const { return grid_with_solution_->time().n_steps(); }

    double x_min() const {
        auto g = grid_with_solution_->x();
        return g.empty() ? 0.0 : g[0];
    }

    double x_max() const {
        auto g = grid_with_solution_->x();
        return g.empty() ? 0.0 : g[g.size() - 1];
    }

private:
    AmericanSolverWorkspace(std::shared_ptr<Grid<double>> grid_sol,
                           std::pmr::vector<double>&& pmr_buffer,
                           PDEWorkspace workspace_spans)
        : grid_with_solution_(std::move(grid_sol))
        , pmr_buffer_(std::move(pmr_buffer))
        , workspace_spans_(workspace_spans)
    {}

    std::shared_ptr<Grid<double>> grid_with_solution_;
    std::pmr::vector<double> pmr_buffer_;  // Contiguous PMR buffer for workspace
    PDEWorkspace workspace_spans_;     // Spans into pmr_buffer_
};

}  // namespace mango
