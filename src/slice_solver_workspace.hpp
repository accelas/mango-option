/**
 * @file slice_solver_workspace.hpp
 * @brief Reusable workspace for solving multiple PDEs with different parameters
 *
 * When building price tables, we solve many PDEs that differ only in coefficients
 * (volatility, rate, dividend) but share the same spatial grid. This workspace
 * allows reusing expensive allocations across multiple solver instances.
 */

#pragma once

#include "grid.hpp"
#include "operators/grid_spacing.hpp"
#include <memory>
#include <vector>

namespace mango {

/**
 * Reusable workspace for slice-based PDE solving.
 *
 * Eliminates redundant allocations when solving multiple PDEs with:
 * - Same spatial grid structure
 * - Different PDE coefficients (σ, r, q)
 *
 * Example usage:
 * ```cpp
 * SliceSolverWorkspace workspace(x_min, x_max, n_space);
 *
 * for (auto [sigma, rate] : parameter_grid) {
 *     AmericanOptionSolver solver(params, grid_config, workspace);
 *     auto result = solver.solve();
 * }
 * ```
 *
 * Memory savings:
 * - Grid buffer: ~800 bytes per reuse
 * - GridSpacing: ~800 bytes per reuse
 * - Total: ~1.6 KB per solver instance avoided
 *
 * For 200 solvers (typical 4D table): ~320 KB saved
 */
class SliceSolverWorkspace {
public:
    /**
     * Create workspace with specified grid parameters.
     *
     * @param x_min Minimum log-moneyness
     * @param x_max Maximum log-moneyness
     * @param n_space Number of spatial grid points
     */
    SliceSolverWorkspace(double x_min, double x_max, size_t n_space)
        : x_min_(x_min)
        , x_max_(x_max)
        , n_space_(n_space)
        , grid_buffer_(GridSpec<>::uniform(x_min, x_max, n_space).generate())
    {
        // Create GridSpacing once (reused across all spatial operators)
        auto grid_view = GridView<double>(grid_buffer_.span());
        grid_spacing_ = std::make_shared<operators::GridSpacing<double>>(grid_view);
    }

    /// Get the shared grid buffer
    const GridBuffer<double>& grid_buffer() const { return grid_buffer_; }

    /// Get grid span (for passing to PDESolver)
    std::span<const double> grid_span() const { return grid_buffer_.span(); }

    /// Get the shared GridSpacing (for passing to operator factory)
    std::shared_ptr<operators::GridSpacing<double>> grid_spacing() const {
        return grid_spacing_;
    }

    /// Grid parameters (for validation)
    double x_min() const { return x_min_; }
    double x_max() const { return x_max_; }
    size_t n_space() const { return n_space_; }

private:
    double x_min_;
    double x_max_;
    size_t n_space_;

    // Shared allocations
    GridBuffer<double> grid_buffer_;
    std::shared_ptr<operators::GridSpacing<double>> grid_spacing_;
};

}  // namespace mango
