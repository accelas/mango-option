/**
 * @file slice_solver_workspace.hpp
 * @brief Reusable workspace for slice-based PDE solves (shared grid + SIMD-aligned storage)
 *
 * When building price tables, we solve many PDEs that differ only in coefficients
 * (volatility, rate, dividend) but share the same spatial grid. This workspace
 * allows reusing expensive allocations across multiple solver instances.
 */

#pragma once

#include "workspace.hpp"
#include "operators/grid_spacing.hpp"
#include "grid.hpp"
#include <memory>
#include <span>

namespace mango {

/**
 * Reusable workspace for slice-based PDE solving.
 *
 * Eliminates redundant allocations when solving multiple PDEs with:
 * - Same spatial grid structure
 * - Different PDE coefficients (σ, r, q)
 *
 * Pre-allocates the spatial grid, GridSpacing metadata, and WorkspaceStorage so
 * multiple PDE solves that share the same grid can reuse SIMD-aligned buffers
 * without repeated heap traffic. Intended to be owned per-thread in OpenMP regions.
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
 * Memory savings (vs creating grid+spacing+workspace per solver):
 * - Grid buffer: ~800 bytes per reuse
 * - GridSpacing: ~800 bytes per reuse
 * - WorkspaceStorage: ~10n doubles per reuse (SIMD-aligned)
 *
 * For 200 solvers (typical 4D table): significant savings
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
    SliceSolverWorkspace(double x_min,
                         double x_max,
                         size_t n_space)
        : grid_buffer_(GridSpec<>::uniform(x_min, x_max, n_space).value().generate())
        , grid_view_(grid_buffer_.span())
        , grid_spacing_(std::make_shared<operators::GridSpacing<double>>(grid_view_))
        , workspace_(std::make_shared<WorkspaceStorage>(n_space, grid_view_.span()))
    {}

    /// Spatial grid span
    std::span<const double> grid_span() const { return grid_view_.span(); }

    /// Shared GridSpacing for spatial operators
    std::shared_ptr<operators::GridSpacing<double>> grid_spacing() const { return grid_spacing_; }

    /// Shared WorkspaceStorage for PDESolver internals
    std::shared_ptr<WorkspaceStorage> workspace() const { return workspace_; }

    /// Grid parameters (for validation)
    double x_min() const { return grid_view_.span().front(); }
    double x_max() const { return grid_view_.span().back(); }
    size_t n_space() const { return grid_view_.span().size(); }

private:
    GridBuffer<double> grid_buffer_;
    GridView<double> grid_view_;
    std::shared_ptr<operators::GridSpacing<double>> grid_spacing_;
    std::shared_ptr<WorkspaceStorage> workspace_;
};

}  // namespace mango
