/**
 * @file slice_solver_workspace.hpp
 * @brief Reusable workspace for slice-based PDE solves (shared grid + storage)
 */

#pragma once

#include "workspace.hpp"
#include "operators/grid_spacing.hpp"
#include "grid.hpp"
#include <memory>
#include <span>

namespace mango {

/**
 * SliceSolverWorkspace
 *
 * Pre-allocates the spatial grid, GridSpacing metadata, and WorkspaceStorage so
 * multiple PDE solves that share the same grid can reuse buffers without
 * repeated heap traffic. Intended to be owned per-thread in OpenMP regions.
 */
class SliceSolverWorkspace {
public:
    SliceSolverWorkspace(double x_min,
                         double x_max,
                         size_t n_space,
                         size_t cache_block_threshold = 5000)
        : grid_buffer_(GridSpec<>::uniform(x_min, x_max, n_space).generate())
        , grid_view_(grid_buffer_.span())
        , grid_spacing_(std::make_shared<operators::GridSpacing<double>>(grid_view_))
        , workspace_(std::make_shared<WorkspaceStorage>(n_space, grid_view_.span(), cache_block_threshold))
    {}

    /// Spatial grid span
    std::span<const double> grid_span() const { return grid_view_.span(); }

    /// Shared GridSpacing for spatial operators
    std::shared_ptr<operators::GridSpacing<double>> grid_spacing() const { return grid_spacing_; }

    /// Shared WorkspaceStorage for PDESolver internals
    std::shared_ptr<WorkspaceStorage> workspace() const { return workspace_; }

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
