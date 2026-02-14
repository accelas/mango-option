// SPDX-License-Identifier: MIT
#pragma once

#include "mango/pde/operators/spatial_operator.hpp"
#include "mango/pde/core/grid.hpp"
#include "mango/pde/core/pde_workspace.hpp"
#include <memory>

namespace mango::operators {

/// Factory function to create spatial operator with appropriate stencil
/// Returns operator with owned PDE and GridSpacing (safe lifetimes)
template<typename PDE, typename T = double>
auto create_spatial_operator(PDE pde, const GridView<T>& grid, PDEWorkspace& workspace) {
    auto spacing = std::make_shared<GridSpacing<T>>(grid);
    return SpatialOperator<PDE, T>(std::move(pde), spacing, workspace);
}

/// Overload with explicit spacing (for reuse across multiple operators)
template<typename PDE, typename T = double>
auto create_spatial_operator(PDE pde, std::shared_ptr<GridSpacing<T>> spacing,
                             PDEWorkspace& workspace) {
    return SpatialOperator<PDE, T>(std::move(pde), spacing, workspace);
}

} // namespace mango::operators
