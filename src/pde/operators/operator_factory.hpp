#pragma once

#include "spatial_operator.hpp"
#include "src/pde/core/grid.hpp"
#include <memory>

namespace mango::operators {

/// Factory function to create spatial operator with appropriate stencil
/// Returns operator with owned PDE and GridSpacing (safe lifetimes)
template<typename PDE, typename T = double>
auto create_spatial_operator(
    PDE pde,  // Pass by value to allow perfect forwarding of temporaries
    const GridView<T>& grid)
{
    auto spacing = std::make_shared<GridSpacing<T>>(grid);
    return SpatialOperator<PDE, T>(std::move(pde), spacing);
}

/// Overload with explicit spacing (for reuse across multiple operators)
template<typename PDE, typename T = double>
auto create_spatial_operator(
    PDE pde,  // Pass by value
    std::shared_ptr<GridSpacing<T>> spacing)
{
    return SpatialOperator<PDE, T>(std::move(pde), spacing);
}

} // namespace mango::operators
