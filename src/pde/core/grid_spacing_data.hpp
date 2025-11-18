#pragma once

#include <vector>
#include <span>
#include <cstddef>
#include <cmath>

namespace mango {

/// Uniform grid spacing data (minimal storage: 4 values)
///
/// For uniform grids, spacing is constant everywhere.
/// Memory: 32 bytes (3 doubles + 1 size_t)
template<typename T = double>
struct UniformSpacing {
    T dx;           ///< Grid spacing
    T dx_inv;       ///< 1/dx (precomputed for performance)
    T dx_inv_sq;    ///< 1/dx² (precomputed for performance)
    size_t n;       ///< Number of grid points

    /// Construct from spacing and grid size
    ///
    /// @param spacing Grid spacing (dx)
    /// @param size Number of grid points
    UniformSpacing(T spacing, size_t size)
        : dx(spacing)
        , dx_inv(T(1) / spacing)
        , dx_inv_sq(dx_inv * dx_inv)
        , n(size)
    {}
};

} // namespace mango
