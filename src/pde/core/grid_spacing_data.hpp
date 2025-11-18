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

/// Non-uniform grid spacing data (precomputed weight arrays)
///
/// For non-uniform grids, precomputes all spacing-dependent values
/// needed for finite difference operators.
///
/// Memory layout: [dx_left_inv | dx_right_inv | dx_center_inv | w_left | w_right]
/// Each section has size (n-2) for interior points
///
/// Memory: ~40 bytes overhead + 5×(n-2)×sizeof(T)
///         For n=100, double: ~4 KB
template<typename T = double>
struct NonUniformSpacing {
    size_t n;  ///< Number of grid points

    /// Precomputed arrays (single contiguous buffer)
    /// Layout: [dx_left_inv | dx_right_inv | dx_center_inv | w_left | w_right]
    std::vector<T> precomputed;

    /// Construct from non-uniform grid points
    ///
    /// @param x Grid points (must be sorted, size >= 3)
    explicit NonUniformSpacing(std::span<const T> x)
        : n(x.size())
    {
        const size_t interior = n - 2;
        precomputed.resize(5 * interior);

        // Precompute all spacing arrays for interior points i=1..n-2
        for (size_t i = 1; i <= n - 2; ++i) {
            const T dx_left = x[i] - x[i-1];
            const T dx_right = x[i+1] - x[i];
            const T dx_center = T(0.5) * (dx_left + dx_right);

            const size_t idx = i - 1;  // Index into arrays (0-based)

            precomputed[idx] = T(1) / dx_left;
            precomputed[interior + idx] = T(1) / dx_right;
            precomputed[2 * interior + idx] = T(1) / dx_center;
            precomputed[3 * interior + idx] = dx_right / (dx_left + dx_right);
            precomputed[4 * interior + idx] = dx_left / (dx_left + dx_right);
        }
    }

    /// Get inverse left spacing for each interior point
    /// Returns: 1/(x[i] - x[i-1]) for i=1..n-2
    std::span<const T> dx_left_inv() const {
        const size_t interior = n - 2;
        return {precomputed.data(), interior};
    }

    /// Get inverse right spacing for each interior point
    /// Returns: 1/(x[i+1] - x[i]) for i=1..n-2
    std::span<const T> dx_right_inv() const {
        const size_t interior = n - 2;
        return {precomputed.data() + interior, interior};
    }

    /// Get inverse center spacing for each interior point
    /// Returns: 2/(dx_left + dx_right) for i=1..n-2
    std::span<const T> dx_center_inv() const {
        const size_t interior = n - 2;
        return {precomputed.data() + 2 * interior, interior};
    }

    /// Get left weight for weighted first derivative
    /// Returns: dx_right/(dx_left + dx_right) for i=1..n-2
    std::span<const T> w_left() const {
        const size_t interior = n - 2;
        return {precomputed.data() + 3 * interior, interior};
    }

    /// Get right weight for weighted first derivative
    /// Returns: dx_left/(dx_left + dx_right) for i=1..n-2
    std::span<const T> w_right() const {
        const size_t interior = n - 2;
        return {precomputed.data() + 4 * interior, interior};
    }
};

} // namespace mango
