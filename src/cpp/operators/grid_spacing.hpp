#pragma once

#include "../grid.hpp"
#include <vector>
#include <span>
#include <cmath>
#include <cassert>

namespace mango::operators {

/**
 * GridSpacing: Computes and caches grid spacing information
 *
 * This adapter wraps a GridView and provides efficient access to spacing
 * information needed for finite difference stencils. For uniform grids,
 * spacing is constant (computed once). For non-uniform grids, per-point
 * spacing is pre-computed and cached.
 *
 * Single Responsibility: Grid metric computation
 */
template<typename T = double>
class GridSpacing {
public:
    /**
     * Create grid spacing from a grid view
     * @param grid Grid points (non-owning view)
     */
    explicit GridSpacing(GridView<T> grid)
        : grid_(grid)
        , is_uniform_(grid.is_uniform())
    {
        const size_t n = grid.size();
        if (n < 2) return;

        if (is_uniform_) {
            // Uniform grid: compute once
            dx_uniform_ = (grid.x_max() - grid.x_min()) / static_cast<T>(n - 1);
            dx_uniform_inv_ = T(1) / dx_uniform_;
            dx_uniform_inv_sq_ = dx_uniform_inv_ * dx_uniform_inv_;
        } else {
            // Non-uniform grid: pre-compute all spacings
            dx_array_.reserve(n - 1);
            for (size_t i = 0; i < n - 1; ++i) {
                dx_array_.push_back(grid[i + 1] - grid[i]);
            }
        }
    }

    // Query if grid is uniform
    bool is_uniform() const { return is_uniform_; }

    // Get uniform spacing (only valid if is_uniform())
    T spacing() const {
        return dx_uniform_;
    }

    T spacing_inv() const {
        return dx_uniform_inv_;
    }

    T spacing_inv_sq() const {
        return dx_uniform_inv_sq_;
    }

    // Get spacing at point i: dx[i] = x[i+1] - x[i]
    // Valid for i in [0, n-2]
    T spacing_at(size_t i) const {
        if (is_uniform_) {
            return dx_uniform_;
        } else {
            return dx_array_[i];
        }
    }

    // Get all spacings (for non-uniform grids)
    std::span<const T> spacings() const {
        return dx_array_;
    }

    // Left and right spacing for non-uniform centered differences
    // Preconditions:
    //   left_spacing(i):  requires 1 <= i < size()
    //   right_spacing(i): requires 0 <= i < size()-1
    T left_spacing(size_t i) const {
        assert(i >= 1 && i < grid_.size() && "left_spacing: index out of bounds");
        if (is_uniform_) {
            return dx_uniform_;
        } else {
            return dx_array_[i - 1];  // dx[i-1] = x[i] - x[i-1]
        }
    }

    T right_spacing(size_t i) const {
        assert(i < grid_.size() - 1 && "right_spacing: index out of bounds");
        if (is_uniform_) {
            return dx_uniform_;
        } else {
            return dx_array_[i];  // dx[i] = x[i+1] - x[i]
        }
    }

    // Minimum size for 3-point stencil
    static constexpr size_t min_stencil_size() { return 3; }

    // Access to underlying grid
    const GridView<T>& grid() const { return grid_; }
    size_t size() const { return grid_.size(); }

private:
    GridView<T> grid_;
    bool is_uniform_;

    // Uniform grid: single spacing value (pre-computed)
    T dx_uniform_{};
    T dx_uniform_inv_{};     // 1/dx
    T dx_uniform_inv_sq_{};  // 1/dx²

    // Non-uniform grid: array of spacings (pre-computed)
    std::vector<T> dx_array_;
};

} // namespace mango::operators
