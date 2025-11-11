#pragma once

#include "../grid.hpp"
#include <vector>
#include <span>
#include <cmath>
#include <cassert>

namespace mango::operators {

/**
 * GridSpacing: Grid spacing information for finite difference operators
 *
 * For UNIFORM grids:
 *   - Stores constant spacing (dx, dx_inv, dx_inv_sq)
 *   - Zero memory overhead for precomputed arrays
 *
 * For NON-UNIFORM grids:
 *   - Eagerly precomputes weight arrays during construction:
 *     * dx_left_inv[i]   = 1 / (x[i] - x[i-1])
 *     * dx_right_inv[i]  = 1 / (x[i+1] - x[i])
 *     * dx_center_inv[i] = 2 / (dx_left + dx_right)
 *     * w_left[i]        = dx_right / (dx_left + dx_right)
 *     * w_right[i]       = dx_left / (dx_left + dx_right)
 *   - Single contiguous buffer (5×(n-2)×8 bytes, ~4KB for n=100)
 *   - Zero-copy span accessors (fail-fast if called on uniform grid)
 *
 * USE CASE:
 *   Tanh-clustered grids for adaptive mesh refinement around strikes/barriers
 *   in option pricing. Grids are fixed during PDE solve, so one-time
 *   precomputation cost (~1-2 µs) is amortized over many time steps.
 *
 * SIMD INTEGRATION:
 *   CenteredDifferenceSIMD loads precomputed arrays via element_aligned spans,
 *   avoiding per-lane divisions. Expected speedup: 3-6x over scalar non-uniform.
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
        , n_(grid.size())
    {
        if (n_ < 2) return;

        if (is_uniform_) {
            // Uniform grid: compute once
            dx_uniform_ = (grid.x_max() - grid.x_min()) / static_cast<T>(n_ - 1);
            dx_uniform_inv_ = T(1) / dx_uniform_;
            dx_uniform_inv_sq_ = dx_uniform_inv_ * dx_uniform_inv_;
        } else {
            // Non-uniform grid: pre-compute all spacings
            dx_array_.reserve(n_ - 1);
            for (size_t i = 0; i < n_ - 1; ++i) {
                dx_array_.push_back(grid[i + 1] - grid[i]);
            }

            // Precompute non-uniform data for SIMD kernels
            precompute_non_uniform_data();
        }
    }

    // Query if grid is uniform
    bool is_uniform() const { return is_uniform_; }

    // Get uniform spacing (only valid if is_uniform())
    T spacing() const {
        assert(is_uniform_ && "spacing() requires uniform grid");
        return dx_uniform_;
    }

    T spacing_inv() const {
        assert(is_uniform_ && "spacing_inv() requires uniform grid");
        return dx_uniform_inv_;
    }

    T spacing_inv_sq() const {
        assert(is_uniform_ && "spacing_inv_sq() requires uniform grid");
        return dx_uniform_inv_sq_;
    }

    // Get spacing at point i: dx[i] = x[i+1] - x[i]
    // Valid for i in [0, n-2]
    T spacing_at(size_t i) const {
        assert(i < grid_.size() - 1 && "spacing_at(i) requires i < n-1");
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

    // Zero-copy accessors (fail-fast if called on uniform grid)
    // Returns precomputed values for interior points i=1..n-2 (n-2 points)
    std::span<const T> dx_left_inv() const {
        assert(!is_uniform_ && "dx_left_inv only available for non-uniform grids");
        assert(n_ >= min_stencil_size() && "Grid too small for stencil operations");
        const size_t interior_count = n_ - 2;
        return {precomputed_.data(), interior_count};
    }

    std::span<const T> dx_right_inv() const {
        assert(!is_uniform_ && "dx_right_inv only available for non-uniform grids");
        assert(n_ >= min_stencil_size() && "Grid too small for stencil operations");
        const size_t interior_count = n_ - 2;
        return {precomputed_.data() + interior_count, interior_count};
    }

    std::span<const T> dx_center_inv() const {
        assert(!is_uniform_ && "dx_center_inv only available for non-uniform grids");
        assert(n_ >= min_stencil_size() && "Grid too small for stencil operations");
        const size_t interior_count = n_ - 2;
        return {precomputed_.data() + 2 * interior_count, interior_count};
    }

    std::span<const T> w_left() const {
        assert(!is_uniform_ && "w_left only available for non-uniform grids");
        assert(n_ >= min_stencil_size() && "Grid too small for stencil operations");
        const size_t interior_count = n_ - 2;
        return {precomputed_.data() + 3 * interior_count, interior_count};
    }

    std::span<const T> w_right() const {
        assert(!is_uniform_ && "w_right only available for non-uniform grids");
        assert(n_ >= min_stencil_size() && "Grid too small for stencil operations");
        const size_t interior_count = n_ - 2;
        return {precomputed_.data() + 4 * interior_count, interior_count};
    }

private:
    GridView<T> grid_;
    bool is_uniform_;
    size_t n_;

    // Uniform grid: single spacing value (pre-computed)
    T dx_uniform_{};
    T dx_uniform_inv_{};     // 1/dx
    T dx_uniform_inv_sq_{};  // 1/dx²

    // Non-uniform grid: array of spacings (pre-computed)
    std::vector<T> dx_array_;

    // Single buffer: [dx_left_inv | dx_right_inv | dx_center_inv | w_left | w_right]
    std::vector<T> precomputed_;

    void precompute_non_uniform_data() {
        const size_t interior_count = n_ - 2;  // Points i=1..n-2 (n-2 points with both neighbors)
        precomputed_.resize(5 * interior_count);

        // Compute all arrays in one loop (for interior points i=1..n-2)
        for (size_t i = 1; i <= n_ - 2; ++i) {
            const T dx_left = left_spacing(i);     // x[i] - x[i-1]
            const T dx_right = right_spacing(i);   // x[i+1] - x[i]
            const T dx_center = T(0.5) * (dx_left + dx_right);

            const size_t idx = i - 1;  // Index into precomputed arrays

            precomputed_[idx] = T(1) / dx_left;
            precomputed_[interior_count + idx] = T(1) / dx_right;
            precomputed_[2 * interior_count + idx] = T(1) / dx_center;
            precomputed_[3 * interior_count + idx] = dx_right / (dx_left + dx_right);
            precomputed_[4 * interior_count + idx] = dx_left / (dx_left + dx_right);
        }
    }
};

} // namespace mango::operators
