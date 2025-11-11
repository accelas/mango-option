#pragma once

#include "grid_spacing.hpp"
#include <experimental/simd>
#include <span>
#include <concepts>
#include <cassert>
#include <algorithm>
#include <cmath>

namespace mango::operators {

namespace stdx = std::experimental;

/**
 * CenteredDifferenceSIMD: Vectorized stencil operator
 *
 * Replaces scalar std::fma with std::experimental::simd operations.
 * Uses [[gnu::target_clones]] for ISA-specific code generation.
 *
 * REQUIREMENTS:
 * - Input spans must be PADDED (use workspace.u_current_padded(), etc.)
 * - start must be ≥ 1 (no boundary point)
 * - end must be ≤ u.size() - 1 (no boundary point)
 * - Boundary conditions handled separately by caller
 */
template<std::floating_point T = double>
class CenteredDifferenceSIMD {
public:
    using simd_t = stdx::native_simd<T>;
    static constexpr size_t simd_width = simd_t::size();

    explicit CenteredDifferenceSIMD(const GridSpacing<T>& spacing,
                                   size_t l1_tile_size = 1024)
        : spacing_(spacing)
        , l1_tile_size_(l1_tile_size)
    {}

    /**
     * Vectorized second derivative kernel (uniform grid)
     *
     * Marked with target_clones for ISA-specific code generation:
     * - default: SSE2 baseline (simd_width = 2 for double)
     * - avx2: Haswell+ (simd_width = 4 for double)
     * - avx512f: Skylake-X+ (simd_width = 8 for double)
     *
     * Verify with: objdump -d <binary> | grep -A20 compute_second_derivative_uniform
     */
    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_second_derivative_uniform(
        std::span<const T> u,
        std::span<T> d2u_dx2,
        size_t start,
        size_t end) const
    {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");

        const T dx2_inv = spacing_.spacing_inv_sq();
        const simd_t dx2_inv_vec(dx2_inv);
        const simd_t minus_two(T(-2));

        // Vectorized main loop
        size_t i = start;
        for (; i + simd_width <= end; i += simd_width) {
            // SoA layout ensures contiguous loads (no gather needed)
            simd_t u_left, u_center, u_right;
            u_left.copy_from(u.data() + i - 1, stdx::element_aligned);
            u_center.copy_from(u.data() + i, stdx::element_aligned);
            u_right.copy_from(u.data() + i + 1, stdx::element_aligned);

            // d2u/dx2 = (u[i+1] + u[i-1] - 2*u[i]) * dx2_inv
            const simd_t sum = u_left + u_right;
            const simd_t result = stdx::fma(sum, dx2_inv_vec,
                                           minus_two * u_center * dx2_inv_vec);

            result.copy_to(d2u_dx2.data() + i, stdx::element_aligned);
        }

        // Scalar tail (zero-padded arrays allow safe i+1 access)
        for (; i < end; ++i) {
            d2u_dx2[i] = std::fma(u[i+1] + u[i-1], dx2_inv, T(-2) * u[i] * dx2_inv);
        }
    }

    /**
     * Tiled second derivative (cache-friendly)
     *
     * Operator decides tile size based on stencil width and cache target.
     */
    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_second_derivative_tiled(
        std::span<const T> u,
        std::span<T> d2u_dx2,
        size_t start,
        size_t end) const
    {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");

        for (size_t tile_start = start; tile_start < end; tile_start += l1_tile_size_) {
            const size_t tile_end = std::min(tile_start + l1_tile_size_, end);
            compute_second_derivative_uniform(u, d2u_dx2, tile_start, tile_end);
        }
    }

    /**
     * First derivative (vectorized, uniform grid)
     */
    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_first_derivative_uniform(
        std::span<const T> u,
        std::span<T> du_dx,
        size_t start,
        size_t end) const
    {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");

        const T half_dx_inv = spacing_.spacing_inv() * T(0.5);
        const simd_t half_dx_inv_vec(half_dx_inv);

        size_t i = start;
        for (; i + simd_width <= end; i += simd_width) {
            simd_t u_left, u_right;
            u_left.copy_from(u.data() + i - 1, stdx::element_aligned);
            u_right.copy_from(u.data() + i + 1, stdx::element_aligned);

            const simd_t result = (u_right - u_left) * half_dx_inv_vec;
            result.copy_to(du_dx.data() + i, stdx::element_aligned);
        }

        for (; i < end; ++i) {
            du_dx[i] = (u[i+1] - u[i-1]) * half_dx_inv;
        }
    }

    /**
     * Second derivative (vectorized, non-uniform grid)
     *
     * Uses precomputed spacing arrays from GridSpacing to avoid divisions.
     * SIMD kernel loads dx_left_inv, dx_right_inv, dx_center_inv for each point.
     * Scalar tail uses same precomputed arrays for exact numerical match.
     */
    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_second_derivative_non_uniform(
        std::span<const T> u,
        std::span<T> d2u_dx2,
        size_t start,
        size_t end) const
    {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");
        assert(!spacing_.is_uniform() && "Use compute_second_derivative_uniform for uniform grids");

        // Get precomputed arrays (zero-copy spans)
        auto dx_left_inv = spacing_.dx_left_inv();
        auto dx_right_inv = spacing_.dx_right_inv();
        auto dx_center_inv = spacing_.dx_center_inv();

        // Vectorized main loop
        size_t i = start;
        for (; i + simd_width <= end; i += simd_width) {
            // Load u values
            simd_t u_left, u_center, u_right;
            u_left.copy_from(u.data() + i - 1, stdx::element_aligned);
            u_center.copy_from(u.data() + i, stdx::element_aligned);
            u_right.copy_from(u.data() + i + 1, stdx::element_aligned);

            // Load precomputed spacing inverses
            simd_t dxl_inv, dxr_inv, dxc_inv;
            dxl_inv.copy_from(dx_left_inv.data() + i - 1, stdx::element_aligned);
            dxr_inv.copy_from(dx_right_inv.data() + i - 1, stdx::element_aligned);
            dxc_inv.copy_from(dx_center_inv.data() + i - 1, stdx::element_aligned);

            // d²u/dx² = ((u[i+1] - u[i])/dx_right - (u[i] - u[i-1])/dx_left) / dx_center
            const simd_t forward_diff = (u_right - u_center) * dxr_inv;
            const simd_t backward_diff = (u_center - u_left) * dxl_inv;
            const simd_t result = (forward_diff - backward_diff) * dxc_inv;

            result.copy_to(d2u_dx2.data() + i, stdx::element_aligned);
        }

        // Scalar tail: use precomputed values for exact match
        for (; i < end; ++i) {
            const T dxl_inv = dx_left_inv[i - 1];
            const T dxr_inv = dx_right_inv[i - 1];
            const T dxc_inv = dx_center_inv[i - 1];

            const T forward_diff = (u[i+1] - u[i]) * dxr_inv;
            const T backward_diff = (u[i] - u[i-1]) * dxl_inv;
            d2u_dx2[i] = (forward_diff - backward_diff) * dxc_inv;
        }
    }

    size_t tile_size() const { return l1_tile_size_; }

private:
    const GridSpacing<T>& spacing_;
    size_t l1_tile_size_;
};

} // namespace mango::operators
