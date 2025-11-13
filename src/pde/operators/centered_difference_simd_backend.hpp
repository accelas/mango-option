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
 * SimdBackend: Vectorized stencil operator (renamed from CenteredDifferenceSIMD)
 *
 * Replaces scalar std::fma with std::experimental::simd operations.
 * Uses [[gnu::target_clones]] for ISA-specific code generation.
 *
 * SUPPORTED GRIDS:
 * - Uniform: Uses constant spacing (spacing_inv, spacing_inv_sq)
 * - Non-uniform: Uses precomputed weight arrays from GridSpacing
 *
 * NON-UNIFORM GRID SUPPORT:
 * For non-uniform (tanh-clustered) grids, GridSpacing precomputes:
 *   dx_left_inv[i], dx_right_inv[i], dx_center_inv[i], w_left[i], w_right[i]
 * in a single contiguous buffer during construction.
 *
 * SIMD kernels load these values via zero-copy spans, avoiding per-lane
 * divisions. Expected speedup: 3-6x over scalar non-uniform code.
 *
 * USAGE:
 *   // Explicit dispatch (performance-critical paths)
 *   if (spacing.is_uniform()) {
 *     stencil.compute_second_derivative_uniform(u, d2u_dx2, 1, n-1);
 *   } else {
 *     stencil.compute_second_derivative_non_uniform(u, d2u_dx2, 1, n-1);
 *   }
 *
 *   // Convenience wrapper (tests, examples)
 *   stencil.compute_second_derivative(u, d2u_dx2, 1, n-1);  // Auto-dispatch
 *
 * REQUIREMENTS:
 * - Input spans must be PADDED (use workspace.u_current_padded(), etc.)
 * - start must be ≥ 1 (no boundary point)
 * - end must be ≤ u.size() - 1 (no boundary point)
 * - Boundary conditions handled separately by caller
 */
template<std::floating_point T = double>
class SimdBackend {
public:
    using simd_t = stdx::native_simd<T>;
    static constexpr size_t simd_width = simd_t::size();

    explicit SimdBackend(const GridSpacing<T>& spacing)
        : spacing_(spacing)
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

    /**
     * First derivative (vectorized, non-uniform grid)
     *
     * Uses precomputed weight arrays from GridSpacing to achieve second-order
     * accuracy on non-uniform grids. SIMD kernel loads w_left, w_right,
     * dx_left_inv, dx_right_inv for each point.
     * Scalar tail uses same precomputed arrays for exact numerical match.
     *
     * Formula: du/dx = w_left * (u[i] - u[i-1])/dx_left + w_right * (u[i+1] - u[i])/dx_right
     * where w_left = dx_right / (dx_left + dx_right), w_right = dx_left / (dx_left + dx_right)
     */
    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_first_derivative_non_uniform(
        std::span<const T> u,
        std::span<T> du_dx,
        size_t start,
        size_t end) const
    {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");
        assert(!spacing_.is_uniform() && "Use compute_first_derivative_uniform for uniform grids");

        // Get precomputed arrays
        auto w_left = spacing_.w_left();
        auto w_right = spacing_.w_right();
        auto dx_left_inv = spacing_.dx_left_inv();
        auto dx_right_inv = spacing_.dx_right_inv();

        // Vectorized main loop
        size_t i = start;
        for (; i + simd_width <= end; i += simd_width) {
            // Load u values
            simd_t u_left, u_center, u_right;
            u_left.copy_from(u.data() + i - 1, stdx::element_aligned);
            u_center.copy_from(u.data() + i, stdx::element_aligned);
            u_right.copy_from(u.data() + i + 1, stdx::element_aligned);

            // Load precomputed weights and inverses
            simd_t wl, wr, dxl_inv, dxr_inv;
            wl.copy_from(w_left.data() + i - 1, stdx::element_aligned);
            wr.copy_from(w_right.data() + i - 1, stdx::element_aligned);
            dxl_inv.copy_from(dx_left_inv.data() + i - 1, stdx::element_aligned);
            dxr_inv.copy_from(dx_right_inv.data() + i - 1, stdx::element_aligned);

            // du/dx = w_left * (u[i] - u[i-1])/dx_left + w_right * (u[i+1] - u[i])/dx_right
            const simd_t term1 = wl * (u_center - u_left) * dxl_inv;
            const simd_t term2 = wr * (u_right - u_center) * dxr_inv;
            const simd_t result = term1 + term2;

            result.copy_to(du_dx.data() + i, stdx::element_aligned);
        }

        // Scalar tail: use precomputed values
        for (; i < end; ++i) {
            const T wl = w_left[i - 1];
            const T wr = w_right[i - 1];
            const T dxl_inv = dx_left_inv[i - 1];
            const T dxr_inv = dx_right_inv[i - 1];

            const T term1 = wl * (u[i] - u[i-1]) * dxl_inv;
            const T term2 = wr * (u[i+1] - u[i]) * dxr_inv;
            du_dx[i] = term1 + term2;
        }
    }

    /**
     * Convenience wrapper for second derivative (automatic dispatch)
     *
     * Automatically dispatches to uniform or non-uniform implementation
     * based on grid type. Both variants get ISA-specific code generation
     * via [[gnu::target_clones]].
     *
     * Use this for tests, examples, or when grid type is runtime-determined.
     * For performance-critical paths with known grid type at compile time,
     * prefer explicit compute_second_derivative_uniform() or
     * compute_second_derivative_non_uniform() to avoid branch overhead.
     */
    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_second_derivative(
        std::span<const T> u,
        std::span<T> d2u_dx2,
        size_t start,
        size_t end) const
    {
        if (spacing_.is_uniform()) {
            compute_second_derivative_uniform(u, d2u_dx2, start, end);
        } else {
            compute_second_derivative_non_uniform(u, d2u_dx2, start, end);
        }
    }

    /**
     * Convenience wrapper for first derivative (automatic dispatch)
     *
     * Automatically dispatches to uniform or non-uniform implementation
     * based on grid type. Both variants get ISA-specific code generation
     * via [[gnu::target_clones]].
     *
     * Use this for tests, examples, or when grid type is runtime-determined.
     * For performance-critical paths with known grid type at compile time,
     * prefer explicit compute_first_derivative_uniform() or
     * compute_first_derivative_non_uniform() to avoid branch overhead.
     */
    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_first_derivative(
        std::span<const T> u,
        std::span<T> du_dx,
        size_t start,
        size_t end) const
    {
        if (spacing_.is_uniform()) {
            compute_first_derivative_uniform(u, du_dx, start, end);
        } else {
            compute_first_derivative_non_uniform(u, du_dx, start, end);
        }
    }

private:
    const GridSpacing<T>& spacing_;
};

} // namespace mango::operators
