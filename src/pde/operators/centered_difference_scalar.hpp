#pragma once

#include "src/pde/core/grid.hpp"
#include "src/support/parallel.hpp"
#include <span>
#include <cmath>
#include <cassert>

namespace mango::operators {

/**
 * ScalarBackend: Scalar implementation with compiler auto-vectorization
 *
 * Uses #pragma omp simd hints for automatic vectorization.
 * For non-uniform grids, loads from precomputed GridSpacing arrays.
 */
template<std::floating_point T = double>
class ScalarBackend {
public:
    explicit ScalarBackend(const GridSpacing<T>& spacing)
        : spacing_(spacing)
    {}

    // Uniform grid second derivative
    void compute_second_derivative_uniform(
        std::span<const T> u, std::span<T> d2u_dx2,
        size_t start, size_t end) const
    {
        const T dx2_inv = spacing_.spacing_inv_sq();

        MANGO_PRAGMA_SIMD
        for (size_t i = start; i < end; ++i) {
            d2u_dx2[i] = std::fma(u[i+1] + u[i-1], dx2_inv,
                                 -T(2)*u[i]*dx2_inv);
        }
    }

    // Uniform grid first derivative
    void compute_first_derivative_uniform(
        std::span<const T> u, std::span<T> du_dx,
        size_t start, size_t end) const
    {
        const T half_dx_inv = spacing_.spacing_inv() * T(0.5);

        MANGO_PRAGMA_SIMD
        for (size_t i = start; i < end; ++i) {
            du_dx[i] = (u[i+1] - u[i-1]) * half_dx_inv;
        }
    }

    // Non-uniform grid second derivative - USES PRECOMPUTED ARRAYS
    void compute_second_derivative_non_uniform(
        std::span<const T> u, std::span<T> d2u_dx2,
        size_t start, size_t end) const
    {
        auto dx_left_inv = spacing_.dx_left_inv();
        auto dx_right_inv = spacing_.dx_right_inv();
        auto dx_center_inv = spacing_.dx_center_inv();

        MANGO_PRAGMA_SIMD
        for (size_t i = start; i < end; ++i) {
            const T dxl_inv = dx_left_inv[i - 1];
            const T dxr_inv = dx_right_inv[i - 1];
            const T dxc_inv = dx_center_inv[i - 1];

            const T forward_diff = (u[i+1] - u[i]) * dxr_inv;
            const T backward_diff = (u[i] - u[i-1]) * dxl_inv;
            d2u_dx2[i] = (forward_diff - backward_diff) * dxc_inv;
        }
    }

    // Non-uniform grid first derivative - USES PRECOMPUTED ARRAYS
    void compute_first_derivative_non_uniform(
        std::span<const T> u, std::span<T> du_dx,
        size_t start, size_t end) const
    {
        auto w_left = spacing_.w_left();
        auto w_right = spacing_.w_right();
        auto dx_left_inv = spacing_.dx_left_inv();
        auto dx_right_inv = spacing_.dx_right_inv();

        MANGO_PRAGMA_SIMD
        for (size_t i = start; i < end; ++i) {
            const T wl = w_left[i - 1];
            const T wr = w_right[i - 1];
            const T dxl_inv = dx_left_inv[i - 1];
            const T dxr_inv = dx_right_inv[i - 1];

            const T term1 = wl * (u[i] - u[i-1]) * dxl_inv;
            const T term2 = wr * (u[i+1] - u[i]) * dxr_inv;
            du_dx[i] = term1 + term2;
        }
    }

    // Auto-dispatch second derivative
    void compute_second_derivative(
        std::span<const T> u, std::span<T> d2u_dx2,
        size_t start, size_t end) const
    {
        if (spacing_.is_uniform()) {
            compute_second_derivative_uniform(u, d2u_dx2, start, end);
        } else {
            compute_second_derivative_non_uniform(u, d2u_dx2, start, end);
        }
    }

    // Auto-dispatch first derivative
    void compute_first_derivative(
        std::span<const T> u, std::span<T> du_dx,
        size_t start, size_t end) const
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
