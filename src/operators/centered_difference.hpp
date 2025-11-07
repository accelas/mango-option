#pragma once

#include "grid_spacing.hpp"
#include "../parallel.hpp"
#include <span>
#include <cmath>

namespace mango::operators {

/**
 * CenteredDifference: Centered finite difference stencil
 *
 * Computes first and second derivatives using centered differences.
 * Adapts automatically to uniform vs non-uniform grids via GridSpacing.
 *
 * Uniform grid formulas:
 *   du/dx = (u[i+1] - u[i-1]) / (2*dx)
 *   d²u/dx² = (u[i+1] - 2*u[i] + u[i-1]) / dx²
 *
 * Non-uniform grid formulas:
 *   du/dx = w_left*(u[i] - u[i-1])/dx_left + w_right*(u[i+1] - u[i])/dx_right
 *     where w_left = dx_right/(dx_left + dx_right), w_right = dx_left/(dx_left + dx_right)
 *   d²u/dx² = 2 * ((u[i+1] - u[i])/dx_right - (u[i] - u[i-1])/dx_left) / (dx_left + dx_right)
 *
 * Single Responsibility: Finite difference discretization
 *
 * Note: Uses std::fma for improved precision and performance, so T must be
 * a standard floating-point type (float, double, long double).
 */
template<std::floating_point T = double>
class CenteredDifference {
public:
    /**
     * Construct centered difference stencil
     * @param spacing Grid spacing information
     */
    explicit CenteredDifference(const GridSpacing<T>& spacing)
        : spacing_(spacing)
    {}

    /**
     * Compute first derivative at interior point i
     * @param u Field values
     * @param i Interior point index (must satisfy 1 <= i < n-1)
     * @return du/dx at point i
     */
    T first_derivative(std::span<const T> u, size_t i) const {
        if (spacing_.is_uniform()) {
            // Uniform: (u[i+1] - u[i-1]) / (2*dx)
            return (u[i+1] - u[i-1]) * spacing_.spacing_inv() * T(0.5);
        } else {
            // Non-uniform: weighted three-point (2nd order)
            const T dx_left = spacing_.spacing_at(i - 1);
            const T dx_right = spacing_.spacing_at(i);
            const T w_left = dx_right / (dx_left + dx_right);
            const T w_right = dx_left / (dx_left + dx_right);
            // Use FMA for the weighted sum
            return std::fma(w_left, (u[i] - u[i-1]) / dx_left,
                           w_right * (u[i+1] - u[i]) / dx_right);
        }
    }

    /**
     * Compute second derivative at interior point i
     * @param u Field values
     * @param i Interior point index (must satisfy 1 <= i < n-1)
     * @return d²u/dx² at point i
     */
    T second_derivative(std::span<const T> u, size_t i) const {
        if (spacing_.is_uniform()) {
            // Uniform: (u[i+1] - 2*u[i] + u[i-1]) / dx²
            // Use FMA: (u[i+1] + u[i-1]) * dx2_inv - 2*u[i]*dx2_inv
            return std::fma(u[i+1] + u[i-1], spacing_.spacing_inv_sq(),
                           -T(2)*u[i]*spacing_.spacing_inv_sq());
        } else {
            // Non-uniform: 2 * ((u[i+1] - u[i])/dx_right - (u[i] - u[i-1])/dx_left) / (dx_left + dx_right)
            const T dx_left = spacing_.spacing_at(i - 1);
            const T dx_right = spacing_.spacing_at(i);
            const T dx_center = T(0.5) * (dx_left + dx_right);

            const T forward_diff = (u[i+1] - u[i]) / dx_right;
            const T backward_diff = (u[i] - u[i-1]) / dx_left;

            return (forward_diff - backward_diff) / dx_center;
        }
    }

    /**
     * Compute both derivatives at interior point i
     * @param u Field values
     * @param i Interior point index (must satisfy 1 <= i < n-1)
     * @return Pair of (du/dx, d²u/dx²)
     */
    std::pair<T, T> derivatives(std::span<const T> u, size_t i) const {
        if (spacing_.is_uniform()) {
            // Uniform: optimize by computing both at once
            const T du_dx = (u[i+1] - u[i-1]) * spacing_.spacing_inv() * T(0.5);
            // Use FMA for second derivative
            const T d2u_dx2 = std::fma(u[i+1] + u[i-1], spacing_.spacing_inv_sq(),
                                      -T(2)*u[i]*spacing_.spacing_inv_sq());
            return {du_dx, d2u_dx2};
        } else {
            // Non-uniform: compute independently
            return {first_derivative(u, i), second_derivative(u, i)};
        }
    }

    // Optimized vectorized path for uniform grids (fused kernel)
    // Evaluator: (T d2u_dx2, T du_dx, T u) -> T
    template<typename Evaluator>
    void apply_uniform(std::span<const T> u,
                      std::span<T> Lu,
                      size_t start,
                      size_t end,
                      Evaluator&& eval) const {
        const T half_dx_inv = spacing_.spacing_inv() * T(0.5);
        const T dx2_inv = spacing_.spacing_inv_sq();

        MANGO_PRAGMA_SIMD
        for (size_t i = start; i < end; ++i) {
            const T du_dx = (u[i+1] - u[i-1]) * half_dx_inv;
            // d2u_dx2 = (u[i+1] + u[i-1] - 2*u[i]) * dx2_inv
            // Use FMA: std::fma(x, y, z) = x*y + z
            const T d2u_dx2 = std::fma(u[i+1] + u[i-1], dx2_inv, -T(2)*u[i]*dx2_inv);
            Lu[i] = eval(d2u_dx2, du_dx, u[i]);  // Lambda inlines away
        }
    }

    // General path for non-uniform grids
    template<typename Evaluator>
    void apply_non_uniform(std::span<const T> u,
                          std::span<T> Lu,
                          size_t start,
                          size_t end,
                          Evaluator&& eval) const {
        // Note: SIMD on non-uniform grids less efficient due to per-point divisions,
        // but modern compilers can still vectorize with proper hints
        MANGO_PRAGMA_SIMD
        for (size_t i = start; i < end; ++i) {
            const T dx_left = spacing_.left_spacing(i);    // x[i] - x[i-1]
            const T dx_right = spacing_.right_spacing(i);  // x[i+1] - x[i]
            const T dx_center = T(0.5) * (dx_left + dx_right);

            // First derivative: weighted three-point (2nd order on non-uniform grids)
            const T w_left = dx_right / (dx_left + dx_right);
            const T w_right = dx_left / (dx_left + dx_right);
            // Use FMA for the weighted sum
            const T du_dx = std::fma(w_left, (u[i] - u[i-1]) / dx_left,
                                     w_right * (u[i+1] - u[i]) / dx_right);

            // Second derivative: non-uniform centered difference
            const T forward_diff = (u[i+1] - u[i]) / dx_right;
            const T backward_diff = (u[i] - u[i-1]) / dx_left;
            const T d2u_dx2 = (forward_diff - backward_diff) / dx_center;

            Lu[i] = eval(d2u_dx2, du_dx, u[i]);
        }
    }

    // All-points first derivative (for Greeks computation)
    void compute_all_first(std::span<const T> u,
                          std::span<T> du_dx,
                          size_t start,
                          size_t end) const {
        if (spacing_.is_uniform()) {
            const T half_dx_inv = spacing_.spacing_inv() * T(0.5);
            MANGO_PRAGMA_SIMD
            for (size_t i = start; i < end; ++i) {
                du_dx[i] = (u[i+1] - u[i-1]) * half_dx_inv;
            }
        } else {
            MANGO_PRAGMA_SIMD
            for (size_t i = start; i < end; ++i) {
                const T dx_left = spacing_.left_spacing(i);
                const T dx_right = spacing_.right_spacing(i);
                // Weighted three-point (2nd order on non-uniform grids)
                const T w_left = dx_right / (dx_left + dx_right);
                const T w_right = dx_left / (dx_left + dx_right);
                // Use FMA for the weighted sum
                du_dx[i] = std::fma(w_left, (u[i] - u[i-1]) / dx_left,
                                    w_right * (u[i+1] - u[i]) / dx_right);
            }
        }
    }

    // All-points second derivative (for Greeks computation)
    void compute_all_second(std::span<const T> u,
                           std::span<T> d2u_dx2,
                           size_t start,
                           size_t end) const {
        if (spacing_.is_uniform()) {
            const T dx2_inv = spacing_.spacing_inv_sq();
            MANGO_PRAGMA_SIMD
            for (size_t i = start; i < end; ++i) {
                // Use FMA: (u[i+1] + u[i-1]) * dx2_inv - 2*u[i]*dx2_inv
                d2u_dx2[i] = std::fma(u[i+1] + u[i-1], dx2_inv, -T(2)*u[i]*dx2_inv);
            }
        } else {
            MANGO_PRAGMA_SIMD
            for (size_t i = start; i < end; ++i) {
                const T dx_left = spacing_.left_spacing(i);
                const T dx_right = spacing_.right_spacing(i);
                const T dx_center = T(0.5) * (dx_left + dx_right);

                const T forward_diff = (u[i+1] - u[i]) / dx_right;
                const T backward_diff = (u[i] - u[i-1]) / dx_left;
                d2u_dx2[i] = (forward_diff - backward_diff) / dx_center;
            }
        }
    }

private:
    const GridSpacing<T>& spacing_;
};

} // namespace mango::operators
