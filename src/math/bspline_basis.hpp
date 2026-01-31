// SPDX-License-Identifier: MIT
/**
 * @file bspline_basis.hpp
 * @brief Generic B-spline basis function evaluation
 *
 * Provides fundamental B-spline operations:
 * - Clamped knot vector construction
 * - Knot span finding (binary search)
 * - Cox-de Boor basis recursion (degree p=3)
 * - Basis function derivatives
 *
 * These utilities are dimension-agnostic and support separable
 * tensor-product B-spline fitting in arbitrary dimensions.
 *
 * References:
 * - de Boor, "A Practical Guide to Splines" (2001)
 * - Piegl & Tiller, "The NURBS Book" (1997)
 */

#pragma once

#include <vector>
#include <span>
#include <concepts>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cassert>

namespace mango {

/// Create clamped knot vector for cubic B-splines (std::vector overload)
///
/// For n data points, creates n+4 knots with repeated endpoints:
///   [x₀, x₀, x₀, x₀, t₁, ..., tₘ, xₙ₋₁, xₙ₋₁, xₙ₋₁, xₙ₋₁]
///
/// The interior knots t₁...tₘ are placed between data sites to ensure
/// the Schoenberg-Whitney condition holds (collocation matrix is non-singular).
///
/// **Boundary interpolation:** Clamping ensures B-spline interpolates exactly
/// at the first and last data points (multiplicity p+1 = 4 for cubics).
///
/// @tparam T Floating point type
/// @param x Data grid points (must be sorted)
/// @return Clamped knot vector (size = n + 4)
template<std::floating_point T>
[[nodiscard]] std::vector<T> clamped_knots_cubic(const std::vector<T>& x) {
    const int n = static_cast<int>(x.size());
    std::vector<T> t(n + 4);

    // Left clamp: repeat first point 4 times
    std::fill_n(t.begin(), 4, x.front());

    // Interior knots positioned strictly between data sites (midpoints)
    if (n > 4) {
        const int interior = n - 4;
        const int intervals = n - 1;

        for (int idx = 0; idx < interior; ++idx) {
            // Proportional placement: map idx → continuous position
            const T ratio = static_cast<T>(idx + 1) / static_cast<T>(interior + 1);
            T pos = ratio * static_cast<T>(intervals);

            // Find interval containing this position
            int low = static_cast<int>(std::floor(pos));
            if (low >= intervals) {
                low = intervals - 1;
            }

            // Interpolate knot position within interval
            const T frac = pos - static_cast<T>(low);
            const T left = x[low];
            const T right = x[low + 1];
            T knot = (T{1} - frac) * left + frac * right;

            // Clamp to interior of interval (avoid coinciding with data sites)
            const T spacing = right - left;
            const T eps = std::max(T{1e-12} * spacing,
                                  std::numeric_limits<T>::epsilon() *
                                      std::max(std::abs(right), T{1}));
            knot = std::clamp(knot, left + eps, right - eps);

            t[4 + idx] = knot;
        }
    }

    // Right clamp: repeat last point 4 times
    std::fill_n(t.end() - 4, 4, x.back());

    return t;
}

/// Create clamped knot vector for cubic B-splines (std::span overload)
///
/// Overload for std::span arguments (used by some workspace code).
///
/// @tparam T Floating point type
/// @param x Data grid points as span (must be sorted)
/// @return Clamped knot vector (size = n + 4)
template<std::floating_point T>
[[nodiscard]] std::vector<T> clamped_knots_cubic(std::span<const T> x) {
    // Convert span to vector and call vector overload
    return clamped_knots_cubic(std::vector<T>(x.begin(), x.end()));
}

/// Find knot span containing x using binary search
///
/// Returns index i such that t[i] ≤ x < t[i+1] for cubic B-splines (degree 3).
/// Handles boundary cases correctly (clamping to valid span range).
///
/// **Algorithm:** Binary search in [p, n) where p=3 (degree), n = control points
///
/// Time: O(log n)
///
/// @tparam T Floating point type
/// @param t Knot vector (clamped, size = n + 4 for cubic)
/// @param x Query point
/// @return Knot span index i ∈ [3, n-1]
template<std::floating_point T>
[[nodiscard]] int find_span_cubic(const std::vector<T>& t, T x) noexcept {
    constexpr int DEGREE = 3;
    const int n_ctrl = static_cast<int>(t.size()) - DEGREE - 1;
    const int min_span = DEGREE;
    const int max_span = std::max(min_span, n_ctrl - 1);

    // Boundary cases: clamp to valid span range
    if (x <= t[min_span]) {
        return min_span;
    }
    if (x >= t[n_ctrl]) {
        return max_span;
    }

    // Binary search in [min_span, n_ctrl]
    auto it = std::upper_bound(t.begin() + min_span, t.begin() + n_ctrl + 1, x);
    int i = static_cast<int>(std::distance(t.begin(), it)) - 1;

    // Clamp result to valid range [min_span, max_span]
    i = std::clamp(i, min_span, max_span);

    return i;
}

/// Evaluate cubic B-spline basis functions using Cox-de Boor recursion
///
/// Computes the 4 nonzero cubic basis functions at x for knot span i.
/// Uses de Boor's stable recursive formula with proper handling of zero denominators.
///
/// **Output ordering:** N[0] = Bᵢ(x), N[1] = Bᵢ₋₁(x), N[2] = Bᵢ₋₂(x), N[3] = Bᵢ₋₃(x)
///
/// **Cox-de Boor recursion:**
///   Nᵢ,ₚ(x) = [(x - tᵢ)/(tᵢ₊ₚ - tᵢ)] · Nᵢ,ₚ₋₁(x) + [(tᵢ₊ₚ₊₁ - x)/(tᵢ₊ₚ₊₁ - tᵢ₊₁)] · Nᵢ₊₁,ₚ₋₁(x)
///
/// **Numerical stability:** Checks for zero denominators before division
///
/// Time: O(p²) = O(9) for cubic
///
/// @tparam T Floating point type
/// @param t Knot vector
/// @param i Knot span index (from find_span_cubic)
/// @param x Evaluation point
/// @param N Output: 4 basis function values N[0..3]
template<std::floating_point T>
void cubic_basis_nonuniform(
    const std::vector<T>& t,
    int i,
    T x,
    T N[4]) noexcept
{
    const int n = static_cast<int>(t.size());

    // Exact interpolation at right boundary (avoids numerical errors)
    if (std::abs(x - t.back()) < T{1e-14}) {
        N[0] = T{1};
        N[1] = T{0};
        N[2] = T{0};
        N[3] = T{0};
        return;
    }

    // Degree 0: piecewise constants (indicator functions)
    T N0[4] = {T{0}, T{0}, T{0}, T{0}};
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 1 < n) {
            N0[k] = (t[idx] <= x && x < t[idx + 1]) ? T{1} : T{0};
        }
    }

    // Degree 1: linear combination
    T N1[4] = {T{0}, T{0}, T{0}, T{0}};
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 2 < n) {
            const T leftDen  = t[idx + 1] - t[idx];
            const T rightDen = t[idx + 2] - t[idx + 1];

            const T left  = (leftDen > T{0}) ? (x - t[idx]) / leftDen * N0[k] : T{0};
            const T right = (rightDen > T{0} && k > 0) ?
                           (t[idx + 2] - x) / rightDen * N0[k - 1] : T{0};

            N1[k] = left + right;
        }
    }

    // Degree 2: quadratic
    T N2[4] = {T{0}, T{0}, T{0}, T{0}};
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 3 < n) {
            const T leftDen  = t[idx + 2] - t[idx];
            const T rightDen = t[idx + 3] - t[idx + 1];

            const T left  = (leftDen > T{0}) ? (x - t[idx]) / leftDen * N1[k] : T{0};
            const T right = (rightDen > T{0} && k > 0) ?
                           (t[idx + 3] - x) / rightDen * N1[k - 1] : T{0};

            N2[k] = left + right;
        }
    }

    // Degree 3: cubic (final result)
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 4 < n) {
            const T leftDen  = t[idx + 3] - t[idx];
            const T rightDen = t[idx + 4] - t[idx + 1];

            const T left  = (leftDen > T{0}) ? (x - t[idx]) / leftDen * N2[k] : T{0};
            const T right = (rightDen > T{0} && k > 0) ?
                           (t[idx + 4] - x) / rightDen * N2[k - 1] : T{0};

            N[k] = left + right;
        } else {
            N[k] = T{0};
        }
    }
}

/// Evaluate derivatives of cubic B-spline basis functions
///
/// Computes dNᵢ/dx for the 4 nonzero cubic basis functions at x.
/// Uses Cox-de Boor derivative formula expressed in terms of quadratic basis functions.
///
/// **Derivative formula:**
///   B'ᵢ,ₚ(x) = p/(tᵢ₊ₚ - tᵢ) · Bᵢ,ₚ₋₁(x) - p/(tᵢ₊ₚ₊₁ - tᵢ₊₁) · Bᵢ₊₁,ₚ₋₁(x)
///
/// For cubic (p=3), derivatives are expressed in terms of quadratic (p=2) basis.
///
/// **Output ordering:** dN[0] = B'ᵢ(x), dN[1] = B'ᵢ₋₁(x), dN[2] = B'ᵢ₋₂(x), dN[3] = B'ᵢ₋₃(x)
///
/// Time: O(p²) = O(9) for cubic
///
/// @tparam T Floating point type
/// @param t Knot vector
/// @param i Knot span index
/// @param x Evaluation point
/// @param dN Output: 4 basis function derivatives dN[0..3]
template<std::floating_point T>
void cubic_basis_derivative_nonuniform(
    const std::vector<T>& t,
    int i,
    T x,
    T dN[4]) noexcept
{
    const int n = static_cast<int>(t.size());

    // Build quadratic basis functions (degree 2) for derivative computation
    // Degree 0: piecewise constants
    T N0[4] = {T{0}, T{0}, T{0}, T{0}};
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 1 < n) {
            N0[k] = (t[idx] <= x && x < t[idx + 1]) ? T{1} : T{0};
        }
    }

    // Degree 1: linear
    T N1[4] = {T{0}, T{0}, T{0}, T{0}};
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 2 < n) {
            const T leftDen  = t[idx + 1] - t[idx];
            const T rightDen = t[idx + 2] - t[idx + 1];

            const T left  = (leftDen > T{0}) ? (x - t[idx]) / leftDen * N0[k] : T{0};
            const T right = (rightDen > T{0} && k > 0) ?
                           (t[idx + 2] - x) / rightDen * N0[k - 1] : T{0};

            N1[k] = left + right;
        }
    }

    // Degree 2: quadratic (needed for cubic derivatives)
    T N2[4] = {T{0}, T{0}, T{0}, T{0}};
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 3 < n) {
            const T leftDen  = t[idx + 2] - t[idx];
            const T rightDen = t[idx + 3] - t[idx + 1];

            const T left  = (leftDen > T{0}) ? (x - t[idx]) / leftDen * N1[k] : T{0};
            const T right = (rightDen > T{0} && k > 0) ?
                           (t[idx + 3] - x) / rightDen * N1[k - 1] : T{0};

            N2[k] = left + right;
        }
    }

    // Apply derivative formula for cubic (p=3):
    // B'ᵢ,₃(x) = 3/(tᵢ₊₃ - tᵢ) · Bᵢ,₂(x) - 3/(tᵢ₊₄ - tᵢ₊₁) · Bᵢ₊₁,₂(x)
    constexpr T p = T{3};
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 4 < n) {
            const T leftDen  = t[idx + 3] - t[idx];
            const T rightDen = t[idx + 4] - t[idx + 1];

            // First term: p/(tᵢ₊ₚ - tᵢ) · Bᵢ,ₚ₋₁(x)
            const T left = (leftDen > T{0}) ? (p / leftDen) * N2[k] : T{0};

            // Second term: -p/(tᵢ₊ₚ₊₁ - tᵢ₊₁) · Bᵢ₊₁,ₚ₋₁(x)
            const T right = (rightDen > T{0} && k > 0) ? (p / rightDen) * N2[k - 1] : T{0};

            dN[k] = left - right;
        } else {
            dN[k] = T{0};
        }
    }
}

/// Compute second derivatives of 4 nonzero cubic B-spline basis functions
///
/// Uses the recurrence: B''_{i,3}(x) = 3 · [ B'_{i,2}(x)/(t_{i+3}-t_i) - B'_{i+1,2}(x)/(t_{i+4}-t_{i+1}) ]
/// where B'_{i,2}(x) = 2 · [ B_{i,1}(x)/(t_{i+2}-t_i) - B_{i+1,1}(x)/(t_{i+3}-t_{i+1}) ]
///
/// @tparam T Floating point type (float, double, etc.)
/// @param t Knot vector (size = n_control_points + 4 for cubic)
/// @param i Knot span index from find_span_cubic()
/// @param x Evaluation point
/// @param d2N Output: 4 basis function second derivatives d2N[0..3]
template<std::floating_point T>
void cubic_basis_second_derivative_nonuniform(
    const std::vector<T>& t,
    int i,
    T x,
    T d2N[4]) noexcept
{
    const int n = static_cast<int>(t.size());

    // Degree 0: piecewise constants
    T N0[4] = {T{0}, T{0}, T{0}, T{0}};
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 1 < n) {
            N0[k] = (t[idx] <= x && x < t[idx + 1]) ? T{1} : T{0};
        }
    }

    // Degree 1: linear basis
    T N1[4] = {T{0}, T{0}, T{0}, T{0}};
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 2 < n) {
            const T leftDen  = t[idx + 1] - t[idx];
            const T rightDen = t[idx + 2] - t[idx + 1];

            const T left  = (leftDen > T{0}) ? (x - t[idx]) / leftDen * N0[k] : T{0};
            const T right = (rightDen > T{0} && k > 0) ?
                           (t[idx + 2] - x) / rightDen * N0[k - 1] : T{0};

            N1[k] = left + right;
        }
    }

    // First derivatives of quadratic basis (degree 2):
    // B'_{i,2}(x) = 2/(t_{i+2}-t_i) · B_{i,1}(x) - 2/(t_{i+3}-t_{i+1}) · B_{i+1,1}(x)
    T dN2[4] = {T{0}, T{0}, T{0}, T{0}};
    constexpr T p2 = T{2};
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 3 < n) {
            const T leftDen  = t[idx + 2] - t[idx];
            const T rightDen = t[idx + 3] - t[idx + 1];

            const T left = (leftDen > T{0}) ? (p2 / leftDen) * N1[k] : T{0};
            const T right = (rightDen > T{0} && k > 0) ? (p2 / rightDen) * N1[k - 1] : T{0};

            dN2[k] = left - right;
        }
    }

    // Second derivative of cubic (p=3):
    // B''_{i,3}(x) = 3/(t_{i+3}-t_i) · B'_{i,2}(x) - 3/(t_{i+4}-t_{i+1}) · B'_{i+1,2}(x)
    constexpr T p3 = T{3};
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 4 < n) {
            const T leftDen  = t[idx + 3] - t[idx];
            const T rightDen = t[idx + 4] - t[idx + 1];

            const T left = (leftDen > T{0}) ? (p3 / leftDen) * dN2[k] : T{0};
            const T right = (rightDen > T{0} && k > 0) ? (p3 / rightDen) * dN2[k - 1] : T{0};

            d2N[k] = left - right;
        } else {
            d2N[k] = T{0};
        }
    }
}

/// Clamp query point to valid B-spline domain
///
/// For evaluation at boundaries, uses nextafter to ensure x < xmax
/// (handles half-open interval [xmin, xmax) correctly).
///
/// @tparam T Floating point type
/// @param x Query point
/// @param xmin Minimum value
/// @param xmax Maximum value
/// @return Clamped value
template<std::floating_point T>
[[nodiscard]] inline T clamp_query(T x, T xmin, T xmax) noexcept {
    if (x <= xmin) return xmin;
    if (x >= xmax) {
        return std::nextafter(xmax, -std::numeric_limits<T>::infinity());
    }
    return x;
}

}  // namespace mango
