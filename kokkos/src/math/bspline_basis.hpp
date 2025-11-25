#pragma once

/// @file bspline_basis.hpp
/// @brief B-spline basis functions for Kokkos (GPU-compatible)
///
/// Provides fundamental B-spline operations as KOKKOS_INLINE_FUNCTIONs:
/// - Knot span finding (binary search)
/// - Cox-de Boor basis recursion (degree p=3)
/// - Basis function derivatives
///
/// All functions work with raw pointers for GPU compatibility.

#include <Kokkos_Core.hpp>
#include <cmath>
#include <algorithm>

namespace mango::kokkos {

/// Find knot span containing x using binary search (device-compatible)
///
/// Returns index i such that t[i] <= x < t[i+1] for cubic B-splines.
/// Handles boundary cases correctly.
///
/// @param t Knot array pointer
/// @param n_knots Number of knots
/// @param x Query point
/// @return Knot span index i in [3, n_ctrl-1]
KOKKOS_INLINE_FUNCTION
int find_span_cubic(const double* t, int n_knots, double x) noexcept {
    constexpr int DEGREE = 3;
    const int n_ctrl = n_knots - DEGREE - 1;
    const int min_span = DEGREE;
    const int max_span = (n_ctrl > min_span) ? (n_ctrl - 1) : min_span;

    // Boundary cases
    if (x <= t[min_span]) return min_span;
    if (x >= t[n_ctrl]) return max_span;

    // Binary search in [min_span, n_ctrl]
    int low = min_span;
    int high = n_ctrl;
    while (low < high) {
        int mid = (low + high + 1) / 2;
        if (t[mid] <= x) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    // Clamp result
    if (low < min_span) low = min_span;
    if (low > max_span) low = max_span;
    return low;
}

/// Evaluate cubic B-spline basis functions using Cox-de Boor recursion
///
/// Computes the 4 nonzero cubic basis functions at x for knot span i.
/// Output: N[0] = B_i(x), N[1] = B_{i-1}(x), N[2] = B_{i-2}(x), N[3] = B_{i-3}(x)
///
/// @param t Knot array pointer
/// @param n_knots Number of knots
/// @param i Knot span index
/// @param x Evaluation point
/// @param N Output: 4 basis function values
KOKKOS_INLINE_FUNCTION
void cubic_basis_nonuniform(const double* t, int n_knots, int i, double x, double N[4]) noexcept {
    const int n = n_knots;

    // Exact interpolation at right boundary
    if (Kokkos::abs(x - t[n - 1]) < 1e-14) {
        N[0] = 1.0; N[1] = 0.0; N[2] = 0.0; N[3] = 0.0;
        return;
    }

    // Degree 0: piecewise constants
    double N0[4] = {0.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 1 < n) {
            N0[k] = (t[idx] <= x && x < t[idx + 1]) ? 1.0 : 0.0;
        }
    }

    // Degree 1: linear
    double N1[4] = {0.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 2 < n) {
            const double leftDen = t[idx + 1] - t[idx];
            const double rightDen = t[idx + 2] - t[idx + 1];

            const double left = (leftDen > 0.0) ? (x - t[idx]) / leftDen * N0[k] : 0.0;
            const double right = (rightDen > 0.0 && k > 0) ?
                (t[idx + 2] - x) / rightDen * N0[k - 1] : 0.0;

            N1[k] = left + right;
        }
    }

    // Degree 2: quadratic
    double N2[4] = {0.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 3 < n) {
            const double leftDen = t[idx + 2] - t[idx];
            const double rightDen = t[idx + 3] - t[idx + 1];

            const double left = (leftDen > 0.0) ? (x - t[idx]) / leftDen * N1[k] : 0.0;
            const double right = (rightDen > 0.0 && k > 0) ?
                (t[idx + 3] - x) / rightDen * N1[k - 1] : 0.0;

            N2[k] = left + right;
        }
    }

    // Degree 3: cubic (final result)
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 4 < n) {
            const double leftDen = t[idx + 3] - t[idx];
            const double rightDen = t[idx + 4] - t[idx + 1];

            const double left = (leftDen > 0.0) ? (x - t[idx]) / leftDen * N2[k] : 0.0;
            const double right = (rightDen > 0.0 && k > 0) ?
                (t[idx + 4] - x) / rightDen * N2[k - 1] : 0.0;

            N[k] = left + right;
        } else {
            N[k] = 0.0;
        }
    }
}

/// Evaluate derivatives of cubic B-spline basis functions
///
/// Output: dN[0] = B'_i(x), dN[1] = B'_{i-1}(x), etc.
///
/// @param t Knot array pointer
/// @param n_knots Number of knots
/// @param i Knot span index
/// @param x Evaluation point
/// @param dN Output: 4 basis derivatives
KOKKOS_INLINE_FUNCTION
void cubic_basis_derivative(const double* t, int n_knots, int i, double x, double dN[4]) noexcept {
    const int n = n_knots;

    // Build quadratic basis for derivative computation
    double N0[4] = {0.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 1 < n) {
            N0[k] = (t[idx] <= x && x < t[idx + 1]) ? 1.0 : 0.0;
        }
    }

    double N1[4] = {0.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 2 < n) {
            const double leftDen = t[idx + 1] - t[idx];
            const double rightDen = t[idx + 2] - t[idx + 1];

            const double left = (leftDen > 0.0) ? (x - t[idx]) / leftDen * N0[k] : 0.0;
            const double right = (rightDen > 0.0 && k > 0) ?
                (t[idx + 2] - x) / rightDen * N0[k - 1] : 0.0;

            N1[k] = left + right;
        }
    }

    double N2[4] = {0.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 3 < n) {
            const double leftDen = t[idx + 2] - t[idx];
            const double rightDen = t[idx + 3] - t[idx + 1];

            const double left = (leftDen > 0.0) ? (x - t[idx]) / leftDen * N1[k] : 0.0;
            const double right = (rightDen > 0.0 && k > 0) ?
                (t[idx + 3] - x) / rightDen * N1[k - 1] : 0.0;

            N2[k] = left + right;
        }
    }

    // Derivative formula: B'_{i,3}(x) = 3/(t_{i+3}-t_i)*B_{i,2} - 3/(t_{i+4}-t_{i+1})*B_{i+1,2}
    constexpr double p = 3.0;
    for (int k = 0; k < 4; ++k) {
        const int idx = i - k;
        if (idx >= 0 && idx + 4 < n) {
            const double leftDen = t[idx + 3] - t[idx];
            const double rightDen = t[idx + 4] - t[idx + 1];

            const double left = (leftDen > 0.0) ? (p / leftDen) * N2[k] : 0.0;
            const double right = (rightDen > 0.0 && k > 0) ? (p / rightDen) * N2[k - 1] : 0.0;

            dN[k] = left - right;
        } else {
            dN[k] = 0.0;
        }
    }
}

/// Clamp query point to valid B-spline domain
KOKKOS_INLINE_FUNCTION
double clamp_query(double x, double xmin, double xmax) noexcept {
    if (x <= xmin) return xmin;
    if (x >= xmax) return xmax - 1e-14;  // Half-open interval
    return x;
}

/// Create clamped knot vector for cubic B-splines (host-only)
///
/// For n data points, creates n+4 knots with repeated endpoints.
/// This is a HOST function - call before GPU kernel launch.
///
/// @param x Data grid points (sorted)
/// @param knots Output knot view (must be pre-allocated with size n+4)
template <typename MemSpace>
void create_clamped_knots_cubic(
    Kokkos::View<double*, MemSpace> x,
    Kokkos::View<double*, MemSpace> knots)
{
    const int n = static_cast<int>(x.extent(0));

    // Create host mirrors
    auto x_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
    auto knots_h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, knots);

    // Left clamp: repeat first point 4 times
    for (int i = 0; i < 4; ++i) {
        knots_h(i) = x_h(0);
    }

    // Interior knots
    if (n > 4) {
        const int interior = n - 4;
        const int intervals = n - 1;

        for (int idx = 0; idx < interior; ++idx) {
            const double ratio = static_cast<double>(idx + 1) / static_cast<double>(interior + 1);
            double pos = ratio * static_cast<double>(intervals);

            int low = static_cast<int>(pos);
            if (low >= intervals) low = intervals - 1;

            const double frac = pos - static_cast<double>(low);
            const double left_val = x_h(low);
            const double right_val = x_h(low + 1);
            double knot = (1.0 - frac) * left_val + frac * right_val;

            // Clamp to interior
            const double spacing = right_val - left_val;
            const double eps = 1e-12 * spacing;
            if (knot < left_val + eps) knot = left_val + eps;
            if (knot > right_val - eps) knot = right_val - eps;

            knots_h(4 + idx) = knot;
        }
    }

    // Right clamp: repeat last point 4 times
    for (int i = 0; i < 4; ++i) {
        knots_h(n + i) = x_h(n - 1);
    }

    Kokkos::deep_copy(knots, knots_h);
}

}  // namespace mango::kokkos
