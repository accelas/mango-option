/**
 * @file bspline_utils.hpp
 * @brief Shared utility functions for B-spline evaluation
 *
 * Common functions used by both bspline_4d.hpp and bspline_collocation_1d.hpp.
 * Extracted to avoid ODR violations when both headers are included.
 */

#pragma once

#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

namespace mango {

/// Create clamped knot vector for cubic B-splines
///
/// For n data points, creates n+4 knots with repeated endpoints:
/// [x[0], x[0], x[0], x[0], x[1], ..., x[n-4], x[n-1], x[n-1], x[n-1], x[n-1]]
///
/// The interior knots x[1] through x[n-4] are placed between the clamped endpoints.
/// This ensures the collocation matrix is non-singular (Schoenberg-Whitney condition).
///
/// @param x Grid points (must be sorted)
/// @return Clamped knot vector
inline std::vector<double> clamped_knots_cubic(const std::vector<double>& x) {
    const int n = static_cast<int>(x.size());
    std::vector<double> t(n + 4);

    // Left clamp: repeat first point 4 times
    std::fill_n(t.begin(), 4, x.front());

    // Interior knots positioned strictly between data sites (midpoints)
    if (n > 4) {
        const int interior = n - 4;
        const int intervals = n - 1;
        for (int idx = 0; idx < interior; ++idx) {
            const double ratio = static_cast<double>(idx + 1) /
                                 static_cast<double>(interior + 1);
            double pos = ratio * static_cast<double>(intervals);
            int low = static_cast<int>(std::floor(pos));
            if (low >= intervals) {
                low = intervals - 1;
            }
            const double frac = pos - static_cast<double>(low);
            const double left = x[low];
            const double right = x[low + 1];
            double knot = (1.0 - frac) * left + frac * right;

            const double spacing = right - left;
            const double eps = std::max(1e-12 * spacing,
                                        std::numeric_limits<double>::epsilon() *
                                            std::max(std::abs(right), 1.0));
            knot = std::clamp(knot, left + eps, right - eps);

            t[4 + idx] = knot;
        }
    }

    // Right clamp: repeat last point 4 times
    std::fill_n(t.end() - 4, 4, x.back());

    return t;
}

/// Find knot span containing x using binary search
///
/// Returns index i such that t[i] <= x < t[i+1]
///
/// @param t Knot vector
/// @param x Query point
/// @return Knot span index
inline int find_span_cubic(const std::vector<double>& t, double x) {
    constexpr int DEGREE = 3;
    const int n_ctrl = static_cast<int>(t.size()) - DEGREE - 1;
    const int min_span = DEGREE;
    const int max_span = std::max(min_span, n_ctrl - 1);

    if (x <= t[min_span]) {
        return min_span;
    }
    if (x >= t[n_ctrl]) {
        return max_span;
    }

    auto it = std::upper_bound(t.begin() + min_span, t.begin() + n_ctrl + 1, x);
    int i = static_cast<int>(std::distance(t.begin(), it)) - 1;

    if (i < min_span) i = min_span;
    if (i > max_span) i = max_span;

    return i;
}

/// Evaluate cubic basis functions using Cox-de Boor recursion
///
/// Computes the 4 nonzero cubic basis functions at x for knot span i.
/// Uses de Boor's recursive formula with proper handling of zero denominators.
///
/// @param t Knot vector
/// @param i Knot span index
/// @param x Evaluation point
/// @param N Output: 4 basis function values N[0..3]
///          N[0] corresponds to basis i, N[1] to i-1, N[2] to i-2, N[3] to i-3
inline void cubic_basis_nonuniform(const std::vector<double>& t, int i, double x, double N[4]) {
    const int n = static_cast<int>(t.size());

    // Ensure exact interpolation at the right boundary.
    if (std::abs(x - t.back()) < 1e-14) {
        N[0] = 1.0;
        N[1] = 0.0;
        N[2] = 0.0;
        N[3] = 0.0;
        return;
    }

    // Degree 0: piecewise constants
    double N0[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) {
        int idx = i - k;
        if (idx >= 0 && idx + 1 < n) {
            N0[k] = (t[idx] <= x && x < t[idx + 1]) ? 1.0 : 0.0;
        }
    }

    // Degree 1: linear combination
    double N1[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) {
        int idx = i - k;
        if (idx >= 0 && idx + 2 < n) {
            double leftDen  = t[idx + 1] - t[idx];
            double rightDen = t[idx + 2] - t[idx + 1];

            double left  = (leftDen > 0.0) ? (x - t[idx]) / leftDen * N0[k] : 0.0;
            double right = (rightDen > 0.0 && k > 0) ? (t[idx + 2] - x) / rightDen * N0[k - 1] : 0.0;

            N1[k] = left + right;
        }
    }

    // Degree 2: quadratic
    double N2[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) {
        int idx = i - k;
        if (idx >= 0 && idx + 3 < n) {
            double leftDen  = t[idx + 2] - t[idx];
            double rightDen = t[idx + 3] - t[idx + 1];

            double left  = (leftDen > 0.0) ? (x - t[idx]) / leftDen * N1[k] : 0.0;
            double right = (rightDen > 0.0 && k > 0) ? (t[idx + 3] - x) / rightDen * N1[k - 1] : 0.0;

            N2[k] = left + right;
        }
    }

    // Degree 3: cubic
    for (int k = 0; k < 4; ++k) {
        int idx = i - k;
        if (idx >= 0 && idx + 4 < n) {
            double leftDen  = t[idx + 3] - t[idx];
            double rightDen = t[idx + 4] - t[idx + 1];

            double left  = (leftDen > 0.0) ? (x - t[idx]) / leftDen * N2[k] : 0.0;
            double right = (rightDen > 0.0 && k > 0) ? (t[idx + 4] - x) / rightDen * N2[k - 1] : 0.0;

            N[k] = left + right;
        } else {
            N[k] = 0.0;
        }
    }
}

/// Evaluate cubic basis function derivatives using Cox-de Boor recursion
///
/// Computes the derivatives of the 4 nonzero cubic basis functions at x.
/// Uses the derivative formula: B'_{i,p}(x) = p/(t[i+p]-t[i])*B_{i,p-1}(x) - p/(t[i+p+1]-t[i+1])*B_{i+1,p-1}(x)
///
/// For cubic (p=3), derivatives are expressed in terms of quadratic basis functions.
///
/// @param t Knot vector
/// @param i Knot span index
/// @param x Evaluation point
/// @param dN Output: 4 basis function derivatives dN[0..3]
///           dN[0] corresponds to d/dx B_i(x), dN[1] to d/dx B_{i-1}(x), etc.
inline void cubic_basis_derivative_nonuniform(const std::vector<double>& t, int i, double x, double dN[4]) {
    const int n = static_cast<int>(t.size());

    // We need quadratic basis functions (degree 2) to compute cubic derivatives
    // Build up from degree 0 → 1 → 2 using Cox-de Boor recursion

    // Degree 0: piecewise constants
    double N0[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) {
        int idx = i - k;
        if (idx >= 0 && idx + 1 < n) {
            N0[k] = (t[idx] <= x && x < t[idx + 1]) ? 1.0 : 0.0;
        }
    }

    // Degree 1: linear
    double N1[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) {
        int idx = i - k;
        if (idx >= 0 && idx + 2 < n) {
            double leftDen  = t[idx + 1] - t[idx];
            double rightDen = t[idx + 2] - t[idx + 1];

            double left  = (leftDen > 0.0) ? (x - t[idx]) / leftDen * N0[k] : 0.0;
            double right = (rightDen > 0.0 && k > 0) ? (t[idx + 2] - x) / rightDen * N0[k - 1] : 0.0;

            N1[k] = left + right;
        }
    }

    // Degree 2: quadratic (needed for cubic derivatives)
    double N2[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) {
        int idx = i - k;
        if (idx >= 0 && idx + 3 < n) {
            double leftDen  = t[idx + 2] - t[idx];
            double rightDen = t[idx + 3] - t[idx + 1];

            double left  = (leftDen > 0.0) ? (x - t[idx]) / leftDen * N1[k] : 0.0;
            double right = (rightDen > 0.0 && k > 0) ? (t[idx + 3] - x) / rightDen * N1[k - 1] : 0.0;

            N2[k] = left + right;
        }
    }

    // Apply derivative formula for cubic (p=3):
    // B'_{i,3}(x) = 3/(t[i+3]-t[i]) * B_{i,2}(x) - 3/(t[i+4]-t[i+1]) * B_{i+1,2}(x)
    constexpr double p = 3.0;
    for (int k = 0; k < 4; ++k) {
        int idx = i - k;
        if (idx >= 0 && idx + 4 < n) {
            double leftDen  = t[idx + 3] - t[idx];
            double rightDen = t[idx + 4] - t[idx + 1];

            // First term: p/(t[i+p]-t[i]) * B_{i,p-1}(x)
            double left = (leftDen > 0.0) ? (p / leftDen) * N2[k] : 0.0;

            // Second term: -p/(t[i+p+1]-t[i+1]) * B_{i+1,p-1}(x)
            double right = (rightDen > 0.0 && k > 0) ? (p / rightDen) * N2[k - 1] : 0.0;

            dN[k] = left - right;
        } else {
            dN[k] = 0.0;
        }
    }
}

}  // namespace mango
