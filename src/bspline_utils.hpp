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

namespace mango {

/// Create clamped knot vector for cubic B-splines
///
/// For n data points, creates n+4 knots with repeated endpoints:
/// [x[0], x[0], x[0], x[0], x[1], ..., x[n-2], x[n-1], x[n-1], x[n-1], x[n-1]]
///
/// @param x Grid points (must be sorted)
/// @return Clamped knot vector
inline std::vector<double> clamped_knots_cubic(const std::vector<double>& x) {
    const int n = static_cast<int>(x.size());
    std::vector<double> t(n + 4);

    // Left clamp: repeat first point 4 times
    std::fill_n(t.begin(), 4, x.front());

    // Interior knots
    for (int i = 1; i < n - 1; ++i) {
        t[3 + i] = x[i];
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
    auto it = std::upper_bound(t.begin(), t.end(), x);
    int i = static_cast<int>(std::distance(t.begin(), it)) - 1;

    // Clamp to valid range
    if (i < 0) i = 0;
    if (i >= static_cast<int>(t.size()) - 2) {
        i = static_cast<int>(t.size()) - 2;
    }

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

}  // namespace mango
