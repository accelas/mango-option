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
#include <array>
#include <experimental/simd>

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

// ============================================================================
// SIMD Cox-de Boor Basis Functions
// ============================================================================

// SIMD vectorization for cubic B-spline basis evaluation
// Processes 4 basis functions simultaneously using std::experimental::simd
//
// Performance: ~2.5x speedup over scalar Cox-de Boor recursion
// Uses [[gnu::target_clones]] for automatic AVX2/AVX512 dispatch
//
// NOTE: These are new SIMD implementations, not replacements for the scalar
// functions above. The scalar functions remain unchanged for backward compatibility.

namespace stdx = std::experimental;

// SIMD type aliases for 4-wide vectors (4 basis functions)
using simd4d = stdx::fixed_size_simd<double, 4>;
using simd4_mask = stdx::fixed_size_simd_mask<double, 4>;

/// Vectorized degree-0 initialization (piecewise constants)
///
/// Computes N_{i,0}(x) = 1 if t_i <= x < t_{i+1}, else 0
/// for 4 basis functions simultaneously
///
/// @param t Knot vector
/// @param i Knot span index
/// @param x Evaluation point
/// @return SIMD vector with 4 degree-0 basis values
[[gnu::target_clones("default","avx2","avx512f")]]
inline simd4d cubic_basis_degree0_simd(
    const std::vector<double>& t,
    int i,
    double x)
{
    // Gather knot values for 4 basis functions
    // Lane 0: basis i   → [t[i], t[i+1]]
    // Lane 1: basis i-1 → [t[i-1], t[i]]
    // Lane 2: basis i-2 → [t[i-2], t[i-1]]
    // Lane 3: basis i-3 → [t[i-3], t[i-2]]
    std::array<double, 4> t_left, t_right;
    for (int lane = 0; lane < 4; ++lane) {
        int idx = i - lane;
        t_left[lane] = t[idx];
        t_right[lane] = t[idx + 1];
    }

    // Load into SIMD vectors
    simd4d t_left_vec, t_right_vec;
    t_left_vec.copy_from(t_left.data(), stdx::element_aligned);
    t_right_vec.copy_from(t_right.data(), stdx::element_aligned);

    // Vectorized interval check: t_left <= x < t_right
    simd4d x_vec(x);  // Broadcast x to all lanes
    auto in_interval = (t_left_vec <= x_vec) && (x_vec < t_right_vec);

    // Return 1.0 if in interval, 0.0 otherwise
    // Use vectorized blend: select true_val for mask, else false_val
    simd4d result(0.0);
    stdx::where(in_interval, result) = simd4d(1.0);
    return result;
}

/// Vectorized Cox-de Boor recursion for cubic B-splines
///
/// Computes 4 cubic basis functions using SIMD vectorization:
/// N_{i,k}(x) = (x - t_i)/(t_{i+k} - t_i) * N_{i,k-1}(x)
///            + (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * N_{i+1,k-1}(x)
///
/// Processes degrees 0 → 1 → 2 → 3 with full vectorization across 4 lanes.
/// Handles division by zero for uniform/repeated knots.
///
/// @param t Knot vector
/// @param i Knot span index
/// @param x Evaluation point
/// @param N Output: 4 basis function values N[0..3]
///          N[0] = N_{i,3}(x), N[1] = N_{i-1,3}(x), N[2] = N_{i-2,3}(x), N[3] = N_{i-3,3}(x)
[[gnu::target_clones("default","avx2","avx512f")]]
inline void cubic_basis_nonuniform_simd(
    const std::vector<double>& t,
    int i,
    double x,
    double N[4])
{
    const int n = static_cast<int>(t.size());

    // Handle right boundary exactly (same as scalar version)
    if (std::abs(x - t.back()) < 1e-14) {
        N[0] = 1.0;
        N[1] = 0.0;
        N[2] = 0.0;
        N[3] = 0.0;
        return;
    }

    // Degree 0: piecewise constants
    simd4d N_curr = cubic_basis_degree0_simd(t, i, x);

    // Degrees 1-3: recursive Cox-de Boor formula
    for (int p = 1; p <= 3; ++p) {
        // Gather denominator knot differences for left and right terms
        // Left term:  (t[idx+p] - t[idx])
        // Right term: (t[idx+p+1] - t[idx+1])
        std::array<double, 4> denom_left, denom_right;
        for (int lane = 0; lane < 4; ++lane) {
            int idx = i - lane;
            if (idx >= 0 && idx + p + 1 < n) {
                denom_left[lane] = t[idx + p] - t[idx];
                denom_right[lane] = t[idx + p + 1] - t[idx + 1];
            } else {
                denom_left[lane] = 0.0;
                denom_right[lane] = 0.0;
            }
        }

        simd4d denom_left_vec, denom_right_vec;
        denom_left_vec.copy_from(denom_left.data(), stdx::element_aligned);
        denom_right_vec.copy_from(denom_right.data(), stdx::element_aligned);

        // Gather numerator knot values
        // Left numerator:  (x - t[idx])
        // Right numerator: (t[idx+p+1] - x)
        std::array<double, 4> t_base, t_end;
        for (int lane = 0; lane < 4; ++lane) {
            int idx = i - lane;
            if (idx >= 0 && idx + p + 1 < n) {
                t_base[lane] = t[idx];
                t_end[lane] = t[idx + p + 1];
            } else {
                t_base[lane] = 0.0;
                t_end[lane] = 0.0;
            }
        }

        simd4d t_base_vec, t_end_vec;
        t_base_vec.copy_from(t_base.data(), stdx::element_aligned);
        t_end_vec.copy_from(t_end.data(), stdx::element_aligned);

        // Compute left and right terms
        simd4d x_vec(x);
        simd4d left_num = x_vec - t_base_vec;
        simd4d right_num = t_end_vec - x_vec;

        // Handle division by zero (uniform/repeated knots)
        auto left_valid = denom_left_vec != simd4d(0.0);
        auto right_valid = denom_right_vec != simd4d(0.0);

        // Compute left term: (x - t[idx]) / (t[idx+p] - t[idx]) * N_curr[k]
        simd4d left_term(0.0);
        stdx::where(left_valid, left_term) = (left_num / denom_left_vec) * N_curr;

        // Shift N_curr by one lane for right term: N_curr[k-1]
        // Lane 0 gets 0.0 (no k-1), lanes 1-3 get N_curr[0-2]
        std::array<double, 4> shifted{0.0, N_curr[0], N_curr[1], N_curr[2]};
        simd4d N_curr_shifted;
        N_curr_shifted.copy_from(shifted.data(), stdx::element_aligned);

        // Compute right term: (t[idx+p+1] - x) / (t[idx+p+1] - t[idx+1]) * N_curr[k-1]
        simd4d right_term(0.0);
        stdx::where(right_valid, right_term) = (right_num / denom_right_vec) * N_curr_shifted;

        // Combine left and right terms
        N_curr = left_term + right_term;
    }

    // Store result
    N_curr.copy_to(N, stdx::element_aligned);
}

}  // namespace mango
