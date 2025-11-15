/**
 * @file bspline_utils.cpp
 * @brief SIMD implementations of B-spline utility functions
 *
 * This file contains the SIMD-vectorized implementations of cubic B-spline
 * basis functions using [[gnu::target_clones]] for multi-ISA support.
 *
 * These functions are defined here (not in the header) to avoid IFUNC
 * circular dependencies in shared library builds.
 */

#include "bspline_utils.hpp"

namespace mango {

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
simd4d cubic_basis_degree0_simd(
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
void cubic_basis_nonuniform_simd(
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
