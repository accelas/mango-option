/**
 * @file bspline_collocation.hpp
 * @brief 1D cubic B-spline collocation solver
 *
 * Solves the collocation system B·c = f where:
 * - B[i,j] = N_j(x_i) is the collocation matrix
 * - c are the B-spline control point coefficients
 * - f are the function values at grid points
 *
 * The banded structure (4-diagonal for cubic B-splines) is exploited
 * via the banded matrix solver for O(n) complexity instead of O(n³).
 *
 * This is a generic 1D solver used as a building block for separable
 * multi-dimensional tensor-product B-spline fitting.
 *
 * Features:
 * - Banded LU factorization (O(n) time, O(n) space)
 * - Condition number estimation for numerical diagnostics
 * - Residual computation for quality assessment
 * - Style matches cubic_spline_solver.hpp and thomas_solver.hpp
 */

#pragma once

#include "src/math/banded_matrix_solver.hpp"
#include "src/math/bspline_basis.hpp"
#include "src/support/parallel.hpp"
#include <expected>
#include <span>
#include <vector>
#include <concepts>
#include <optional>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>

namespace mango {

/// Successful result of 1D B-spline collocation fitting
template<std::floating_point T>
struct BSplineCollocationResult {
    std::vector<T> coefficients;   ///< Fitted control points
    T max_residual;                 ///< Max |B*c - f|
    T condition_estimate;           ///< Rough condition number estimate
};

/// Configuration for B-spline collocation solver
template<std::floating_point T>
struct BSplineCollocationConfig {
    T tolerance = T{1e-9};  ///< Maximum allowed residual
};

/// 1D Cubic B-spline collocation solver
///
/// Fits B-spline coefficients to interpolate function values at grid points.
/// Uses banded LU factorization for efficient solution.
///
/// **Algorithm:**
/// 1. Build collocation matrix B[i,j] = N_j(x_i) in banded format
/// 2. Solve banded system B·c = f via LU factorization
/// 3. Verify residuals ||B·c - f||∞ < tolerance
/// 4. Estimate condition number for numerical diagnostics
///
/// Time:  O(n) for factorization and solve
/// Space: O(n) for banded storage
///
/// @tparam T Floating point type (float, double, long double)
template<std::floating_point T>
class BSplineCollocation1D {
public:
    /// Factory method to create BSplineCollocation1D instance
    ///
    /// @param grid Data grid points (sorted, ≥4 points)
    /// @return Solver instance or error message
    [[nodiscard]] static std::expected<BSplineCollocation1D, std::string> create(
        std::vector<T> grid)
    {
        // Validate grid size
        if (grid.size() < 4) {
            return std::unexpected(
                "Grid must have ≥4 points for cubic B-splines, got " +
                std::to_string(grid.size()) + " points");
        }

        // Validate grid is sorted
        if (!std::is_sorted(grid.begin(), grid.end())) {
            return std::unexpected("Grid must be sorted in ascending order");
        }

        // Check for near-duplicate points
        constexpr T MIN_SPACING = T{1e-14};
        for (size_t i = 1; i < grid.size(); ++i) {
            const T spacing = grid[i] - grid[i-1];
            if (spacing < MIN_SPACING) {
                return std::unexpected(
                    "Grid points too close together (spacing < 1e-14) at index " +
                    std::to_string(i) + ": [" +
                    std::to_string(grid[i-1]) + ", " + std::to_string(grid[i]) + "]");
            }
        }

        // Check for zero-width grid
        if (grid.back() - grid.front() < MIN_SPACING) {
            return std::unexpected("Grid has zero width (all points nearly identical)");
        }

        // All validations passed
        return BSplineCollocation1D(std::move(grid));
    }

    /// Fit B-spline coefficients via collocation
    ///
    /// Solves B·c = f where B is the collocation matrix.
    /// Returns fitted coefficients or error message.
    ///
    /// @param values Function values at grid points (size n)
    /// @param config Solver configuration
    /// @return Fit result with coefficients and diagnostics
    [[nodiscard]] std::expected<BSplineCollocationResult<T>, std::string> fit(
        const std::vector<T>& values,
        const BSplineCollocationConfig<T>& config = {})
    {
        if (values.size() != n_) {
            return std::unexpected("Value array size mismatch");
        }

        // Validate input values for NaN/Inf
        for (size_t i = 0; i < n_; ++i) {
            if (std::isnan(values[i])) {
                return std::unexpected(
                    "Input values contain NaN at index " + std::to_string(i));
            }
            if (std::isinf(values[i])) {
                return std::unexpected(
                    "Input values contain infinite value at index " + std::to_string(i));
            }
        }

        // Build collocation matrix
        build_collocation_matrix();

        // Create banded matrix with LAPACK format (kl=0, ku=3 for cubic B-splines)
        BandedMatrix<T> A(n_, 0, 3);
        for (size_t i = 0; i < n_; ++i) {
            for (size_t k = 0; k < 4; ++k) {
                const int col = band_col_start_[i] + static_cast<int>(k);
                if (col >= 0 && col < static_cast<int>(n_)) {
                    A(i, static_cast<size_t>(col)) = band_values_[i * 4 + k];
                }
            }
        }

        // Factorize banded system
        BandedLUWorkspace<T> workspace(n_, 0, 3);
        auto factor_result = factorize_banded(A, workspace);
        if (!factor_result.ok()) {
            return std::unexpected(
                "Failed to factorize collocation system: " +
                std::string(factor_result.message()));
        }

        // Solve for coefficients
        std::vector<T> coeffs(n_);
        auto solve_result = solve_banded(workspace, std::span<const T>(values), std::span<T>(coeffs));
        if (!solve_result.ok()) {
            return std::unexpected(
                "Failed to solve collocation system: " +
                std::string(solve_result.message()));
        }

        // Compute residuals: ||B·c - f||
        const T max_residual = compute_residual(coeffs, values);

        // Check residual tolerance
        if (max_residual > config.tolerance) {
            return std::unexpected(
                "Residual " + std::to_string(max_residual) +
                " exceeds tolerance " + std::to_string(config.tolerance));
        }

        // Estimate condition number
        const T norm_A = compute_matrix_norm1();
        const T cond_est = estimate_banded_condition(workspace, norm_A);

        return BSplineCollocationResult<T>{
            .coefficients = std::move(coeffs),
            .max_residual = max_residual,
            .condition_estimate = cond_est
        };
    }

    /// Fit with external coefficient buffer (zero-allocation variant)
    ///
    /// @param values Function values at grid points
    /// @param coeffs_out Pre-allocated buffer for coefficients (size n_)
    /// @param config Solver configuration
    /// @return Fit result WITHOUT coefficients vector (uses coeffs_out)
    [[nodiscard]] std::expected<BSplineCollocationResult<T>, std::string> fit_with_buffer(
        std::span<const T> values,
        std::span<T> coeffs_out,
        const BSplineCollocationConfig<T>& config = {})
    {
        if (values.size() != n_) {
            return std::unexpected("Value array size mismatch");
        }
        if (coeffs_out.size() != n_) {
            return std::unexpected("Coefficients buffer size mismatch");
        }

        // Validate input values
        for (size_t i = 0; i < n_; ++i) {
            if (std::isnan(values[i])) {
                return std::unexpected(
                    "Input values contain NaN at index " + std::to_string(i));
            }
            if (std::isinf(values[i])) {
                return std::unexpected(
                    "Input values contain infinite value at index " + std::to_string(i));
            }
        }

        // Build collocation matrix
        build_collocation_matrix();

        // Create banded matrix with LAPACK format (kl=0, ku=3 for cubic B-splines)
        BandedMatrix<T> A(n_, 0, 3);
        for (size_t i = 0; i < n_; ++i) {
            for (size_t k = 0; k < 4; ++k) {
                const int col = band_col_start_[i] + static_cast<int>(k);
                if (col >= 0 && col < static_cast<int>(n_)) {
                    A(i, static_cast<size_t>(col)) = band_values_[i * 4 + k];
                }
            }
        }

        // Factorize and solve
        BandedLUWorkspace<T> workspace(n_, 0, 3);
        auto factor_result = factorize_banded(A, workspace);
        if (!factor_result.ok()) {
            return std::unexpected(
                "Factorization failed: " + std::string(factor_result.message()));
        }

        auto solve_result = solve_banded(workspace, values, coeffs_out);
        if (!solve_result.ok()) {
            return std::unexpected(
                "Solve failed: " + std::string(solve_result.message()));
        }

        // Compute residuals
        const T max_residual = compute_residual_from_span(coeffs_out, values);

        if (max_residual > config.tolerance) {
            return std::unexpected(
                "Residual " + std::to_string(max_residual) +
                " exceeds tolerance " + std::to_string(config.tolerance));
        }

        // Estimate condition number
        const T norm_A = compute_matrix_norm1();
        const T cond_est = estimate_banded_condition(workspace, norm_A);

        // Return result without copying coefficients
        return BSplineCollocationResult<T>{
            .coefficients = {},
            .max_residual = max_residual,
            .condition_estimate = cond_est
        };
    }

    /// Get grid size
    [[nodiscard]] size_t size() const noexcept { return n_; }

private:
    /// Private constructor (use factory method)
    explicit BSplineCollocation1D(std::vector<T> grid)
        : grid_(std::move(grid))
        , n_(grid_.size())
    {
        // Build knot vector (clamped cubic)
        knots_ = clamped_knots_cubic<T>(grid_);

        // Pre-allocate banded storage (4 entries per row for cubic B-splines)
        band_values_.resize(n_ * 4, T{0});
        band_col_start_.resize(n_, 0);
    }

    std::vector<T> grid_;               ///< Data grid points
    std::vector<T> knots_;              ///< Knot vector (clamped)
    size_t n_;                          ///< Number of grid points

    // Banded storage: each row has exactly 4 non-zero entries
    std::vector<T> band_values_;        ///< Banded matrix values (n×4, row-major)
    std::vector<int> band_col_start_;   ///< First column index for each row's band

    /// Build collocation matrix B[i,j] = N_j(x_i) in banded format
    void build_collocation_matrix() {
        for (size_t i = 0; i < n_; ++i) {
            const T x = grid_[i];

            // Find knot span
            const int span = find_span_cubic(knots_, x);

            // Evaluate 4 non-zero basis functions at x
            T basis[4];
            cubic_basis_nonuniform(knots_, span, x, basis);

            // Store in banded format
            band_col_start_[i] = std::max(0, span - 3);

            // Fill band values (left to right order)
            for (int k = 0; k < 4; ++k) {
                const int col = span - k;
                if (col >= 0 && col < static_cast<int>(n_)) {
                    const int band_idx = col - band_col_start_[i];
                    if (band_idx >= 0 && band_idx < 4) {
                        band_values_[i * 4 + band_idx] = basis[k];
                    }
                }
            }
        }
    }

    /// Compute max residual ||B·c - f||∞ using banded storage
    [[nodiscard]] T compute_residual(
        const std::vector<T>& coeffs,
        const std::vector<T>& values) const
    {
        T max_res = T{0};

        for (size_t i = 0; i < n_; ++i) {
            // Compute (B·c)[i] - only sum over 4 non-zero entries
            T Bc_i = T{0};
            const int j_start = band_col_start_[i];
            const int j_end = std::min(j_start + 4, static_cast<int>(n_));

            for (int j = j_start; j < j_end; ++j) {
                const int band_idx = j - j_start;
                const T b_ij = band_values_[i * 4 + band_idx];
                Bc_i = std::fma(b_ij, coeffs[j], Bc_i);
            }

            const T residual = std::abs(Bc_i - values[i]);
            max_res = std::max(max_res, residual);
        }

        return max_res;
    }

    /// Compute residual from span coefficients
    [[nodiscard]] T compute_residual_from_span(
        std::span<const T> coeffs,
        std::span<const T> values) const
    {
        T max_residual = T{0};

        for (size_t i = 0; i < n_; ++i) {
            T Bc_i = T{0};
            const int col_start = band_col_start_[i];

            for (size_t k = 0; k < 4 &&
                 (col_start + static_cast<int>(k)) < static_cast<int>(n_); ++k)
            {
                Bc_i = std::fma(band_values_[i * 4 + k],
                               coeffs[col_start + k],
                               Bc_i);
            }

            const T residual = std::abs(Bc_i - values[i]);
            max_residual = std::max(max_residual, residual);
        }

        return max_residual;
    }

    /// Compute 1-norm of collocation matrix ||B||₁ = max column sum
    [[nodiscard]] T compute_matrix_norm1() const {
        std::vector<T> col_sums(n_, T{0});

        for (size_t i = 0; i < n_; ++i) {
            const int j_start = band_col_start_[i];
            const int j_end = std::min(j_start + 4, static_cast<int>(n_));

            for (int j = j_start; j < j_end; ++j) {
                const int band_idx = j - j_start;
                const T b_ij = band_values_[i * 4 + band_idx];
                col_sums[j] += std::abs(b_ij);
            }
        }

        return *std::max_element(col_sums.begin(), col_sums.end());
    }
};

}  // namespace mango
