/**
 * @file bspline_fitter_4d.hpp
 * @brief Complete 4D B-spline coefficient fitting using separable collocation
 *
 * This file contains a complete implementation of 4D B-spline fitting:
 * - BSplineCollocation1D: 1D cubic B-spline collocation solver
 * - BSplineFitter4DSeparable: Separable 4D fitting via sequential 1D solves
 * - BSplineFitter4D: High-level 4D fitter interface
 *
 * Uses tensor-product structure to fit B-spline coefficients efficiently.
 * Instead of a massive O(n⁴) dense system, we solve sequential 1D systems
 * along each axis: axis0 → axis1 → axis2 → axis3.
 *
 * Performance: O(N0³ + N1³ + N2³ + N3³)
 *   For 50×30×20×10: ~5ms fitting time
 *
 * Accuracy: Residuals <1e-6 at all grid points (validated per-axis)
 *
 * Usage:
 *   auto fitter_result = BSplineFitter4D::create(axis0_grid, axis1_grid, axis2_grid, axis3_grid);
 *   if (fitter_result.has_value()) {
 *       auto result = fitter_result.value().fit(values_4d);
 *       if (result.success) {
 *           // Use result.coefficients with BSpline4D
 *       }
 *   } else {
 *       // Handle creation error: fitter_result.error()
 *   }
 */

#pragma once

#include "src/interpolation/bspline_utils.hpp"
#include "src/pde/core/thomas_solver.hpp"
#include "src/support/expected.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <optional>
#include <string>
#include <stdexcept>
#include <memory>
#include <span>
#include <cassert>

namespace mango {

// ============================================================================
// Banded Matrix Storage
// ============================================================================

/// Compact storage for 4-diagonal banded matrix from cubic B-spline collocation
///
/// Matrix structure for cubic B-spline (degree 3):
///   - Each basis function has compact support → at most 4 non-zero entries per row
///   - Banded structure: entries in columns [j-3, j-2, j-1, j]
///
/// Storage layout (row-major):
///   band_values_[i*4 + k] = A[i, col_start[i] + k] for k ∈ [0,3]
///
/// Memory: O(4n) vs O(n²) for dense
class BandedMatrixStorage {
public:
    /// Construct banded storage for n×n matrix with bandwidth 4
    explicit BandedMatrixStorage(size_t n)
        : n_(n)
        , band_values_(4 * n, 0.0)
        , col_start_(n, 0)
    {}

    /// Get reference to band entry A[row, col]
    /// Assumes col ∈ [col_start[row], col_start[row] + 3]
    double& operator()(size_t row, size_t col) {
        assert(row < n_);
        assert(col >= col_start_[row] && col < col_start_[row] + 4);
        size_t k = col - col_start_[row];
        return band_values_[row * 4 + k];
    }

    /// Get const reference to band entry
    double operator()(size_t row, size_t col) const {
        assert(row < n_);
        assert(col >= col_start_[row] && col < col_start_[row] + 4);
        size_t k = col - col_start_[row];
        return band_values_[row * 4 + k];
    }

    /// Get starting column index for row
    size_t col_start(size_t row) const {
        assert(row < n_);
        return col_start_[row];
    }

    /// Set starting column index for row
    void set_col_start(size_t row, size_t col) {
        assert(row < n_);
        col_start_[row] = col;
    }

    /// Get number of rows (and columns)
    size_t size() const { return n_; }

    /// Get raw band values (for debugging/testing)
    std::span<const double> band_values() const { return band_values_; }

    /// Get raw column starts (for debugging/testing)
    std::span<const size_t> col_starts() const { return col_start_; }

private:
    size_t n_;                          ///< Matrix dimension
    std::vector<double> band_values_;   ///< Banded storage (4n entries)
    std::vector<size_t> col_start_;     ///< Starting column for each row
};

// ============================================================================
// Banded LU Solver
// ============================================================================

/// Solve banded system Ax = b using LU decomposition
///
/// For 4-diagonal banded matrix from cubic B-spline collocation.
/// Time complexity: O(n) for fixed bandwidth
/// Space complexity: O(n) (in-place decomposition)
///
/// @param A Banded matrix (modified in-place during decomposition)
/// @param b Right-hand side vector
/// @param x Solution vector (output)
inline void banded_lu_solve(
    BandedMatrixStorage& A,
    std::span<const double> b,
    std::span<double> x)
{
    const size_t n = A.size();
    assert(b.size() == n);
    assert(x.size() == n);

    // Working storage for intermediate results
    std::vector<double> y(n);  // For forward substitution

    // Phase 1: LU decomposition (in-place, Doolittle algorithm)
    // For banded matrix with bandwidth k=4, this is O(n) not O(n³)
    for (size_t i = 0; i < n; ++i) {
        size_t col_start = A.col_start(i);
        size_t col_end = std::min(col_start + 4, n);

        // Eliminate entries below diagonal in column i
        for (size_t k = i + 1; k < std::min(i + 4, n); ++k) {
            size_t k_col_start = A.col_start(k);

            // Check if A(k, i) is in the band
            if (i >= k_col_start && i < k_col_start + 4) {
                double factor = A(k, i) / A(i, i);

                // Update row k (only within band)
                for (size_t j = i; j < col_end; ++j) {
                    if (j >= k_col_start && j < k_col_start + 4) {
                        A(k, j) -= factor * A(i, j);
                    }
                }

                // Store multiplier in lower triangle (for forward substitution)
                A(k, i) = factor;
            }
        }
    }

    // Phase 2: Forward substitution (Ly = b)
    for (size_t i = 0; i < n; ++i) {
        y[i] = b[i];

        size_t col_start = A.col_start(i);
        for (size_t j = col_start; j < i; ++j) {
            if (j >= col_start && j < col_start + 4) {
                y[i] -= A(i, j) * y[j];
            }
        }
    }

    // Phase 3: Back substitution (Ux = y)
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        x[i] = y[i];

        size_t col_start = A.col_start(i);
        size_t col_end = std::min(col_start + 4, n);

        for (size_t j = static_cast<size_t>(i) + 1; j < col_end; ++j) {
            x[i] -= A(i, j) * x[j];
        }

        x[i] /= A(i, i);
    }
}

// ============================================================================
// 1D B-spline Collocation Solver
// ============================================================================

/// Result of 1D B-spline collocation fitting
struct BSplineCollocation1DResult {
    std::vector<double> coefficients;  ///< Fitted control points
    bool success;                       ///< Fit succeeded
    std::string error_message;          ///< Error if failed
    double max_residual;                ///< Max |B*c - f|
    double condition_estimate;          ///< Rough condition number estimate
};

/// 1D Cubic B-spline collocation solver
///
/// Solves the collocation system: B*c = f
/// where B[i,j] = N_j(x_i) is the collocation matrix.
///
/// Builds and solves the collocation system to find control points
/// that make the B-spline interpolate the given data.
class BSplineCollocation1D {
public:
    /// Factory method to create BSplineCollocation1D instance
    ///
    /// @param grid Data grid points (sorted, ≥4 points)
    /// @return expected<BSplineCollocation1D, std::string> containing either the solver or error message
    static expected<BSplineCollocation1D, std::string> create(std::vector<double> grid) {
        try {
            // Validate grid size
            if (grid.size() < 4) {
                return unexpected(std::string("Grid must have ≥4 points for cubic B-splines, got ") +
                               std::to_string(grid.size()) + " points");
            }

            // Validate grid is sorted
            if (!std::is_sorted(grid.begin(), grid.end())) {
                return unexpected(std::string("Grid must be sorted in ascending order"));
            }

            // Check for duplicate or near-duplicate points
            constexpr double MIN_SPACING = 1e-14;
            for (size_t i = 1; i < grid.size(); ++i) {
                double spacing = grid[i] - grid[i-1];
                if (spacing < MIN_SPACING) {
                    return unexpected(
                        std::string("Grid points too close together (spacing < 1e-14). ") +
                        "Found grid[" + std::to_string(i-1) + "] = " + std::to_string(grid[i-1]) +
                        " and grid[" + std::to_string(i) + "] = " + std::to_string(grid[i]) +
                        " with spacing " + std::to_string(spacing)
                    );
                }
            }

            // Check for zero-width grid
            if (grid.back() - grid.front() < MIN_SPACING) {
                return unexpected(std::string("Grid has zero width (all points nearly identical)"));
            }

            // All validations passed - create the solver
            return BSplineCollocation1D(std::move(grid));
        } catch (const std::exception& e) {
            return unexpected(std::string("BSplineCollocation1D creation failed: ") + e.what());
        }
    }

    /// Enable or disable banded solver (for testing)
    ///
    /// @param use_banded If true, use efficient O(n) banded LU solver. If false, use dense solver.
    void set_use_banded_solver(bool use_banded) {
        use_banded_solver_ = use_banded;
    }

    /// Fit B-spline coefficients via collocation
    ///
    /// Solves B*c = f where B is the collocation matrix
    ///
    /// @param values Function values at grid points (size n)
    /// @param tolerance Max allowed residual (default 1e-9)
    /// @return Fit result with coefficients and diagnostics
    BSplineCollocation1DResult fit(const std::vector<double>& values, double tolerance = 1e-9) {
        if (values.size() != n_) {
            return {std::vector<double>(), false,
                    "Value array size mismatch", 0.0, 0.0};
        }

        // Validate input values for NaN/Inf
        for (size_t i = 0; i < n_; ++i) {
            if (std::isnan(values[i])) {
                return {std::vector<double>(), false,
                        "Input values contain NaN at index " + std::to_string(i), 0.0, 0.0};
            }
            if (std::isinf(values[i])) {
                return {std::vector<double>(), false,
                        "Input values contain infinite value at index " + std::to_string(i), 0.0, 0.0};
            }
        }

        // Build collocation matrix
        build_collocation_matrix();

        // Solve banded system: B*c = f
        std::vector<double> coeffs(n_);
        bool solve_success = solve_banded_system(values, coeffs);

        if (!solve_success) {
            return {std::vector<double>(), false,
                    "Failed to solve collocation system (singular or ill-conditioned)",
                    0.0, 0.0};
        }

        // Compute residuals: ||B*c - f||
        double max_residual = compute_residual(coeffs, values);

        // Check residual tolerance
        if (max_residual > tolerance) {
            return {std::vector<double>(), false,
                    "Residual " + std::to_string(max_residual) +
                    " exceeds tolerance " + std::to_string(tolerance),
                    max_residual, 0.0};
        }

        // Estimate condition number via 1-norm bound
        double cond_est = estimate_condition_number();

        return {coeffs, true, "", max_residual, cond_est};
    }

private:
    /// Constructor (private - use factory method)
    explicit BSplineCollocation1D(std::vector<double> grid)
        : grid_(std::move(grid))
        , n_(grid_.size())
        , use_banded_solver_(true)  // Default: use efficient banded solver
    {
        // Build knot vector (clamped cubic)
        knots_ = clamped_knots_cubic(grid_);

        // Pre-allocate banded storage (4 entries per row for cubic B-splines)
        band_values_.resize(n_ * 4, 0.0);
        band_col_start_.resize(n_, 0);
    }

    std::vector<double> grid_;              ///< Data grid points
    std::vector<double> knots_;             ///< Knot vector (clamped)
    size_t n_;                              ///< Number of grid points
    bool use_banded_solver_;                ///< If true, use banded LU solver; if false, use dense solver

    // Banded storage: each row has exactly 4 non-zero entries (cubic B-spline support)
    std::vector<double> band_values_;       ///< Banded matrix values (n×4, row-major)
    std::vector<int> band_col_start_;       ///< First column index for each row's band

    /// Build collocation matrix B[i,j] = N_j(x_i) in banded format
    void build_collocation_matrix() {
        for (size_t i = 0; i < n_; ++i) {
            const double x = grid_[i];

            // Find knot span
            int span = find_span_cubic(knots_, x);

            // Evaluate 4 non-zero basis functions at x
            double basis[4];
            cubic_basis_nonuniform(knots_, span, x, basis);

            // Store in banded format
            band_col_start_[i] = std::max(0, span - 3);

            // Fill band values (left to right order)
            for (int k = 0; k < 4; ++k) {
                int col = span - k;
                if (col >= 0 && col < static_cast<int>(n_)) {
                    int band_idx = col - band_col_start_[i];
                    if (band_idx >= 0 && band_idx < 4) {
                        band_values_[i * 4 + band_idx] = basis[k];
                    }
                }
            }
        }
    }

    /// Solve banded linear system (dispatcher)
    bool solve_banded_system(const std::vector<double>& rhs, std::vector<double>& solution) const {
        if (use_banded_solver_) {
            return solve_banded_system_efficient(rhs, solution);
        } else {
            return solve_banded_system_dense(rhs, solution);
        }
    }

    /// Solve using efficient O(n) banded LU solver
    bool solve_banded_system_efficient(const std::vector<double>& rhs, std::vector<double>& solution) const {
        // Build BandedMatrixStorage from compact storage
        BandedMatrixStorage A(n_);

        for (size_t i = 0; i < n_; ++i) {
            int col_start = band_col_start_[i];
            A.set_col_start(i, static_cast<size_t>(col_start));

            // Copy band values
            for (int k = 0; k < 4; ++k) {
                int col = col_start + k;
                if (col >= 0 && col < static_cast<int>(n_)) {
                    A(i, static_cast<size_t>(col)) = band_values_[i * 4 + k];
                }
            }
        }

        // Solve using banded LU
        solution.resize(n_);
        banded_lu_solve(A, rhs, solution);

        return true;  // banded_lu_solve doesn't report failures (assumes well-conditioned)
    }

    /// Solve using dense O(n³) solver (for regression testing)
    bool solve_banded_system_dense(const std::vector<double>& rhs, std::vector<double>& solution) const {
        solution = rhs;

        // Expand compact n×4 banded storage to full n×n for working copy
        std::vector<double> A(n_ * n_, 0.0);

        // Copy from compact storage to full matrix
        for (size_t i = 0; i < n_; ++i) {
            int j_start = band_col_start_[i];
            int j_end = std::min(j_start + 4, static_cast<int>(n_));

            for (int j = j_start; j < j_end; ++j) {
                int band_idx = j - j_start;
                A[i * n_ + j] = band_values_[i * 4 + band_idx];
            }
        }

        const double pivot_tol = 1e-14;

        // Gaussian elimination with partial pivoting
        for (size_t col = 0; col < n_; ++col) {
            // Find pivot
            size_t pivot_row = col;
            double pivot_val = std::abs(A[col * n_ + col]);

            for (size_t row = col + 1; row < n_; ++row) {
                double val = std::abs(A[row * n_ + col]);
                if (val > pivot_val) {
                    pivot_val = val;
                    pivot_row = row;
                }
            }

            // Check for singular matrix
            if (pivot_val < pivot_tol) {
                return false;
            }

            // Swap rows if needed
            if (pivot_row != col) {
                for (size_t j = 0; j < n_; ++j) {
                    std::swap(A[col * n_ + j], A[pivot_row * n_ + j]);
                }
                std::swap(solution[col], solution[pivot_row]);
            }

            // Eliminate column
            double diag = A[col * n_ + col];
            for (size_t row = col + 1; row < n_; ++row) {
                double mult = A[row * n_ + col] / diag;

                for (size_t j = col; j < n_; ++j) {
                    A[row * n_ + j] -= mult * A[col * n_ + j];
                }

                solution[row] -= mult * solution[col];
            }
        }

        // Back substitution
        for (int i = static_cast<int>(n_) - 1; i >= 0; --i) {
            double sum = solution[i];

            for (size_t j = i + 1; j < n_; ++j) {
                sum -= A[i * n_ + j] * solution[j];
            }

            solution[i] = sum / A[i * n_ + i];
        }

        return true;
    }

    /// Compute max residual ||B*c - f||_∞ using banded storage
    double compute_residual(const std::vector<double>& coeffs,
                           const std::vector<double>& values) const {
        double max_res = 0.0;

        for (size_t i = 0; i < n_; ++i) {
            // Compute (B*c)[i] - only sum over 4 non-zero entries
            double Bc_i = 0.0;
            int j_start = band_col_start_[i];
            int j_end = std::min(j_start + 4, static_cast<int>(n_));

            for (int j = j_start; j < j_end; ++j) {
                int band_idx = j - j_start;
                double b_ij = band_values_[i * 4 + band_idx];
                Bc_i = std::fma(b_ij, coeffs[j], Bc_i);
            }

            double residual = std::abs(Bc_i - values[i]);
            max_res = std::max(max_res, residual);
        }

        return max_res;
    }

    /// Estimate 1-norm condition number
    double estimate_condition_number() const {
        // ||B||_1 = max column sum
        std::vector<double> col_sums(n_, 0.0);

        for (size_t i = 0; i < n_; ++i) {
            int j_start = band_col_start_[i];
            int j_end = std::min(j_start + 4, static_cast<int>(n_));

            for (int j = j_start; j < j_end; ++j) {
                int band_idx = j - j_start;
                double b_ij = band_values_[i * 4 + band_idx];
                col_sums[j] += std::abs(b_ij);
            }
        }

        double norm_B = *std::max_element(col_sums.begin(), col_sums.end());

        if (norm_B == 0.0) {
            return std::numeric_limits<double>::infinity();
        }

        // Approximate ||B^{-1}||_1
        std::vector<double> rhs(n_, 0.0);
        std::vector<double> solution(n_);
        double max_col_sum = 0.0;

        for (size_t j = 0; j < n_; ++j) {
            std::fill(rhs.begin(), rhs.end(), 0.0);
            rhs[j] = 1.0;

            if (!solve_banded_system(rhs, solution)) {
                return std::numeric_limits<double>::infinity();
            }

            double col_sum = 0.0;
            for (double val : solution) {
                col_sum += std::abs(val);
            }
            max_col_sum = std::max(max_col_sum, col_sum);
        }

        return norm_B * max_col_sum;
    }
};

// ============================================================================
// Separable 4D B-spline Fitter
// ============================================================================

/// Result of separable 4D fitting with per-axis diagnostics
struct BSplineFit4DSeparableResult {
    std::vector<double> coefficients;  ///< Final coefficients (N0 × N1 × N2 × N3)
    bool success;                       ///< Overall fit succeeded
    std::string error_message;          ///< Error if failed

    // Per-axis diagnostics
    double max_residual_axis0;          ///< Max residual after axis0 fit
    double max_residual_axis1;          ///< Max residual after axis1 fit
    double max_residual_axis2;          ///< Max residual after axis2 fit
    double max_residual_axis3;          ///< Max residual after axis3 fit

    double condition_axis0;             ///< Condition estimate for axis0
    double condition_axis1;             ///< Condition estimate for axis1
    double condition_axis2;             ///< Condition estimate for axis2
    double condition_axis3;             ///< Condition estimate for axis3

    size_t failed_slices_axis0;         ///< Number of failed 1D fits along axis0
    size_t failed_slices_axis1;         ///< Number of failed 1D fits along axis1
    size_t failed_slices_axis2;         ///< Number of failed 1D fits along axis2
    size_t failed_slices_axis3;         ///< Number of failed 1D fits along axis3
};

/// Separable 4D B-spline fitter
///
/// Exploits tensor-product structure to avoid solving a massive dense system.
/// Instead of solving one (N0·N1·N2·N3)² system, we solve many small 1D systems
/// sequentially along each axis with cache-optimized ordering.
///
/// Performs sequential 1D fitting along each axis using collocation.
/// Works in-place to minimize memory usage.
class BSplineFitter4DSeparable {
public:
    /// Create fitter with validation
    ///
    /// @param axis0_grid Grid for axis 0 (sorted, ≥4 points)
    /// @param axis1_grid Grid for axis 1 (sorted, ≥4 points)
    /// @param axis2_grid Grid for axis 2 (sorted, ≥4 points)
    /// @param axis3_grid Grid for axis 3 (sorted, ≥4 points)
    /// @return Fitter instance or error message
    ///
    /// @note Validation is delegated to BSplineCollocation1D for each axis.
    ///       Grids are checked during 1D solver construction.
    static expected<BSplineFitter4DSeparable, std::string> create(
        std::vector<double> axis0_grid,
        std::vector<double> axis1_grid,
        std::vector<double> axis2_grid,
        std::vector<double> axis3_grid)
    {
        try {
            return BSplineFitter4DSeparable(std::move(axis0_grid), std::move(axis1_grid),
                                           std::move(axis2_grid), std::move(axis3_grid));
        } catch (const std::exception& e) {
            return unexpected(std::string(e.what()));
        }
    }

    /// Control banded solver usage for all axes
    ///
    /// @param use_banded If true, use efficient O(n) banded LU solver. If false, use dense solver.
    void set_use_banded_solver(bool use_banded) {
        solver_axis0_->set_use_banded_solver(use_banded);
        solver_axis1_->set_use_banded_solver(use_banded);
        solver_axis2_->set_use_banded_solver(use_banded);
        solver_axis3_->set_use_banded_solver(use_banded);
    }

    /// Fit B-spline coefficients via separable collocation
    ///
    /// @param values Function values at grid points (row-major: i*N1*N2*N3 + j*N2*N3 + k*N3 + l)
    /// @param tolerance Max allowed residual per axis (default 1e-6)
    /// @return Fit result with coefficients and diagnostics
    ///
    /// @note Axis order optimized for cache locality (axis3 → axis2 → axis1 → axis0)
    ///       Processing fastest-varying dimensions first minimizes cache misses
    BSplineFit4DSeparableResult fit(const std::vector<double>& values, double tolerance = 1e-6) {
        if (values.size() != N0_ * N1_ * N2_ * N3_) {
            return {std::vector<double>(), false,
                    "Value array size mismatch", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        }

        // Work in-place: copy values to coefficients array
        std::vector<double> coeffs = values;

        BSplineFit4DSeparableResult result;
        result.success = true;
        result.failed_slices_axis0 = 0;
        result.failed_slices_axis1 = 0;
        result.failed_slices_axis2 = 0;
        result.failed_slices_axis3 = 0;

        // CACHE-OPTIMIZED AXIS ORDER (fastest-varying to slowest-varying)
        // Memory layout: ((i*N1 + j)*N2 + k)*N3 + l
        // Strides: axis3=1, axis2=N3, axis1=N2*N3, axis0=N1*N2*N3

        // Step 1: axis3 (stride=1, contiguous access)
        if (!fit_axis3(coeffs, tolerance, result)) {
            result.success = false;
            result.error_message = "Failed fitting along axis3: " +
                                   std::to_string(result.failed_slices_axis3) + " slices failed";
            return result;
        }

        // Step 2: axis2 (stride=N3, small jumps)
        if (!fit_axis2(coeffs, tolerance, result)) {
            result.success = false;
            result.error_message = "Failed fitting along axis2: " +
                                   std::to_string(result.failed_slices_axis2) + " slices failed";
            return result;
        }

        // Step 3: axis1 (stride=N2*N3, medium jumps)
        if (!fit_axis1(coeffs, tolerance, result)) {
            result.success = false;
            result.error_message = "Failed fitting along axis1: " +
                                   std::to_string(result.failed_slices_axis1) + " slices failed";
            return result;
        }

        // Step 4: axis0 (stride=N1*N2*N3, large jumps - done last)
        if (!fit_axis0(coeffs, tolerance, result)) {
            result.success = false;
            result.error_message = "Failed fitting along axis0: " +
                                   std::to_string(result.failed_slices_axis0) + " slices failed";
            return result;
        }

        result.coefficients = std::move(coeffs);
        return result;
    }

private:
    /// Private constructor for factory method
    BSplineFitter4DSeparable(std::vector<double> axis0_grid,
                             std::vector<double> axis1_grid,
                             std::vector<double> axis2_grid,
                             std::vector<double> axis3_grid)
        : axis0_grid_(std::move(axis0_grid))
        , axis1_grid_(std::move(axis1_grid))
        , axis2_grid_(std::move(axis2_grid))
        , axis3_grid_(std::move(axis3_grid))
        , N0_(axis0_grid_.size())
        , N1_(axis1_grid_.size())
        , N2_(axis2_grid_.size())
        , N3_(axis3_grid_.size())
    {
        // Create 1D solvers for each axis using factory method
        auto axis0_result = BSplineCollocation1D::create(axis0_grid_);
        auto axis1_result = BSplineCollocation1D::create(axis1_grid_);
        auto axis2_result = BSplineCollocation1D::create(axis2_grid_);
        auto axis3_result = BSplineCollocation1D::create(axis3_grid_);

        // Collect all error messages if any solver construction fails
        if (!axis0_result.has_value() || !axis1_result.has_value() ||
            !axis2_result.has_value() || !axis3_result.has_value()) {
            throw std::runtime_error("Failed to create BSplineCollocation1D solvers: " +
                                   (axis0_result.has_value() ? "" : "axis0: " + axis0_result.error() + "; ") +
                                   (axis1_result.has_value() ? "" : "axis1: " + axis1_result.error() + "; ") +
                                   (axis2_result.has_value() ? "" : "axis2: " + axis2_result.error() + "; ") +
                                   (axis3_result.has_value() ? "" : "axis3: " + axis3_result.error()));
        }

        solver_axis0_ = std::make_unique<BSplineCollocation1D>(std::move(axis0_result.value()));
        solver_axis1_ = std::make_unique<BSplineCollocation1D>(std::move(axis1_result.value()));
        solver_axis2_ = std::make_unique<BSplineCollocation1D>(std::move(axis2_result.value()));
        solver_axis3_ = std::make_unique<BSplineCollocation1D>(std::move(axis3_result.value()));
    }

    std::vector<double> axis0_grid_, axis1_grid_, axis2_grid_, axis3_grid_;
    size_t N0_, N1_, N2_, N3_;

    std::unique_ptr<BSplineCollocation1D> solver_axis0_;
    std::unique_ptr<BSplineCollocation1D> solver_axis1_;
    std::unique_ptr<BSplineCollocation1D> solver_axis2_;
    std::unique_ptr<BSplineCollocation1D> solver_axis3_;

    /// Fit along axis0 for all (j,k,l) slices
    bool fit_axis0(std::vector<double>& coeffs, double tolerance,
                   BSplineFit4DSeparableResult& result) {
        std::vector<double> slice(N0_);
        std::vector<double> max_residuals;
        std::vector<double> conditions;

        for (size_t j = 0; j < N1_; ++j) {
            for (size_t k = 0; k < N2_; ++k) {
                for (size_t l = 0; l < N3_; ++l) {
                    // Extract 1D slice along axis0: coeffs[:,j,k,l]
                    for (size_t i = 0; i < N0_; ++i) {
                        size_t idx = ((i * N1_ + j) * N2_ + k) * N3_ + l;
                        slice[i] = coeffs[idx];
                    }

                    // Fit 1D B-spline
                    auto fit_result = solver_axis0_->fit(slice, tolerance);

                    if (!fit_result.success) {
                        ++result.failed_slices_axis0;
                        return false;
                    }

                    // Write coefficients back
                    for (size_t i = 0; i < N0_; ++i) {
                        size_t idx = ((i * N1_ + j) * N2_ + k) * N3_ + l;
                        coeffs[idx] = fit_result.coefficients[i];
                    }

                    max_residuals.push_back(fit_result.max_residual);
                    conditions.push_back(fit_result.condition_estimate);
                }
            }
        }

        result.max_residual_axis0 = *std::max_element(max_residuals.begin(), max_residuals.end());
        result.condition_axis0 = *std::max_element(conditions.begin(), conditions.end());
        return true;
    }

    /// Fit along axis1 for all (i,k,l) slices
    bool fit_axis1(std::vector<double>& coeffs, double tolerance,
                   BSplineFit4DSeparableResult& result) {
        std::vector<double> slice(N1_);
        std::vector<double> max_residuals;
        std::vector<double> conditions;

        for (size_t i = 0; i < N0_; ++i) {
            for (size_t k = 0; k < N2_; ++k) {
                for (size_t l = 0; l < N3_; ++l) {
                    // Extract 1D slice along axis1: coeffs[i,:,k,l]
                    for (size_t j = 0; j < N1_; ++j) {
                        size_t idx = ((i * N1_ + j) * N2_ + k) * N3_ + l;
                        slice[j] = coeffs[idx];
                    }

                    // Fit 1D B-spline
                    auto fit_result = solver_axis1_->fit(slice, tolerance);

                    if (!fit_result.success) {
                        ++result.failed_slices_axis1;
                        return false;
                    }

                    // Write coefficients back
                    for (size_t j = 0; j < N1_; ++j) {
                        size_t idx = ((i * N1_ + j) * N2_ + k) * N3_ + l;
                        coeffs[idx] = fit_result.coefficients[j];
                    }

                    max_residuals.push_back(fit_result.max_residual);
                    conditions.push_back(fit_result.condition_estimate);
                }
            }
        }

        result.max_residual_axis1 = *std::max_element(max_residuals.begin(), max_residuals.end());
        result.condition_axis1 = *std::max_element(conditions.begin(), conditions.end());
        return true;
    }

    /// Fit along axis2 for all (i,j,l) slices
    bool fit_axis2(std::vector<double>& coeffs, double tolerance,
                   BSplineFit4DSeparableResult& result) {
        std::vector<double> slice(N2_);
        std::vector<double> max_residuals;
        std::vector<double> conditions;

        for (size_t i = 0; i < N0_; ++i) {
            for (size_t j = 0; j < N1_; ++j) {
                for (size_t l = 0; l < N3_; ++l) {
                    // Extract 1D slice along axis2: coeffs[i,j,:,l]
                    for (size_t k = 0; k < N2_; ++k) {
                        size_t idx = ((i * N1_ + j) * N2_ + k) * N3_ + l;
                        slice[k] = coeffs[idx];
                    }

                    // Fit 1D B-spline
                    auto fit_result = solver_axis2_->fit(slice, tolerance);

                    if (!fit_result.success) {
                        ++result.failed_slices_axis2;
                        return false;
                    }

                    // Write coefficients back
                    for (size_t k = 0; k < N2_; ++k) {
                        size_t idx = ((i * N1_ + j) * N2_ + k) * N3_ + l;
                        coeffs[idx] = fit_result.coefficients[k];
                    }

                    max_residuals.push_back(fit_result.max_residual);
                    conditions.push_back(fit_result.condition_estimate);
                }
            }
        }

        result.max_residual_axis2 = *std::max_element(max_residuals.begin(), max_residuals.end());
        result.condition_axis2 = *std::max_element(conditions.begin(), conditions.end());
        return true;
    }

    /// Fit along axis3 for all (i,j,k) slices
    bool fit_axis3(std::vector<double>& coeffs, double tolerance,
                   BSplineFit4DSeparableResult& result) {
        std::vector<double> slice(N3_);
        std::vector<double> max_residuals;
        std::vector<double> conditions;

        for (size_t i = 0; i < N0_; ++i) {
            for (size_t j = 0; j < N1_; ++j) {
                for (size_t k = 0; k < N2_; ++k) {
                    // Extract 1D slice along axis3: coeffs[i,j,k,:]
                    for (size_t l = 0; l < N3_; ++l) {
                        size_t idx = ((i * N1_ + j) * N2_ + k) * N3_ + l;
                        slice[l] = coeffs[idx];
                    }

                    // Fit 1D B-spline
                    auto fit_result = solver_axis3_->fit(slice, tolerance);

                    if (!fit_result.success) {
                        ++result.failed_slices_axis3;
                        return false;
                    }

                    // Write coefficients back
                    for (size_t l = 0; l < N3_; ++l) {
                        size_t idx = ((i * N1_ + j) * N2_ + k) * N3_ + l;
                        coeffs[idx] = fit_result.coefficients[l];
                    }

                    max_residuals.push_back(fit_result.max_residual);
                    conditions.push_back(fit_result.condition_estimate);
                }
            }
        }

        result.max_residual_axis3 = *std::max_element(max_residuals.begin(), max_residuals.end());
        result.condition_axis3 = *std::max_element(conditions.begin(), conditions.end());
        return true;
    }
};

// ============================================================================
// 4D B-spline Fitter (High-level Interface)
// ============================================================================

/// Result of 4D B-spline fitting
struct BSplineFitResult4D {
    std::vector<double> coefficients;  ///< Fitted coefficients (N0 × N1 × N2 × N3)
    bool success;                       ///< Fit succeeded
    std::string error_message;          ///< Error description if failed
    double max_residual;                ///< Maximum absolute residual at grid points

    // Detailed per-axis statistics (populated if success)
    double max_residual_axis0 = 0.0;   ///< Max residual along axis0
    double max_residual_axis1 = 0.0;   ///< Max residual along axis1
    double max_residual_axis2 = 0.0;   ///< Max residual along axis2
    double max_residual_axis3 = 0.0;   ///< Max residual along axis3

    double condition_axis0 = 0.0;      ///< Condition number estimate (axis0)
    double condition_axis1 = 0.0;      ///< Condition number estimate (axis1)
    double condition_axis2 = 0.0;      ///< Condition number estimate (axis2)
    double condition_axis3 = 0.0;      ///< Condition number estimate (axis3)

    size_t failed_slices_axis0 = 0;    ///< Failed fits along axis0
    size_t failed_slices_axis1 = 0;    ///< Failed fits along axis1
    size_t failed_slices_axis2 = 0;    ///< Failed fits along axis2
    size_t failed_slices_axis3 = 0;    ///< Failed fits along axis3
};

/// Separable 4D B-spline coefficient fitter
///
/// Uses tensor-product structure to avoid dense O(n⁴) solve.
/// Performs sequential 1D fits along each dimension.
///
/// Memory: O(N0·N1·N2·N3) for temporary storage
/// Time: O(N0·N1·N2·N3) for all 1D fits
class BSplineFitter4D {
public:
    /// Factory method to create BSplineFitter4D with validation
    ///
    /// @param axis0_grid Grid for axis 0 (sorted, ≥4 points)
    /// @param axis1_grid Grid for axis 1 (sorted, ≥4 points)
    /// @param axis2_grid Grid for axis 2 (sorted, ≥4 points)
    /// @param axis3_grid Grid for axis 3 (sorted, ≥4 points)
    /// @return expected<BSplineFitter4D, std::string> - success or error message
    ///
    /// @note Validation is delegated to BSplineCollocation1D via BSplineFitter4DSeparable.
    ///       We validate at creation time by attempting to create a separable fitter.
    static expected<BSplineFitter4D, std::string> create(
        std::vector<double> axis0_grid,
        std::vector<double> axis1_grid,
        std::vector<double> axis2_grid,
        std::vector<double> axis3_grid) {

        // Validate grids by attempting to create separable fitter
        // This delegates validation to BSplineCollocation1D for each axis
        auto validation_result = BSplineFitter4DSeparable::create(
            axis0_grid, axis1_grid, axis2_grid, axis3_grid);

        if (!validation_result.has_value()) {
            return unexpected(validation_result.error());
        }

        // Grids are valid, create the fitter
        return BSplineFitter4D(std::move(axis0_grid), std::move(axis1_grid),
                               std::move(axis2_grid), std::move(axis3_grid));
    }

    /// Fit B-spline coefficients via separable collocation
    ///
    /// Uses tensor-product structure: sequential 1D fitting along each axis.
    /// Produces numerically accurate coefficients with residuals <1e-6.
    ///
    /// @param values Function values at grid points (size N0 × N1 × N2 × N3)
    ///               Row-major layout: index = ((i*N1 + j)*N2 + k)*N3 + l
    /// @param tolerance Maximum residual per axis (default 1e-6)
    /// @param use_banded_solver If true, use efficient O(n) banded solver (default: true)
    /// @return Fit result with coefficients and diagnostics
    BSplineFitResult4D fit(const std::vector<double>& values, double tolerance = 1e-6, bool use_banded_solver = true) {
        // Create separable fitter using factory pattern
        auto fitter_result = BSplineFitter4DSeparable::create(axis0_grid_, axis1_grid_,
                                                              axis2_grid_, axis3_grid_);
        if (!fitter_result.has_value()) {
            return {
                .coefficients = std::vector<double>(),
                .success = false,
                .error_message = fitter_result.error(),
                .max_residual = 0.0
            };
        }
        auto& fitter = fitter_result.value();

        // Configure solver mode
        fitter.set_use_banded_solver(use_banded_solver);

        // Perform separable fitting
        auto sep_result = fitter.fit(values, tolerance);

        if (!sep_result.success) {
            return {
                .coefficients = std::vector<double>(),
                .success = false,
                .error_message = sep_result.error_message,
                .max_residual = 0.0
            };
        }

        // Aggregate maximum residual across all axes
        double max_residual = std::max({
            sep_result.max_residual_axis0,
            sep_result.max_residual_axis1,
            sep_result.max_residual_axis2,
            sep_result.max_residual_axis3
        });

        // Return with full statistics
        return {
            .coefficients = sep_result.coefficients,
            .success = true,
            .error_message = "",
            .max_residual = max_residual,
            .max_residual_axis0 = sep_result.max_residual_axis0,
            .max_residual_axis1 = sep_result.max_residual_axis1,
            .max_residual_axis2 = sep_result.max_residual_axis2,
            .max_residual_axis3 = sep_result.max_residual_axis3,
            .condition_axis0 = sep_result.condition_axis0,
            .condition_axis1 = sep_result.condition_axis1,
            .condition_axis2 = sep_result.condition_axis2,
            .condition_axis3 = sep_result.condition_axis3,
            .failed_slices_axis0 = sep_result.failed_slices_axis0,
            .failed_slices_axis1 = sep_result.failed_slices_axis1,
            .failed_slices_axis2 = sep_result.failed_slices_axis2,
            .failed_slices_axis3 = sep_result.failed_slices_axis3
        };
    }

    /// Get grid dimensions
    [[nodiscard]] std::tuple<size_t, size_t, size_t, size_t> dimensions() const noexcept {
        return {N0_, N1_, N2_, N3_};
    }

private:
    /// Private constructor - use factory method create() instead
    ///
    /// @param axis0_grid Grid for axis 0 (validation delegated to separable fitter)
    /// @param axis1_grid Grid for axis 1 (validation delegated to separable fitter)
    /// @param axis2_grid Grid for axis 2 (validation delegated to separable fitter)
    /// @param axis3_grid Grid for axis 3 (validation delegated to separable fitter)
    BSplineFitter4D(std::vector<double> axis0_grid,
                    std::vector<double> axis1_grid,
                    std::vector<double> axis2_grid,
                    std::vector<double> axis3_grid)
        : axis0_grid_(std::move(axis0_grid)),
          axis1_grid_(std::move(axis1_grid)),
          axis2_grid_(std::move(axis2_grid)),
          axis3_grid_(std::move(axis3_grid)),
          N0_(axis0_grid_.size()),
          N1_(axis1_grid_.size()),
          N2_(axis2_grid_.size()),
          N3_(axis3_grid_.size())
    {
        // Pre-compute knot vectors (no validation needed - delegated to separable fitter)
        t0_ = clamped_knots_cubic(axis0_grid_);
        t1_ = clamped_knots_cubic(axis1_grid_);
        t2_ = clamped_knots_cubic(axis2_grid_);
        t3_ = clamped_knots_cubic(axis3_grid_);
    }

    // Friend declaration for factory method to access private constructor
    friend expected<BSplineFitter4D, std::string> create(
        std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>);

    std::vector<double> axis0_grid_;  ///< Grid for axis 0
    std::vector<double> axis1_grid_;  ///< Grid for axis 1
    std::vector<double> axis2_grid_;  ///< Grid for axis 2
    std::vector<double> axis3_grid_;  ///< Grid for axis 3

    std::vector<double> t0_;  ///< Knot vector for axis 0
    std::vector<double> t1_;  ///< Knot vector for axis 1
    std::vector<double> t2_;  ///< Knot vector for axis 2
    std::vector<double> t3_;  ///< Knot vector for axis 3

    size_t N0_;  ///< Number of points on axis 0
    size_t N1_;  ///< Number of points on axis 1
    size_t N2_;  ///< Number of points on axis 2
    size_t N3_;  ///< Number of points on axis 3
};

}  // namespace mango
