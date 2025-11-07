/**
 * @file bspline_collocation_1d.hpp
 * @brief 1D cubic B-spline collocation solver for coefficient fitting
 *
 * Solves the collocation system: B*c = f
 * where:
 *   - B[i,j] = N_j(x_i) is the collocation matrix (basis j at grid point i)
 *   - c are the unknown control points (coefficients)
 *   - f are the known function values at grid points
 *
 * For clamped cubic B-splines:
 *   - n data points → n basis functions → n×n system
 *   - Collocation matrix is banded (width 4 for cubic)
 *   - System is well-conditioned for reasonable grids
 *
 * Algorithm:
 *   1. Build collocation matrix B (evaluate all basis at all grid points)
 *   2. Solve banded system using custom solver with pivoting
 *   3. Validate residuals ||B*c - f|| < tolerance
 *
 * Performance: O(n) for banded solve (width w=4 is constant)
 *
 * Usage:
 *   BSplineCollocation1D solver(grid);
 *   auto result = solver.fit(values);
 *   if (result.success) {
 *       // Use result.coefficients with BSpline evaluator
 *   }
 */

#pragma once

#include "bspline_basis_1d.hpp"
#include "bspline_utils.hpp"
#include "thomas_solver.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <optional>
#include <string>
#include <stdexcept>

namespace mango {

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
/// Builds and solves the collocation system to find control points
/// that make the B-spline interpolate the given data.
class BSplineCollocation1D {
public:
    /// Constructor
    ///
    /// @param grid Data grid points (sorted, ≥4 points)
    explicit BSplineCollocation1D(std::vector<double> grid)
        : grid_(std::move(grid))
        , n_(grid_.size())
    {
        if (n_ < 4) {
            throw std::invalid_argument("Grid must have ≥4 points for cubic B-splines");
        }
        if (!std::is_sorted(grid_.begin(), grid_.end())) {
            throw std::invalid_argument("Grid must be sorted");
        }

        // Check for duplicate or near-duplicate points
        constexpr double MIN_SPACING = 1e-12;
        for (size_t i = 1; i < n_; ++i) {
            double spacing = grid_[i] - grid_[i-1];
            if (spacing < MIN_SPACING) {
                throw std::invalid_argument(
                    "Grid points too close together (spacing < 1e-12). "
                    "Found grid[" + std::to_string(i-1) + "] = " + std::to_string(grid_[i-1]) +
                    " and grid[" + std::to_string(i) + "] = " + std::to_string(grid_[i]) +
                    " with spacing " + std::to_string(spacing)
                );
            }
        }

        // Check for zero-width grid
        if (grid_.back() - grid_.front() < MIN_SPACING) {
            throw std::invalid_argument("Grid has zero width (all points nearly identical)");
        }

        // Build knot vector (clamped cubic)
        knots_ = clamped_knots_cubic(grid_);

        // Pre-allocate collocation matrix (banded storage)
        // For cubic B-splines, bandwidth = 4
        collocation_matrix_.resize(n_ * 4, 0.0);
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

        // Estimate condition number (crude: max/min diagonal ratio)
        double cond_est = estimate_condition();

        return {coeffs, true, "", max_residual, cond_est};
    }

private:
    std::vector<double> grid_;              ///< Data grid points
    std::vector<double> knots_;             ///< Knot vector (clamped)
    size_t n_;                              ///< Number of grid points
    std::vector<double> collocation_matrix_; ///< Full n×n matrix (row-major)

    /// Build collocation matrix B[i,j] = N_j(x_i)
    ///
    /// For each grid point i, evaluate all n basis functions.
    /// Most entries are zero due to local support, but we store full matrix
    /// for simplicity (matrices are small, typically n ≤ 50).
    void build_collocation_matrix() {
        collocation_matrix_.assign(n_ * n_, 0.0);

        for (size_t i = 0; i < n_; ++i) {
            const double x = grid_[i];

            // Find knot span
            int span = find_span_cubic(knots_, x);

            // Evaluate 4 non-zero basis functions at x
            double basis[4];
            cubic_basis_nonuniform(knots_, span, x, basis);

            // Store in full matrix: B[i,j] at index [i*n + j]
            // basis[0] corresponds to N_{span}
            // basis[1] corresponds to N_{span-1}, etc.
            for (int k = 0; k < 4; ++k) {
                int j = span - k;  // Basis function index
                if (j >= 0 && j < static_cast<int>(n_)) {
                    collocation_matrix_[i * n_ + j] = basis[k];
                }
            }
        }
    }

    /// Solve linear system using Gaussian elimination with partial pivoting
    ///
    /// @param rhs Right-hand side (function values)
    /// @param solution Output coefficients
    /// @return True if solve succeeded
    bool solve_banded_system(const std::vector<double>& rhs, std::vector<double>& solution) {
        // Copy to working arrays
        solution = rhs;
        std::vector<double> A = collocation_matrix_;

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

    /// Compute max residual ||B*c - f||_∞
    double compute_residual(const std::vector<double>& coeffs,
                           const std::vector<double>& values) const {
        double max_res = 0.0;

        for (size_t i = 0; i < n_; ++i) {
            // Compute (B*c)[i]
            // PERFORMANCE: Use FMA for tighter precision and better codegen
            double Bc_i = 0.0;
            for (size_t j = 0; j < n_; ++j) {
                Bc_i = std::fma(collocation_matrix_[i * n_ + j], coeffs[j], Bc_i);
            }

            double residual = std::abs(Bc_i - values[i]);
            max_res = std::max(max_res, residual);
        }

        return max_res;
    }

    /// Estimate condition number (crude: max/min diagonal ratio)
    double estimate_condition() const {
        double min_diag = std::numeric_limits<double>::max();
        double max_diag = 0.0;

        for (size_t i = 0; i < n_; ++i) {
            double diag = std::abs(collocation_matrix_[i * n_ + i]);
            min_diag = std::min(min_diag, diag);
            max_diag = std::max(max_diag, diag);
        }

        return (min_diag > 0.0) ? (max_diag / min_diag) : 1e100;
    }
};

}  // namespace mango
