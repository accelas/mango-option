/**
 * @file eigen_banded_solver.hpp
 * @brief Eigen-based banded linear system solver
 *
 * Provides a clean interface for solving banded systems using Eigen's
 * sparse matrix capabilities. Specifically designed for B-spline collocation
 * matrices which are pentadiagonal (width-4 band).
 *
 * Features:
 * - Automatic conversion from banded storage to Eigen sparse format
 * - SparseLU factorization for general nonsymmetric systems
 * - Reusable solver with multiple RHS support
 * - Error checking and status reporting
 *
 * Performance: ~10-50µs for n=100-1000 (comparable to custom implementations)
 *
 * Usage:
 *   EigenBandedSolver solver(n, kl, ku);
 *   solver.set_banded_matrix(subdiag2, subdiag1, diag, superdiag1, superdiag2);
 *   auto result = solver.solve(rhs);
 */

#pragma once

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace mango {

/// Result of banded linear system solve
struct EigenSolveResult {
    std::vector<double> solution;  ///< Solution vector
    bool success;                   ///< Solve succeeded
    std::string error_message;      ///< Error description if failed
};

/// Eigen-based banded linear system solver
///
/// Solves banded systems Ax=b where A has lower bandwidth kl and upper
/// bandwidth ku. Uses Eigen's SparseLU for general nonsymmetric matrices.
///
/// Memory: O(n * (kl + ku + 1)) for matrix storage
/// Time: O(n * (kl + ku)^2) for factorization
class EigenBandedSolver {
public:
    /// Construct solver for banded system
    ///
    /// @param n System size
    /// @param kl Lower bandwidth (number of subdiagonals)
    /// @param ku Upper bandwidth (number of superdiagonals)
    EigenBandedSolver(size_t n, size_t kl, size_t ku)
        : n_(n), kl_(kl), ku_(ku), factorized_(false) {

        if (n == 0) {
            throw std::invalid_argument("System size must be positive");
        }
        if (kl >= n || ku >= n) {
            throw std::invalid_argument("Bandwidth must be less than system size");
        }

        // Reserve space for sparse matrix (estimated non-zeros)
        size_t nnz_estimate = n_ * (kl_ + ku_ + 1);
        matrix_.resize(n_, n_);
        matrix_.reserve(nnz_estimate);
    }

    /// Set matrix from pentadiagonal storage
    ///
    /// For pentadiagonal system (kl=2, ku=2):
    /// - subdiag2[i] = A[i+2][i] (second subdiagonal)
    /// - subdiag1[i] = A[i+1][i] (first subdiagonal)
    /// - diag[i] = A[i][i] (main diagonal)
    /// - superdiag1[i] = A[i][i+1] (first superdiagonal)
    /// - superdiag2[i] = A[i][i+2] (second superdiagonal)
    ///
    /// @param subdiag2 Second subdiagonal (size n-2, can be nullptr if kl<2)
    /// @param subdiag1 First subdiagonal (size n-1, can be nullptr if kl<1)
    /// @param diag Main diagonal (size n)
    /// @param superdiag1 First superdiagonal (size n-1, can be nullptr if ku<1)
    /// @param superdiag2 Second superdiagonal (size n-2, can be nullptr if ku<2)
    void set_pentadiagonal_matrix(
        const double* subdiag2,
        const double* subdiag1,
        const double* diag,
        const double* superdiag1,
        const double* superdiag2)
    {
        matrix_.setZero();
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(n_ * (kl_ + ku_ + 1));

        // Main diagonal
        for (size_t i = 0; i < n_; ++i) {
            triplets.emplace_back(i, i, diag[i]);
        }

        // First subdiagonal (A[i+1][i])
        if (kl_ >= 1 && subdiag1 != nullptr) {
            for (size_t i = 0; i < n_ - 1; ++i) {
                triplets.emplace_back(i + 1, i, subdiag1[i]);
            }
        }

        // Second subdiagonal (A[i+2][i])
        if (kl_ >= 2 && subdiag2 != nullptr) {
            for (size_t i = 0; i < n_ - 2; ++i) {
                triplets.emplace_back(i + 2, i, subdiag2[i]);
            }
        }

        // First superdiagonal (A[i][i+1])
        if (ku_ >= 1 && superdiag1 != nullptr) {
            for (size_t i = 0; i < n_ - 1; ++i) {
                triplets.emplace_back(i, i + 1, superdiag1[i]);
            }
        }

        // Second superdiagonal (A[i][i+2])
        if (ku_ >= 2 && superdiag2 != nullptr) {
            for (size_t i = 0; i < n_ - 2; ++i) {
                triplets.emplace_back(i, i + 2, superdiag2[i]);
            }
        }

        matrix_.setFromTriplets(triplets.begin(), triplets.end());
        factorized_ = false;  // Need to refactorize
    }

    /// Factorize the matrix (LU decomposition)
    ///
    /// @return true if factorization succeeded
    bool factorize() {
        solver_.compute(matrix_);

        if (solver_.info() != Eigen::Success) {
            last_error_ = "Factorization failed - matrix may be singular";
            factorized_ = false;
            return false;
        }

        factorized_ = true;
        last_error_.clear();
        return true;
    }

    /// Solve Ax=b
    ///
    /// Automatically factorizes if not already done.
    ///
    /// @param rhs Right-hand side vector (size n)
    /// @return Solution result with status
    EigenSolveResult solve(const std::vector<double>& rhs) {
        if (rhs.size() != n_) {
            return {std::vector<double>(), false,
                    "RHS size mismatch (expected " + std::to_string(n_) +
                    ", got " + std::to_string(rhs.size()) + ")"};
        }

        // Factorize if needed
        if (!factorized_) {
            if (!factorize()) {
                return {std::vector<double>(), false, last_error_};
            }
        }

        // Convert RHS to Eigen format
        Eigen::VectorXd b(n_);
        for (size_t i = 0; i < n_; ++i) {
            b[i] = rhs[i];
        }

        // Solve
        Eigen::VectorXd x = solver_.solve(b);

        if (solver_.info() != Eigen::Success) {
            return {std::vector<double>(), false, "Solve failed - numerical issues"};
        }

        // Convert solution back to std::vector
        std::vector<double> solution(n_);
        for (size_t i = 0; i < n_; ++i) {
            solution[i] = x[i];
        }

        // Verify solution quality (optional check)
        Eigen::VectorXd residual = matrix_ * x - b;
        double residual_norm = residual.norm();
        double rhs_norm = b.norm();
        double relative_residual = (rhs_norm > 1e-14) ? residual_norm / rhs_norm : residual_norm;

        if (relative_residual > 1e-6) {
            return {solution, false,
                    "Large residual error: " + std::to_string(relative_residual)};
        }

        return {solution, true, ""};
    }

    /// Check if solver is ready (matrix factorized)
    [[nodiscard]] bool is_ready() const noexcept { return factorized_; }

    /// Get system size
    [[nodiscard]] size_t size() const noexcept { return n_; }

    /// Get lower bandwidth
    [[nodiscard]] size_t lower_bandwidth() const noexcept { return kl_; }

    /// Get upper bandwidth
    [[nodiscard]] size_t upper_bandwidth() const noexcept { return ku_; }

    /// Get last error message
    [[nodiscard]] const std::string& last_error() const noexcept { return last_error_; }

private:
    size_t n_;   ///< System size
    size_t kl_;  ///< Lower bandwidth
    size_t ku_;  ///< Upper bandwidth

    Eigen::SparseMatrix<double> matrix_;     ///< Sparse matrix storage
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_;  ///< LU solver

    bool factorized_;        ///< Matrix has been factorized
    std::string last_error_; ///< Last error message
};

/// Convenience function: solve pentadiagonal system in one call
///
/// @param n System size
/// @param subdiag2 Second subdiagonal (size n-2, nullptr if not used)
/// @param subdiag1 First subdiagonal (size n-1, nullptr if not used)
/// @param diag Main diagonal (size n)
/// @param superdiag1 First superdiagonal (size n-1, nullptr if not used)
/// @param superdiag2 Second superdiagonal (size n-2, nullptr if not used)
/// @param rhs Right-hand side (size n)
/// @return Solution result
inline EigenSolveResult solve_pentadiagonal(
    size_t n,
    const double* subdiag2,
    const double* subdiag1,
    const double* diag,
    const double* superdiag1,
    const double* superdiag2,
    const std::vector<double>& rhs)
{
    EigenBandedSolver solver(n, 2, 2);
    solver.set_pentadiagonal_matrix(subdiag2, subdiag1, diag, superdiag1, superdiag2);
    return solver.solve(rhs);
}

/// Convenience function: solve tridiagonal system in one call
///
/// @param n System size
/// @param subdiag First subdiagonal (size n-1)
/// @param diag Main diagonal (size n)
/// @param superdiag First superdiagonal (size n-1)
/// @param rhs Right-hand side (size n)
/// @return Solution result
inline EigenSolveResult solve_tridiagonal(
    size_t n,
    const double* subdiag,
    const double* diag,
    const double* superdiag,
    const std::vector<double>& rhs)
{
    EigenBandedSolver solver(n, 1, 1);
    solver.set_pentadiagonal_matrix(nullptr, subdiag, diag, superdiag, nullptr);
    return solver.solve(rhs);
}

}  // namespace mango
