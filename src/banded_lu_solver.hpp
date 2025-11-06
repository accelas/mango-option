/**
 * @file banded_lu_solver.hpp
 * @brief Banded LU decomposition for sparse linear systems
 *
 * Implements LU factorization for banded matrices without pivoting.
 * Optimized for diagonally dominant systems arising from B-spline
 * least-squares fitting.
 *
 * Features:
 * - Compact band storage (only non-zero diagonals)
 * - O(n·bw²) complexity instead of O(n³) for dense
 * - Modern C++20 with concepts and std::span
 * - RAII workspace management
 * - Detailed error reporting
 *
 * References:
 * - Golub & Van Loan, "Matrix Computations" (4th ed.), §4.3
 * - LAPACK DGBSV documentation
 */

#pragma once

#include "solver_common.hpp"
#include <span>
#include <vector>
#include <cmath>
#include <optional>
#include <string_view>
#include <algorithm>
#include <cassert>

namespace mango {

/// Result type for banded LU solver
template<FloatingPoint T>
struct BandedLUResult {
    bool success;
    std::optional<std::string_view> error;

    /// Implicit conversion to bool for easy checking
    constexpr explicit operator bool() const noexcept { return success; }

    /// Check if solver succeeded
    [[nodiscard]] constexpr bool ok() const noexcept { return success; }

    /// Get error message (empty if successful)
    [[nodiscard]] constexpr std::string_view message() const noexcept {
        return error.value_or("");
    }

    /// Create success result
    [[nodiscard]] static constexpr BandedLUResult ok_result() noexcept {
        return BandedLUResult{.success = true, .error = std::nullopt};
    }

    /// Create error result
    [[nodiscard]] static constexpr BandedLUResult error_result(std::string_view msg) noexcept {
        return BandedLUResult{.success = false, .error = msg};
    }
};

/// Configuration for banded LU solver
template<FloatingPoint T>
struct BandedLUConfig {
    /// Tolerance for singularity detection
    T singularity_tol = static_cast<T>(1e-15);

    /// Whether to check for diagonal dominance (stricter condition)
    bool check_diagonal_dominance = false;
};

/// Banded LU Decomposition Solver
///
/// Solves Ax = b where A is a banded matrix with lower bandwidth kl
/// and upper bandwidth ku. Uses LU decomposition without pivoting.
///
/// Band Storage Format (row-wise):
///   For matrix A with bands kl below and ku above diagonal:
///   - lower_bands[k] contains the (k+1)-th subdiagonal (k=0..kl-1)
///   - diag contains the main diagonal
///   - upper_bands[k] contains the (k+1)-th superdiagonal (k=0..ku-1)
///
/// Example for kl=2, ku=2, n=5:
///   A = [d0 u0 u1  0  0]
///       [l0 d1 u0 u1  0]
///       [l1 l0 d2 u0 u1]
///       [ 0 l1 l0 d3 u0]
///       [ 0  0 l1 l0 d4]
///
/// Storage:
///   lower_bands[0] = [l0, l0, l0, l0]      (1st subdiagonal, length n-1)
///   lower_bands[1] = [l1, l1, l1]          (2nd subdiagonal, length n-2)
///   diag           = [d0, d1, d2, d3, d4]  (main diagonal, length n)
///   upper_bands[0] = [u0, u0, u0, u0]      (1st superdiagonal, length n-1)
///   upper_bands[1] = [u1, u1, u1]          (2nd superdiagonal, length n-2)
///
/// Complexity: O(n · kl · ku)
/// Workspace:  O(n · (kl + ku))
///
/// @tparam T Floating point type (float, double, long double)
template<FloatingPoint T>
class BandedLU {
public:
    /// Construct banded LU solver
    ///
    /// @param n System size
    /// @param lower_bw Lower bandwidth (number of subdiagonals)
    /// @param upper_bw Upper bandwidth (number of superdiagonals)
    BandedLU(size_t n, size_t lower_bw, size_t upper_bw)
        : n_(n),
          kl_(lower_bw),
          ku_(upper_bw),
          is_factorized_(false)
    {
        assert(n > 0 && "System size must be positive");
        assert(kl_ < n && "Lower bandwidth must be < n");
        assert(ku_ < n && "Upper bandwidth must be < n");

        // Allocate workspace for LU factors
        // During LU, upper bandwidth can grow from ku to kl+ku
        const size_t total_upper = kl_ + ku_;

        // Storage: kl lower bands + 1 diagonal + (kl+ku) upper bands
        diag_.resize(n);
        lower_bands_.resize(kl_);
        for (size_t k = 0; k < kl_; ++k) {
            lower_bands_[k].resize(n - k - 1);
        }

        upper_bands_.resize(total_upper);
        for (size_t k = 0; k < total_upper; ++k) {
            upper_bands_[k].resize(n - k - 1);
        }

        // Workspace for permutation and intermediate results
        workspace_.resize(n);
    }

    /// Factorize the matrix A = LU (without pivoting)
    ///
    /// @param lower_bands Input lower bands [kl bands of decreasing length]
    /// @param diag Input main diagonal [n elements]
    /// @param upper_bands Input upper bands [ku bands of decreasing length]
    /// @param config Solver configuration
    /// @return Result indicating success/failure
    [[nodiscard]] BandedLUResult<T> factorize(
        std::span<const std::span<const T>> lower_bands,
        std::span<const T> diag,
        std::span<const std::span<const T>> upper_bands,
        const BandedLUConfig<T>& config = {}) noexcept
    {
        using Result = BandedLUResult<T>;

        // Validate input dimensions
        if (lower_bands.size() != kl_) {
            return Result::error_result("Number of lower bands must match kl");
        }
        if (upper_bands.size() != ku_) {
            return Result::error_result("Number of upper bands must match ku");
        }
        if (diag.size() != n_) {
            return Result::error_result("Diagonal size must be n");
        }

        // Validate band lengths
        for (size_t k = 0; k < kl_; ++k) {
            if (lower_bands[k].size() != n_ - k - 1) {
                return Result::error_result("Invalid lower band length");
            }
        }
        for (size_t k = 0; k < ku_; ++k) {
            if (upper_bands[k].size() != n_ - k - 1) {
                return Result::error_result("Invalid upper band length");
            }
        }

        // Copy input to working storage
        std::copy(diag.begin(), diag.end(), diag_.begin());

        for (size_t k = 0; k < kl_; ++k) {
            std::copy(lower_bands[k].begin(), lower_bands[k].end(),
                     lower_bands_[k].begin());
        }

        for (size_t k = 0; k < ku_; ++k) {
            std::copy(upper_bands[k].begin(), upper_bands[k].end(),
                     upper_bands_[k].begin());
        }

        // Initialize extended upper bands to zero
        for (size_t k = ku_; k < kl_ + ku_; ++k) {
            std::fill(upper_bands_[k].begin(), upper_bands_[k].end(), T{0});
        }

        // Optional: Check diagonal dominance
        if (config.check_diagonal_dominance) {
            for (size_t i = 0; i < n_; ++i) {
                T diag_abs = std::abs(diag_[i]);
                T off_diag_sum = T{0};

                // Sum lower bands
                for (size_t k = 0; k < kl_ && i >= k + 1; ++k) {
                    off_diag_sum += std::abs(lower_bands_[k][i - k - 1]);
                }

                // Sum upper bands
                for (size_t k = 0; k < ku_ && i + k + 1 < n_; ++k) {
                    off_diag_sum += std::abs(upper_bands_[k][i]);
                }

                if (diag_abs < off_diag_sum) {
                    return Result::error_result("Matrix not diagonally dominant");
                }
            }
        }

        // ========== LU Factorization (without pivoting) ==========
        //
        // For each row i, eliminate all elements in column i below the diagonal
        // using row operations. This modifies the matrix in-place to produce L and U.

        for (size_t i = 0; i < n_; ++i) {
            // Check for singularity
            if (std::abs(diag_[i]) < config.singularity_tol) {
                is_factorized_ = false;
                return Result::error_result("Singular or near-singular matrix");
            }

            // Eliminate entries below diagonal in column i
            const size_t n_elim = std::min(kl_, n_ - i - 1);

            for (size_t k = 0; k < n_elim; ++k) {
                const size_t row = i + k + 1;

                // Compute multiplier: L[row, i] = A[row, i] / A[i, i]
                const T multiplier = lower_bands_[k][i] / diag_[i];
                lower_bands_[k][i] = multiplier;  // Store in L

                // Update row 'row' by subtracting multiplier * row i
                // This affects: diag[row] and upper bands starting from row

                // Update diagonal element
                if (k == 0) {
                    // Next row's diagonal -= multiplier * this row's first super
                    if (i < n_ - 1 && ku_ > 0) {
                        diag_[row] -= multiplier * upper_bands_[0][i];
                    }
                } else {
                    // row is further below, updates its diagonal via upper band k-1
                    if (k - 1 < ku_) {
                        diag_[row] -= multiplier * upper_bands_[k - 1][i];
                    }
                }

                // Update upper bands in row 'row'
                for (size_t j = 0; j < ku_ + kl_ && i + j + 1 < n_; ++j) {
                    const size_t band_idx = k + j;
                    if (j < ku_) {
                        // Original upper band
                        if (row + j < n_ - 1) {
                            const size_t target_band = (band_idx < ku_ + kl_) ? band_idx : 0;
                            if (target_band < upper_bands_.size() && i < upper_bands_[j].size()) {
                                if (row < n_ && target_band < upper_bands_.size()) {
                                    const size_t write_idx = row;
                                    if (write_idx < upper_bands_[target_band].size() &&
                                        i < upper_bands_[j].size()) {
                                        upper_bands_[target_band][write_idx] -=
                                            multiplier * upper_bands_[j][i];
                                    }
                                }
                            }
                        }
                    } else {
                        // Extended upper band (fill-in from elimination)
                        const size_t ext_band = band_idx;
                        if (ext_band < upper_bands_.size()) {
                            const size_t col_offset = j - ku_;
                            if (row + col_offset < n_ && ext_band < upper_bands_.size()) {
                                const size_t write_idx = row;
                                const size_t read_band = j;
                                if (write_idx < upper_bands_[ext_band].size() &&
                                    read_band < upper_bands_.size() &&
                                    i < upper_bands_[read_band].size()) {
                                    upper_bands_[ext_band][write_idx] -=
                                        multiplier * upper_bands_[read_band][i];
                                }
                            }
                        }
                    }
                }
            }
        }

        is_factorized_ = true;
        return Result::ok_result();
    }

    /// Solve Ax = b using the factorized matrix
    ///
    /// Must call factorize() first. Performs forward/backward substitution.
    ///
    /// @param rhs Right-hand side vector [n elements]
    /// @param solution Output solution vector [n elements]
    /// @return Result indicating success/failure
    [[nodiscard]] BandedLUResult<T> solve(
        std::span<const T> rhs,
        std::span<T> solution) const noexcept
    {
        using Result = BandedLUResult<T>;

        if (!is_factorized_) {
            return Result::error_result("Matrix not factorized");
        }

        if (rhs.size() != n_) {
            return Result::error_result("RHS size must be n");
        }

        if (solution.size() != n_) {
            return Result::error_result("Solution size must be n");
        }

        // Copy RHS to solution for in-place solve
        std::copy(rhs.begin(), rhs.end(), solution.begin());

        // ========== Forward Substitution: Solve Ly = b ==========
        // L is unit lower triangular (diagonal of 1s implicit)

        for (size_t i = 0; i < n_; ++i) {
            // solution[i] already contains b[i]
            // Subtract L[i,j] * solution[j] for j < i

            const size_t n_lower = std::min(kl_, i);
            for (size_t k = 0; k < n_lower; ++k) {
                const size_t j = i - k - 1;
                solution[i] -= lower_bands_[k][j] * solution[j];
            }
        }

        // ========== Backward Substitution: Solve Ux = y ==========
        // U is upper triangular with diagonal in diag_

        for (size_t i = n_; i > 0; --i) {
            const size_t row = i - 1;

            // Subtract U[row,j] * solution[j] for j > row
            const size_t n_upper = std::min(kl_ + ku_, n_ - row - 1);
            for (size_t k = 0; k < n_upper; ++k) {
                const size_t col = row + k + 1;
                if (k < upper_bands_.size() && row < upper_bands_[k].size()) {
                    solution[row] -= upper_bands_[k][row] * solution[col];
                }
            }

            // Divide by diagonal
            solution[row] /= diag_[row];
        }

        return Result::ok_result();
    }

    /// Get system size
    [[nodiscard]] constexpr size_t size() const noexcept { return n_; }

    /// Get lower bandwidth
    [[nodiscard]] constexpr size_t lower_bandwidth() const noexcept { return kl_; }

    /// Get upper bandwidth
    [[nodiscard]] constexpr size_t upper_bandwidth() const noexcept { return ku_; }

    /// Check if matrix has been factorized
    [[nodiscard]] constexpr bool is_factorized() const noexcept { return is_factorized_; }

private:
    size_t n_;                              ///< System size
    size_t kl_;                             ///< Lower bandwidth
    size_t ku_;                             ///< Upper bandwidth
    bool is_factorized_;                    ///< Factorization status

    std::vector<T> diag_;                   ///< Main diagonal (U)
    std::vector<std::vector<T>> lower_bands_; ///< Lower bands (L, unit diagonal implicit)
    std::vector<std::vector<T>> upper_bands_; ///< Upper bands (U, with fill-in)
    std::vector<T> workspace_;              ///< Temporary storage
};

/// Convenience function: factorize and solve in one call
///
/// For repeated solves with the same matrix, create a BandedLU object
/// and reuse it to avoid repeated factorization.
///
/// @tparam T Floating point type
/// @param n System size
/// @param lower_bw Lower bandwidth
/// @param upper_bw Upper bandwidth
/// @param lower_bands Input lower bands
/// @param diag Input main diagonal
/// @param upper_bands Input upper bands
/// @param rhs Right-hand side
/// @param solution Output solution
/// @param config Solver configuration
/// @return Result indicating success/failure
template<FloatingPoint T>
[[nodiscard]] inline BandedLUResult<T> solve_banded(
    size_t n,
    size_t lower_bw,
    size_t upper_bw,
    std::span<const std::span<const T>> lower_bands,
    std::span<const T> diag,
    std::span<const std::span<const T>> upper_bands,
    std::span<const T> rhs,
    std::span<T> solution,
    const BandedLUConfig<T>& config = {})
{
    BandedLU<T> solver(n, lower_bw, upper_bw);

    auto factor_result = solver.factorize(lower_bands, diag, upper_bands, config);
    if (!factor_result) {
        return factor_result;
    }

    return solver.solve(rhs, solution);
}

}  // namespace mango
