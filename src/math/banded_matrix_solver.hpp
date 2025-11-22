/**
 * @file banded_matrix_solver.hpp
 * @brief Banded matrix solver for narrow-bandwidth linear systems
 *
 * Provides compact storage and LAPACK-backed LU factorization for banded
 * matrices arising from cubic B-spline collocation and finite difference methods.
 *
 * Key features:
 * - Compact banded storage: O(bandwidth × n) vs O(n²) dense
 * - LAPACK DGBTRF/DGBTRS for numerical stability
 * - Condition number estimation via DGBCON
 * - Style matches thomas_solver.hpp (free functions, RAII workspace)
 */

#pragma once

#include <span>
#include <vector>
#include <concepts>
#include <optional>
#include <string_view>
#include <algorithm>
#include <limits>
#include <cassert>
#include <lapacke.h>

namespace mango {

/// Result type for banded matrix operations
template<std::floating_point T>
struct BandedResult {
    bool success;
    std::optional<std::string_view> error;

    /// Implicit conversion to bool for easy checking
    constexpr explicit operator bool() const noexcept { return success; }

    /// Check if operation succeeded
    [[nodiscard]] constexpr bool ok() const noexcept { return success; }

    /// Get error message (empty if successful)
    [[nodiscard]] constexpr std::string_view message() const noexcept {
        return error.value_or("");
    }

    /// Create success result
    [[nodiscard]] static constexpr BandedResult ok_result() noexcept {
        return BandedResult{.success = true, .error = std::nullopt};
    }

    /// Create error result
    [[nodiscard]] static constexpr BandedResult error_result(std::string_view msg) noexcept {
        return BandedResult{.success = false, .error = msg};
    }
};

/// Compact storage for banded matrices
///
/// For a matrix with bandwidth k (k non-zero entries per row):
/// - Memory: O(k × n) vs O(n²) for dense storage
/// - Access: A(row, col) requires col ∈ [col_start(row), col_start(row) + k)
///
/// Common use cases:
/// - Cubic B-spline collocation matrices (bandwidth = 4)
/// - Finite difference Jacobians (bandwidth = 3 for centered differences)
///
/// @tparam T Floating point type (float, double, long double)
template<std::floating_point T>
class BandedMatrix {
public:
    /// Construct banded matrix with fixed bandwidth
    ///
    /// @param n Matrix dimension (n × n)
    /// @param bandwidth Maximum non-zero entries per row
    explicit BandedMatrix(size_t n, size_t bandwidth)
        : n_(n)
        , bandwidth_(bandwidth)
        , band_values_(bandwidth * n, T{0})
        , col_start_(n, 0)
    {
        assert(bandwidth > 0 && "Bandwidth must be positive");
        assert(n > 0 && "Matrix dimension must be positive");
    }

    /// Access band entry A(row, col) for modification
    ///
    /// @pre col must be in [col_start(row), col_start(row) + bandwidth)
    [[nodiscard]] T& operator()(size_t row, size_t col) {
        assert(row < n_ && "Row index out of bounds");
        assert(col >= col_start_[row] && col < col_start_[row] + bandwidth_ &&
               "Column outside band storage");

        const size_t k = col - col_start_[row];
        return band_values_[row * bandwidth_ + k];
    }

    /// Access band entry A(row, col) for read-only
    [[nodiscard]] T operator()(size_t row, size_t col) const {
        assert(row < n_ && "Row index out of bounds");
        assert(col >= col_start_[row] && col < col_start_[row] + bandwidth_ &&
               "Column outside band storage");

        const size_t k = col - col_start_[row];
        return band_values_[row * bandwidth_ + k];
    }

    /// Get starting column index for row's band
    [[nodiscard]] size_t col_start(size_t row) const noexcept {
        assert(row < n_);
        return col_start_[row];
    }

    /// Set starting column index for row's band
    void set_col_start(size_t row, size_t col) noexcept {
        assert(row < n_);
        assert(col < n_);
        col_start_[row] = col;
    }

    /// Get matrix dimension
    [[nodiscard]] size_t size() const noexcept { return n_; }

    /// Get bandwidth
    [[nodiscard]] size_t bandwidth() const noexcept { return bandwidth_; }

    /// Get raw band values (for testing/debugging)
    [[nodiscard]] std::span<const T> band_values() const noexcept {
        return band_values_;
    }

    /// Get raw column starts (for testing/debugging)
    [[nodiscard]] std::span<const size_t> col_starts() const noexcept {
        return col_start_;
    }

private:
    size_t n_;                       ///< Matrix dimension
    size_t bandwidth_;               ///< Non-zero entries per row
    std::vector<T> band_values_;     ///< Compact band storage (bandwidth × n)
    std::vector<size_t> col_start_;  ///< Starting column for each row's band
};

/// RAII workspace for banded LU factorization
///
/// Manages LAPACK storage, pivot indices, and condition number estimation.
/// Reusable across multiple factorizations of same-sized matrices.
///
/// Example:
///   BandedLUWorkspace<double> ws(n, bandwidth);
///   auto result = factorize_banded(A, ws);
///   if (result.ok()) {
///       solve_banded(ws, b, x);
///   }
template<std::floating_point T>
class BandedLUWorkspace {
public:
    /// Construct workspace for n×n banded matrix
    ///
    /// @param n Matrix dimension
    /// @param bandwidth Maximum bandwidth (non-zero entries per row)
    explicit BandedLUWorkspace(size_t n, size_t bandwidth = 4)
        : n_(n)
        , bandwidth_(bandwidth)
        , kl_(0)
        , ku_(0)
        , ldab_(0)
        , factored_(false)
    {
        // Pre-allocate maximum possible LAPACK storage
        // Actual size determined during factorization
        const lapack_int max_ldab = static_cast<lapack_int>(2 * bandwidth + bandwidth + 1);
        lapack_storage_.reserve(static_cast<size_t>(max_ldab) * n);
        pivot_indices_.reserve(n);
    }

    /// Get matrix dimension
    [[nodiscard]] size_t size() const noexcept { return n_; }

    /// Check if factorization is valid
    [[nodiscard]] bool is_factored() const noexcept { return factored_; }

    /// Reset factorization state (for reuse with different matrix values)
    void reset() noexcept {
        factored_ = false;
        kl_ = 0;
        ku_ = 0;
        ldab_ = 0;
        lapack_storage_.clear();
        pivot_indices_.clear();
    }

    /// Resize workspace for different matrix size
    void resize(size_t new_n, size_t new_bandwidth = 4) {
        n_ = new_n;
        bandwidth_ = new_bandwidth;
        reset();

        const lapack_int max_ldab = static_cast<lapack_int>(2 * bandwidth_ + bandwidth_ + 1);
        lapack_storage_.reserve(static_cast<size_t>(max_ldab) * n_);
        pivot_indices_.reserve(n_);
    }

private:
    size_t n_;                          ///< Matrix dimension
    size_t bandwidth_;                  ///< Bandwidth hint
    lapack_int kl_;                     ///< Actual sub-diagonals (computed)
    lapack_int ku_;                     ///< Actual super-diagonals (computed)
    lapack_int ldab_;                   ///< Leading dimension for LAPACK storage
    bool factored_;                     ///< True if factorization computed

    std::vector<T> lapack_storage_;         ///< LAPACK band storage (column-major)
    std::vector<lapack_int> pivot_indices_; ///< Pivot indices from DGBTRF

    // Friend declarations for solver functions
    template<std::floating_point U>
    friend BandedResult<U> factorize_banded(
        const BandedMatrix<U>& A,
        BandedLUWorkspace<U>& workspace) noexcept;

    template<std::floating_point U>
    friend BandedResult<U> solve_banded(
        const BandedLUWorkspace<U>& workspace,
        std::span<const U> b,
        std::span<U> x) noexcept;

    template<std::floating_point U>
    friend U estimate_banded_condition(
        const BandedLUWorkspace<U>& workspace,
        U norm_A) noexcept;
};

/// Compute LU factorization of banded matrix
///
/// Uses LAPACKE_dgbtrf (band Gaussian elimination with partial pivoting).
/// Factorization is stored in workspace for subsequent solve operations.
///
/// Time:  O(bandwidth² × n)
/// Space: O(bandwidth × n) in workspace
///
/// @tparam T Floating point type (currently only double supported by LAPACKE)
/// @param A Banded matrix to factorize
/// @param workspace Workspace for LU factors (modified in-place)
/// @return Result indicating success/failure
template<std::floating_point T>
[[nodiscard]] BandedResult<T> factorize_banded(
    const BandedMatrix<T>& A,
    BandedLUWorkspace<T>& workspace) noexcept
{
    static_assert(std::same_as<T, double>,
                 "LAPACKE banded solvers currently only support double precision");

    using Result = BandedResult<T>;

    const lapack_int n = static_cast<lapack_int>(A.size());
    if (n == 0) {
        return Result::error_result("Matrix dimension must be > 0");
    }

    workspace.factored_ = false;

    // Determine actual sub- and super-diagonal counts
    lapack_int kl = 0;  // Sub-diagonals
    lapack_int ku = 0;  // Super-diagonals

    for (lapack_int i = 0; i < n; ++i) {
        const size_t col_start = A.col_start(static_cast<size_t>(i));
        for (size_t k = 0; k < A.bandwidth(); ++k) {
            const size_t col = col_start + k;
            if (col >= A.size()) continue;

            const lapack_int col_idx = static_cast<lapack_int>(col);
            kl = std::max(kl, i - col_idx);
            ku = std::max(ku, col_idx - i);
        }
    }

    workspace.kl_ = kl;
    workspace.ku_ = ku;

    // LAPACK band storage: leading dimension = 2*kl + ku + 1
    workspace.ldab_ = 2 * kl + ku + 1;
    workspace.lapack_storage_.assign(
        static_cast<size_t>(workspace.ldab_) * static_cast<size_t>(n), T{0});

    // Convert custom band format to LAPACK column-major band storage
    for (lapack_int i = 0; i < n; ++i) {
        const size_t col_start = A.col_start(static_cast<size_t>(i));
        for (size_t k = 0; k < A.bandwidth(); ++k) {
            const size_t col = col_start + k;
            if (col >= A.size()) continue;

            const T value = A.band_values()[static_cast<size_t>(i) * A.bandwidth() + k];
            const lapack_int col_idx = static_cast<lapack_int>(col);
            const lapack_int row_idx = kl + ku + i - col_idx;

            if (row_idx >= 0 && row_idx < workspace.ldab_) {
                const size_t storage_idx =
                    static_cast<size_t>(row_idx + col_idx * workspace.ldab_);
                workspace.lapack_storage_[storage_idx] = value;
            }
        }
    }

    // Perform LU factorization with partial pivoting
    workspace.pivot_indices_.resize(static_cast<size_t>(n));
    const lapack_int info = LAPACKE_dgbtrf(
        LAPACK_COL_MAJOR,
        n, n, kl, ku,
        workspace.lapack_storage_.data(),
        workspace.ldab_,
        workspace.pivot_indices_.data()
    );

    if (info < 0) {
        return Result::error_result("LAPACKE_dgbtrf: invalid argument");
    }
    if (info > 0) {
        return Result::error_result("Matrix is singular");
    }

    workspace.factored_ = true;
    return Result::ok_result();
}

/// Solve banded system LU·x = b using pre-computed factorization
///
/// Uses forward and back substitution (LAPACKE_dgbtrs) with cached LU factors.
/// Much faster than re-factorizing for multiple right-hand sides.
///
/// Time:  O(bandwidth × n)
/// Space: O(1) (operates in-place on x)
///
/// @tparam T Floating point type (currently only double)
/// @param workspace Workspace with LU factorization (from factorize_banded)
/// @param b Right-hand side vector
/// @param x Solution vector (output, can alias b for in-place solve)
/// @return Result indicating success/failure
template<std::floating_point T>
[[nodiscard]] BandedResult<T> solve_banded(
    const BandedLUWorkspace<T>& workspace,
    std::span<const T> b,
    std::span<T> x) noexcept
{
    using Result = BandedResult<T>;

    if (!workspace.factored_) {
        return Result::error_result("Matrix not factorized");
    }
    if (b.size() != workspace.n_ || x.size() != workspace.n_) {
        return Result::error_result("Dimension mismatch");
    }

    // Copy b into x (LAPACKE_dgbtrs solves in-place)
    std::copy(b.begin(), b.end(), x.begin());

    const lapack_int n = static_cast<lapack_int>(workspace.n_);
    const lapack_int nrhs = 1;
    const lapack_int info = LAPACKE_dgbtrs(
        LAPACK_COL_MAJOR,
        'N',  // No transpose
        n, workspace.kl_, workspace.ku_, nrhs,
        workspace.lapack_storage_.data(),
        workspace.ldab_,
        workspace.pivot_indices_.data(),
        x.data(),
        n
    );

    if (info < 0) {
        return Result::error_result("LAPACKE_dgbtrs: invalid argument");
    }
    if (info > 0) {
        return Result::error_result("LAPACKE_dgbtrs: zero pivot");
    }

    return Result::ok_result();
}

/// Estimate condition number of banded matrix using 1-norm
///
/// Uses LAPACKE_dgbcon to estimate κ₁(A) = ||A||₁ · ||A^{-1}||₁ from LU factors.
/// Much faster than computing true condition number via singular values.
///
/// @tparam T Floating point type
/// @param workspace Workspace with LU factorization
/// @param norm_A 1-norm of original matrix ||A||₁
/// @return Condition number estimate (∞ if ill-conditioned)
template<std::floating_point T>
[[nodiscard]] T estimate_banded_condition(
    const BandedLUWorkspace<T>& workspace,
    T norm_A) noexcept
{
    if (!workspace.factored_ || norm_A == T{0}) {
        return std::numeric_limits<T>::infinity();
    }

    double rcond = 0.0;  // Reciprocal condition number (output)
    const lapack_int n = static_cast<lapack_int>(workspace.n_);

    const lapack_int info = LAPACKE_dgbcon(
        LAPACK_COL_MAJOR,
        '1',  // 1-norm
        n, workspace.kl_, workspace.ku_,
        workspace.lapack_storage_.data(),
        workspace.ldab_,
        workspace.pivot_indices_.data(),
        norm_A,
        &rcond
    );

    if (info != 0 || rcond == 0.0) {
        return std::numeric_limits<T>::infinity();
    }

    return static_cast<T>(1.0 / rcond);
}

/// Compute 1-norm of banded matrix: ||A||₁ = max column sum
///
/// @tparam T Floating point type
/// @param A Banded matrix
/// @return 1-norm ||A||₁
template<std::floating_point T>
[[nodiscard]] T banded_norm1(const BandedMatrix<T>& A) noexcept {
    const size_t n = A.size();
    std::vector<T> col_sums(n, T{0});

    for (size_t i = 0; i < n; ++i) {
        const size_t col_start = A.col_start(i);
        const size_t col_end = std::min(col_start + A.bandwidth(), n);

        for (size_t j = col_start; j < col_end; ++j) {
            col_sums[j] += std::abs(A(i, j));
        }
    }

    return *std::max_element(col_sums.begin(), col_sums.end());
}

}  // namespace mango
