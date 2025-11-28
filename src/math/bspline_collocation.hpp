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
#include "src/math/bspline_collocation_workspace.hpp"
#include "src/support/error_types.hpp"
#include "src/support/parallel.hpp"
#include <experimental/mdspan>
#include <expected>
#include <span>
#include <vector>
#include <concepts>
#include <optional>
#include <algorithm>
#include <cmath>
#include <limits>
#include <lapacke.h>

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
/// @tparam Bandwidth Number of non-zero entries per row (degree + 1)
template<std::floating_point T, size_t Bandwidth = 4>
class BSplineCollocation1D {
public:
    /// Bandwidth for B-splines (degree + 1, default 4 for cubic)
    static constexpr size_t BANDWIDTH = Bandwidth;

    /// mdspan type for internal band storage (n × Bandwidth, row-major)
    using band_extents_type = std::experimental::extents<size_t, std::dynamic_extent, BANDWIDTH>;
    using band_view_type = std::experimental::mdspan<T, band_extents_type>;
    using const_band_view_type = std::experimental::mdspan<const T, band_extents_type>;

    /// Factory method to create BSplineCollocation1D instance
    ///
    /// @param grid Data grid points (sorted, ≥4 points)
    /// @return Solver instance or error
    [[nodiscard]] static std::expected<BSplineCollocation1D, InterpolationError> create(
        std::vector<T> grid)
    {
        // Validate grid size
        if (grid.size() < 4) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::InsufficientGridPoints,
                grid.size()});
        }

        // Validate grid is sorted
        if (!std::is_sorted(grid.begin(), grid.end())) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::GridNotSorted,
                grid.size()});
        }

        // Check for near-duplicate points
        constexpr T MIN_SPACING = T{1e-14};
        for (size_t i = 1; i < grid.size(); ++i) {
            const T spacing = grid[i] - grid[i-1];
            if (spacing < MIN_SPACING) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::GridNotSorted,
                    grid.size(),
                    i});
            }
        }

        // Check for zero-width grid
        if (grid.back() - grid.front() < MIN_SPACING) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::ZeroWidthGrid,
                grid.size()});
        }

        // All validations passed
        return BSplineCollocation1D(std::move(grid));
    }

    /// Fit B-spline coefficients via collocation
    ///
    /// Solves B·c = f where B is the collocation matrix.
    /// Returns fitted coefficients or error.
    ///
    /// @param values Function values at grid points (size n)
    /// @param config Solver configuration
    /// @return Fit result with coefficients and diagnostics
    [[nodiscard]] std::expected<BSplineCollocationResult<T>, InterpolationError> fit(
        const std::vector<T>& values,
        const BSplineCollocationConfig<T>& config = {})
    {
        if (values.size() != n_) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::ValueSizeMismatch,
                values.size()});
        }

        // Validate input values for NaN/Inf
        for (size_t i = 0; i < n_; ++i) {
            if (std::isnan(values[i])) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::NaNInput,
                    n_,
                    i});
            }
            if (std::isinf(values[i])) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::InfInput,
                    n_,
                    i});
            }
        }

        // Build collocation matrix
        build_collocation_matrix();

        // Create banded matrix with bandwidth=4 for cubic B-splines
        BandedMatrix<T> A(n_, 4);
        for (size_t i = 0; i < n_; ++i) {
            // Set the starting column for this row's band
            A.set_col_start(i, static_cast<size_t>(band_col_start_[i]));

            // Fill the 4 non-zero entries for this row
            for (size_t k = 0; k < 4; ++k) {
                const int col = band_col_start_[i] + static_cast<int>(k);
                if (col >= 0 && col < static_cast<int>(n_)) {
                    A(i, static_cast<size_t>(col)) = band_values_[i * 4 + k];
                }
            }
        }

        // Factorize banded system
        BandedLUWorkspace<T> workspace(n_, 4);
        auto factor_result = factorize_banded(A, workspace);
        if (!factor_result.ok()) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::FittingFailed,
                n_});
        }

        // Solve for coefficients
        std::vector<T> coeffs(n_);
        auto solve_result = solve_banded(workspace, std::span<const T>(values), std::span<T>(coeffs));
        if (!solve_result.ok()) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::FittingFailed,
                n_});
        }

        // Compute residuals: ||B·c - f||
        const T max_residual = compute_residual(coeffs, values);

        // Check residual tolerance
        if (max_residual > config.tolerance) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::FittingFailed,
                n_,
                0,
                static_cast<double>(max_residual)});
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
    [[nodiscard]] std::expected<BSplineCollocationResult<T>, InterpolationError> fit_with_buffer(
        std::span<const T> values,
        std::span<T> coeffs_out,
        const BSplineCollocationConfig<T>& config = {})
    {
        if (values.size() != n_) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::ValueSizeMismatch,
                values.size()});
        }
        if (coeffs_out.size() != n_) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::BufferSizeMismatch,
                coeffs_out.size()});
        }

        // Validate input values
        for (size_t i = 0; i < n_; ++i) {
            if (std::isnan(values[i])) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::NaNInput,
                    n_,
                    i});
            }
            if (std::isinf(values[i])) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::InfInput,
                    n_,
                    i});
            }
        }

        // Build collocation matrix
        build_collocation_matrix();

        // Create banded matrix with bandwidth=4 for cubic B-splines
        BandedMatrix<T> A(n_, 4);
        for (size_t i = 0; i < n_; ++i) {
            // Set the starting column for this row's band
            A.set_col_start(i, static_cast<size_t>(band_col_start_[i]));

            // Fill the 4 non-zero entries for this row
            for (size_t k = 0; k < 4; ++k) {
                const int col = band_col_start_[i] + static_cast<int>(k);
                if (col >= 0 && col < static_cast<int>(n_)) {
                    A(i, static_cast<size_t>(col)) = band_values_[i * 4 + k];
                }
            }
        }

        // Factorize and solve
        BandedLUWorkspace<T> workspace(n_, 4);
        auto factor_result = factorize_banded(A, workspace);
        if (!factor_result.ok()) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::FittingFailed,
                n_});
        }

        auto solve_result = solve_banded(workspace, values, coeffs_out);
        if (!solve_result.ok()) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::FittingFailed,
                n_});
        }

        // Compute residuals
        const T max_residual = compute_residual_from_span(coeffs_out, values);

        if (max_residual > config.tolerance) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::FittingFailed,
                n_,
                0,
                static_cast<double>(max_residual)});
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

    /// Fit with external workspace (zero-allocation variant)
    ///
    /// Uses BSplineCollocationWorkspace for all temporary storage.
    /// Coefficients are written to ws.coeffs().
    ///
    /// @param values Function values at grid points (size n_)
    /// @param ws Pre-allocated workspace (must have size() == n_)
    /// @param config Solver configuration
    /// @return Fit result (coefficients are in ws.coeffs(), not in result)
    [[nodiscard]] std::expected<BSplineCollocationResult<T>, InterpolationError>
    fit_with_workspace(
        std::span<const T> values,
        BSplineCollocationWorkspace<T>& ws,
        const BSplineCollocationConfig<T>& config = {})
    {
        if (values.size() != n_) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::ValueSizeMismatch,
                values.size()});
        }
        if (ws.size() != n_) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::BufferSizeMismatch,
                ws.size()});
        }

        // Validate input values
        for (size_t i = 0; i < n_; ++i) {
            if (std::isnan(values[i])) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::NaNInput, n_, i});
            }
            if (std::isinf(values[i])) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::InfInput, n_, i});
            }
        }

        // Build collocation matrix into workspace band_storage
        build_collocation_matrix_to_workspace(ws);

        // Factorize using workspace lapack_storage and pivots
        auto factor_result = factorize_banded_workspace(ws);
        if (!factor_result.ok()) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::FittingFailed, n_});
        }

        // Solve into ws.coeffs()
        auto solve_result = solve_banded_workspace(ws, values);
        if (!solve_result.ok()) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::FittingFailed, n_});
        }

        // Compute residuals
        const T max_residual = compute_residual_from_span(ws.coeffs(), values);

        if (max_residual > config.tolerance) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::FittingFailed, n_, 0,
                static_cast<double>(max_residual)});
        }

        // Estimate condition number
        const T norm_A = compute_matrix_norm1();
        const T cond_est = estimate_banded_condition_workspace(ws, norm_A);

        return BSplineCollocationResult<T>{
            .coefficients = {},  // Caller uses ws.coeffs()
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

        // Pre-allocate banded storage (BANDWIDTH entries per row)
        band_values_.resize(n_ * BANDWIDTH, T{0});
        band_col_start_.resize(n_, 0);
    }

    std::vector<T> grid_;               ///< Data grid points
    std::vector<T> knots_;              ///< Knot vector (clamped)
    size_t n_;                          ///< Number of grid points

    // Banded storage: each row has exactly 4 non-zero entries
    std::vector<T> band_values_;        ///< Banded matrix values (n×4, row-major)
    std::vector<int> band_col_start_;   ///< First column index for each row's band

    /// Get mdspan view of band_values_ for clean 2D indexing
    ///
    /// Returns view with extents (n_, BANDWIDTH) for band_[i, k] access
    [[nodiscard]] band_view_type band_view() noexcept {
        return band_view_type(band_values_.data(), n_);
    }

    [[nodiscard]] const_band_view_type band_view() const noexcept {
        return const_band_view_type(band_values_.data(), n_);
    }

    /// Build collocation matrix B[i,j] = N_j(x_i) in banded format
    void build_collocation_matrix() {
        auto band = band_view();

        for (size_t i = 0; i < n_; ++i) {
            const T x = grid_[i];

            // Find knot span
            const int span = find_span_cubic(knots_, x);

            // Evaluate BANDWIDTH non-zero basis functions at x
            T basis[BANDWIDTH];
            cubic_basis_nonuniform(knots_, span, x, basis);

            // Store in banded format
            band_col_start_[i] = std::max(0, span - static_cast<int>(BANDWIDTH - 1));

            // Fill band values (left to right order)
            for (size_t k = 0; k < BANDWIDTH; ++k) {
                const int col = span - static_cast<int>(k);
                if (col >= 0 && col < static_cast<int>(n_)) {
                    const int band_idx = col - band_col_start_[i];
                    if (band_idx >= 0 && band_idx < static_cast<int>(BANDWIDTH)) {
                        band[i, static_cast<size_t>(band_idx)] = basis[k];
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
        auto band = band_view();
        T max_res = T{0};

        for (size_t i = 0; i < n_; ++i) {
            // Compute (B·c)[i] - only sum over BANDWIDTH non-zero entries
            T Bc_i = T{0};
            const int j_start = band_col_start_[i];
            const int j_end = std::min(j_start + static_cast<int>(BANDWIDTH), static_cast<int>(n_));

            for (int j = j_start; j < j_end; ++j) {
                const size_t band_idx = static_cast<size_t>(j - j_start);
                const T b_ij = band[i, band_idx];
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
        auto band = band_view();
        T max_residual = T{0};

        for (size_t i = 0; i < n_; ++i) {
            T Bc_i = T{0};
            const int col_start = band_col_start_[i];

            for (size_t k = 0; k < BANDWIDTH &&
                 (col_start + static_cast<int>(k)) < static_cast<int>(n_); ++k)
            {
                Bc_i = std::fma(band[i, k], coeffs[col_start + k], Bc_i);
            }

            const T residual = std::abs(Bc_i - values[i]);
            max_residual = std::max(max_residual, residual);
        }

        return max_residual;
    }

    /// Compute 1-norm of collocation matrix ||B||₁ = max column sum
    [[nodiscard]] T compute_matrix_norm1() const {
        auto band = band_view();
        std::vector<T> col_sums(n_, T{0});

        for (size_t i = 0; i < n_; ++i) {
            const int j_start = band_col_start_[i];
            const int j_end = std::min(j_start + static_cast<int>(BANDWIDTH), static_cast<int>(n_));

            for (int j = j_start; j < j_end; ++j) {
                const size_t band_idx = static_cast<size_t>(j - j_start);
                const T b_ij = band[i, band_idx];
                col_sums[j] += std::abs(b_ij);
            }
        }

        return *std::max_element(col_sums.begin(), col_sums.end());
    }

    /// Build collocation matrix into workspace band storage
    ///
    /// Writes matrix directly to workspace in LAPACK banded format.
    /// For bandwidth=4 cubic B-splines, LAPACK uses ldab=10 storage.
    void build_collocation_matrix_to_workspace(BSplineCollocationWorkspace<T>& ws) {
        // First build into internal storage (same as regular method)
        build_collocation_matrix();

        // Zero the workspace band storage
        auto ws_band_storage = ws.band_storage();
        std::fill(ws_band_storage.begin(), ws_band_storage.end(), T{0});

        // Get mdspan views for clean indexing
        auto internal_band = band_view();
        auto ws_band = ws.band_view();

        // Copy from internal format to LAPACK banded format via mdspan
        for (size_t i = 0; i < n_; ++i) {
            const int col_start = band_col_start_[i];
            const int col_end = std::min(col_start + static_cast<int>(BANDWIDTH), static_cast<int>(n_));

            for (int j = col_start; j < col_end; ++j) {
                const size_t band_idx = static_cast<size_t>(j - col_start);
                const T value = internal_band[i, band_idx];

                // ws_band handles LAPACK banded format offset calculation
                ws_band[i, static_cast<size_t>(j)] = value;
            }
        }
    }

    /// Factorize banded matrix using workspace storage
    ///
    /// Performs LU factorization using LAPACK dgbtrf directly on workspace.
    [[nodiscard]] BandedResult<T> factorize_banded_workspace(BSplineCollocationWorkspace<T>& ws) {
        static_assert(std::same_as<T, double>,
                     "LAPACKE banded solvers currently only support double precision");

        using Result = BandedResult<T>;
        using Workspace = BSplineCollocationWorkspace<T>;

        const lapack_int n = static_cast<lapack_int>(n_);
        const lapack_int kl = Workspace::KL;
        const lapack_int ku = Workspace::KU;
        const lapack_int ldab = static_cast<lapack_int>(Workspace::LDAB);

        // Copy band_storage to lapack_storage (dgbtrf modifies in-place)
        auto band_storage = ws.band_storage();
        auto lapack_storage = ws.lapack_storage();
        std::copy(band_storage.begin(), band_storage.end(), lapack_storage.begin());

        // Perform LU factorization
        const lapack_int info = LAPACKE_dgbtrf(
            LAPACK_COL_MAJOR,
            n, n, kl, ku,
            lapack_storage.data(),
            ldab,
            ws.pivots().data()
        );

        if (info < 0) {
            return Result::error_result("LAPACKE_dgbtrf: invalid argument");
        }
        if (info > 0) {
            return Result::error_result("Matrix is singular");
        }

        return Result::ok_result();
    }

    /// Solve banded system using workspace storage
    ///
    /// Solves LU·x = b using pre-computed factorization in workspace.
    [[nodiscard]] BandedResult<T> solve_banded_workspace(
        BSplineCollocationWorkspace<T>& ws,
        std::span<const T> b)
    {
        using Result = BandedResult<T>;
        using Workspace = BSplineCollocationWorkspace<T>;

        if (b.size() != n_) {
            return Result::error_result("Dimension mismatch");
        }

        const lapack_int n = static_cast<lapack_int>(n_);
        const lapack_int kl = Workspace::KL;
        const lapack_int ku = Workspace::KU;
        const lapack_int ldab = static_cast<lapack_int>(Workspace::LDAB);
        const lapack_int nrhs = 1;

        // Copy b into ws.coeffs() (dgbtrs solves in-place)
        auto coeffs = ws.coeffs();
        std::copy(b.begin(), b.end(), coeffs.begin());

        // Solve using LU factors in lapack_storage
        const lapack_int info = LAPACKE_dgbtrs(
            LAPACK_COL_MAJOR,
            'N',  // No transpose
            n, kl, ku, nrhs,
            ws.lapack_storage().data(),
            ldab,
            ws.pivots().data(),
            coeffs.data(),
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

    /// Estimate condition number using workspace storage
    ///
    /// Uses LAPACK dgbcon to estimate condition number from LU factors.
    [[nodiscard]] T estimate_banded_condition_workspace(
        const BSplineCollocationWorkspace<T>& ws,
        T norm_A) const
    {
        using Workspace = BSplineCollocationWorkspace<T>;

        if (norm_A == T{0}) {
            return std::numeric_limits<T>::infinity();
        }

        const lapack_int n = static_cast<lapack_int>(n_);
        const lapack_int kl = Workspace::KL;
        const lapack_int ku = Workspace::KU;
        const lapack_int ldab = static_cast<lapack_int>(Workspace::LDAB);

        double rcond = 0.0;  // Reciprocal condition number (output)

        const lapack_int info = LAPACKE_dgbcon(
            LAPACK_COL_MAJOR,
            '1',  // 1-norm
            n, kl, ku,
            ws.lapack_storage().data(),
            ldab,
            ws.pivots().data(),
            norm_A,
            &rcond
        );

        if (info != 0 || rcond == 0.0) {
            return std::numeric_limits<T>::infinity();
        }

        return static_cast<T>(1.0 / rcond);
    }
};

}  // namespace mango
