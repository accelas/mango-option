// SPDX-License-Identifier: MIT
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
 * via a pluggable banded solver backend for O(n) complexity instead of
 * O(n³). The backend (default: LAPACK) owns the factor storage layout,
 * the pivot representation, and every call into the underlying
 * linear-algebra library — this header contains no LAPACK calls.
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

#include "mango/math/banded_solver_backend.hpp"
#include "mango/math/lapack_banded_backend.hpp"
#include "mango/math/bspline/bspline_basis.hpp"
#include "mango/math/bspline/bspline_collocation_workspace.hpp"
#include "mango/support/error_types.hpp"
#include "mango/support/parallel.hpp"
#include <experimental/mdspan>
#include <expected>
#include <span>
#include <vector>
#include <concepts>
#include <algorithm>
#include <cmath>
#include <cstdint>

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

/// Result of a one-time banded LU factorization of the collocation matrix.
///
/// Produced by `BSplineCollocation1D::factorize()` and consumed by
/// `solve_factored()`. Default-constructible so callers can hold an
/// "empty"/malformed instance (e.g. to exercise error paths).
///
/// The representation of `lu` and `pivots` is backend-defined: `lu` holds
/// `Backend::factor_storage_size(n, bandwidth)` factor values and `pivots`
/// holds `Backend::pivot_storage_size(n, bandwidth)` pivots in the
/// backend's own pivot type.
template<std::floating_point T, typename Backend = LapackBandedBackend>
    requires BandedSolverBackend<Backend, T>
struct BSplineCollocationFactorization {
    std::vector<T> lu;                                 ///< Backend factor storage
    std::vector<typename Backend::pivot_type> pivots;  ///< Backend pivot storage
    T condition_estimate{};  ///< 1-norm condition estimate (inf on failure)
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
/// @tparam Bandwidth Reserved; must be 4 (cubic) until basis evaluation is
///     generalized
/// @tparam Backend Banded solver backend policy (default: LAPACK)
template<std::floating_point T, std::size_t Bandwidth = 4,
         typename Backend = LapackBandedBackend>
    requires BandedSolverBackend<Backend, T>
class BSplineCollocation1D {
public:
    /// Bandwidth for B-splines (reserved; only 4 = cubic is supported)
    static constexpr size_t BANDWIDTH = Bandwidth;

    // Reserved parameter: only cubic (Bandwidth == 4) is supported —
    // cubic_basis_nonuniform unconditionally writes 4 entries, so any
    // smaller Bandwidth is a stack buffer overflow (issue #441 item 9).
    static_assert(Bandwidth == 4,
                  "BSplineCollocation1D supports only Bandwidth == 4 (cubic)");

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
        const BSplineCollocationConfig<T>& config = {}) const
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

        // Pack the collocation matrix into backend factor storage and
        // LU-factorize it in place.
        std::vector<T> factors(Backend::factor_storage_size(n_, BANDWIDTH));
        std::vector<typename Backend::pivot_type> pivots(
            Backend::pivot_storage_size(n_, BANDWIDTH));
        Backend::pack(std::span<const T>{band_values_},
                      std::span<const int>{band_col_start_},
                      n_, BANDWIDTH, std::span<T>{factors});
        if (!Backend::factorize(std::span<T>{factors},
                                std::span<typename Backend::pivot_type>{pivots},
                                n_, BANDWIDTH).ok()) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::FittingFailed,
                n_});
        }

        // Solve for coefficients (in-place on a copy of the RHS)
        std::vector<T> coeffs(n_);
        std::copy(values.begin(), values.end(), coeffs.begin());
        if (!Backend::solve(std::span<const T>{factors},
                            std::span<const typename Backend::pivot_type>{pivots},
                            std::span<T>{coeffs}, n_, BANDWIDTH).ok()) {
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
        const T cond_est = Backend::condition(
            std::span<const T>{factors},
            std::span<const typename Backend::pivot_type>{pivots},
            norm_A, n_, BANDWIDTH);

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
        const BSplineCollocationConfig<T>& config = {}) const
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

        // Pack the collocation matrix into backend factor storage and
        // LU-factorize it in place.
        std::vector<T> factors(Backend::factor_storage_size(n_, BANDWIDTH));
        std::vector<typename Backend::pivot_type> pivots(
            Backend::pivot_storage_size(n_, BANDWIDTH));
        Backend::pack(std::span<const T>{band_values_},
                      std::span<const int>{band_col_start_},
                      n_, BANDWIDTH, std::span<T>{factors});
        if (!Backend::factorize(std::span<T>{factors},
                                std::span<typename Backend::pivot_type>{pivots},
                                n_, BANDWIDTH).ok()) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::FittingFailed,
                n_});
        }

        // Solve in place on coeffs_out (copy of the RHS)
        std::copy(values.begin(), values.end(), coeffs_out.begin());
        if (!Backend::solve(std::span<const T>{factors},
                            std::span<const typename Backend::pivot_type>{pivots},
                            coeffs_out, n_, BANDWIDTH).ok()) {
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
        const T cond_est = Backend::condition(
            std::span<const T>{factors},
            std::span<const typename Backend::pivot_type>{pivots},
            norm_A, n_, BANDWIDTH);

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
    /// Only available with the LAPACK backend: the workspace's byte
    /// layout (LDAB band storage, lapack_int pivot region) is a contract
    /// with LAPACK's banded format.
    ///
    /// @param values Function values at grid points (size n_)
    /// @param ws Pre-allocated workspace (must have size() == n_)
    /// @param config Solver configuration
    /// @return Fit result (coefficients are in ws.coeffs(), not in result)
    [[nodiscard]] std::expected<BSplineCollocationResult<T>, InterpolationError>
    fit_with_workspace(
        std::span<const T> values,
        BSplineCollocationWorkspace<T, BANDWIDTH>& ws,
        const BSplineCollocationConfig<T>& config = {}) const
        requires std::same_as<Backend, LapackBandedBackend>
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

        // Copy band_storage to lapack_storage (factorization is in-place)
        auto band_storage = ws.band_storage();
        auto lapack_storage = ws.lapack_storage();
        std::copy(band_storage.begin(), band_storage.end(), lapack_storage.begin());

        // Factorize using workspace lapack_storage and pivots
        if (!Backend::factorize(lapack_storage, ws.pivots(), n_, BANDWIDTH).ok()) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::FittingFailed, n_});
        }

        // Solve into ws.coeffs() (in-place on a copy of the RHS)
        auto coeffs = ws.coeffs();
        std::copy(values.begin(), values.end(), coeffs.begin());
        if (!Backend::solve(lapack_storage, ws.pivots(), coeffs, n_, BANDWIDTH).ok()) {
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
        const T cond_est = Backend::condition(
            lapack_storage, ws.pivots(), norm_A, n_, BANDWIDTH);

        return BSplineCollocationResult<T>{
            .coefficients = {},  // Caller uses ws.coeffs()
            .max_residual = max_residual,
            .condition_estimate = cond_est
        };
    }

    /// Factorize the collocation matrix once (const, race-free).
    ///
    /// The collocation matrix depends only on the grid, which is fixed at
    /// construction, so a single factorization may be shared read-only
    /// across concurrent `solve_factored()` calls on this instance (e.g.
    /// one per OpenMP thread solving a different right-hand side/slice).
    /// This method itself allocates a private LU buffer and touches no
    /// solver state, so concurrent calls to `factorize()` are also safe.
    ///
    /// @return Factorization (LU factors + pivots + condition estimate) or
    ///     error if the matrix is singular
    [[nodiscard]] std::expected<BSplineCollocationFactorization<T, Backend>, InterpolationError>
    factorize() const
    {
        BSplineCollocationFactorization<T, Backend> fact;
        fact.lu.assign(Backend::factor_storage_size(n_, BANDWIDTH), T{0});
        fact.pivots.resize(Backend::pivot_storage_size(n_, BANDWIDTH));
        Backend::pack(std::span<const T>{band_values_},
                      std::span<const int>{band_col_start_},
                      n_, BANDWIDTH, std::span<T>{fact.lu});

        if (!Backend::factorize(std::span<T>{fact.lu},
                                std::span<typename Backend::pivot_type>{fact.pivots},
                                n_, BANDWIDTH).ok()) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::FittingFailed, n_});
        }

        const T norm_A = compute_matrix_norm1();
        fact.condition_estimate = Backend::condition(
            std::span<const T>{fact.lu},
            std::span<const typename Backend::pivot_type>{fact.pivots},
            norm_A, n_, BANDWIDTH);
        return fact;
    }

    /// Solve the collocation system using a pre-computed factorization.
    ///
    /// Thread-safety: `fact` is read-only here and `values`/`coeffs_out`
    /// are caller-owned per-call buffers, so concurrent calls on the same
    /// `BSplineCollocation1D` instance (and even the same `fact`) with
    /// distinct `values`/`coeffs_out` buffers are race-free. It is the
    /// caller's responsibility to pass a `fact` produced by a
    /// `factorize()` call on *this* solver (or another solver built from
    /// the same grid) — a factorization from a differently-sized or
    /// differently-gridded solver is not detected beyond a size check.
    ///
    /// @param fact Factorization from `factorize()`
    /// @param values Function values at grid points (size n_)
    /// @param coeffs_out Pre-allocated buffer for coefficients (size n_);
    ///     must not overlap `values` — the residual check needs the
    ///     original RHS after the in-place solve, so aliasing buffers are
    ///     rejected with BufferSizeMismatch
    /// @param config Solver configuration
    /// @return Max residual or error
    [[nodiscard]] std::expected<T, InterpolationError> solve_factored(
        const BSplineCollocationFactorization<T, Backend>& fact,
        std::span<const T> values, std::span<T> coeffs_out,
        const BSplineCollocationConfig<T>& config = {}) const
    {
        if (values.size() != n_) return std::unexpected(InterpolationError{
            InterpolationErrorCode::ValueSizeMismatch, values.size()});
        if (coeffs_out.size() != n_) return std::unexpected(InterpolationError{
            InterpolationErrorCode::BufferSizeMismatch, coeffs_out.size()});
        // Compare as uintptr_t for a total order: raw < on pointers into
        // unrelated allocations is unspecified in C++.
        const auto v_lo = reinterpret_cast<std::uintptr_t>(values.data());
        const auto v_hi = reinterpret_cast<std::uintptr_t>(values.data() + values.size());
        const auto c_lo = reinterpret_cast<std::uintptr_t>(coeffs_out.data());
        const auto c_hi = reinterpret_cast<std::uintptr_t>(coeffs_out.data() + coeffs_out.size());
        if (v_lo < c_hi && c_lo < v_hi) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::BufferSizeMismatch, coeffs_out.size()});
        }
        if (fact.lu.size() != Backend::factor_storage_size(n_, BANDWIDTH)) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::BufferSizeMismatch, fact.lu.size()});
        }
        if (fact.pivots.size() != Backend::pivot_storage_size(n_, BANDWIDTH)) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::BufferSizeMismatch, fact.pivots.size()});
        }
        for (size_t i = 0; i < n_; ++i) {
            if (std::isnan(values[i])) return std::unexpected(InterpolationError{
                InterpolationErrorCode::NaNInput, n_, i});
            if (std::isinf(values[i])) return std::unexpected(InterpolationError{
                InterpolationErrorCode::InfInput, n_, i});
        }
        std::copy(values.begin(), values.end(), coeffs_out.begin());
        if (!Backend::solve(std::span<const T>{fact.lu},
                            std::span<const typename Backend::pivot_type>{fact.pivots},
                            coeffs_out, n_, BANDWIDTH).ok()) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::FittingFailed, n_});
        }
        const T max_residual = compute_residual_from_span(coeffs_out, values);
        if (max_residual > config.tolerance) return std::unexpected(InterpolationError{
            InterpolationErrorCode::FittingFailed, n_, 0,
            static_cast<double>(max_residual)});
        return max_residual;
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

        // Build the collocation matrix once; it depends only on the grid.
        build_collocation_matrix();
    }

    std::vector<T> grid_;               ///< Data grid points
    std::vector<T> knots_;              ///< Knot vector (clamped)
    size_t n_;                          ///< Number of grid points

    // Banded storage: each row has exactly 4 non-zero entries.
    // Immutable after construction (issue #435) — built once in the
    // constructor and never rewritten, so concurrent fit() calls on a
    // shared instance never race on this state.
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
    /// Packs the internal band representation into the workspace's
    /// LAPACK-banded band_storage via the backend.
    void build_collocation_matrix_to_workspace(BSplineCollocationWorkspace<T, BANDWIDTH>& ws) const {
        Backend::pack(std::span<const T>{band_values_},
                      std::span<const int>{band_col_start_},
                      n_, BANDWIDTH, ws.band_storage());
    }
};

}  // namespace mango
