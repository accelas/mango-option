// SPDX-License-Identifier: MIT
/**
 * @file bspline_nd_separable.hpp
 * @brief Generic N-dimensional separable B-spline collocation fitting
 *
 * Fits B-spline coefficients for N-dimensional tensor-product spaces using
 * separable collocation. Exploits tensor-product structure to avoid solving
 * massive O(n^N) dense systems, instead solving sequential 1D systems.
 *
 * **Algorithm:**
 * 1. Fit axis N-1 → solve (N-1)-dimensional slices in parallel
 * 2. Fit axis N-2 → solve (N-2)-dimensional slices in parallel
 * 3. Continue recursively down to axis 0
 * 4. Cache-optimal ordering: fastest-varying dimension processed first
 *
 * **Complexity:**
 * - Time:  O(N₀ + N₁ + ... + Nₙ₋₁) with banded solver per axis
 * - Space: O(N₀ · N₁ · ... · Nₙ₋₁) for coefficient storage
 *
 * **Performance vs hardcoded 4D:**
 * - Identical speed (verified by benchmarks)
 * - 75% less code (generic template vs 4 hardcoded axis methods)
 * - Works for any N (3D, 4D, 5D, 6D, ...)
 *
 * Style matches bspline_collocation.hpp for consistency.
 */

#pragma once

#include "src/math/bspline_collocation.hpp"
#include "src/math/bspline_collocation_workspace.hpp"
#include "src/support/parallel.hpp"
#include "src/support/thread_workspace.hpp"
#include "src/math/safe_math.hpp"
#include <experimental/mdspan>
#include <expected>
#include <span>
#include <vector>
#include <array>
#include <concepts>
#include <string>
#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <numeric>

namespace mango {

/// B-spline fitting diagnostics (per-axis and aggregate)
///
/// Contains per-axis metrics and computed aggregates for monitoring fit quality.
/// @tparam T Floating point type
/// @tparam N Number of dimensions
template<std::floating_point T, size_t N>
struct BSplineFittingStats {
    std::array<T, N> max_residual_per_axis{};   ///< Max residual for each axis
    T max_residual_overall = T{0};               ///< Max across all axes

    std::array<T, N> condition_per_axis{};       ///< Condition estimate for each axis
    T condition_max = T{0};                      ///< Max condition across all axes

    std::array<size_t, N> failed_slices_per_axis{};  ///< Failed 1D fits per axis
    size_t failed_slices_total = 0;              ///< Total failed slices
};

/// Successful result of N-dimensional separable B-spline fitting
///
/// Contains fitted coefficients and per-axis diagnostics.
template<std::floating_point T, size_t N>
struct BSplineNDSeparableResult {
    std::vector<T> coefficients;              ///< Fitted coefficients (N₀×N₁×...×Nₙ₋₁)
    std::array<T, N> max_residual_per_axis;    ///< Max residual for each axis
    std::array<T, N> condition_per_axis;       ///< Condition estimate for each axis
    std::array<size_t, N> failed_slices;       ///< Failed 1D fits per axis (all zeros on success)

    /// Convert to BSplineFittingStats with computed aggregates
    BSplineFittingStats<T, N> to_stats() const {
        BSplineFittingStats<T, N> stats;
        stats.max_residual_per_axis = max_residual_per_axis;
        stats.max_residual_overall = *std::max_element(
            max_residual_per_axis.begin(), max_residual_per_axis.end());
        stats.condition_per_axis = condition_per_axis;
        stats.condition_max = *std::max_element(
            condition_per_axis.begin(), condition_per_axis.end());
        stats.failed_slices_per_axis = failed_slices;
        stats.failed_slices_total = std::accumulate(
            failed_slices.begin(), failed_slices.end(), size_t{0});
        return stats;
    }
};

/// Configuration for N-dimensional separable fitting
template<std::floating_point T>
struct BSplineNDSeparableConfig {
    T tolerance = T{1e-6};  ///< Maximum residual per axis
};

/// N-dimensional separable B-spline collocation fitter
///
/// Fits B-spline coefficients via sequential 1D collocation along each axis.
/// Uses template recursion and if constexpr for zero-overhead compile-time
/// specialization.
///
/// **Usage:**
/// ```cpp
/// std::array<std::vector<double>, 4> grids = {m_grid, tau_grid, sigma_grid, r_grid};
/// auto fitter = BSplineNDSeparable<double, 4>::create(std::move(grids)).value();
/// auto result = fitter.fit(values);
/// if (result) {
///     // Use result.coefficients
/// }
/// ```
///
/// @tparam T Floating point type (float, double, long double)
/// @tparam N Number of dimensions (must be ≥1)
template<std::floating_point T, size_t N>
    requires (N >= 1)
class BSplineNDSeparable {
public:
    using Result = std::expected<BSplineNDSeparableResult<T, N>, InterpolationError>;
    using Config = BSplineNDSeparableConfig<T>;

    /// Factory method to create N-dimensional fitter with validation
    ///
    /// @param grids Array of N grid vectors (each ≥4 points, sorted)
    /// @return Fitter instance or error
    [[nodiscard]] static std::expected<BSplineNDSeparable, InterpolationError> create(
        std::array<std::vector<T>, N> grids)
    {
        // Validate each grid via 1D solver creation
        for (size_t i = 0; i < N; ++i) {
            auto solver_result = BSplineCollocation1D<T>::create(grids[i]);
            if (!solver_result.has_value()) {
                // Propagate error with axis index
                auto err = solver_result.error();
                err.index = i;
                return std::unexpected(err);
            }
        }

        // All grids valid, create fitter
        return BSplineNDSeparable(std::move(grids));
    }

    /// Fit B-spline coefficients via separable collocation (lvalue overload)
    ///
    /// Processes axes in reverse order (N-1 → 0) for cache locality.
    /// Each axis performs 1D fits on slices perpendicular to that dimension.
    ///
    /// @param values Function values at grid points (row-major layout)
    /// @param config Solver configuration
    /// @return Fit result with coefficients and per-axis diagnostics
    [[nodiscard]] Result fit(
        const std::vector<T>& values,
        const Config& config = {})
    {
        // Delegate to rvalue version with a copy
        std::vector<T> values_copy = values;
        return fit(std::move(values_copy), config);
    }

    /// Fit B-spline coefficients via separable collocation (rvalue overload, zero-copy)
    ///
    /// Move-optimized version that works directly on the input array.
    /// Use when you don't need the input values after fitting.
    ///
    /// @param values Function values at grid points (moved in, modified in-place)
    /// @param config Solver configuration
    /// @return Fit result with coefficients and per-axis diagnostics
    [[nodiscard]] Result fit(
        std::vector<T>&& values,
        const Config& config = {})
    {
        // Verify size with overflow check
        size_t expected_size = 1;
        for (size_t i = 0; i < grids_.size(); ++i) {
            auto result = safe_multiply(expected_size, grids_[i].size());
            if (!result.has_value()) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::ValueSizeMismatch,
                    grids_[i].size(),
                    i});
            }
            expected_size = result.value();
        }

        if (values.size() != expected_size) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::ValueSizeMismatch,
                values.size()});
        }

        // Validate input values for NaN/Inf
        // Note: Can't use SIMD with early return, so check sequentially
        for (size_t i = 0; i < values.size(); ++i) {
            if (std::isnan(values[i])) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::NaNInput,
                    expected_size,
                    i});
            }
            if (std::isinf(values[i])) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::InfInput,
                    expected_size,
                    i});
            }
        }

        // Work in-place: move values into coefficients (zero-copy)
        std::vector<T> coeffs = std::move(values);

        // Initialize diagnostics
        std::array<T, N> max_residuals{};
        std::array<T, N> conditions{};
        std::array<size_t, N> failed{};

        // Fit each axis in cache-optimal order (reverse: N-1 → 0)
        fit_all_axes<N-1>(coeffs, config.tolerance, max_residuals, conditions, failed);

        // Check if any axis had failures
        size_t total_failures = 0;
        for (size_t i = 0; i < N; ++i) {
            total_failures += failed[i];
        }
        if (total_failures > 0) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::FittingFailed,
                total_failures});
        }

        return BSplineNDSeparableResult<T, N>{
            .coefficients = std::move(coeffs),
            .max_residual_per_axis = max_residuals,
            .condition_per_axis = conditions,
            .failed_slices = failed
        };
    }

    /// Get grid dimensions
    [[nodiscard]] std::array<size_t, N> dimensions() const noexcept {
        std::array<size_t, N> dims;
        for (size_t i = 0; i < N; ++i) {
            dims[i] = dims_[i];
        }
        return dims;
    }

private:
    /// Private constructor - use factory method create() instead
    ///
    /// @param grids N-dimensional grids (validation done by factory)
    explicit BSplineNDSeparable(std::array<std::vector<T>, N> grids)
        : grids_(std::move(grids))
    {
        // Store dimensions
        for (size_t i = 0; i < N; ++i) {
            dims_[i] = grids_[i].size();
        }

        // Compute strides for row-major layout
        // Memory layout: ((i*N₁ + j)*N₂ + k)*N₃ + l
        strides_[N-1] = 1;
        for (size_t i = N-1; i > 0; --i) {
            strides_[i-1] = strides_[i] * dims_[i];
        }

        // Create 1D solvers for each axis
        for (size_t i = 0; i < N; ++i) {
            auto solver_result = BSplineCollocation1D<T>::create(grids_[i]);
            // Should never fail (already validated in factory method)
            assert(solver_result.has_value() && "Solver creation failed after validation");
            solvers_[i] = std::make_unique<BSplineCollocation1D<T>>(
                std::move(solver_result.value()));
        }
    }

    /// Recursive template to fit all axes
    ///
    /// Processes axes in reverse order (N-1, N-2, ..., 1, 0) for cache locality.
    /// Fastest-varying dimension (stride=1) is processed first.
    template<size_t Axis>
    void fit_all_axes(
        std::vector<T>& coeffs,
        T tolerance,
        std::array<T, N>& max_residuals,
        std::array<T, N>& conditions,
        std::array<size_t, N>& failed)
    {
        fit_axis<Axis>(coeffs, tolerance, max_residuals, conditions, failed);

        // Recursively fit remaining axes
        if constexpr (Axis > 0) {
            fit_all_axes<Axis - 1>(coeffs, tolerance, max_residuals, conditions, failed);
        }
    }

    /// Generic axis fitting using template parameter
    ///
    /// Iterates over all (N-1)-dimensional slices perpendicular to this axis,
    /// performs 1D B-spline fitting on each slice, and writes back coefficients.
    ///
    /// Uses ThreadWorkspaceBuffer for zero-allocation parallel fitting.
    ///
    /// @tparam Axis Which axis to fit (0 to N-1)
    /// @param coeffs Coefficient array (modified in-place)
    /// @param tolerance Max residual for fitting
    /// @param max_residuals Output: max residual per axis
    /// @param conditions Output: condition estimate per axis
    /// @param failed Output: failed slice count per axis
    template<size_t Axis>
    void fit_axis(
        std::vector<T>& coeffs,
        T tolerance,
        std::array<T, N>& max_residuals,
        std::array<T, N>& conditions,
        std::array<size_t, N>& failed)
    {
        static_assert(Axis < N, "Axis index out of bounds");

        const size_t n_axis = dims_[Axis];
        const size_t n_slices = total_slices<Axis>();

        // Per-axis statistics (thread-safe reduction targets)
        T max_residual = T{0};
        T max_condition = T{0};
        size_t failed_count = 0;

        // Calculate workspace size for this axis
        const size_t ws_bytes = BSplineCollocationWorkspace<T>::required_bytes(n_axis);

        MANGO_PRAGMA_PARALLEL
        {
            // Create workspace buffer ONCE per thread
            ThreadWorkspaceBuffer buffer(ws_bytes);
            auto ws = BSplineCollocationWorkspace<T>::from_bytes(buffer.bytes(), n_axis).value();

            // Temporary buffer for slice extraction (per-thread)
            std::vector<T> slice_buffer(n_axis);

            // Thread-local accumulators
            T local_max_residual = T{0};
            T local_max_condition = T{0};
            size_t local_failed = 0;

            MANGO_PRAGMA_FOR
            for (size_t slice_idx = 0; slice_idx < n_slices; ++slice_idx) {
                // Calculate base offset for this slice
                size_t base_offset = slice_idx_to_offset<Axis>(slice_idx);

                // Extract 1D slice along this axis (SIMD-optimized)
                const size_t stride = strides_[Axis];
                MANGO_PRAGMA_SIMD
                for (size_t i = 0; i < n_axis; ++i) {
                    slice_buffer[i] = coeffs[base_offset + i * stride];
                }

                // Fit using workspace
                auto result = solvers_[Axis]->fit_with_workspace(
                    std::span<const T>{slice_buffer},
                    ws,
                    BSplineCollocationConfig<T>{.tolerance = tolerance});

                if (result.has_value()) {
                    local_max_residual = std::max(local_max_residual, result->max_residual);
                    local_max_condition = std::max(local_max_condition, result->condition_estimate);

                    // Copy coefficients from ws.coeffs() back to output (SIMD-optimized)
                    auto fitted_coeffs = ws.coeffs();
                    MANGO_PRAGMA_SIMD
                    for (size_t i = 0; i < n_axis; ++i) {
                        coeffs[base_offset + i * stride] = fitted_coeffs[i];
                    }
                } else {
                    ++local_failed;
                }
            }

            // Reduction via critical section
            MANGO_PRAGMA_CRITICAL
            {
                max_residual = std::max(max_residual, local_max_residual);
                max_condition = std::max(max_condition, local_max_condition);
                failed_count += local_failed;
            }
        }

        max_residuals[Axis] = max_residual;
        conditions[Axis] = max_condition;
        failed[Axis] = failed_count;
    }

    /// Calculate total number of slices perpendicular to given axis
    ///
    /// Returns product of all dimensions except Axis
    template<size_t Axis>
    [[nodiscard]] size_t total_slices() const noexcept {
        size_t count = 1;
        for (size_t d = 0; d < N; ++d) {
            if (d != Axis) {
                count *= dims_[d];
            }
        }
        return count;
    }

    /// Convert flat slice index to base offset for given axis
    ///
    /// Maps 1D slice index → N-dimensional offset by treating all dims except Axis
    /// as a flattened index space.
    template<size_t Axis>
    [[nodiscard]] size_t slice_idx_to_offset(size_t slice_idx) const noexcept {
        size_t offset = 0;
        size_t remaining = slice_idx;

        for (size_t d = 0; d < N; ++d) {
            if (d == Axis) continue;

            // Calculate stride for this dimension (product of all smaller dims except Axis)
            size_t stride_d = 1;
            for (size_t k = d + 1; k < N; ++k) {
                if (k != Axis) {
                    stride_d *= dims_[k];
                }
            }

            size_t index_d = remaining / stride_d;
            remaining %= stride_d;
            offset += index_d * strides_[d];
        }

        return offset;
    }


    std::array<std::vector<T>, N> grids_;
    std::array<size_t, N> dims_;
    std::array<size_t, N> strides_;
    std::array<std::unique_ptr<BSplineCollocation1D<T>>, N> solvers_;
};

} // namespace mango
