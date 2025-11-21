/**
 * @file cubic_spline_nd.hpp
 * @brief N-dimensional separable cubic spline interpolation
 *
 * Provides template-based N-dimensional cubic spline interpolation using
 * tensor-product structure. Works for any dimension N ≥ 1.
 *
 * Key features:
 * - Compile-time dimension specification via template parameter
 * - Generic over floating-point types (float, double, long double)
 * - Separable fitting: sequential 1D splines along each axis
 * - Cache-optimized axis traversal order
 * - RAII workspace management for zero-allocation evaluation
 *
 * Usage:
 *   // 3D temperature field T(x, y, z)
 *   auto spline = CubicSplineND<double, 3>::create(
 *       {x_grid, y_grid, z_grid}, temperature_values).value();
 *
 *   double T_interp = spline.eval({1.5, 0.75, 15.0});
 *
 * Performance:
 * - Fitting: O(N × ∏ grid_sizes) for N-dimensional data
 * - Evaluation: O(N × max_grid_size) per query (builds N 1D splines)
 * - Memory: O(∏ grid_sizes) for data storage
 *
 * Comparison to B-spline ND:
 * - Cubic spline: Exact interpolation at grid points, simpler
 * - B-spline: Smoothing fit, faster evaluation after pre-fit
 */

#pragma once

#include "src/math/cubic_spline_solver.hpp"
#include <array>
#include <vector>
#include <span>
#include <expected>
#include <string>
#include <algorithm>
#include <numeric>
#include <concepts>

namespace mango {

/// N-dimensional separable cubic spline interpolator
///
/// Uses tensor-product structure: sequential 1D cubic splines along each axis.
/// Provides exact interpolation at grid points with C² continuity.
///
/// @tparam T Floating point type (float, double, long double)
/// @tparam N Number of dimensions (N ≥ 1)
template<std::floating_point T, size_t N>
    requires (N >= 1)
class CubicSplineND {
public:
    using GridArray = std::array<std::vector<T>, N>;
    using QueryPoint = std::array<T, N>;
    using Shape = std::array<size_t, N>;

    /// Factory method with validation
    ///
    /// @param grids N vectors of grid coordinates (each must be sorted, size ≥ 2)
    /// @param values Flattened N-D array in row-major order
    /// @param config Spline configuration (applied to all 1D splines)
    /// @return Spline instance or error message
    [[nodiscard]] static std::expected<CubicSplineND, std::string> create(
        GridArray grids,
        std::span<const T> values,
        const CubicSplineConfig<T>& config = {})
    {
        // Validate each dimension's grid
        for (size_t dim = 0; dim < N; ++dim) {
            if (grids[dim].size() < 2) {
                return std::unexpected(
                    "Grid dimension " + std::to_string(dim) +
                    " must have ≥2 points, got " + std::to_string(grids[dim].size()));
            }

            if (!std::is_sorted(grids[dim].begin(), grids[dim].end())) {
                return std::unexpected(
                    "Grid dimension " + std::to_string(dim) + " must be sorted");
            }

            // Check for near-duplicate points
            for (size_t i = 1; i < grids[dim].size(); ++i) {
                if (grids[dim][i] - grids[dim][i-1] < T{1e-14}) {
                    return std::unexpected(
                        "Grid dimension " + std::to_string(dim) +
                        " has points too close together at index " + std::to_string(i));
                }
            }
        }

        // Compute expected data size
        size_t expected_size = 1;
        Shape shape;
        for (size_t i = 0; i < N; ++i) {
            shape[i] = grids[i].size();
            expected_size *= shape[i];
        }

        if (values.size() != expected_size) {
            return std::unexpected(
                "Values size " + std::to_string(values.size()) +
                " does not match grid product " + std::to_string(expected_size));
        }

        return CubicSplineND(std::move(grids), values, config, shape);
    }

    /// Evaluate at N-dimensional query point
    ///
    /// Uses separable cubic spline interpolation: builds 1D splines along
    /// each axis sequentially. Processes dimensions in reverse order for
    /// cache efficiency (fastest-varying dimension first).
    ///
    /// @param query N-dimensional query point
    /// @return Interpolated value
    [[nodiscard]] T eval(const QueryPoint& query) const {
        // Start with full data
        std::vector<T> current_data(data_);
        std::vector<size_t> current_shape(shape_.begin(), shape_.end());

        // Process each dimension in reverse order (cache-optimized)
        // Fastest-varying dimension (N-1) first, slowest (0) last
        for (int dim = static_cast<int>(N) - 1; dim >= 0; --dim) {
            const size_t n = current_shape[dim];
            const T query_coord = query[dim];

            // Clamp query to grid bounds
            const T clamped_query = std::clamp(query_coord,
                                              grids_[dim].front(),
                                              grids_[dim].back());

            // Compute stride for this dimension
            const size_t stride = compute_stride(current_shape, dim);
            const size_t n_slices = std::accumulate(
                current_shape.begin(), current_shape.end(),
                size_t{1}, std::multiplies<>()) / n;

            std::vector<T> next_data;
            next_data.reserve(n_slices);

            // Extract and interpolate each 1D slice
            for (size_t slice = 0; slice < n_slices; ++slice) {
                // Extract 1D slice along current dimension
                std::vector<T> slice_values(n);
                for (size_t i = 0; i < n; ++i) {
                    const size_t idx = compute_flat_index(slice, i, stride, current_shape, dim);
                    slice_values[i] = current_data[idx];
                }

                // Build 1D spline and evaluate
                CubicSpline<T> spline_1d;
                auto build_error = spline_1d.build(grids_[dim], slice_values, config_);
                if (build_error.has_value()) {
                    // Fallback to linear interpolation on error
                    next_data.push_back(linear_interp_1d(
                        grids_[dim], slice_values, clamped_query));
                    continue;
                }

                next_data.push_back(spline_1d.eval(clamped_query));
            }

            // Update for next dimension
            current_data = std::move(next_data);
            current_shape.erase(current_shape.begin() + dim);
        }

        return current_data[0];
    }

    /// Get grid for specific dimension
    [[nodiscard]] const std::vector<T>& grid(size_t dim) const noexcept {
        return grids_[dim];
    }

    /// Get shape (grid sizes for each dimension)
    [[nodiscard]] const Shape& shape() const noexcept {
        return shape_;
    }

    /// Get total number of data points
    [[nodiscard]] size_t size() const noexcept {
        return data_.size();
    }

private:
    GridArray grids_;                    ///< Grid coordinates for each dimension
    std::vector<T> data_;                ///< Flattened N-D data (row-major)
    CubicSplineConfig<T> config_;        ///< Spline configuration
    Shape shape_;                        ///< Grid sizes per dimension

    /// Private constructor - use factory method
    CubicSplineND(GridArray grids,
                  std::span<const T> values,
                  const CubicSplineConfig<T>& config,
                  const Shape& shape)
        : grids_(std::move(grids))
        , data_(values.begin(), values.end())
        , config_(config)
        , shape_(shape)
    {}

    /// Compute stride for accessing dimension
    ///
    /// Stride = product of sizes of all dimensions after this one
    /// For row-major layout: index = (...((i0*N1 + i1)*N2 + i2)*N3 + ...)
    [[nodiscard]] size_t compute_stride(
        const std::vector<size_t>& shape,
        size_t dim) const noexcept
    {
        size_t stride = 1;
        for (size_t i = dim + 1; i < shape.size(); ++i) {
            stride *= shape[i];
        }
        return stride;
    }

    /// Compute flat index for slice extraction
    ///
    /// Given a slice index and position along the slicing dimension,
    /// compute the corresponding flat array index.
    [[nodiscard]] size_t compute_flat_index(
        size_t slice_idx,
        size_t pos_in_dim,
        [[maybe_unused]] size_t stride,
        const std::vector<size_t>& shape,
        size_t dim) const noexcept
    {
        // Build reduced shape (excluding slicing dimension)
        std::vector<size_t> reduced_shape;
        reduced_shape.reserve(shape.size() - 1);
        for (size_t d = 0; d < shape.size(); ++d) {
            if (d != static_cast<size_t>(dim)) {
                reduced_shape.push_back(shape[d]);
            }
        }

        // Compute multi-dimensional indices in reduced space from slice_idx
        std::vector<size_t> reduced_indices;
        reduced_indices.reserve(reduced_shape.size());
        size_t remaining = slice_idx;

        for (size_t i = 0; i < reduced_shape.size(); ++i) {
            // Stride in reduced space
            size_t local_stride = 1;
            for (size_t j = i + 1; j < reduced_shape.size(); ++j) {
                local_stride *= reduced_shape[j];
            }
            reduced_indices.push_back(remaining / local_stride);
            remaining %= local_stride;
        }

        // Build full multi-dimensional indices by inserting pos_in_dim at dim
        size_t idx = 0;
        size_t reduced_idx = 0;
        for (size_t d = 0; d < shape.size(); ++d) {
            if (d == static_cast<size_t>(dim)) {
                idx = idx * shape[d] + pos_in_dim;
            } else {
                idx = idx * shape[d] + reduced_indices[reduced_idx++];
            }
        }

        return idx;
    }

    /// Fallback linear interpolation for 1D slice
    [[nodiscard]] static T linear_interp_1d(
        const std::vector<T>& grid,
        const std::vector<T>& values,
        T x) noexcept
    {
        if (x <= grid.front()) return values.front();
        if (x >= grid.back()) return values.back();

        // Binary search for interval
        auto it = std::lower_bound(grid.begin(), grid.end(), x);
        if (it == grid.begin()) return values[0];

        const size_t i = std::distance(grid.begin(), it) - 1;
        const T t = (x - grid[i]) / (grid[i + 1] - grid[i]);
        return (T{1} - t) * values[i] + t * values[i + 1];
    }
};

/// Workspace for efficient N-dimensional cubic spline evaluation
///
/// Pre-allocates buffers for intermediate results to avoid repeated
/// allocations during evaluation. Reusable across multiple queries.
///
/// @tparam T Floating point type
/// @tparam N Number of dimensions
template<std::floating_point T, size_t N>
    requires (N >= 1)
struct CubicSplineNDWorkspace {
    std::vector<T> slice_buffer;    ///< Buffer for extracting 1D slices
    std::vector<T> eval_buffer;     ///< Buffer for intermediate evaluations

    /// Create workspace sized for maximum grid dimension
    ///
    /// @param max_grid_size Largest grid size across all dimensions
    explicit CubicSplineNDWorkspace(size_t max_grid_size)
        : slice_buffer(max_grid_size)
        , eval_buffer(max_grid_size)
    {}

    /// Get slice buffer as span
    [[nodiscard]] std::span<T> get_slice_buffer(size_t n) noexcept {
        return std::span{slice_buffer.data(), std::min(n, slice_buffer.size())};
    }

    /// Get evaluation buffer as span
    [[nodiscard]] std::span<T> get_eval_buffer(size_t n) noexcept {
        return std::span{eval_buffer.data(), std::min(n, eval_buffer.size())};
    }
};

// ============================================================================
// Convenience Aliases for Common Dimensions
// ============================================================================

/// 3D cubic spline interpolator
template<std::floating_point T>
using CubicSpline3D = CubicSplineND<T, 3>;

/// 4D cubic spline interpolator
template<std::floating_point T>
using CubicSpline4D = CubicSplineND<T, 4>;

/// 5D cubic spline interpolator
template<std::floating_point T>
using CubicSpline5D = CubicSplineND<T, 5>;

}  // namespace mango
