/**
 * @file bspline_nd.hpp
 * @brief N-dimensional tensor-product B-spline interpolation
 *
 * Provides template-based N-dimensional B-spline interpolation using
 * tensor-product structure. Works for any dimension N ≥ 1.
 *
 * Key features:
 * - Compile-time dimension specification via template parameter
 * - Clamped cubic B-splines with Cox-de Boor recursion
 * - FMA optimization for fast evaluation
 * - Recursive tensor-product evaluation
 * - Zero-copy construction from workspace
 *
 * Usage:
 *   // 4D B-spline for option pricing
 *   BSplineND<4> spline_4d(grids, knots, coefficients);
 *   double price = spline_4d.eval({1.05, 0.25, 0.20, 0.05});
 *
 *   // 5D B-spline with dividend dimension
 *   BSplineND<5> spline_5d(grids, knots, coefficients);
 *   double price = spline_5d.eval({1.05, 0.25, 0.20, 0.05, 0.02});
 *
 * Performance: Comparable to hardcoded BSpline4D (~135ns per query)
 */

#pragma once

#include "src/math/bspline_basis.hpp"
#include <experimental/mdspan>
#include <array>
#include <vector>
#include <span>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <expected>
#include <string>

namespace mango {

/// Clamp query point to valid domain with half-open interval
///
/// For right boundary, uses nextafter to ensure x < xmax (not x <= xmax)
/// to avoid issues with half-open interval [xmin, xmax)
template<std::floating_point T>
inline T clamp_bspline_query(T x, T xmin, T xmax) {
    if (x <= xmin) return xmin;
    if (x >= xmax) {
        return std::nextafter(xmax, -std::numeric_limits<T>::infinity());
    }
    return x;
}

/// N-dimensional tensor-product B-spline interpolator
///
/// Uses tensor-product structure: sequential cubic B-splines along each axis.
/// Provides exact interpolation at grid points with C² continuity.
///
/// @tparam T Floating point type (float, double, long double)
/// @tparam N Number of dimensions (N ≥ 1)
template<std::floating_point T, size_t N>
    requires (N >= 1)
class BSplineND {
public:
    using GridArray = std::array<std::vector<T>, N>;
    using KnotArray = std::array<std::vector<T>, N>;
    using QueryPoint = std::array<T, N>;
    using Shape = std::array<size_t, N>;

    // NEW: mdspan for N-dimensional coefficient array
    using CoeffExtents = std::experimental::dextents<size_t, N>;
    using CoeffMdspan = std::experimental::mdspan<T, CoeffExtents, std::experimental::layout_right>;

    /// Factory method with validation
    ///
    /// @param grids N vectors of grid coordinates (each must be sorted, size ≥ 4)
    /// @param knots N vectors of knot sequences (clamped cubic)
    /// @param coeffs Flattened N-D coefficient array in row-major order
    /// @return BSplineND instance or error message
    [[nodiscard]] static std::expected<BSplineND, std::string> create(
        GridArray grids,
        KnotArray knots,
        std::vector<T> coeffs)
    {
        // Validate grid sizes
        for (size_t dim = 0; dim < N; ++dim) {
            if (grids[dim].size() < 4) {
                return std::unexpected("Grid dimension " + std::to_string(dim) +
                                     " must have at least 4 points");
            }
            if (knots[dim].size() != grids[dim].size() + 4) {
                return std::unexpected("Knot dimension " + std::to_string(dim) +
                                     " size mismatch");
            }
        }

        // Compute expected coefficient array size
        size_t expected_size = 1;
        for (size_t dim = 0; dim < N; ++dim) {
            expected_size *= grids[dim].size();
        }

        if (coeffs.size() != expected_size) {
            return std::unexpected("Coefficient size " + std::to_string(coeffs.size()) +
                                 " does not match grid dimensions (expected " +
                                 std::to_string(expected_size) + ")");
        }

        return BSplineND(std::move(grids), std::move(knots), std::move(coeffs));
    }

    /// Evaluate B-spline at query point
    ///
    /// @param query N-dimensional query point
    /// @return Interpolated value
    T eval(const QueryPoint& query) const {
        // Clamp queries to domain
        QueryPoint clamped;
        for (size_t dim = 0; dim < N; ++dim) {
            clamped[dim] = clamp_bspline_query(
                query[dim],
                grids_[dim].front(),
                grids_[dim].back()
            );
        }

        // Find knot spans for all dimensions
        std::array<int, N> spans;
        for (size_t dim = 0; dim < N; ++dim) {
            spans[dim] = find_span_cubic(knots_[dim], clamped[dim]);
        }

        // Evaluate basis functions for all dimensions
        std::array<std::array<T, 4>, N> basis_weights;
        for (size_t dim = 0; dim < N; ++dim) {
            cubic_basis_nonuniform(knots_[dim], spans[dim], clamped[dim],
                                 basis_weights[dim].data());
        }

        // Tensor-product evaluation with recursive loop unrolling
        return eval_tensor_product<0>(spans, basis_weights, std::array<int, N>{});
    }

    /// Get grid dimensions
    [[nodiscard]] Shape dimensions() const noexcept {
        Shape dims;
        for (size_t i = 0; i < N; ++i) {
            dims[i] = grids_[i].size();
        }
        return dims;
    }

    /// Get grid for specific dimension
    [[nodiscard]] const std::vector<T>& grid(size_t dim) const {
        assert(dim < N && "Dimension index out of bounds");
        return grids_[dim];
    }

    /// Get knots for specific dimension
    [[nodiscard]] const std::vector<T>& knots(size_t dim) const {
        assert(dim < N && "Dimension index out of bounds");
        return knots_[dim];
    }

    /// Get coefficient array
    [[nodiscard]] const std::vector<T>& coefficients() const noexcept {
        return coeffs_;
    }

private:
    GridArray grids_;        ///< Grid points for each dimension
    KnotArray knots_;        ///< Knot vectors for each dimension
    std::vector<T> coeffs_;  ///< Coefficient storage
    CoeffMdspan coeffs_view_;///< N-dimensional view of coeffs_
    Shape dims_;             ///< Cached grid dimensions

    /// Private constructor (use factory method)
    BSplineND(GridArray grids, KnotArray knots, std::vector<T> coeffs)
        : grids_(std::move(grids))
        , knots_(std::move(knots))
        , coeffs_(std::move(coeffs))
        , coeffs_view_(nullptr, CoeffExtents{})  // Initialized below
    {
        // Extract dimensions
        for (size_t i = 0; i < N; ++i) {
            dims_[i] = grids_[i].size();
        }

        // Create mdspan view with proper extents
        coeffs_view_ = create_coeffs_view(coeffs_.data(), dims_);
    }

    /// Recursive tensor-product evaluation
    ///
    /// Evaluates N-dimensional tensor product using compile-time recursion.
    /// Each recursion level handles one dimension, unrolling the 4-point
    /// cubic B-spline support at compile time.
    ///
    /// @tparam Dim Current dimension being processed (0 to N-1)
    /// @param spans Knot span indices for each dimension
    /// @param weights Basis function weights for each dimension
    /// @param indices Current multi-index being evaluated
    /// @return Accumulated sum for this recursion level
    template<size_t Dim>
    T eval_tensor_product(
        const std::array<int, N>& spans,
        const std::array<std::array<T, 4>, N>& weights,
        std::array<int, N> indices) const
    {
        T sum = 0.0;

        // Unroll loop over 4 basis functions in this dimension
        for (int offset = 0; offset < 4; ++offset) {
            const int idx = spans[Dim] - offset;

            // Bounds check
            if (static_cast<unsigned>(idx) >= static_cast<unsigned>(dims_[Dim])) {
                continue;
            }

            indices[Dim] = idx;
            const T weight = weights[Dim][offset];

            if constexpr (Dim == N - 1) {
                // Base case: innermost dimension
                // Compute flat index and accumulate with FMA
                const size_t flat_idx = compute_flat_index(indices);
                sum = std::fma(coeffs_[flat_idx], weight, sum);
            } else {
                // Recursive case: descend to next dimension
                const T nested_sum = eval_tensor_product<Dim + 1>(spans, weights, indices);
                sum = std::fma(nested_sum, weight, sum);
            }
        }

        return sum;
    }

    /// Helper to create mdspan with variadic extents
    static CoeffMdspan create_coeffs_view(T* data, const Shape& dims) {
        return create_view_impl(data, dims, std::make_index_sequence<N>{});
    }

    template<size_t... Is>
    static CoeffMdspan create_view_impl(T* data, const Shape& dims,
                                        std::index_sequence<Is...>) {
        return CoeffMdspan(data, dims[Is]...);
    }

    /// Compute flat index from N-dimensional index (row-major order)
    ///
    /// Converts multi-dimensional index to flat array index.
    /// Example for 3D: idx = i*dim1*dim2 + j*dim2 + k
    ///
    /// @param indices Multi-dimensional index
    /// @return Flat index into coefficient array
    size_t compute_flat_index(const std::array<int, N>& indices) const noexcept {
        size_t idx = 0;
        size_t stride = 1;

        // Compute index in row-major order (last dimension varies fastest)
        for (size_t dim = N; dim > 0; --dim) {
            const size_t d = dim - 1;
            idx += static_cast<size_t>(indices[d]) * stride;
            stride *= dims_[d];
        }

        return idx;
    }
};

} // namespace mango
