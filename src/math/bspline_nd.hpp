// SPDX-License-Identifier: MIT
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
#include "src/support/error_types.hpp"
#include "src/math/safe_math.hpp"
#include <experimental/mdspan>
#include <array>
#include <vector>
#include <span>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <expected>

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
    /// @return BSplineND instance or error
    [[nodiscard]] static std::expected<BSplineND, InterpolationError> create(
        GridArray grids,
        KnotArray knots,
        std::vector<T> coeffs)
    {
        // Validate grid sizes
        for (size_t dim = 0; dim < N; ++dim) {
            if (grids[dim].size() < 4) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::InsufficientGridPoints,
                    grids[dim].size(),
                    dim});
            }
            if (knots[dim].size() != grids[dim].size() + 4) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::DimensionMismatch,
                    knots[dim].size(),
                    dim});
            }
        }

        // Compute expected coefficient array size with overflow check
        size_t expected_size = 1;
        for (size_t dim = 0; dim < N; ++dim) {
            auto result = safe_multiply(expected_size, grids[dim].size());
            if (!result.has_value()) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::ValueSizeMismatch,
                    grids[dim].size(),
                    dim});
            }
            expected_size = result.value();
        }

        if (coeffs.size() != expected_size) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::CoefficientSizeMismatch,
                coeffs.size()});
        }

        return BSplineND(std::move(grids), std::move(knots), std::move(coeffs));
    }

    /// Evaluate B-spline at query point
    ///
    /// @param query N-dimensional query point
    /// @return Interpolated value
    T eval(const QueryPoint& query) const {
        // Process all dimensions in single loop for better cache locality
        QueryPoint clamped;
        std::array<int, N> spans;
        std::array<std::array<T, 4>, N> basis_weights;

        for (size_t dim = 0; dim < N; ++dim) {
            // Clamp query to domain
            clamped[dim] = clamp_bspline_query(
                query[dim],
                grids_[dim].front(),
                grids_[dim].back()
            );

            // Find knot span
            spans[dim] = find_span_cubic(knots_[dim], clamped[dim]);

            // Evaluate basis functions
            cubic_basis_nonuniform(knots_[dim], spans[dim], clamped[dim],
                                 basis_weights[dim].data());
        }

        // Tensor-product evaluation with recursive loop unrolling
        return eval_tensor_product<0>(spans, basis_weights, std::array<int, N>{});
    }

    /// Evaluate partial derivative of B-spline at query point (analytic)
    ///
    /// Computes ∂f/∂xₐ using analytic B-spline derivative formula.
    /// Uses derivative basis functions for the specified axis, regular basis
    /// for all other axes.
    ///
    /// @param axis Dimension to differentiate (0 to N-1)
    /// @param query N-dimensional query point
    /// @return Partial derivative ∂f/∂x_axis
    T eval_partial(size_t axis, const QueryPoint& query) const {
        assert(axis < N && "Axis index out of bounds");

        // Process all dimensions in single loop for better cache locality
        QueryPoint clamped;
        std::array<int, N> spans;
        std::array<std::array<T, 4>, N> basis_weights;

        for (size_t dim = 0; dim < N; ++dim) {
            // Clamp query to domain
            clamped[dim] = clamp_bspline_query(
                query[dim],
                grids_[dim].front(),
                grids_[dim].back()
            );

            // Find knot span
            spans[dim] = find_span_cubic(knots_[dim], clamped[dim]);

            // Evaluate basis functions (derivative basis for target axis, regular for others)
            if (dim == axis) {
                cubic_basis_derivative_nonuniform(knots_[dim], spans[dim], clamped[dim],
                                                  basis_weights[dim].data());
            } else {
                cubic_basis_nonuniform(knots_[dim], spans[dim], clamped[dim],
                                      basis_weights[dim].data());
            }
        }

        // Tensor-product evaluation (same as eval, but with derivative weights in one axis)
        return eval_tensor_product<0>(spans, basis_weights, std::array<int, N>{});
    }

    /// Second partial derivative along a single axis
    ///
    /// Uses analytical B-spline second derivatives (O(h²) accuracy).
    ///
    /// @param axis Axis along which to differentiate (0 to N-1)
    /// @param query N-dimensional query point
    /// @return Second partial derivative ∂²f/∂x²_axis
    T eval_second_partial(size_t axis, const QueryPoint& query) const {
        assert(axis < N && "Axis index out of bounds");

        QueryPoint clamped;
        std::array<int, N> spans;
        std::array<std::array<T, 4>, N> basis_weights;

        for (size_t dim = 0; dim < N; ++dim) {
            clamped[dim] = clamp_bspline_query(
                query[dim],
                grids_[dim].front(),
                grids_[dim].back()
            );

            spans[dim] = find_span_cubic(knots_[dim], clamped[dim]);

            if (dim == axis) {
                cubic_basis_second_derivative_nonuniform(knots_[dim], spans[dim], clamped[dim],
                                                         basis_weights[dim].data());
            } else {
                cubic_basis_nonuniform(knots_[dim], spans[dim], clamped[dim],
                                      basis_weights[dim].data());
            }
        }

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
    [[nodiscard]] const std::vector<T>& grid(size_t dim) const noexcept {
        assert(dim < N && "Dimension index out of bounds");
        return grids_[dim];
    }

    /// Get knots for specific dimension
    [[nodiscard]] const std::vector<T>& knots(size_t dim) const noexcept {
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

    /// Access N-dimensional coefficient array via mdspan
    ///
    /// Uses variadic template expansion to convert std::array to mdspan subscript.
    template<size_t... Is>
    static T access_coeffs_impl(const CoeffMdspan& view, const std::array<int, N>& indices,
                                std::index_sequence<Is...>) {
        return view[indices[Is]...];  // Expands to view[indices[0], indices[1], ...]
    }

    static T access_coeffs(const CoeffMdspan& view, const std::array<int, N>& indices) {
        return access_coeffs_impl(view, indices, std::make_index_sequence<N>{});
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
                // Base case: use mdspan multi-dimensional indexing
                const T coeff = access_coeffs(coeffs_view_, indices);
                sum = std::fma(coeff, weight, sum);
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
};

} // namespace mango
