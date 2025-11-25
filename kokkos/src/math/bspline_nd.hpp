#pragma once

/// @file bspline_nd.hpp
/// @brief N-dimensional tensor-product B-spline interpolation for Kokkos
///
/// Provides GPU-portable N-dimensional B-spline interpolation using
/// tensor-product structure. Works for any dimension N ≥ 1.
///
/// Key features:
/// - Compile-time dimension specification via template parameter
/// - Clamped cubic B-splines with Cox-de Boor recursion
/// - FMA optimization for fast evaluation
/// - Recursive tensor-product evaluation
/// - KOKKOS_INLINE_FUNCTION for GPU kernels

#include <Kokkos_Core.hpp>
#include <expected>
#include <array>
#include <cmath>
#include <limits>
#include "kokkos/src/math/bspline_basis.hpp"
#include "kokkos/src/support/execution_space.hpp"

namespace mango::kokkos {

/// Error codes for B-spline construction
enum class BSplineNDError {
    InsufficientGridPoints,
    DimensionMismatch,
    CoefficientSizeMismatch,
    InvalidDimension
};

/// N-dimensional tensor-product B-spline interpolator
///
/// Uses tensor-product structure: sequential cubic B-splines along each axis.
/// Provides exact interpolation at grid points with C² continuity.
///
/// @tparam MemSpace Kokkos memory space (HostSpace, CudaSpace, etc.)
/// @tparam N Number of dimensions (N ≥ 1, typically 4 for option pricing)
template <typename MemSpace, size_t N>
    requires (N >= 1)
class BSplineND {
public:
    using view_type = Kokkos::View<double*, MemSpace>;
    using QueryPoint = std::array<double, N>;
    using Shape = std::array<size_t, N>;

    /// Factory method with validation
    ///
    /// @param grids N Views of grid coordinates (each must be sorted, size ≥ 4)
    /// @param knots N Views of knot sequences (clamped cubic)
    /// @param coeffs Flattened N-D coefficient array in row-major order
    /// @return BSplineND instance or error
    [[nodiscard]] static std::expected<BSplineND, BSplineNDError> create(
        std::array<view_type, N> grids,
        std::array<view_type, N> knots,
        view_type coeffs)
    {
        // Validate grid sizes
        size_t expected_size = 1;
        std::array<size_t, N> dims;
        std::array<int, N> n_knots;

        for (size_t dim = 0; dim < N; ++dim) {
            dims[dim] = grids[dim].extent(0);
            n_knots[dim] = static_cast<int>(knots[dim].extent(0));

            if (dims[dim] < 4) {
                return std::unexpected(BSplineNDError::InsufficientGridPoints);
            }
            if (static_cast<size_t>(n_knots[dim]) != dims[dim] + 4) {
                return std::unexpected(BSplineNDError::DimensionMismatch);
            }
            expected_size *= dims[dim];
        }

        if (coeffs.extent(0) != expected_size) {
            return std::unexpected(BSplineNDError::CoefficientSizeMismatch);
        }

        return BSplineND(std::move(grids), std::move(knots), std::move(coeffs),
                         dims, n_knots, expected_size);
    }

    /// Evaluate B-spline at query point (host-side)
    ///
    /// This method creates host mirrors and evaluates on CPU.
    /// For GPU batched evaluation, use eval_batch() instead.
    ///
    /// @param query N-dimensional query point
    /// @return Interpolated value
    double eval(const QueryPoint& query) const {
        // Create host mirrors for evaluation
        std::array<typename view_type::HostMirror, N> grids_h;
        std::array<typename view_type::HostMirror, N> knots_h;

        for (size_t d = 0; d < N; ++d) {
            grids_h[d] = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grids_[d]);
            knots_h[d] = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, knots_[d]);
        }
        auto coeffs_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, coeffs_);

        return eval_impl(query, grids_h, knots_h, coeffs_h);
    }

    /// Get grid dimensions
    [[nodiscard]] Shape dimensions() const noexcept {
        return dims_;
    }

    /// Get grid View for specific dimension
    [[nodiscard]] view_type grid(size_t dim) const noexcept {
        return grids_[dim];
    }

    /// Get knots View for specific dimension
    [[nodiscard]] view_type knots(size_t dim) const noexcept {
        return knots_[dim];
    }

    /// Get coefficient View
    [[nodiscard]] view_type coefficients() const noexcept {
        return coeffs_;
    }

    /// Raw data pointers for kernel use
    [[nodiscard]] const std::array<view_type, N>& grids_raw() const noexcept {
        return grids_;
    }

    [[nodiscard]] const std::array<view_type, N>& knots_raw() const noexcept {
        return knots_;
    }

    [[nodiscard]] view_type coeffs_raw() const noexcept {
        return coeffs_;
    }

    [[nodiscard]] const std::array<size_t, N>& dims_raw() const noexcept {
        return dims_;
    }

    [[nodiscard]] const std::array<int, N>& n_knots_raw() const noexcept {
        return n_knots_;
    }

private:
    std::array<view_type, N> grids_;   ///< Grid points for each dimension
    std::array<view_type, N> knots_;   ///< Knot vectors for each dimension
    view_type coeffs_;                  ///< Coefficient storage (flattened)
    std::array<size_t, N> dims_;       ///< Grid dimensions
    std::array<int, N> n_knots_;       ///< Knot counts per dimension
    size_t total_coeffs_;              ///< Total coefficient count

    /// Private constructor (use factory method)
    BSplineND(std::array<view_type, N> grids,
              std::array<view_type, N> knots,
              view_type coeffs,
              std::array<size_t, N> dims,
              std::array<int, N> n_knots,
              size_t total_coeffs)
        : grids_(std::move(grids))
        , knots_(std::move(knots))
        , coeffs_(std::move(coeffs))
        , dims_(dims)
        , n_knots_(n_knots)
        , total_coeffs_(total_coeffs)
    {}

    /// Evaluation implementation using host mirrors
    template <typename GridArrayH, typename KnotArrayH, typename CoeffsH>
    double eval_impl(const QueryPoint& query,
                     const GridArrayH& grids_h,
                     const KnotArrayH& knots_h,
                     const CoeffsH& coeffs_h) const
    {
        // Process all dimensions: clamp, find span, compute basis
        QueryPoint clamped;
        std::array<int, N> spans;
        std::array<std::array<double, 4>, N> basis_weights;

        for (size_t dim = 0; dim < N; ++dim) {
            // Clamp query to domain
            double xmin = grids_h[dim](0);
            double xmax = grids_h[dim](dims_[dim] - 1);
            clamped[dim] = clamp_query(query[dim], xmin, xmax);

            // Find knot span
            spans[dim] = find_span_cubic(knots_h[dim].data(), n_knots_[dim], clamped[dim]);

            // Evaluate basis functions
            cubic_basis_nonuniform(knots_h[dim].data(), n_knots_[dim],
                                   spans[dim], clamped[dim],
                                   basis_weights[dim].data());
        }

        // Tensor-product evaluation with recursive template
        return eval_tensor_product<0>(spans, basis_weights, coeffs_h, std::array<int, N>{});
    }

    /// Recursive tensor-product evaluation (compile-time unrolling)
    template <size_t Dim, typename CoeffsH>
    double eval_tensor_product(
        const std::array<int, N>& spans,
        const std::array<std::array<double, 4>, N>& weights,
        const CoeffsH& coeffs_h,
        std::array<int, N> indices) const
    {
        double sum = 0.0;

        // Unroll loop over 4 basis functions in this dimension
        for (int offset = 0; offset < 4; ++offset) {
            const int idx = spans[Dim] - offset;

            // Bounds check
            if (static_cast<unsigned>(idx) >= static_cast<unsigned>(dims_[Dim])) {
                continue;
            }

            indices[Dim] = idx;
            const double weight = weights[Dim][offset];

            if constexpr (Dim == N - 1) {
                // Base case: compute linear index and access coefficient
                const size_t linear_idx = compute_linear_index(indices);
                const double coeff = coeffs_h(linear_idx);
                sum = std::fma(coeff, weight, sum);
            } else {
                // Recursive case: descend to next dimension
                const double nested_sum = eval_tensor_product<Dim + 1>(spans, weights, coeffs_h, indices);
                sum = std::fma(nested_sum, weight, sum);
            }
        }

        return sum;
    }

    /// Compute linear index from N-dimensional indices (row-major order)
    size_t compute_linear_index(const std::array<int, N>& indices) const noexcept {
        size_t linear = 0;
        size_t stride = 1;

        // Compute from last to first dimension (row-major)
        for (int d = static_cast<int>(N) - 1; d >= 0; --d) {
            linear += static_cast<size_t>(indices[d]) * stride;
            stride *= dims_[d];
        }

        return linear;
    }
};

/// Device-callable evaluation functor for batched B-spline queries
///
/// This struct holds raw pointers/data needed for GPU kernel evaluation.
/// Create from BSplineND and pass to parallel_for kernels.
template <size_t N>
struct BSplineNDEvaluator {
    // Grid data (device pointers)
    const double* grids[N];
    const double* knots[N];
    const double* coeffs;
    size_t dims[N];
    int n_knots[N];

    /// Evaluate B-spline at query point (device-callable)
    KOKKOS_INLINE_FUNCTION
    double operator()(const double query[N]) const noexcept {
        double clamped[N];
        int spans[N];
        double basis_weights[N][4];

        // Process all dimensions
        for (size_t dim = 0; dim < N; ++dim) {
            // Clamp query to domain
            double xmin = grids[dim][0];
            double xmax = grids[dim][dims[dim] - 1];
            clamped[dim] = clamp_query(query[dim], xmin, xmax);

            // Find knot span
            spans[dim] = find_span_cubic(knots[dim], n_knots[dim], clamped[dim]);

            // Evaluate basis functions
            cubic_basis_nonuniform(knots[dim], n_knots[dim],
                                   spans[dim], clamped[dim],
                                   basis_weights[dim]);
        }

        // Iterative tensor-product evaluation (no recursion on device)
        return eval_tensor_product_iterative(spans, basis_weights);
    }

private:
    /// Iterative tensor-product for N dimensions
    ///
    /// For small N (typically 4-5), uses loop unrolling via compile-time
    /// dimension count.
    KOKKOS_INLINE_FUNCTION
    double eval_tensor_product_iterative(const int spans[N],
                                          const double weights[N][4]) const noexcept {
        // For N=4, we have 4^4 = 256 terms max
        // Use nested loops for explicit unrolling
        if constexpr (N == 4) {
            return eval_4d(spans, weights);
        } else if constexpr (N == 3) {
            return eval_3d(spans, weights);
        } else if constexpr (N == 2) {
            return eval_2d(spans, weights);
        } else if constexpr (N == 1) {
            return eval_1d(spans, weights);
        } else {
            // Generic N-dimensional (slower but works)
            return eval_generic(spans, weights);
        }
    }

    KOKKOS_INLINE_FUNCTION
    double eval_1d(const int spans[N], const double weights[N][4]) const noexcept {
        double sum = 0.0;
        for (int i0 = 0; i0 < 4; ++i0) {
            const int idx0 = spans[0] - i0;
            if (static_cast<unsigned>(idx0) >= dims[0]) continue;

            sum = Kokkos::fma(coeffs[idx0], weights[0][i0], sum);
        }
        return sum;
    }

    KOKKOS_INLINE_FUNCTION
    double eval_2d(const int spans[N], const double weights[N][4]) const noexcept {
        double sum = 0.0;
        for (int i0 = 0; i0 < 4; ++i0) {
            const int idx0 = spans[0] - i0;
            if (static_cast<unsigned>(idx0) >= dims[0]) continue;

            for (int i1 = 0; i1 < 4; ++i1) {
                const int idx1 = spans[1] - i1;
                if (static_cast<unsigned>(idx1) >= dims[1]) continue;

                const size_t linear = static_cast<size_t>(idx0) * dims[1] +
                                      static_cast<size_t>(idx1);
                const double w = weights[0][i0] * weights[1][i1];
                sum = Kokkos::fma(coeffs[linear], w, sum);
            }
        }
        return sum;
    }

    KOKKOS_INLINE_FUNCTION
    double eval_3d(const int spans[N], const double weights[N][4]) const noexcept {
        double sum = 0.0;
        for (int i0 = 0; i0 < 4; ++i0) {
            const int idx0 = spans[0] - i0;
            if (static_cast<unsigned>(idx0) >= dims[0]) continue;

            for (int i1 = 0; i1 < 4; ++i1) {
                const int idx1 = spans[1] - i1;
                if (static_cast<unsigned>(idx1) >= dims[1]) continue;

                for (int i2 = 0; i2 < 4; ++i2) {
                    const int idx2 = spans[2] - i2;
                    if (static_cast<unsigned>(idx2) >= dims[2]) continue;

                    const size_t linear = (static_cast<size_t>(idx0) * dims[1] +
                                           static_cast<size_t>(idx1)) * dims[2] +
                                          static_cast<size_t>(idx2);
                    const double w = weights[0][i0] * weights[1][i1] * weights[2][i2];
                    sum = Kokkos::fma(coeffs[linear], w, sum);
                }
            }
        }
        return sum;
    }

    KOKKOS_INLINE_FUNCTION
    double eval_4d(const int spans[N], const double weights[N][4]) const noexcept {
        double sum = 0.0;
        for (int i0 = 0; i0 < 4; ++i0) {
            const int idx0 = spans[0] - i0;
            if (static_cast<unsigned>(idx0) >= dims[0]) continue;

            for (int i1 = 0; i1 < 4; ++i1) {
                const int idx1 = spans[1] - i1;
                if (static_cast<unsigned>(idx1) >= dims[1]) continue;

                for (int i2 = 0; i2 < 4; ++i2) {
                    const int idx2 = spans[2] - i2;
                    if (static_cast<unsigned>(idx2) >= dims[2]) continue;

                    for (int i3 = 0; i3 < 4; ++i3) {
                        const int idx3 = spans[3] - i3;
                        if (static_cast<unsigned>(idx3) >= dims[3]) continue;

                        const size_t linear =
                            ((static_cast<size_t>(idx0) * dims[1] +
                              static_cast<size_t>(idx1)) * dims[2] +
                             static_cast<size_t>(idx2)) * dims[3] +
                            static_cast<size_t>(idx3);

                        const double w = weights[0][i0] * weights[1][i1] *
                                         weights[2][i2] * weights[3][i3];
                        sum = Kokkos::fma(coeffs[linear], w, sum);
                    }
                }
            }
        }
        return sum;
    }

    KOKKOS_INLINE_FUNCTION
    double eval_generic(const int spans[N], const double weights[N][4]) const noexcept {
        // Generic N-dimensional evaluation using iteration
        // Uses a counter array to iterate through all 4^N combinations
        int counters[N];
        for (size_t d = 0; d < N; ++d) counters[d] = 0;

        double sum = 0.0;

        while (true) {
            // Compute indices and weight
            double w = 1.0;
            size_t linear = 0;
            size_t stride = 1;
            bool valid = true;

            for (int d = static_cast<int>(N) - 1; d >= 0; --d) {
                const int idx = spans[d] - counters[d];
                if (static_cast<unsigned>(idx) >= dims[d]) {
                    valid = false;
                    break;
                }
                linear += static_cast<size_t>(idx) * stride;
                stride *= dims[d];
                w *= weights[d][counters[d]];
            }

            if (valid) {
                sum = Kokkos::fma(coeffs[linear], w, sum);
            }

            // Increment counters (like counting in base 4)
            int carry = 1;
            for (int d = static_cast<int>(N) - 1; d >= 0 && carry; --d) {
                counters[d] += carry;
                if (counters[d] >= 4) {
                    counters[d] = 0;
                } else {
                    carry = 0;
                }
            }

            if (carry) break;  // All combinations done
        }

        return sum;
    }
};

/// Create evaluator from BSplineND for kernel use
template <typename MemSpace, size_t N>
BSplineNDEvaluator<N> make_evaluator(const BSplineND<MemSpace, N>& spline) {
    BSplineNDEvaluator<N> eval;

    for (size_t d = 0; d < N; ++d) {
        eval.grids[d] = spline.grids_raw()[d].data();
        eval.knots[d] = spline.knots_raw()[d].data();
        eval.dims[d] = spline.dims_raw()[d];
        eval.n_knots[d] = spline.n_knots_raw()[d];
    }
    eval.coeffs = spline.coeffs_raw().data();

    return eval;
}

}  // namespace mango::kokkos
