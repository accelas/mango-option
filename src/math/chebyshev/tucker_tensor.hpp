// SPDX-License-Identifier: MIT
#pragma once

#include "mango/support/parallel.hpp"

#include <Eigen/Dense>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace mango {

/// Result of a Tucker (HOSVD) decomposition of an N-dimensional tensor.
template <size_t N>
struct TuckerResult {
    std::vector<double> core;
    std::array<Eigen::MatrixXd, N> factors;
    std::array<size_t, N> shape;
    std::array<size_t, N> ranks;
};

// ---------------------------------------------------------------------------
// mode_unfold<N>: Generic mode-n unfolding of a row-major tensor.
//
// For a tensor T with shape [d0, d1, ..., d_{N-1}] stored in row-major
// (C-order), mode_unfold(T, shape, m) returns a matrix M where:
//   - Row = subscript[m]
//   - Column = remaining subscripts packed right-to-left (skipping mode m)
//
// This matches the column ordering in the existing 3D/4D implementations.
// ---------------------------------------------------------------------------
template <size_t N>
[[nodiscard]] Eigen::MatrixXd
mode_unfold(const std::vector<double>& T,
            const std::array<size_t, N>& shape,
            size_t mode) {
    static_assert(N >= 2, "Tucker decomposition requires N >= 2");
    assert(mode < N);

    size_t n_rows = shape[mode];
    size_t total = 1;
    for (size_t d = 0; d < N; ++d) total *= shape[d];
    size_t n_cols = total / n_rows;

    // Precompute row-major strides for the full tensor
    std::array<size_t, N> strides{};
    strides[N - 1] = 1;
    for (int d = static_cast<int>(N) - 2; d >= 0; --d)
        strides[d] = strides[d + 1] * shape[d + 1];

    Eigen::MatrixXd M(static_cast<Eigen::Index>(n_rows),
                      static_cast<Eigen::Index>(n_cols));

    for (size_t flat = 0; flat < total; ++flat) {
        // Decompose flat index into N-dim subscript
        std::array<size_t, N> idx{};
        size_t remaining = flat;
        for (size_t d = 0; d < N; ++d) {
            idx[d] = remaining / strides[d];
            remaining %= strides[d];
        }

        size_t row = idx[mode];

        // Column: pack remaining indices right-to-left (skipping mode)
        size_t col = 0;
        size_t stride = 1;
        for (int d = static_cast<int>(N) - 1; d >= 0; --d) {
            if (static_cast<size_t>(d) == mode) continue;
            col += idx[d] * stride;
            stride *= shape[d];
        }

        M(static_cast<Eigen::Index>(row),
          static_cast<Eigen::Index>(col)) = T[flat];
    }
    return M;
}

// ---------------------------------------------------------------------------
// repack_after_contraction: Given the SVD result G = U^T * M_unfold (where
// M_unfold is the mode-d unfolding), repack into a row-major vector with
// the contracted mode's dimension replaced by its rank.
//
// G has shape (rank_d, n_cols) where n_cols = product of all dims except d,
// with column ordering from mode_unfold (right-to-left skipping mode d).
//
// We need to produce a row-major vector with the updated shape (where
// shape[d] has been replaced by rank_d).
// ---------------------------------------------------------------------------
namespace detail {

template <size_t N>
std::vector<double>
repack_after_contraction(const Eigen::MatrixXd& G,
                         const std::array<size_t, N>& old_shape,
                         size_t mode,
                         size_t rank) {
    // Build the new shape
    std::array<size_t, N> new_shape = old_shape;
    new_shape[mode] = rank;

    size_t new_total = 1;
    for (size_t d = 0; d < N; ++d) new_total *= new_shape[d];

    // Compute strides for the new row-major layout
    std::array<size_t, N> new_strides{};
    new_strides[N - 1] = 1;
    for (int d = static_cast<int>(N) - 2; d >= 0; --d)
        new_strides[d] = new_strides[d + 1] * new_shape[d + 1];

    // Compute the column strides from the mode_unfold column ordering.
    // Columns are packed right-to-left skipping mode d:
    //   col = sum over d'!=mode of idx[d'] * col_stride[d']
    // where col_stride is accumulated right-to-left.
    std::array<size_t, N> col_strides{};
    {
        size_t s = 1;
        for (int d = static_cast<int>(N) - 1; d >= 0; --d) {
            if (static_cast<size_t>(d) == mode) {
                col_strides[d] = 0;  // not used
                continue;
            }
            col_strides[d] = s;
            s *= old_shape[d];
        }
    }

    size_t n_cols = static_cast<size_t>(G.cols());

    // Compute strides for decoding columns back to non-mode indices.
    // We need to reverse the right-to-left packing to recover each idx[d].
    // The non-mode dimensions, from right to left, contribute:
    //   idx[d_last] * 1, idx[d_prev] * shape[d_last], ...
    // Build an ordered list of (dimension, cumulative_size) for decoding.
    struct DimInfo {
        size_t dim;
        size_t size;
    };
    // Collect non-mode dimensions in right-to-left order (matching column packing)
    std::array<DimInfo, N - 1> decode_order{};
    size_t n_decode = 0;
    for (int d = static_cast<int>(N) - 1; d >= 0; --d) {
        if (static_cast<size_t>(d) == mode) continue;
        decode_order[n_decode++] = {static_cast<size_t>(d), old_shape[d]};
    }

    // Decode strides: the first (rightmost) dimension has stride 1,
    // next has stride = size of first, etc. These are cumulative products.
    std::array<size_t, N - 1> decode_strides{};
    if (n_decode > 0) {
        decode_strides[0] = 1;
        for (size_t i = 1; i < n_decode; ++i)
            decode_strides[i] = decode_strides[i - 1] * decode_order[i - 1].size;
    }

    std::vector<double> result(new_total);

    for (size_t r = 0; r < rank; ++r) {
        for (size_t j = 0; j < n_cols; ++j) {
            // Decode column j to get non-mode subscripts
            std::array<size_t, N> idx{};
            idx[mode] = r;

            size_t rem = j;
            // Decode from highest stride to lowest
            for (int i = static_cast<int>(n_decode) - 1; i >= 0; --i) {
                idx[decode_order[i].dim] = rem / decode_strides[i];
                rem %= decode_strides[i];
            }

            // Compute flat index in new row-major layout
            size_t flat = 0;
            for (size_t d = 0; d < N; ++d)
                flat += idx[d] * new_strides[d];

            result[flat] = G(static_cast<Eigen::Index>(r),
                             static_cast<Eigen::Index>(j));
        }
    }

    return result;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// tucker_hosvd<N>: Generic Higher-Order SVD (truncated Tucker decomposition).
//
// Algorithm:
// 1. For each mode 0..N-1: unfold, SVD, truncate via epsilon threshold.
// 2. Sequential contraction: for each mode, contract with U^T, repack to
//    row-major with updated shape.
// ---------------------------------------------------------------------------
template <size_t N>
[[nodiscard]] TuckerResult<N>
tucker_hosvd(const std::vector<double>& T,
             const std::array<size_t, N>& shape,
             double epsilon) {
    static_assert(N >= 2, "Tucker decomposition requires N >= 2");

    TuckerResult<N> result;
    result.shape = shape;

    // Step 1: Compute truncated SVD for each mode
    for (size_t mode = 0; mode < N; ++mode) {
        Eigen::MatrixXd M = mode_unfold<N>(T, shape, mode);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU);
        const auto& sigma = svd.singularValues();
        double sigma_0 = sigma(0);
        size_t rank = 1;
        for (Eigen::Index i = 1; i < sigma.size(); ++i) {
            if (sigma(i) / sigma_0 >= epsilon)
                rank++;
            else
                break;
        }
        result.ranks[mode] = rank;
        result.factors[mode] = svd.matrixU().leftCols(
            static_cast<Eigen::Index>(rank));
    }

    // Step 2: Sequential contraction
    // Working data starts as the original tensor
    std::vector<double> working = T;
    std::array<size_t, N> working_shape = shape;

    for (size_t mode = 0; mode < N; ++mode) {
        // Unfold along current mode
        Eigen::MatrixXd M = mode_unfold<N>(working, working_shape, mode);

        // Contract: G = U^T * M
        Eigen::MatrixXd G = result.factors[mode].transpose() * M;

        // Repack G into row-major with updated shape
        working = detail::repack_after_contraction<N>(
            G, working_shape, mode, result.ranks[mode]);
        working_shape[mode] = result.ranks[mode];
    }

    result.core = std::move(working);
    return result;
}

// ---------------------------------------------------------------------------
// tucker_reconstruct<N>: Reconstruct full tensor from Tucker decomposition.
// Used for testing round-trip accuracy.
// ---------------------------------------------------------------------------
template <size_t N>
[[nodiscard]] std::vector<double>
tucker_reconstruct(const TuckerResult<N>& tucker) {
    size_t total = 1;
    for (size_t d = 0; d < N; ++d) total *= tucker.shape[d];

    size_t core_total = 1;
    for (size_t d = 0; d < N; ++d) core_total *= tucker.ranks[d];

    // Precompute strides for output tensor
    std::array<size_t, N> out_strides{};
    out_strides[N - 1] = 1;
    for (int d = static_cast<int>(N) - 2; d >= 0; --d)
        out_strides[d] = out_strides[d + 1] * tucker.shape[d + 1];

    // Precompute strides for core tensor
    std::array<size_t, N> core_strides{};
    core_strides[N - 1] = 1;
    for (int d = static_cast<int>(N) - 2; d >= 0; --d)
        core_strides[d] = core_strides[d + 1] * tucker.ranks[d + 1];

    std::vector<double> T(total, 0.0);

    // For each output element
    for (size_t out_flat = 0; out_flat < total; ++out_flat) {
        // Decompose into subscripts
        std::array<size_t, N> idx{};
        size_t rem = out_flat;
        for (size_t d = 0; d < N; ++d) {
            idx[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }

        // Sum over all core elements
        double val = 0.0;
        for (size_t core_flat = 0; core_flat < core_total; ++core_flat) {
            // Decompose core index into rank subscripts
            std::array<size_t, N> ridx{};
            size_t crem = core_flat;
            for (size_t d = 0; d < N; ++d) {
                ridx[d] = crem / core_strides[d];
                crem %= core_strides[d];
            }

            double product = tucker.core[core_flat];
            for (size_t d = 0; d < N; ++d)
                product *= tucker.factors[d](
                    static_cast<Eigen::Index>(idx[d]),
                    static_cast<Eigen::Index>(ridx[d]));
            val += product;
        }
        T[out_flat] = val;
    }

    return T;
}

// ---------------------------------------------------------------------------
// TuckerTensor<N>: Compressed N-dimensional tensor via Tucker decomposition.
//
// Supports contraction with per-axis coefficient vectors, analogous to
// RawTensor but with compressed storage.
// ---------------------------------------------------------------------------
template <size_t N>
class TuckerTensor {
public:
    /// Build a TuckerTensor from a flat row-major tensor.
    /// epsilon controls SVD truncation: singular values with
    /// sigma_i / sigma_0 < epsilon are discarded.
    static TuckerTensor build(std::vector<double> values,
                              const std::array<size_t, N>& shape,
                              double epsilon) {
        auto result = tucker_hosvd<N>(values, shape, epsilon);
        TuckerTensor t;
        t.core_ = std::move(result.core);
        t.factors_ = std::move(result.factors);
        t.shape_ = result.shape;
        t.ranks_ = result.ranks;
        return t;
    }

    /// Contract with per-axis coefficient vectors (length shape[d] each).
    /// Internally: projected[d] = factors[d]^T * coeffs[d] (length ranks[d])
    /// Then contract core tensor via sequential axis reduction with SIMD.
    MANGO_TARGET_CLONES("default", "avx2")
    [[nodiscard]] double
    contract(const std::array<std::vector<double>, N>& coeffs) const {
        // Step 1: Project each coefficient vector into the rank-space
        // projected[d] = U_d^T * coeffs[d], length ranks_[d]
        std::array<std::vector<double>, N> projected;
        for (size_t d = 0; d < N; ++d) {
            size_t r = ranks_[d];
            size_t n = shape_[d];
            projected[d].resize(r);
            const auto& U = factors_[d];
            const double* c = coeffs[d].data();
            for (size_t i = 0; i < r; ++i) {
                double sum = 0.0;
                MANGO_PRAGMA_SIMD
                for (size_t j = 0; j < n; ++j) {
                    sum = std::fma(
                        U(static_cast<Eigen::Index>(j),
                          static_cast<Eigen::Index>(i)),
                        c[j], sum);
                }
                projected[d][i] = sum;
            }
        }

        // Step 2: Sequential axis contraction on the core tensor
        // Same pattern as RawTensor: contract last axis first, then next, ...
        std::vector<double> buf = core_;
        size_t buf_size = core_.size();

        for (int d = static_cast<int>(N) - 1; d >= 0; --d) {
            size_t axis_len = ranks_[d];
            size_t outer = buf_size / axis_len;
            const double* p = projected[d].data();

            std::vector<double> next(outer);
            for (size_t i = 0; i < outer; ++i) {
                const double* row = buf.data() + i * axis_len;
                double sum = 0.0;
                MANGO_PRAGMA_SIMD
                for (size_t j = 0; j < axis_len; ++j) {
                    sum = std::fma(row[j], p[j], sum);
                }
                next[i] = sum;
            }
            buf = std::move(next);
            buf_size = outer;
        }
        return buf[0];
    }

    /// Total number of stored doubles (core + all factor matrices).
    [[nodiscard]] size_t compressed_size() const {
        size_t total = core_.size();
        for (size_t d = 0; d < N; ++d)
            total += static_cast<size_t>(factors_[d].rows() * factors_[d].cols());
        return total;
    }

    [[nodiscard]] std::array<size_t, N> ranks() const { return ranks_; }

private:
    std::vector<double> core_;
    std::array<Eigen::MatrixXd, N> factors_;
    std::array<size_t, N> shape_{};
    std::array<size_t, N> ranks_{};
};

}  // namespace mango
