// SPDX-License-Identifier: MIT
#pragma once

#include "mango/support/parallel.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace mango {

/// Raw (uncompressed) N-dimensional tensor storage.
/// Stores all values in row-major order. Contraction is performed
/// via sequential axis reduction: the last axis is contracted first
/// (SIMD-friendly inner loop), then the next-to-last, and so on,
/// reducing the tensor one dimension at a time.
template <size_t N>
class RawTensor {
public:
    static RawTensor build(std::vector<double> values,
                           const std::array<size_t, N>& shape) {
        RawTensor t;
        t.values_ = std::move(values);
        t.shape_ = shape;
        return t;
    }

    /// Contract with per-axis coefficient vectors.
    /// coeffs[d] has length shape_[d].
    ///
    /// Uses sequential axis contraction from the last axis inward.
    /// The innermost loop is a dot product (SIMD + FMA friendly).
    MANGO_TARGET_CLONES("default", "avx2")
    [[nodiscard]] double
    contract(const std::array<std::vector<double>, N>& coeffs) const {
        // Sequential axis contraction: contract axis N-1 first, then N-2, ...
        // After contracting axis d, the working buffer has dimensions
        // shape[0] x ... x shape[d-1].

        // Start with a copy of values (will be reduced in-place)
        std::vector<double> buf = values_;
        size_t buf_size = values_.size();

        for (int d = static_cast<int>(N) - 1; d >= 0; --d) {
            size_t axis_len = shape_[d];
            size_t outer = buf_size / axis_len;
            const double* c = coeffs[d].data();

            std::vector<double> next(outer);
            for (size_t i = 0; i < outer; ++i) {
                const double* row = buf.data() + i * axis_len;
                double sum = 0.0;
                MANGO_PRAGMA_SIMD
                for (size_t j = 0; j < axis_len; ++j) {
                    sum = std::fma(row[j], c[j], sum);
                }
                next[i] = sum;
            }
            buf = std::move(next);
            buf_size = outer;
        }
        return buf[0];
    }

    [[nodiscard]] size_t compressed_size() const { return values_.size(); }
    [[nodiscard]] const std::array<size_t, N>& shape() const { return shape_; }

private:
    std::vector<double> values_;
    std::array<size_t, N> shape_{};
};

}  // namespace mango
