// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace mango {

/// Raw (uncompressed) N-dimensional tensor storage.
/// Stores all values in row-major order. Contraction iterates
/// all elements weighted by the product of per-axis coefficients.
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
    [[nodiscard]] double
    contract(const std::array<std::vector<double>, N>& coeffs) const {
        size_t total = values_.size();
        double result = 0.0;

        // Precompute strides
        std::array<size_t, N> strides{};
        strides[N - 1] = 1;
        for (int d = static_cast<int>(N) - 2; d >= 0; --d)
            strides[d] = strides[d + 1] * shape_[d + 1];

        for (size_t flat = 0; flat < total; ++flat) {
            double weight = values_[flat];
            size_t remaining = flat;
            for (size_t d = 0; d < N; ++d) {
                size_t idx = remaining / strides[d];
                remaining %= strides[d];
                weight *= coeffs[d][idx];
            }
            result += weight;
        }
        return result;
    }

    [[nodiscard]] size_t compressed_size() const { return values_.size(); }
    [[nodiscard]] const std::array<size_t, N>& shape() const { return shape_; }

private:
    std::vector<double> values_;
    std::array<size_t, N> shape_{};
};

}  // namespace mango
