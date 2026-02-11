// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace mango {

/// Quantized (sigma, rate) key for PDE slice caches.
struct PairKey {
    int64_t sigma_q;
    int64_t rate_q;
    auto operator<=>(const PairKey&) const = default;
};

/// Codec for quantizing double (sigma, rate) pairs into PairKeys.
/// The scale controls precision: 1e6 for B-spline (6 decimal places),
/// 1e12 for Chebyshev (CC-level node reuse stability).
class PairKeyCodec {
public:
    explicit PairKeyCodec(double scale) : scale_(scale) {}

    [[nodiscard]] PairKey make(double sigma, double rate) const {
        return {
            static_cast<int64_t>(std::round(sigma * scale_)),
            static_cast<int64_t>(std::round(rate * scale_))
        };
    }

private:
    double scale_;
};

/// Compare two tau grids for approximate equality.
/// Used for cache invalidation when the tau grid changes.
inline bool tau_grids_equal(const std::vector<double>& a,
                            const std::vector<double>& b,
                            double tol = 1e-10) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

/// Generic (sigma, rate)-keyed cache core.
///
/// PairValue is the stored type per (sigma, rate) pair.
/// The codec controls quantization precision.
template <class PairValue>
class PairCacheCore {
public:
    explicit PairCacheCore(PairKeyCodec codec) : codec_(codec) {}

    [[nodiscard]] bool contains(double sigma, double rate) const {
        return data_.contains(codec_.make(sigma, rate));
    }

    [[nodiscard]] PairValue* get(double sigma, double rate) {
        auto it = data_.find(codec_.make(sigma, rate));
        return it != data_.end() ? &it->second : nullptr;
    }

    [[nodiscard]] const PairValue* get(double sigma, double rate) const {
        auto it = data_.find(codec_.make(sigma, rate));
        return it != data_.end() ? &it->second : nullptr;
    }

    /// Insert-or-access: returns reference to existing or default-constructed value.
    PairValue& upsert(double sigma, double rate) {
        return data_[codec_.make(sigma, rate)];
    }

    void clear() { data_.clear(); }

    [[nodiscard]] size_t size() const { return data_.size(); }

    /// Given wanted sigma and rate arrays, return (sigma_idx, rate_idx) pairs
    /// for combinations NOT yet cached.
    [[nodiscard]] std::vector<std::pair<size_t, size_t>>
    missing_pairs(std::span<const double> sigma_values,
                  std::span<const double> rate_values) const {
        std::vector<std::pair<size_t, size_t>> result;
        for (size_t s = 0; s < sigma_values.size(); ++s) {
            for (size_t r = 0; r < rate_values.size(); ++r) {
                if (!data_.contains(codec_.make(sigma_values[s], rate_values[r]))) {
                    result.push_back({s, r});
                }
            }
        }
        return result;
    }

private:
    PairKeyCodec codec_;
    std::map<PairKey, PairValue> data_;
};

}  // namespace mango
