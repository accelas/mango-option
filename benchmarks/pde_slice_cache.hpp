// SPDX-License-Identifier: MIT
#pragma once

#include "mango/math/cubic_spline_solver.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace mango {

/// Cache of PDE solutions keyed by quantized (sigma, rate) physical values
/// and tau_index.  Each slice is a CubicSpline over the spatial (x) grid at
/// one snapshot time.  Keys use quantized doubles (1e-12 precision) so that
/// the same physical node always maps to the same key regardless of which
/// CC level it appears at.
class PDESliceCache {
public:
    /// Quantize a double to a reproducible integer key.
    static int64_t quantize(double v) {
        return static_cast<int64_t>(std::round(v * 1e12));
    }

    using Key = std::pair<int64_t, int64_t>;  // quantized (sigma, rate)

    /// Store a spline for (sigma, rate, tau_idx).
    void store_slice(double sigma, double rate, size_t tau_idx,
                     std::span<const double> x_grid,
                     std::span<const double> values) {
        auto key = make_key(sigma, rate);
        auto& tau_map = slices_[key];
        auto& entry = tau_map[tau_idx];
        auto err = entry.spline.build(x_grid, values);
        entry.valid = !err.has_value();
    }

    /// Check if any tau slices exist for (sigma, rate).
    [[nodiscard]] bool has_slice(double sigma, double rate) const {
        return slices_.contains(make_key(sigma, rate));
    }

    /// Retrieve the spline for (sigma, rate, tau_idx), or nullptr.
    [[nodiscard]] const CubicSpline<double>*
    get_slice(double sigma, double rate, size_t tau_idx) const {
        auto it = slices_.find(make_key(sigma, rate));
        if (it == slices_.end()) return nullptr;
        auto jt = it->second.find(tau_idx);
        if (jt == it->second.end() || !jt->second.valid) return nullptr;
        return &jt->second.spline;
    }

    /// Number of distinct (sigma, rate) pairs cached.
    [[nodiscard]] size_t num_cached_pairs() const { return slices_.size(); }

    /// Number of tau slices stored for a given (sigma, rate) pair.
    [[nodiscard]] size_t num_tau_slices(double sigma, double rate) const {
        auto it = slices_.find(make_key(sigma, rate));
        if (it == slices_.end()) return 0;
        return it->second.size();
    }

    /// Given wanted sigma values and rate values, return (sigma_idx, rate_idx)
    /// pairs (indices into the input arrays) that are NOT yet cached.
    [[nodiscard]] std::vector<std::pair<size_t, size_t>>
    missing_pairs(std::span<const double> sigma_values,
                  std::span<const double> rate_values) const {
        std::vector<std::pair<size_t, size_t>> result;
        for (size_t s = 0; s < sigma_values.size(); ++s) {
            for (size_t r = 0; r < rate_values.size(); ++r) {
                if (!slices_.contains(make_key(sigma_values[s], rate_values[r]))) {
                    result.push_back({s, r});
                }
            }
        }
        return result;
    }

    /// Record cumulative PDE solve count.
    void record_pde_solves(size_t count) { total_pde_solves_ += count; }

    /// Total PDE solves performed to populate this cache.
    [[nodiscard]] size_t total_pde_solves() const { return total_pde_solves_; }

    /// Clear all cached slices.
    void clear() {
        slices_.clear();
        total_pde_solves_ = 0;
    }

private:
    static Key make_key(double sigma, double rate) {
        return {quantize(sigma), quantize(rate)};
    }

    struct SliceEntry {
        CubicSpline<double> spline;
        bool valid = false;
    };

    std::map<Key, std::map<size_t, SliceEntry>> slices_;
    size_t total_pde_solves_ = 0;
};

}  // namespace mango
