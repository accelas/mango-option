// SPDX-License-Identifier: MIT
#pragma once

#include "mango/math/cubic_spline_solver.hpp"

#include <cstddef>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace mango {

/// Cache of PDE solutions keyed by (sigma_index, rate_index, tau_index).
/// Each slice is a CubicSpline over the spatial (x) grid at one snapshot time.
/// Supports incremental population: solve new (sigma, rate) pairs and add them
/// without re-solving existing pairs.
class PDESliceCache {
public:
    using Key = std::pair<size_t, size_t>;  // (sigma_idx, rate_idx)

    /// Store a spline for (sigma_idx, rate_idx, tau_idx).
    void store_slice(size_t sigma_idx, size_t rate_idx, size_t tau_idx,
                     std::span<const double> x_grid,
                     std::span<const double> values) {
        auto& tau_map = slices_[{sigma_idx, rate_idx}];
        auto& entry = tau_map[tau_idx];
        auto err = entry.spline.build(x_grid, values);
        entry.valid = !err.has_value();
    }

    /// Check if any tau slices exist for (sigma_idx, rate_idx).
    [[nodiscard]] bool has_slice(size_t sigma_idx, size_t rate_idx) const {
        return slices_.contains({sigma_idx, rate_idx});
    }

    /// Retrieve the spline for (sigma_idx, rate_idx, tau_idx), or nullptr.
    [[nodiscard]] const CubicSpline<double>*
    get_slice(size_t sigma_idx, size_t rate_idx, size_t tau_idx) const {
        auto it = slices_.find({sigma_idx, rate_idx});
        if (it == slices_.end()) return nullptr;
        auto jt = it->second.find(tau_idx);
        if (jt == it->second.end() || !jt->second.valid) return nullptr;
        return &jt->second.spline;
    }

    /// Number of distinct (sigma, rate) pairs cached.
    [[nodiscard]] size_t num_cached_pairs() const { return slices_.size(); }

    /// Number of tau slices stored for a given (sigma, rate) pair.
    [[nodiscard]] size_t num_tau_slices(size_t sigma_idx, size_t rate_idx) const {
        auto it = slices_.find({sigma_idx, rate_idx});
        if (it == slices_.end()) return 0;
        return it->second.size();
    }

    /// Given wanted sigma indices and rate indices, return (sigma_idx, rate_idx)
    /// pairs that are NOT yet cached.
    [[nodiscard]] std::vector<Key>
    missing_pairs(std::span<const size_t> sigma_indices,
                  std::span<const size_t> rate_indices) const {
        std::vector<Key> result;
        for (size_t s : sigma_indices) {
            for (size_t r : rate_indices) {
                if (!slices_.contains({s, r})) {
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
    struct SliceEntry {
        CubicSpline<double> spline;
        bool valid = false;
    };

    std::map<Key, std::map<size_t, SliceEntry>> slices_;
    size_t total_pde_solves_ = 0;
};

}  // namespace mango
