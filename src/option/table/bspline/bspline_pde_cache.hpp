// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/american_option_result.hpp"
#include "mango/option/table/pde_cache.hpp"
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace mango {

/// (sigma, rate) -> AmericanOptionResult cache for B-spline adaptive builder.
///
/// Wraps PairCacheCore with tau-grid invalidation: when the tau grid changes,
/// all cached PDE solutions become stale and are cleared.
/// Uses 1e6 quantization (6 decimal places) to match B-spline grid precision.
class BSplinePDECache {
public:
    BSplinePDECache() : core_(PairKeyCodec{1e6}) {}

    void add(double sigma, double rate, std::shared_ptr<AmericanOptionResult> result) {
        core_.upsert(sigma, rate) = std::move(result);
    }

    [[nodiscard]] std::shared_ptr<AmericanOptionResult> get(double sigma, double rate) const {
        auto* p = core_.get(sigma, rate);
        return p ? *p : nullptr;
    }

    [[nodiscard]] bool contains(double sigma, double rate) const {
        return core_.contains(sigma, rate);
    }

    [[nodiscard]] size_t size() const { return core_.size(); }

    void clear() {
        core_.clear();
        current_tau_grid_.clear();
    }

    /// Set the current tau grid for invalidation checking.
    void set_tau_grid(const std::vector<double>& tau_grid) {
        current_tau_grid_ = tau_grid;
    }

    /// Clear cache if tau grid changed since last set_tau_grid / invalidate call.
    void invalidate_if_tau_changed(const std::vector<double>& new_tau) {
        if (!tau_grids_equal(current_tau_grid_, new_tau)) {
            core_.clear();
            current_tau_grid_ = new_tau;
        }
    }

    /// Get pairs that are NOT in the cache.
    [[nodiscard]] std::vector<std::pair<double, double>> get_missing_pairs(
        const std::vector<std::pair<double, double>>& all_pairs) const
    {
        std::vector<std::pair<double, double>> missing;
        for (const auto& [s, r] : all_pairs) {
            if (!core_.contains(s, r)) {
                missing.emplace_back(s, r);
            }
        }
        return missing;
    }

    /// Get indices of pairs that are NOT in the cache.
    [[nodiscard]] std::vector<size_t> get_missing_indices(
        const std::vector<std::pair<double, double>>& all_pairs) const
    {
        std::vector<size_t> indices;
        for (size_t i = 0; i < all_pairs.size(); ++i) {
            if (!core_.contains(all_pairs[i].first, all_pairs[i].second)) {
                indices.push_back(i);
            }
        }
        return indices;
    }

private:
    PairCacheCore<std::shared_ptr<AmericanOptionResult>> core_;
    std::vector<double> current_tau_grid_;
};

}  // namespace mango
