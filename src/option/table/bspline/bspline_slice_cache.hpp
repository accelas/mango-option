// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/american_option_result.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace mango {

/// Cache for (sigma, r) -> AmericanOptionResult mappings
///
/// Used by AdaptiveGridBuilder to avoid re-solving PDE for unchanged slices.
/// Cache is invalidated when tau grid changes (PDE solve depends on maturity).
class SliceCache {
public:
    /// Add a result to the cache (takes ownership via shared_ptr)
    void add(double sigma, double rate, std::shared_ptr<AmericanOptionResult> result) {
        results_[make_key(sigma, rate)] = std::move(result);
    }

    /// Get a result from the cache
    [[nodiscard]] std::shared_ptr<AmericanOptionResult> get(double sigma, double rate) const {
        auto it = results_.find(make_key(sigma, rate));
        if (it != results_.end()) {
            return it->second;
        }
        return nullptr;
    }

    /// Set the current tau grid for invalidation checking
    void set_tau_grid(const std::vector<double>& tau_grid) {
        current_tau_grid_ = tau_grid;
    }

    /// Invalidate cache if tau grid changed
    void invalidate_if_tau_changed(const std::vector<double>& new_tau) {
        if (!grids_equal(current_tau_grid_, new_tau)) {
            results_.clear();
            current_tau_grid_ = new_tau;
        }
    }

    /// Get pairs that are NOT in the cache
    [[nodiscard]] std::vector<std::pair<double, double>> get_missing_pairs(
        const std::vector<std::pair<double, double>>& all_pairs) const
    {
        std::vector<std::pair<double, double>> missing;
        for (const auto& p : all_pairs) {
            if (results_.find(make_key(p.first, p.second)) == results_.end()) {
                missing.push_back(p);
            }
        }
        return missing;
    }

    /// Get indices of pairs that are NOT in the cache
    [[nodiscard]] std::vector<size_t> get_missing_indices(
        const std::vector<std::pair<double, double>>& all_pairs) const
    {
        std::vector<size_t> indices;
        for (size_t i = 0; i < all_pairs.size(); ++i) {
            if (results_.find(make_key(all_pairs[i].first, all_pairs[i].second)) == results_.end()) {
                indices.push_back(i);
            }
        }
        return indices;
    }

    /// Check if a pair exists in cache
    [[nodiscard]] bool contains(double sigma, double rate) const {
        return results_.find(make_key(sigma, rate)) != results_.end();
    }

    /// Get number of cached results
    [[nodiscard]] size_t size() const { return results_.size(); }

    /// Clear all cached results
    void clear() {
        results_.clear();
        current_tau_grid_.clear();
    }

private:
    /// Key for the map - use pair of sigma, rate
    using Key = std::pair<double, double>;

    static Key make_key(double sigma, double rate) {
        // Round to avoid floating point comparison issues
        // 6 decimal places is enough precision for vol/rate
        auto round6 = [](double x) {
            return std::round(x * 1e6) / 1e6;
        };
        return {round6(sigma), round6(rate)};
    }

    static bool grids_equal(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(a[i] - b[i]) > 1e-10) return false;
        }
        return true;
    }

    std::map<Key, std::shared_ptr<AmericanOptionResult>> results_;
    std::vector<double> current_tau_grid_;
};

}  // namespace mango
