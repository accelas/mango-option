// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/american_option_result.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
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

/// Bin-based error attribution for adaptive grid refinement
///
/// Tracks where errors occur in each dimension to identify which
/// dimension and which region needs refinement.
struct ErrorBins {
    static constexpr size_t N_BINS = 5;
    static constexpr size_t N_DIMS = 4;

    /// Count of high-error samples in each bin for each dimension
    std::array<std::array<size_t, N_BINS>, N_DIMS> bin_counts = {};

    /// Total error mass accumulated in each dimension
    std::array<double, N_DIMS> dim_error_mass = {};

    /// Record an error at a normalized position [0,1]^4
    ///
    /// @param normalized_pos Position in [0,1]^4 (clamped if out of range)
    /// @param iv_error IV error at this point
    /// @param threshold Only record if iv_error > threshold
    void record_error(const std::array<double, N_DIMS>& normalized_pos,
                      double iv_error, double threshold) {
        if (iv_error <= threshold) {
            return;
        }

        for (size_t d = 0; d < N_DIMS; ++d) {
            // Clamp to [0, 1] and compute bin
            double pos = std::clamp(normalized_pos[d], 0.0, 1.0);
            size_t bin = static_cast<size_t>(pos * N_BINS);
            bin = std::min(bin, N_BINS - 1);  // Handle pos == 1.0

            bin_counts[d][bin]++;
            dim_error_mass[d] += iv_error;
        }
    }

    /// Find dimension with most concentrated errors
    ///
    /// Returns the dimension where errors are most localized (highest
    /// max bin count relative to total), indicating refinement will help.
    [[nodiscard]] size_t worst_dimension() const {
        double best_score = -1.0;
        size_t best_dim = 0;

        for (size_t d = 0; d < N_DIMS; ++d) {
            // Find max bin count for this dimension
            size_t max_count = std::ranges::max(bin_counts[d]);
            size_t total_count = std::reduce(bin_counts[d].begin(), bin_counts[d].end());

            if (total_count == 0) continue;

            // Score = concentration ratio * error mass
            // Higher when errors are localized AND significant
            double concentration = static_cast<double>(max_count) / static_cast<double>(total_count);
            double score = concentration * dim_error_mass[d];

            if (score > best_score) {
                best_score = score;
                best_dim = d;
            }
        }

        return best_dim;
    }

    /// Get bins with error count >= min_count for a dimension
    [[nodiscard]] std::vector<size_t> problematic_bins(size_t dim, size_t min_count = 2) const {
        auto indices = std::views::iota(size_t{0}, N_BINS)
                     | std::views::filter([&](size_t b) { return bin_counts[dim][b] >= min_count; });
        return std::ranges::to<std::vector<size_t>>(indices);
    }

    /// Clear all bins
    void reset() {
        for (auto& dim_bins : bin_counts) {
            dim_bins.fill(0);
        }
        dim_error_mass.fill(0.0);
    }
};

}  // namespace mango
