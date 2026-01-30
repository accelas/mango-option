// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <ranges>

namespace mango {

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
