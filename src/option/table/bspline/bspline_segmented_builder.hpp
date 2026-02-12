// SPDX-License-Identifier: MIT
#pragma once

#include <expected>
#include <memory>
#include <vector>
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/grid_spec_types.hpp"
#include "mango/support/error_types.hpp"

namespace mango {

// ===========================================================================
// Segmented surface assembly
// ===========================================================================

struct BSplineSegmentConfig {
    std::shared_ptr<const PriceTableSurface> surface;
    double tau_start;
    double tau_end;
};

struct BSplineSegmentedConfig {
    std::vector<BSplineSegmentConfig> segments;
    double K_ref;
};

[[nodiscard]] std::expected<BSplineSegmentedSurface, PriceTableError>
build_segmented_surface(BSplineSegmentedConfig config);

// ===========================================================================
// Multi-K_ref surface assembly
// ===========================================================================

struct BSplineMultiKRefEntry {
    double K_ref;
    BSplineSegmentedSurface surface;
};

[[nodiscard]] std::expected<BSplineMultiKRefInner, PriceTableError>
build_multi_kref_surface(std::vector<BSplineMultiKRefEntry> entries);

/// Orchestrates backward-chained construction of a SegmentedSurface for a
/// single K_ref.  Splits maturity at discrete dividend dates. All segments use
/// NormalizedPrice mode (V/K_ref). The last segment uses payoff IC; earlier
/// segments chain from the previous segment's surface.
class SegmentedPriceTableBuilder {
public:
    struct Config {
        double K_ref;
        OptionType option_type;
        DividendSpec dividends;  ///< Continuous yield + discrete schedule

        /// Grid specification:
        /// - grid.moneyness: log-moneyness ln(S/K_ref)
        /// - grid.vol: volatility
        /// - grid.rate: rate
        IVGrid grid;

        double maturity;  // T in years

        /// Minimum tau points per segment (actual count may be higher)
        int tau_points_per_segment = 5;

        /// Target dt between tau grid points.
        /// When > 0, each segment gets ceil(width / tau_target_dt) + 1 points
        /// (clamped to [tau_points_min, tau_points_max]).
        /// When == 0, falls back to constant tau_points_per_segment.
        double tau_target_dt = 0.0;
        int tau_points_min = 4;   ///< B-spline minimum
        int tau_points_max = 30;  ///< Cap for very wide segments

        /// PDE grid accuracy for each segment's PDE solve.
        /// Default GridAccuracyParams{} gives ~100 spatial points.
        GridAccuracyParams pde_accuracy = {};
    };

    /// Build a SegmentedSurface from the given configuration.
    ///
    /// Algorithm:
    ///   1. Filter dividends outside (0, T), sort, compute segment boundaries in τ.
    ///   2. Expand moneyness grid downward to accommodate spot adjustment.
    ///   3. Build last segment (closest to expiry) with payoff IC.
    ///   4. Build earlier segments backward with chained IC.
    ///   5. Assemble into SegmentedSurface.
    static std::expected<BSplineSegmentedSurface, PriceTableError> build(const Config& config);

private:
    /// Build a single segment of the segmented price surface.
    ///
    /// For segment 0 (closest to expiry), uses payoff IC.
    /// For later segments, chains from the previous segment's surface
    /// with a dividend-adjusted initial condition.
    ///
    /// On success, updates @p prev_surface to the newly built segment's surface.
    static std::expected<BSplineSegmentConfig, PriceTableError>
    build_segment(
        size_t seg_idx,
        const std::vector<double>& boundaries,
        const Config& config,
        const std::vector<double>& expanded_log_m_grid,
        double K_ref,
        const std::vector<Dividend>& dividends,
        std::shared_ptr<const PriceTableSurface>& prev_surface);
};

}  // namespace mango
