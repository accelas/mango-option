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
    std::shared_ptr<const BSplineND<double, 4>> spline;
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

/// Builds a single-reference segmented surface from raw fixed-expiry PDE
/// snapshots. All leaves store V/K_ref; fitted values never feed a PDE solve.
/// Event neighborhoods without both calendar sides are explicitly excluded.
class SegmentedPriceTableBuilder {
public:
    struct Config {
        double K_ref;
        OptionType option_type;
        DividendSpec dividends;  ///< Continuous yield + discrete schedule

        /// Grid specification:
        /// - grid.moneyness: exact log-moneyness sites ln(S/K_ref)
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

        /// PDE grid accuracy for the fixed-expiry solve cohort.
        /// Default GridAccuracyParams{} gives ~100 spatial points.
        GridAccuracyParams pde_accuracy = {};

        /// Exact physical remaining-maturity coordinates. When nonempty,
        /// replaces count-based placement and must include every supported
        /// segment's endpoints and at least four nodes per segment. Event
        /// gaps are excluded; coordinates are never moved or discarded.
        std::vector<double> tau_grid{};
    };

    /// Counts describe requested (tau, sigma, rate) spatial rows, including
    /// the analytic payoff row. Missing rows are refused before fitting.
    struct BuildResult {
        BSplineSegmentedSurface surface;
        size_t pde_solves;
        size_t sample_rows;
        size_t sample_points;
        size_t tau_point_cap_hits;
    };

    static std::expected<BuildResult, PriceTableError>
    build_with_diagnostics(const Config& config);

    /// Resolve exact physical sampling coordinates without solving the PDE.
    /// Shared by adaptive seeding and construction; preserves event ownership.
    static std::expected<std::vector<double>, PriceTableError>
    make_tau_grid(const Config& config);

    /// Build a SegmentedSurface from the given configuration.
    ///
    /// Algorithm:
    ///   1. Filter dividends outside (0, T), sort, compute segment boundaries in τ.
    ///   2. Preserve supplied fit axes; estimate independent PDE spatial coverage.
    ///   3. Solve each (sigma, rate) end to end with exact mandatory samples.
    ///   4. Fit temporal regimes from raw snapshots, refusing missing rows.
    ///   5. Assemble into SegmentedSurface.
    static std::expected<BSplineSegmentedSurface, PriceTableError> build(const Config& config);


};

}  // namespace mango
