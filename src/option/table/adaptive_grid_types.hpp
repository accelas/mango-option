// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/price_table_axes.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/standard_surface.hpp"
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace mango {

/// Configuration for multi-K_ref surface construction.
/// Used by both manual and adaptive grid builders.
struct MultiKRefConfig {
    std::vector<double> K_refs;   ///< explicit list; if empty, use auto selection
    int K_ref_count = 11;         ///< used when K_refs is empty
    double K_ref_span = 0.3;      ///< +/-span around spot for auto mode (log-spaced)
};

/// Grid specification for IV solver: explicit grid points for each axis.
/// Requires >= 4 points per axis (B-spline minimum).
///
/// Defaults cover typical equity option ranges.  When used with adaptive
/// refinement the values serve as domain bounds; otherwise they are the
/// exact interpolation knots.
struct IVGrid {
    /// S/K moneyness ratio (not log).  Converted to log(S/K) internally.
    std::vector<double> moneyness = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3};
    std::vector<double> vol = {0.05, 0.10, 0.20, 0.30, 0.50};
    std::vector<double> rate = {0.01, 0.03, 0.05, 0.10};
};

/// Configuration for adaptive grid refinement
///
/// Defaults match the High accuracy profile (2 bps target).
/// See PriceTableGridProfile::High in price_table_grid_estimator.hpp.
struct AdaptiveGridParams {
    /// Target IV error in absolute terms (default: 2 bps = 2e-5, High profile)
    double target_iv_error = 2e-5;

    /// Maximum refinement iterations (default: 5)
    size_t max_iter = 5;

    /// Maximum points per dimension ceiling (default: 160, High profile)
    size_t max_points_per_dim = 160;

    /// Minimum moneyness grid points (default: 60)
    /// Moneyness requires higher density than other dimensions due to
    /// exercise boundary curvature and PDE → B-spline sampling loss.
    size_t min_moneyness_points = 60;

    /// Number of validation FD solves per iteration (default: 64)
    size_t validation_samples = 64;

    /// Grid growth factor per refinement (default: 1.3)
    double refinement_factor = 1.3;

    /// Random seed for Latin Hypercube sampling (default: 42)
    uint64_t lhs_seed = 42;

    /// Vega floor for error metric (default: 1e-4)
    /// When vega < floor, fall back to price-based tolerance
    double vega_floor = 1e-4;

    /// Maximum tolerable PDE solve failure rate (default: 0.5 = 50%)
    /// Some solves may fail at extreme parameter combinations
    double max_failure_rate = 0.5;
};

/// Per-iteration diagnostics
struct IterationStats {
    size_t iteration = 0;                    ///< Iteration number (0-indexed)
    std::array<size_t, 4> grid_sizes = {};   ///< [m, tau, sigma, r] sizes
    size_t pde_solves_table = 0;             ///< Slices computed for table
    size_t pde_solves_validation = 0;        ///< Fresh solves for validation
    double max_error = 0.0;                  ///< Max IV error observed
    double avg_error = 0.0;                  ///< Mean IV error
    int refined_dim = -1;                    ///< Which dim was refined (-1 if none)
    double elapsed_seconds = 0.0;            ///< Wall-clock time for this iteration
};

/// Final result with full diagnostics
struct AdaptiveResult {
    /// The built price table surface
    std::shared_ptr<const PriceTableSurface> surface = nullptr;

    /// Final axes used for the surface
    PriceTableAxes axes;

    /// Type-erased price function (set by Chebyshev path)
    std::function<double(double spot, double strike, double tau,
                         double sigma, double rate)> price_fn;

    /// Query price from the surface
    /// Returns NaN if no surface is available (build failure or not yet built)
    /// coords: [moneyness, tau, sigma, rate]
    [[nodiscard]] double value(const std::array<double, 4>& coords) const {
        return surface ? surface->value(coords)
                       : std::numeric_limits<double>::quiet_NaN();
    }

    /// Per-iteration history for diagnostics
    std::vector<IterationStats> iterations;

    /// Actual max IV error from final validation
    double achieved_max_error = 0.0;

    /// Actual mean IV error from final validation
    double achieved_avg_error = 0.0;

    /// True iff achieved_max_error <= target_iv_error
    bool target_met = false;

    /// Total PDE solves across all iterations (table + validation)
    size_t total_pde_solves = 0;
};

/// Result from adaptive segmented grid building (multi-K_ref path)
struct SegmentedAdaptiveResult {
    MultiKRefInner surface;
    IVGrid grid;  ///< The grid sizes adaptive chose
    int tau_points_per_segment;
};

}  // namespace mango
