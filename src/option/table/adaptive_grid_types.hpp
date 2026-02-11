// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"
#include <array>
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
/// Requires >= 4 points per axis (interpolation minimum).
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
    /// exercise boundary curvature and interpolation sampling loss.
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

/// Configuration for segmented adaptive grid building
struct SegmentedAdaptiveConfig {
    double spot;
    OptionType option_type;
    double dividend_yield;
    std::vector<Dividend> discrete_dividends;
    double maturity;
    MultiKRefConfig kref_config;
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

}  // namespace mango
