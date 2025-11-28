#pragma once

#include "src/option/table/price_table_axes.hpp"
#include "src/option/table/price_table_surface.hpp"
#include <array>
#include <memory>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace mango {

/// Configuration for adaptive grid refinement
struct AdaptiveGridParams {
    /// Target IV error in absolute terms (default: 5 bps = 0.0005)
    double target_iv_error = 0.0005;

    /// Maximum refinement iterations (default: 5)
    size_t max_iterations = 5;

    /// Maximum points per dimension ceiling (default: 50)
    size_t max_points_per_dim = 50;

    /// Number of validation FD solves per iteration (default: 64)
    size_t validation_samples = 64;

    /// Grid growth factor per refinement (default: 1.3)
    double refinement_factor = 1.3;

    /// Number of bins per dimension for error attribution (default: 5)
    size_t bins_per_dim = 5;

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
    /// The built price table surface (always populated, even if target not met)
    std::shared_ptr<const PriceTableSurface<4>> surface = nullptr;

    /// Final axes used for the surface
    PriceTableAxes<4> axes;

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

}  // namespace mango
