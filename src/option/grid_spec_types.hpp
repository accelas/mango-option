// SPDX-License-Identifier: MIT
#pragma once
#include "mango/pde/core/grid.hpp"
#include <cmath>
#include <variant>
#include <vector>

namespace mango {

/// Equidistribution-optimal sinh concentration for a given domain half-width.
/// Derived from matching sinh density to Black-Scholes gamma profile:
///   α = 2 · arcsinh(n_σ / √2)
constexpr double optimal_sinh_alpha(double n_sigma) {
    // std::asinh/std::sqrt are not constexpr in C++23 standard,
    // but we use constexpr as a hint; works at runtime regardless.
    return 2.0 * std::asinh(n_sigma / std::sqrt(2.0));
}

enum class GridAccuracyProfile { Low, Medium, High, Ultra };

struct GridAccuracyParams {
    /// Domain half-width in units of σ√T.
    /// 5.0 covers ±5 std devs (99.99994% of log-normal density).
    /// Conservative heuristic — not derived from error analysis.
    /// Smaller values (3-4) save points but risk boundary error on
    /// long-dated or high-vol options; larger values (6+) waste points
    /// in regions where the solution equals the boundary value.
    double n_sigma = 5.0;

    /// Sinh clustering strength (default: equidistribution-optimal for n_sigma)
    /// α = 2 · arcsinh(n_σ / √2) ≈ 3.95 for n_sigma=5.0
    double alpha = optimal_sinh_alpha(5.0);

    /// Target spatial truncation error (default: 1e-2 for ~1e-3 price accuracy)
    /// - 1e-2: Fast mode (~100-150 points, ~5ms per option)
    /// - 1e-3: Medium accuracy (~300-400 points, ~50ms per option)
    /// - 1e-6: High accuracy mode (~1200 points, ~300ms per option)
    double tol = 1e-2;

    /// CFL safety factor for time step (default: 0.75)
    double c_t = 0.75;

    /// Minimum spatial grid points (default: 100)
    size_t min_spatial_points = 100;

    /// Maximum spatial grid points (default: 1200)
    size_t max_spatial_points = 1200;

    /// Maximum time steps (default: 5000)
    size_t max_time_steps = 5000;
};

struct PDEGridConfig {
    GridSpec<double> grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, optimal_sinh_alpha(5.0)).value();
    size_t n_time = 1000;
    std::vector<double> mandatory_times = {};
};

using PDEGridSpec = std::variant<PDEGridConfig, GridAccuracyParams>;

GridAccuracyParams make_grid_accuracy(GridAccuracyProfile profile);

}  // namespace mango
