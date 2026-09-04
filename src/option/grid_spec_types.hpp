// SPDX-License-Identifier: MIT
#pragma once
#include "mango/pde/core/grid.hpp"
#include "mango/option/option_spec.hpp"
#include <cmath>
#include <optional>
#include <span>
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

/// Closed interval of log-moneyness x = ln(S/K), relative to the contract's
/// strike (absolute in the solver's x coordinate, NOT an offset from spot).
struct LogMoneynessRange {
    double lo = 0.0;
    double hi = 0.0;

    /// Tight range of a node set given in any order; nullopt for an empty set
    /// or a set containing a non-finite node.
    static std::optional<LogMoneynessRange> of(std::span<const double> nodes);

    /// Largest distance from x0 to either endpoint: the symmetric half-width
    /// about x0 that contains the whole range.
    [[nodiscard]] double reach_from(double x0) const;
};

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

    /// Log-moneyness range the PDE solution must resolve (in addition to
    /// the contract's own spot), because the caller will read the solution
    /// there (price tables evaluate every slice at their moneyness nodes).
    /// The estimated domain, which is symmetric about the contract's x0 =
    /// ln(spot/strike), is widened until the range sits at least
    /// `coverage_clearance_sigmas` diffusion lengths (sigma*sqrt(T)) inside
    /// the boundary, with a 10 % widening of the reach as a floor for tiny
    /// sigma*sqrt(T).  For contracts sharing one estimated grid the largest
    /// sigma*sqrt(T) among them sets the clearance and the largest required
    /// widening is applied to all, so the shared grid covers the range as a
    /// whole.  nullopt: only the contract's spot matters (the n_sigma domain).
    /// Coverage is disabled (the plain n_sigma domain is used) when either
    /// endpoint or the clearance is non-finite; a negative clearance counts
    /// as zero.  `LogMoneynessRange::of` returns nullopt when any node is
    /// non-finite.  An explicit PDEGridConfig always takes precedence: a
    /// caller supplying a concrete grid owns its domain.
    std::optional<LogMoneynessRange> log_moneyness_coverage;

    /// Diffusion lengths of boundary clearance demanded by
    /// `log_moneyness_coverage` (default: 3).
    double coverage_clearance_sigmas = 3.0;
};

struct PDEGridConfig {
    GridSpec<double> grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, optimal_sinh_alpha(5.0)).value();
    size_t n_time = 1000;
    std::vector<double> mandatory_times = {};
};

using PDEGridSpec = std::variant<PDEGridConfig, GridAccuracyParams>;

GridAccuracyParams make_grid_accuracy(GridAccuracyProfile profile);

/// Estimate grid specification from option parameters.
std::pair<GridSpec<double>, TimeDomain> estimate_pde_grid(
    const PricingParams& params,
    const GridAccuracyParams& accuracy = GridAccuracyParams{});

/// Compute global grid for batch processing.
std::pair<GridSpec<double>, TimeDomain> estimate_batch_pde_grid(
    std::span<const PricingParams> params,
    const GridAccuracyParams& accuracy = GridAccuracyParams{});

/// The grid estimate_batch_pde_grid would use for `batch`, as a concrete
/// PDEGridConfig for solve_batch's custom_grid (mandatory_times empty: the
/// batch solver rebuilds per-contract dividend times itself).
PDEGridConfig estimate_batch_pde_grid_config(
    std::span<const PricingParams> batch,
    const GridAccuracyParams& accuracy = GridAccuracyParams{});

}  // namespace mango
