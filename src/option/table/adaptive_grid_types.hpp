// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"
#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <limits>

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

    /// Maximum refinement iterations (default: 8)
    size_t max_iter = 8;

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

    /// Deprecated and ignored since the round-trip metric (spec 2026-09-19
    /// D6); kept for C ABI layout stability until #463 removes it.
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

/// Outcome of scoring one holdout point under the round-trip IV metric:
/// price the reference IV off the built surface, then re-invert the
/// surface's own price back to an IV and compare against the reference.
enum class PointStatus : uint8_t {
    /// The round trip completed and produced an IV error.
    Measured,
    /// The FD reference for this point could not be established, so no
    /// round trip was attempted.  Not charged against the surface.
    ReferenceUnresolved,
    /// The surface's local vega at this point was too small for the
    /// round-trip inversion to recover an IV from the surface's price.
    SurfaceVegaTooSmall,
    /// The round-trip inversion found no root: no surface IV reproduced
    /// the surface's own price within the solver's search bounds.
    SurfaceNoRoot,
    /// The round-trip inversion found more than one candidate root,
    /// so the recovered IV is ambiguous.
    SurfaceAmbiguous,
    /// The round-trip inversion did not converge within its iteration
    /// budget.
    SurfaceNonConvergent,
    /// The surface produced a non-finite price or derivative during the
    /// round-trip inversion.
    SurfaceNonFinite,
};

/// True for every Surface* status: an operational failure of the shipped
/// inversion at this point, as opposed to a reference that never resolved.
constexpr bool is_surface_failure(PointStatus s) noexcept {
    return s != PointStatus::Measured && s != PointStatus::ReferenceUnresolved;
}

/// Result of scoring one holdout point under the round-trip IV metric.
struct PointScore {
    PointStatus status = PointStatus::ReferenceUnresolved;
    /// Round-trip IV error; only meaningful when `status == Measured`.
    double iv_error = std::numeric_limits<double>::quiet_NaN();
    /// |S - V̂|/K between the reference price and the surface's price,
    /// when finite; a diagnostic, not part of the error metric.
    double price_residual = std::numeric_limits<double>::quiet_NaN();
    /// Whether the edge-band rescue path was used for this point; a
    /// diagnostic only, does not affect `status` or `iv_error`.
    bool edge_band_rescue = false;
};

/// Per-iteration diagnostics
///
/// `refined_dim` is a dimension index (0 = moneyness, 1 = tau, 2 = sigma,
/// 3 = rate) or one of three sentinels:
///   * `-1` — no dimension was refined for this entry (the seed build).
///   * `-2` — not an iteration at all: the retention final rebuild of the
///     returned candidate (spec D5/D7).  Excluded from `total_iterations`
///     and never charged to `max_iter`.
///   * `-3` — not a build at all: a segmented-builder probe whose served
///     strike band lies outside the user's strike range, so its refinement
///     loop was skipped and only its seed grid sizes were contributed.
struct IterationStats {
    size_t iteration = 0;                    ///< Iteration number (0-indexed)
    std::array<size_t, 4> grid_sizes = {};   ///< [m, tau, sigma, r] sizes
    size_t pde_solves_table = 0;             ///< Slices computed for table
    size_t pde_solves_validation = 0;        ///< Fresh solves for validation
    double max_error = 0.0;                  ///< Max IV error observed
    double avg_error = 0.0;                  ///< Mean IV error
    int refined_dim = -1;                    ///< Refined dim, or -1/-2/-3 (above)
    double elapsed_seconds = 0.0;            ///< Wall-clock time for this iteration
    bool build_failed = false;               ///< Refinement trial build failed (D5)
    size_t unresolved = 0;                   ///< Points with PointStatus::ReferenceUnresolved
    size_t surface_failures = 0;             ///< Points where is_surface_failure() held
    size_t edge_band_rescues = 0;            ///< Points scored with PointScore::edge_band_rescue set
};

/// Adaptive refinement build diagnostics
struct BuildDiagnostics {
    bool target_met = false;
    double achieved_max_error = 0.0;   // holdout, returned candidate
    double achieved_avg_error = 0.0;
    size_t picked_iteration = 0;
    size_t total_iterations = 0;       // built iterations, excl. final rebuild
    bool final_rebuild = false;
    bool build_failure_fallback = false;
    /// Holdout points with usable FD references (the measurable set).
    size_t holdout_points = 0;
    /// Holdout points whose references failed, or whose evaluation of the
    /// returned surface was non-finite.
    size_t holdout_points_invalid = 0;
    /// Of `holdout_points`, those that actually produced an error for the
    /// returned surface.  The rest were filtered out by the score function
    /// (TV/K or vega floor), where the IV-error metric is undefined; a build
    /// with `holdout_points_measured == 0` is refused, never certified.
    size_t holdout_points_measured = 0;
    /// Holdout points whose FD reference never resolved (PointStatus::ReferenceUnresolved).
    size_t holdout_points_unresolved = 0;
    /// Holdout points whose reference resolved but the round-trip metric
    /// found no supported inversion for them (e.g. vega floor).
    size_t holdout_points_unsupported = 0;
    /// Holdout points where the round-trip inversion of the returned
    /// surface's own price failed (is_surface_failure() held).
    size_t surface_failures = 0;
    /// Holdout points scored via the edge-band rescue path; a diagnostic
    /// count, not part of any pass/fail decision.
    size_t edge_band_rescues = 0;
    /// Largest |S - V̂|/K price residual observed among measured points.
    double max_price_residual = 0.0;
    /// Largest estimated uncertainty in an FD reference used as ground truth.
    double reference_uncertainty_max = 0.0;
    /// Reference solves that used the fine grid.
    size_t reference_solves_fine = 0;
    /// Reference solves that used the coarse grid.
    size_t reference_solves_coarse = 0;
    /// Rows/points from successful segmented sampling builds, including payoff
    /// rows, refinement probes, and final/retry assemblies. Other backends leave zero.
    size_t sample_rows = 0;
    size_t sample_points = 0;
    /// Sum of temporal segments whose requested density hit its point cap.
    size_t tau_point_cap_hits = 0;
    size_t monotonicity_violations = 0;
    size_t monotonicity_points_invalid = 0;
    double worst_vega_slope = 0.0;
    /// Per-iteration forensics; see IterationStats for the refined_dim
    /// sentinels (-2 final rebuild, -3 skipped probe).
    std::vector<IterationStats> iterations;
};

}  // namespace mango
