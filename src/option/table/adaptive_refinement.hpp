// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/price_table.hpp"
#include "mango/option/table/splits/tau_segment.hpp"
#include "mango/option/option_grid.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/support/error_types.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <functional>
#include <limits>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace mango {

// ============================================================================
// Shared types for adaptive grid refinement
// ============================================================================

/// Type-erased surface handle for validation queries during adaptive refinement
struct SurfaceHandle {
    std::function<double(double spot, double strike, double tau,
                         double sigma, double rate)> price;
    /// Surface vega at the same coordinates. The round-trip scorer feeds it to
    /// the shipped inversion's vega pre-check, so a candidate is judged by the
    /// same sensitivity the product would see.
    std::function<double(double spot, double strike, double tau,
                         double sigma, double rate)> vega;
    size_t pde_solves = 0;
};

/// Domain bounds for the refinement loop (spec D2).
///
/// Two domains are carried separately:
///  - `bounds` is the **fit** domain: the span the grids/nodes handed to the
///    builder cover, including any backend-specific support extension.
///  - `sample_bounds` is the **measurement** domain the user actually asked
///    for (their moneyness/tau/vol/rate ranges, after the minimum-spread
///    widening in `expand_domain_bounds`, which is a usability floor rather
///    than headroom).  Every validation sample and every error-bin
///    normalization uses this domain, so accuracy is never measured in the
///    unqueryable support band.
struct RefinementContext {
    double spot;
    double dividend_yield;
    OptionType option_type;
    SurfaceBounds bounds;         ///< fit domain (support incl. headroom)
    SurfaceBounds sample_bounds;  ///< user-facing measurement domain
    /// Optional time-domain admission, fixed across all candidates. Segmented
    /// builders exclude unsupported event neighborhoods before drawing refs;
    /// non-finite prices at admitted times still disqualify a candidate.
    std::function<bool(double)> maturity_is_supported = {};
};

/// Relative holdout improvement required to restart the axis walk (spec D6).
inline constexpr double kMinRelImprovement = 0.02;

/// Result of grid sizing from the refinement loop
///
/// `achieved_max_error` / `achieved_avg_error` / `target_met` describe the
/// *returned* candidate measured on the fixed holdout (spec D5), not the last
/// iteration's fresh samples.
struct RefinementResult {
    std::vector<double> moneyness;
    std::vector<double> tau;
    std::vector<double> vol;
    std::vector<double> rate;
    int tau_points = 0;
    double achieved_max_error = 0.0;
    double achieved_avg_error = 0.0;
    bool target_met = false;
    std::vector<IterationStats> iterations;
    BuildDiagnostics diagnostics;
};

/// Probe-refined grids merged across probe results (issue #461).
///
/// The continuous axes carry the *positions* every probe's refinement loop
/// chose, not just their counts: rebuilding on uniform grids at the probes'
/// maximum sizes threw those positions away, and the rebuilt surface could
/// measure worse than the probe that sized it. B-spline builders also retain
/// global tau positions; count-based backends continue to use tau_points.
struct AggregatedGrids {
    std::vector<double> moneyness;
    std::vector<double> vol;
    std::vector<double> rate;
    int tau_points = 0;
    std::vector<double> tau{};
};

/// Initial grids for seeding the refinement loop (optional for each dimension)
struct InitialGrids {
    std::vector<double> moneyness;
    std::vector<double> tau;
    std::vector<double> vol;
    std::vector<double> rate;
    /// When true, use grids exactly as provided (no seed_grid processing).
    /// Required for Chebyshev paths where CGL node placement must be preserved.
    bool exact = false;
};

/// Result of compute_segment_boundaries: boundaries + gap metadata
struct SegmentBoundaries {
    std::vector<double> bounds;        ///< Sorted segment boundaries
    std::vector<bool> is_gap;          ///< is_gap[s] = true for synthetic dividend gaps
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

    /// Count of surface failures in each bin for each dimension (spec D4).
    /// Recorded unconditionally: a point the shipped inversion could not
    /// round-trip carries no error number, but it does say where the surface
    /// misbehaves, and refinement should be steered there.
    std::array<std::array<size_t, N_BINS>, N_DIMS> failure_counts = {};

    /// Total error mass accumulated in each dimension
    std::array<double, N_DIMS> dim_error_mass = {};

    /// Bin index of a normalized coordinate, clamped into [0, N_BINS).
    [[nodiscard]] static size_t bin_of(double normalized) noexcept {
        double pos = std::clamp(normalized, 0.0, 1.0);
        return std::min(static_cast<size_t>(pos * N_BINS), N_BINS - 1);
    }

    /// Attributed count in one bin: measured high errors plus surface
    /// failures (spec D4).  Both say "refine here"; only the first is a
    /// number, which is why they are stored apart and summed on read.
    [[nodiscard]] size_t attributed(size_t dim, size_t bin) const noexcept {
        return bin_counts[dim][bin] + failure_counts[dim][bin];
    }

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
            bin_counts[d][bin_of(normalized_pos[d])]++;
            dim_error_mass[d] += iv_error;
        }
    }

    /// Record a surface failure at a normalized position [0,1]^4 (spec D4).
    ///
    /// Unconditional: there is no threshold to compare against, because a
    /// failure produces no error number.  It adds no error mass either --
    /// mass is measured error, and this point measured nothing.
    void record_failure(const std::array<double, N_DIMS>& normalized_pos) {
        for (size_t d = 0; d < N_DIMS; ++d) {
            failure_counts[d][bin_of(normalized_pos[d])]++;
        }
    }

    /// Add another set's failure counts (spec D4: a candidate's bins carry
    /// the fresh pass's measured errors plus both passes' failures).
    void merge_failures(const ErrorBins& other) {
        for (size_t d = 0; d < N_DIMS; ++d) {
            for (size_t b = 0; b < N_BINS; ++b) {
                failure_counts[d][b] += other.failure_counts[d][b];
            }
        }
    }

    /// Find dimension with most concentrated errors
    ///
    /// Returns the dimension where errors are most localized (highest
    /// max bin count relative to total), indicating refinement will help.
    ///
    /// Retained as a diagnostic helper; it no longer drives axis selection
    /// (spec D6 -- `pick_refinement_axis` walks every axis by concentration
    /// alone, with a measured-improvement restart).  Kept because it weighs
    /// concentration by error mass, which is the question to ask when reading
    /// a build's bins by hand.
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

    /// Get bins with attributed count >= min_count for a dimension
    /// (measured high errors plus surface failures, spec D4).
    [[nodiscard]] std::vector<size_t> problematic_bins(size_t dim, size_t min_count = 2) const {
        auto indices = std::views::iota(size_t{0}, N_BINS)
                     | std::views::filter([&](size_t b) { return attributed(dim, b) >= min_count; });
        return std::ranges::to<std::vector<size_t>>(indices);
    }

    /// Clear all bins
    void reset() {
        for (auto& dim_bins : bin_counts) {
            dim_bins.fill(0);
        }
        for (auto& dim_bins : failure_counts) {
            dim_bins.fill(0);
        }
        dim_error_mass.fill(0.0);
    }
};

/// One counter per `PointStatus` enumerator, indexed by its underlying value
/// (spec D7).  Kept as a plain array so an evaluation can be summed across
/// candidates without naming each outcome.  `kPointStatusCount` is asserted
/// against the enum, so a new outcome fails the build instead of quietly
/// dropping out of the refusal probe.
using PointStatusCounts = std::array<size_t, kPointStatusCount>;

/// Add `status` to `counts`, ignoring an out-of-range value rather than
/// writing past the array.
constexpr void count_status(PointStatusCounts& counts, PointStatus status) noexcept {
    const auto idx = static_cast<size_t>(status);
    if (idx < counts.size()) ++counts[idx];
}

// ============================================================================
// Callback type aliases
// ============================================================================

/// Builds a surface from current grids, returns handle for querying
using BuildFn = std::function<std::expected<SurfaceHandle, PriceTableError>(
    std::span<const double> moneyness,
    std::span<const double> tau_grid,
    std::span<const double> vol,
    std::span<const double> rate)>;

/// Outcome of a single refinement attempt (spec D6).
struct RefineOutcome {
    bool changed = false;  ///< grids actually changed
    int changed_dim = -1;  ///< the axis that actually changed (may differ
                           ///< from the requested axis only if the backend
                           ///< documents redirection)
};

/// Decides how to grow grids when error exceeds target.
///
/// Called with the requested axis and physical focus intervals (D2:
/// coordinates within sample_bounds identifying where refinement should
/// concentrate; empty means unconstrained/uniform refinement over the
/// whole axis) and the current grids (mutable). Returns the outcome:
/// whether anything changed, and which axis actually changed.
using RefineFn = std::function<RefineOutcome(
    size_t requested_dim,
    std::span<const std::pair<double, double>> focus_intervals,
    std::vector<double>& moneyness,
    std::vector<double>& tau,
    std::vector<double>& vol,
    std::vector<double>& rate)>;

/// Opaque snapshot/restore hooks for backend refinement state (spec D6).
///
/// Restoring the grid vectors is not always enough: the Chebyshev refiners
/// advance per-axis level counters held outside the grids.  Backends with
/// such state provide both hooks; backends whose grids are the whole state
/// (B-spline) leave them empty.  The loop takes a snapshot with every
/// candidate it records and restores it together with the grids whenever the
/// backtracking walk resets to the exploration base.
struct RefineStateHooks {
    std::function<std::shared_ptr<const void>()> snapshot;
    std::function<void(const std::shared_ptr<const void>&)> restore;
};

/// Produces a fresh FD reference price for one validation point
using ValidateFn = std::function<std::expected<double, SolverError>(
    double spot, double strike, double tau,
    double sigma, double rate)>;

/// Per-point reference data from the six-solve stencil (spec D1), computed
/// once per validation/holdout point.
///
/// The stencil prices sigma0 and sigma0 +- target_iv_error on one nested
/// grid pair; each `delta*` is the two-grid Richardson *estimate* of that
/// price's discretisation error (an estimate, never a certificate: a two-grid
/// difference cannot see bias the two grids share).
///
/// Partial stencils are normal.  Only `ref_price` is always present; every
/// other numeric field is NaN when its solve did not happen or did not
/// succeed, and `resolved` is then false.
struct ErrorRefs {
    /// y: the fine-grid reference price at sigma0. Always present.
    double ref_price = std::numeric_limits<double>::quiet_NaN();
    /// lo / hi: fine-grid prices at `sigma_lo` / `sigma_hi`. NaN when unavailable.
    double bracket_lo_price = std::numeric_limits<double>::quiet_NaN();
    double bracket_hi_price = std::numeric_limits<double>::quiet_NaN();
    /// sigma0 -+ target_iv_error, as actually solved.
    double sigma_lo = std::numeric_limits<double>::quiet_NaN();
    double sigma_hi = std::numeric_limits<double>::quiet_NaN();
    /// Richardson error estimates for `ref_price`, `bracket_lo_price` and
    /// `bracket_hi_price`. NaN when unavailable.
    double delta = std::numeric_limits<double>::quiet_NaN();
    double delta_lo = std::numeric_limits<double>::quiet_NaN();
    double delta_hi = std::numeric_limits<double>::quiet_NaN();
    /// Spec D2: the stencil separates in the expected order *and* all three
    /// targets pass the product's query validation.
    bool resolved = false;
    /// Achieved time-step counts of the two grid levels (record only).
    uint32_t fine_steps = 0;
    uint32_t coarse_steps = 0;
};

/// Produce refs for one point (the six-solve stencil of spec D1).
/// A failed or non-finite *base* solve => unexpected (the point is invalid).
/// Any other missing piece => success with `resolved = false` and the base
/// price present.
using PrepareRefsFn = std::function<std::expected<ErrorRefs, SolverError>(
    double spot, double strike, double tau, double sigma, double rate)>;

/// Score one point from interpolated price + cached refs. Pure arithmetic.
///
/// Superseded by `ScoreErrorFn` below; kept only until the refinement loop
/// and the builders move over to the round-trip score.
///
/// Contract (spec D4, final-review amendment 2026-08-29):
///  - `std::nullopt` means the point was **deliberately skipped**: its
///    reference is unresolved (or, on the temporary TV/K bridge, the point
///    has no time value), so the error metric is undefined there and the
///    point carries no evidence either way.  Skipped points are excluded from the
///    max, the average, and the measured count -- they neither certify a
///    surface nor condemn it.
///  - An engaged value must be finite and nonnegative; anything else is a
///    non-viable evaluation and disqualifies the candidate (D5).
using LegacyScoreErrorFn = std::function<std::optional<double>(
    double interp, const ErrorRefs& refs,
    double spot, double strike, double tau,
    double sigma, double rate)>;

/// Score one point by round-tripping the candidate surface (spec D3).
///
/// The scorer prices nothing itself: it hands `surface` to the *shipped*
/// inversion at the three stencil targets (`ref_price` and `ref_price +-
/// delta`) and reports how far the recovered volatilities land from `sigma`.
/// The returned `PointScore::status` therefore describes an operational
/// outcome of that inversion at this point, and `iv_error` is an estimate of
/// the surface's IV error there.
///
/// Precondition: handles must supply `vega` as well as `price`; a handle
/// without it scores `SurfaceNonFinite`.
using ScoreErrorFn = std::function<PointScore(
    const SurfaceHandle& surface, const ErrorRefs& refs,
    double spot, double strike, double tau,
    double sigma, double rate)>;

// ============================================================================
// Shared helper function declarations
// ============================================================================

/// Expand [lo, hi] to at least min_spread wide.
/// If lo_clamp is finite, enforces lo >= lo_clamp (shifting hi to compensate).
void expand_domain_bounds(double& lo, double& hi, double min_spread,
                          double lo_clamp = -std::numeric_limits<double>::infinity());

/// One cubic support band (3 x local knot spacing) of headroom per side.
double spline_support_headroom(double domain_width, size_t n_knots);

/// Select up to 3 probes from a sorted vector: front, back, and nearest to
/// reference_value. Returns all items if size <= 3.
std::vector<double> select_probes(const std::vector<double>& items,
                                  double reference_value);

/// Sum discrete dividends strictly inside (0, maturity) with positive amount.
double total_discrete_dividends(const std::vector<Dividend>& dividends,
                                double maturity);

/// Compute tau-space segment boundaries from dividend schedule.
/// Returns sorted boundaries with gap metadata for dividend dates.
SegmentBoundaries compute_segment_boundaries(
    const std::vector<Dividend>& dividends, double maturity,
    double tau_min, double tau_max);

/// Collapse gap segments into adjacent real segments for TauSegmentSplit.
/// Each real segment's range extends to the midpoint of its adjacent gap.
/// Only real segments are kept; gaps are absorbed.
TauSegmentSplit make_tau_split_from_segments(
    const std::vector<double>& bounds,
    const std::vector<bool>& is_gap,
    double K_ref);

// The option-aware implementations of ValidateFn / PrepareRefsFn /
// ScoreErrorFn (`make_validate_fn`, `make_stencil_refs_fn`,
// `make_round_trip_score_fn`, and the legacy `make_fd_vega_refs_fn` /
// `make_iv_score_fn`) live in adaptive_metrics.hpp: the loop consumes the
// callback types declared above but never depends on the American solver
// behind them.

/// Merge probe results into one set of grids: the sorted union of each
/// probe's knot positions per continuous axis, with positions closer than
/// 1e-9 of the axis span collapsed, then thinned to `max_points_per_dim` by
/// dropping the most crowded interior position first.  Endpoints always
/// survive; surviving positions are never re-spaced.  `tau_points` is the
/// maximum per-segment count across probes.
AggregatedGrids aggregate_probe_grids(const std::vector<RefinementResult>& probe_results,
                                      size_t max_points_per_dim);

/// Insert up to `count` midpoints into the largest gaps of `grid`, one at a
/// time, without exceeding `cap` points.  Used by the segmented final retry
/// so the bump keeps the aggregated positions instead of re-spacing them.
std::vector<double> insert_largest_gap_midpoints(std::vector<double> grid,
                                                 size_t count, size_t cap);

/// Helper to create evenly spaced grid.
/// Requires n >= 2 to avoid divide-by-zero; returns {lo, hi} if n < 2.
std::vector<double> linspace(double lo, double hi, size_t n);

/// Seed a grid from user-provided knots, or fall back to linspace.
/// Ensures domain endpoints are included and minimum 4 points for B-spline.
std::vector<double> seed_grid(const std::vector<double>& user_knots,
                               double lo, double hi, size_t fallback_n = 5);

/// The four working grids the refinement loop starts from.
struct SeededGrids {
    std::vector<double> moneyness;
    std::vector<double> tau;
    std::vector<double> vol;
    std::vector<double> rate;
};

/// Seed the working grids over the fit domain exactly as `run_refinement`
/// does (user knots where given, linspace otherwise, moneyness padded to
/// `params.min_moneyness_points`; `InitialGrids::exact` passes through).
/// Exposed so callers can reproduce the loop's starting sizes without
/// running it.
SeededGrids seed_refinement_grids(const AdaptiveGridParams& params,
                                  const RefinementContext& ctx,
                                  const InitialGrids& initial_grids);

/// Run the iterative adaptive refinement loop (spec D4-D7).
///
/// Builds a surface, measures it against a fixed holdout (references cached
/// once) *and* fresh per-iteration samples, records every candidate, and
/// walks the four axes with greedy coordinate descent plus a measured
/// walk-restart.  The best *viable* candidate is returned -- rebuilt once if
/// it is not the surface most recently built -- or
/// `PriceTableErrorCode::NoViableSurface` when no candidate is safe.
///
/// @param hooks Optional backend-state snapshot/restore (spec D6).  Backends
///              whose refinement state lives entirely in the grids pass none.
std::expected<RefinementResult, PriceTableError> run_refinement(
    const AdaptiveGridParams& params,
    BuildFn build_fn,
    RefineFn refine_fn,
    const RefinementContext& ctx,
    const PrepareRefsFn& prepare_refs,
    const ScoreErrorFn& score,
    const InitialGrids& initial_grids = {},
    const RefineStateHooks& hooks = {});

namespace detail {

// ============================================================================
// Final-surface validation for the segmented builders (spec D9)
// ============================================================================
//
// The segmented builders assemble their *final* surface outside the
// refinement loop (uniform grids aggregated across probes for B-spline, the
// all-K_ref blend for Chebyshev), so the loop's holdout says nothing about
// the object the caller receives.  These helpers give both builders the same
// contract the loop applies to its candidates: references computed once, D4
// validity rules, D5 viability, and an honest score for whichever surface is
// returned.

/// One validation point with its cached references (spec D4).
struct ValidationPoint {
    std::array<double, 4> coords{};  ///< m, tau, sigma, rate
    double strike = 0.0;
    ErrorRefs refs;
};

/// The final validation set: valid points plus the count that could not be
/// referenced (spec D9 step 1).
struct FinalValidationSet {
    std::vector<ValidationPoint> points;
    size_t invalid = 0;
    /// `PrepareRefsFn` invocations made, valid and invalid alike.  This
    /// counts preparations, not PDE solves: how many solves one preparation
    /// runs depends on the factory, and an attempt that stops at the base
    /// solve runs fewer.  `ReferenceSolveCounter` is what records the solves.
    size_t ref_attempts = 0;
};

/// Draw `params.validation_samples` LHS points over the **sample** domain
/// (spec D2) and compute `ErrorRefs` for each exactly once.
///
/// Points whose refs fail or are non-finite are dropped and counted.  Fewer
/// than `max(4, validation_samples / 4)` prepared points -- or fewer than
/// that many *resolved* ones (spec D2) -- ⇒
/// `PriceTableErrorCode::ValidationFailed`: a validation set that cannot
/// measure cannot certify the surface.  The threshold is an operational
/// coverage policy, not a spatial or statistical guarantee; the refusal fires
/// `MANGO_TRACE_ADAPTIVE_VALIDATION_REFUSED` with the counts the error cannot
/// carry.
///
/// @param seed  LHS seed for this set, passed explicitly and deliberately
///              *instead of* `params.lhs_seed`: the final validation must not
///              land on the coordinates the refinement loop already fit and
///              measured, so callers offset the configured seed (the
///              segmented builders use `params.lhs_seed + 999`).
[[nodiscard]] std::expected<FinalValidationSet, PriceTableError>
prepare_final_validation(const AdaptiveGridParams& params,
                         const RefinementContext& ctx,
                         const PrepareRefsFn& prepare_refs,
                         uint64_t seed);

/// Score of one assembled surface over a cached `FinalValidationSet`.
///
/// `measured` counts every point whose score reported `PointStatus::Measured`
/// with a finite, nonnegative error -- including exact zeros -- so
/// `avg_error`'s denominator matches its numerator even for a surface that
/// reproduces every reference exactly.  Points whose reference never resolved
/// are counted in `unresolved` and enter no statistic: the metric is
/// undefined there, and an unresolved reference is not a defect of the
/// surface.  Points where the shipped inversion failed on the surface's own
/// price are counted in `surface_failures`: an outcome, never a number.
struct FinalScore {
    double max_error = 0.0;
    double avg_error = 0.0;
    size_t measured = 0;    ///< points that produced a usable error
    size_t unresolved = 0;  ///< points whose reference did not resolve (D2)
    size_t skipped = 0;     ///< points with a non-finite/negative evaluation
    /// Points excluded by maturity support (spec D4).  Structurally zero
    /// here: this score runs over points that were already admitted at
    /// preparation, which is where unsupported maturities are dropped.  The
    /// field exists so both passes accumulate the same shape; the loop's
    /// `holdout_points_unsupported` comes from that preparation, not here.
    size_t unsupported = 0;
    size_t surface_failures = 0;   ///< points where is_surface_failure() held
    size_t edge_band_rescues = 0;  ///< points scored via the rescue path (D3)
    /// Largest |S - V̂|/K residual seen over the points, measured or not.
    double max_price_residual = 0.0;
    /// Largest finite reference-error *estimate* over the points; 0 when none
    /// was finite.  An estimate, never a certificate.
    double max_delta = 0.0;
    /// Where the surface failures landed, for refinement attribution (D4).
    ErrorBins failure_bins;
    /// Per-outcome totals over the points (spec D7 refusal probe).
    PointStatusCounts status_counts = {};
    bool all_finite = true;

    /// D4 viability: every evaluation finite and nonnegative, at least one
    /// *measurement*, and no surface failure.  `measured > 0` is what stops a
    /// surface whose every point was unresolved from certifying itself with a
    /// vacuous max of 0; `surface_failures == 0` is what stops one the
    /// shipped inversion could not round-trip from being returned at all.
    [[nodiscard]] bool viable() const noexcept {
        return all_finite && measured > 0 && surface_failures == 0;
    }
};

/// Score `handle` on the cached references (interpolations plus arithmetic --
/// no FD solves).  Mirrors the loop's holdout scoring: any non-finite price
/// or score clears `all_finite` and makes the aggregate NaN, so a surface
/// that produced garbage can never report a rosy number.
[[nodiscard]] FinalScore score_final_surface(
    const std::vector<ValidationPoint>& points,
    const SurfaceHandle& handle,
    const ScoreErrorFn& score,
    const RefinementContext& ctx);

/// Retry trigger (spec D9 step 2): the original assembled surface misses the
/// target, or it is not viable at all.
[[nodiscard]] bool needs_final_retry(const FinalScore& original,
                                     double target_iv_error);

/// Which assembled surface the builder returns (spec D9 step 3).
enum class FinalPick { None, Original, Retry };

/// Return the lower-error **viable** surface; `None` when neither is viable
/// (⇒ `PriceTableErrorCode::NoViableSurface`).  A missing retry means the
/// retry was never built (or its build failed).  Ties keep the original: the
/// retry is strictly larger, so equal accuracy is not worth the extra knots.
[[nodiscard]] FinalPick select_final_surface(
    const FinalScore& original,
    const std::optional<FinalScore>& retry);

/// Monotonicity statistics for a returned surface (spec D5).
///
/// Diagnostics only, never a gate: at each validation point's (m, tau, r),
/// scan 7 equally spaced sigma across `ctx.sample_bounds` and count steps
/// where the price falls by more than the point's noise floor.
///
/// The noise floor is a *reporting threshold*: the largest finite reference
/// error estimate among `delta`, `delta_lo` and `delta_hi`, floored at
/// `1e-8 * spot`.  A point with no finite estimate has no floor to report
/// against and is skipped.  `target_iv_error` is unused by the threshold and
/// kept only so callers need not re-derive the scan's context.
void scan_monotonicity(const std::vector<ValidationPoint>& points,
                       const SurfaceHandle& handle,
                       const RefinementContext& ctx,
                       double target_iv_error,
                       BuildDiagnostics& diag);

}  // namespace detail

/// Resolve K_ref values from a MultiKRefConfig.
/// If config.K_refs is non-empty, returns them sorted.
/// Otherwise generates K_ref_count log-spaced values spanning
/// [spot*(1-span), spot*(1+span)].
[[nodiscard]] std::expected<std::vector<double>, PriceTableError>
resolve_k_refs(const MultiKRefConfig& config, double spot);

/// Expand domain bounds for segmented (discrete-dividend) surface building.
///
/// Converts IVGrid moneyness (already log-moneyness) to domain bounds,
/// expands for cumulative discrete dividends, applies minimum spreads,
/// and caps tau at maturity.
///
/// @param domain         IVGrid with moneyness already in log(S/K) space
/// @param maturity       Option maturity (years)
/// @param dividend_yield Continuous dividend yield (unused in expansion, carried for API)
/// @param discrete_dividends Discrete dividend schedule
/// @param min_K_ref      Smallest K_ref value (for dividend expansion denominator)
/// @return Expanded domain bounds, or error if domain is empty
[[nodiscard]] std::expected<SurfaceBounds, PriceTableError>
expand_segmented_domain(const IVGrid& domain,
                        double maturity,
                        double dividend_yield,
                        const std::vector<Dividend>& discrete_dividends,
                        double min_K_ref);

/// Extract domain bounds from OptionGrid (spec D2/D3).
///
/// Produces both the sample domain (user ranges + minimum-spread widening)
/// and the B-spline fit domain (sample domain + `spline_support_headroom` on
/// moneyness only).  `expected_m_knots` is the *expected seeded moneyness
/// density* -- `max(user_moneyness_knots, params.min_moneyness_points)` --
/// not the user strike count; passing the strike count makes the headroom an
/// order of magnitude too wide.
///
/// Chebyshev callers must ignore `bounds` and build their own fit domain
/// from `sample_bounds` via the CC-level extension (spec D3: no double
/// headroom).
std::expected<RefinementContext, PriceTableError>
extract_chain_domain(const OptionGrid& chain, size_t expected_m_knots);

/// Build InitialGrids from OptionGrid (log-moneyness from strikes).
InitialGrids extract_initial_grids(const OptionGrid& chain);

}  // namespace mango
