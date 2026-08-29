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
#include <memory>
#include <numeric>
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
};

/// Absolute holdout-error ceiling above which a candidate surface is treated
/// as garbage and never returned (spec D5).  2,000 bps of IV: an operational
/// garbage detector, deliberately independent of `target_iv_error`.
inline constexpr double kViabilityBound = 0.20;

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

/// Aggregate max grid sizes across probe results
struct MaxGridSizes {
    size_t moneyness = 0, vol = 0, rate = 0;
    int tau_points = 0;
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

/// Per-point reference data, computed once per validation/holdout point.
struct ErrorRefs {
    double ref_price = 0.0;  ///< FD American price
    double vega = 0.0;       ///< FD central-difference American vega
};

/// Produce refs for one point (base solve + two sigma-bump solves).
/// Any failed or non-finite solve => unexpected.
using PrepareRefsFn = std::function<std::expected<ErrorRefs, SolverError>(
    double spot, double strike, double tau, double sigma, double rate)>;

/// Score one point from interpolated price + cached refs. Pure arithmetic.
/// Contract: returns a finite, nonnegative error (the loop treats anything
/// else as a non-viable evaluation).
using ScoreErrorFn = std::function<double(
    double interp, const ErrorRefs& refs,
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

/// Compute IV error from price error and vega, with floor and cap.
double compute_iv_error(double price_error, double vega,
                        double vega_floor, double target_iv_error);

/// Produce ErrorRefs (FD American price + FD central-difference vega) for
/// one point: base solve + two sigma-bump solves.
/// 2 extra PDE solves per point — acceptable at build time.
/// Any failed or non-finite solve => unexpected.
PrepareRefsFn make_fd_vega_refs_fn(const AdaptiveGridParams& params,
                                    const ValidateFn& validate_fn);

/// Score an interpolated price against cached ErrorRefs using the TV/K
/// filter (skips points where TV/K < 1e-4; IV undefined there) and
/// `compute_iv_error` arithmetic (vega floor + target-level noise clamp).
ScoreErrorFn make_iv_score_fn(const AdaptiveGridParams& params,
                              OptionType option_type);

/// Create a ValidateFn that solves a single American option via FD.
ValidateFn make_validate_fn(double dividend_yield,
                            OptionType option_type,
                            const std::vector<Dividend>& discrete_dividends = {});

/// Aggregate max grid sizes across probe results.
MaxGridSizes aggregate_max_sizes(const std::vector<RefinementResult>& probe_results);

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
