// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/adaptive_grid_types.hpp"
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
#include <numeric>
#include <ranges>
#include <span>
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

/// Domain bounds for the refinement loop
struct RefinementContext {
    double spot;
    double dividend_yield;
    OptionType option_type;
    double min_moneyness, max_moneyness;
    double min_tau, max_tau;
    double min_vol, max_vol;
    double min_rate, max_rate;
};

/// Result of grid sizing from the refinement loop
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

/// Decides how to grow grids when error exceeds target.
/// Called with the worst dimension, error bins (for targeted refinement),
/// and current grids (mutable). Returns true if refinement was applied.
using RefineFn = std::function<bool(
    size_t worst_dim,
    const ErrorBins& error_bins,
    std::vector<double>& moneyness,
    std::vector<double>& tau,
    std::vector<double>& vol,
    std::vector<double>& rate)>;

/// Produces a fresh FD reference price for one validation point
using ValidateFn = std::function<std::expected<double, SolverError>(
    double spot, double strike, double tau,
    double sigma, double rate)>;

/// Compute IV error from interpolated/reference prices and option parameters.
/// Signature: (interp, ref_price, spot, strike, tau, sigma, rate, div_yield) -> error
using ComputeErrorFn = std::function<double(
    double interp, double ref_price,
    double spot, double strike, double tau,
    double sigma, double rate, double div_yield)>;

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

/// Error function using FD American vega with TV/K filter.
/// 2 extra PDE solves per validation sample — acceptable at build time.
/// Skips points where TV/K < 1e-4 (IV undefined, error metric meaningless).
ComputeErrorFn make_fd_vega_error_fn(const AdaptiveGridParams& params,
                                      const ValidateFn& validate_fn,
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

/// Run the iterative adaptive refinement loop.
///
/// Repeatedly builds a surface, validates against fresh FD solves, and refines
/// the grid until the target IV error is met or max iterations are reached.
std::expected<RefinementResult, PriceTableError> run_refinement(
    const AdaptiveGridParams& params,
    BuildFn build_fn,
    ValidateFn validate_fn,
    RefineFn refine_fn,
    const RefinementContext& ctx,
    const ComputeErrorFn& compute_error,
    const InitialGrids& initial_grids = {});

/// Resolve K_ref values from a MultiKRefConfig.
/// If config.K_refs is non-empty, returns them sorted.
/// Otherwise generates K_ref_count log-spaced values spanning
/// [spot*(1-span), spot*(1+span)].
[[nodiscard]] std::expected<std::vector<double>, PriceTableError>
resolve_k_refs(const MultiKRefConfig& config, double spot);

/// Domain bounds for segmented surface construction (log-moneyness space).
/// Produced by expand_segmented_domain() — backend-specific headroom
/// (B-spline support, Chebyshev CC margin) is added by the caller.
struct DomainBounds {
    double min_m, max_m;
    double min_tau, max_tau;
    double min_vol, max_vol;
    double min_rate, max_rate;
};

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
[[nodiscard]] std::expected<DomainBounds, PriceTableError>
expand_segmented_domain(const IVGrid& domain,
                        double maturity,
                        double dividend_yield,
                        const std::vector<Dividend>& discrete_dividends,
                        double min_K_ref);

/// Extract domain bounds from OptionGrid, expand, and add headroom.
std::expected<RefinementContext, PriceTableError>
extract_chain_domain(const OptionGrid& chain);

/// Build InitialGrids from OptionGrid (log-moneyness from strikes).
InitialGrids extract_initial_grids(const OptionGrid& chain);

}  // namespace mango
