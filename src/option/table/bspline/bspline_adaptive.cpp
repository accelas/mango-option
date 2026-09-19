// SPDX-License-Identifier: MIT
#include "mango/option/table/bspline/bspline_adaptive.hpp"
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/adaptive_metrics.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/grid_spec_types.hpp"
#include "mango/option/table/bspline/bspline_tensor_accessor.hpp"
#include "mango/option/table/bspline/bspline_pde_cache.hpp"
#include "mango/option/table/bspline/bspline_segmented_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/eep/eep_decomposer.hpp"
#include "mango/option/table/split_surface.hpp"
#include "mango/option/table/splits/multi_kref.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/dividend_utils.hpp"
#include "mango/math/cubic_spline_solver.hpp"
#include "mango/pde/core/time_domain.hpp"
#include "mango/support/ivcalc_trace.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <span>

namespace mango {

namespace {

/// Build a SegmentedPriceTableBuilder::Config from a SegmentedAdaptiveConfig.
/// K_ref is set to 0 -- caller must set it or use build_segmented_surfaces().
SegmentedPriceTableBuilder::Config make_seg_config(
    const SegmentedAdaptiveConfig& config,
    const std::vector<double>& m_grid,
    const std::vector<double>& v_grid,
    const std::vector<double>& r_grid,
    std::span<const double> tau_grid)
{
    GridAccuracyParams accuracy;
    // Spatial extraction feeds a second interpolant. Resolve it more finely
    // than the fitted axis so refinement does not fit a fixed coarse PDE.
    const size_t largest_odd_grid = accuracy.max_spatial_points -
        (accuracy.max_spatial_points % 2 == 0);
    accuracy.min_spatial_points = std::min(largest_odd_grid,
        std::max(size_t{201}, 2 * m_grid.size() + 1));
    return {
        .K_ref = 0.0,
        .option_type = config.option_type,
        .dividends = {.dividend_yield = config.dividend_yield,
                      .discrete_dividends = config.discrete_dividends},
        .grid = {.moneyness = m_grid, .vol = v_grid, .rate = r_grid},
        .maturity = config.maturity,
        .pde_accuracy = accuracy,
        .tau_grid = {tau_grid.begin(), tau_grid.end()},
    };
}

}  // anonymous namespace

// ============================================================================
// B-spline refinement strategy
// ============================================================================

RefineFn make_bspline_refine_fn(const AdaptiveGridParams& params) {
    return [params](size_t requested_dim,
                    std::span<const std::pair<double, double>> focus_intervals,
                    std::vector<double>& moneyness,
                    std::vector<double>& tau,
                    std::vector<double>& vol,
                    std::vector<double>& rate) -> RefineOutcome
    {
        // Insert midpoints in `grid` within the target intervals (empty
        // focus_intervals => the whole [lo, hi] axis, i.e. uniform
        // refinement). Returns true iff at least one midpoint was inserted.
        auto refine_grid_targeted = [&params, focus_intervals](
            std::vector<double>& grid, double lo, double hi) -> bool
        {
            size_t target_size = std::min(
                static_cast<size_t>(grid.size() * params.refinement_factor),
                params.max_points_per_dim
            );

            // Already at or beyond the limit - no refinement possible
            if (target_size <= grid.size()) return false;

            size_t max_new_points = target_size - grid.size();

            // Build set of intervals to refine: caller-supplied focus
            // intervals, or the whole axis when none were given.
            std::vector<std::pair<double, double>> refine_intervals;
            if (focus_intervals.empty()) {
                refine_intervals.push_back({lo, hi});
            } else {
                refine_intervals.assign(focus_intervals.begin(), focus_intervals.end());
            }

            // Insert midpoints only in intervals that need refinement
            std::vector<double> new_grid = grid;
            size_t points_added = 0;

            for (size_t i = 0; i + 1 < grid.size() && points_added < max_new_points; ++i) {
                double midpoint = (grid[i] + grid[i + 1]) / 2.0;

                // Check if midpoint falls in a refine interval
                bool should_refine = false;
                for (const auto& [int_lo, int_hi] : refine_intervals) {
                    if (midpoint >= int_lo && midpoint <= int_hi) {
                        should_refine = true;
                        break;
                    }
                }

                if (should_refine) {
                    new_grid.push_back(midpoint);
                    points_added++;
                }
            }

            if (points_added == 0) return false;

            std::sort(new_grid.begin(), new_grid.end());
            new_grid.erase(std::unique(new_grid.begin(), new_grid.end()),
                           new_grid.end());
            grid = std::move(new_grid);
            return true;
        };

        // Need domain bounds for targeted refinement
        double m_lo = moneyness.front(), m_hi = moneyness.back();
        double t_lo = tau.front(), t_hi = tau.back();
        double v_lo = vol.front(), v_hi = vol.back();
        double r_lo = rate.front(), r_hi = rate.back();

        bool changed = false;
        switch (requested_dim) {
            case 0: changed = refine_grid_targeted(moneyness, m_lo, m_hi); break;
            case 1: changed = refine_grid_targeted(tau, t_lo, t_hi); break;
            case 2: changed = refine_grid_targeted(vol, v_lo, v_hi); break;
            case 3: changed = refine_grid_targeted(rate, r_lo, r_hi); break;
            default: break;
        }
        // Never redirects: changed_dim == requested_dim whenever changed.
        return RefineOutcome{
            .changed = changed,
            .changed_dim = changed ? static_cast<int>(requested_dim) : -1,
        };
    };
}

namespace {

// ============================================================================
// Segmented surface helpers
// ============================================================================

void record_sampling(BuildDiagnostics& diagnostics,
                     const SegmentedPriceTableBuilder::BuildResult& result) {
    diagnostics.sample_rows += result.sample_rows;
    diagnostics.sample_points += result.sample_points;
    diagnostics.tau_point_cap_hits += result.tau_point_cap_hits;
}

/// Build a SegmentedSurface for each K_ref in the list.
/// Takes a Config template with K_ref set per iteration.
std::expected<std::vector<BSplineSegmentedSurface>, PriceTableError>
build_segmented_surfaces(
    SegmentedPriceTableBuilder::Config base_config,
    const std::vector<double>& ref_values,
    size_t& total_pde_solves, BuildDiagnostics& diagnostics)
{
    std::vector<BSplineSegmentedSurface> surfaces;
    surfaces.reserve(ref_values.size());

    for (double ref : ref_values) {
        base_config.K_ref = ref;
        auto result = SegmentedPriceTableBuilder::build_with_diagnostics(base_config);
        if (!result.has_value()) {
            return std::unexpected(result.error());
        }
        total_pde_solves += result->pde_solves;
        record_sampling(diagnostics, *result);
        surfaces.push_back(std::move(result->surface));
    }

    return surfaces;
}


/// Solve missing PDE slices, dispatching on PDEGridSpec variant.
BatchAmericanOptionResult solve_missing_slices(
    BatchAmericanOptionSolver& batch_solver,
    const std::vector<PricingParams>& missing_params,
    std::span<const double> m_grid,
    const PDEGridSpec& pde_grid)
{
    // No missing slices means no solve, irrespective of grid mode.
    if (missing_params.empty()) {
        return {};
    }

    if (const auto* explicit_grid = std::get_if<PDEGridConfig>(&pde_grid)) {
        // Incremental slices obey the same explicit-grid constraint as the
        // initial builder; only automatic requests may estimate a new grid.
        return batch_solver.solve_batch(missing_params, true, nullptr, *explicit_grid);
    }

    if (const auto* accuracy_grid = std::get_if<GridAccuracyParams>(&pde_grid)) {
        GridAccuracyParams accuracy = *accuracy_grid;
        // Every moneyness node is read from the batch solutions, so the
        // solver must resolve the whole node span (spec D12).
        accuracy.log_moneyness_coverage = LogMoneynessRange::of(m_grid);
        // One shared grid per cohort (spec D13): keeps every cached slice on
        // the same x grid and the branch's numbers unchanged.
        auto estimate = estimate_batch_pde_grid_config(missing_params, accuracy);
        if (!estimate) {
            BatchAmericanOptionResult failure;
            failure.failed_count = missing_params.size();
            failure.results.reserve(missing_params.size());
            for (size_t i = 0; i < missing_params.size(); ++i) {
                failure.results.emplace_back(std::unexpected(SolverError{
                    .code = SolverErrorCode::InvalidConfiguration, .iterations = 0}));
            }
            return failure;
        }
        return batch_solver.solve_batch(missing_params, true, nullptr, *estimate);
    }

    // Should not reach here -- PDEGridSpec is a variant with two alternatives
    return {};
}

static BatchAmericanOptionResult merge_results(
    const BSplinePDECache& cache,
    const std::vector<PricingParams>& all_params,
    const std::vector<size_t>& fresh_indices,
    const BatchAmericanOptionResult& fresh_results)
{
    BatchAmericanOptionResult merged;
    merged.results.reserve(all_params.size());
    merged.failed_count = 0;

    // Create a map from fresh_indices to fresh_results for fast lookup
    std::map<size_t, size_t> fresh_map;
    for (size_t i = 0; i < fresh_indices.size(); ++i) {
        fresh_map[fresh_indices[i]] = i;
    }

    // Build merged result vector
    for (size_t i = 0; i < all_params.size(); ++i) {
        auto fresh_it = fresh_map.find(i);
        if (fresh_it != fresh_map.end()) {
            // Use fresh result
            size_t fresh_idx = fresh_it->second;
            if (fresh_idx < fresh_results.results.size()) {
                const auto& fresh = fresh_results.results[fresh_idx];
                if (fresh.has_value()) {
                    // Create new AmericanOptionResult sharing the same grid
                    merged.results.push_back(AmericanOptionResult(
                        fresh.value().grid(), all_params[i]));
                } else {
                    // Copy the error
                    merged.results.push_back(std::unexpected(fresh.error()));
                    merged.failed_count++;
                }
            } else {
                // Should never happen, but handle gracefully
                merged.results.push_back(std::unexpected(SolverError{
                    .code = SolverErrorCode::InvalidConfiguration,
                    .iterations = 0
                }));
                merged.failed_count++;
            }
        } else {
            // Use cached result
            double sigma = all_params[i].volatility;
            double rate = get_zero_rate(all_params[i].rate, all_params[i].maturity);
            auto cached = cache.get(sigma, rate);
            if (cached) {
                merged.results.push_back(AmericanOptionResult(
                    cached->grid(), all_params[i]));
            } else {
                // Cache miss - should never happen
                merged.results.push_back(std::unexpected(SolverError{
                    .code = SolverErrorCode::InvalidConfiguration,
                    .iterations = 0
                }));
                merged.failed_count++;
            }
        }
    }

    return merged;
}

static std::expected<SurfaceHandle, PriceTableError>
build_cached_surface(
    const AdaptiveGridParams& params,
    BSplinePDECache& cache,
    const std::vector<double>& m_grid,
    const std::vector<double>& tau_grid,
    const std::vector<double>& v_grid,
    const std::vector<double>& r_grid,
    double K_ref,
    double dividend_yield,
    const PDEGridSpec& pde_grid,
    OptionType type,
    size_t& build_iteration,
    std::shared_ptr<const BSplineND<double, 4>>& last_spline,
    PriceTableAxes& last_axes)
{
    auto builder_result = PriceTableBuilder::from_vectors(
        m_grid, tau_grid, v_grid, r_grid,
        K_ref, pde_grid, type, dividend_yield,
        params.max_failure_rate);

    if (!builder_result.has_value()) {
        return std::unexpected(builder_result.error());
    }

    auto& [builder, axes] = builder_result.value();

    // Upfront explicit-grid coverage check, mirroring
    // PriceTableBuilderND::build(): an explicit grid narrower than the
    // moneyness fit axis would be silently extrapolated by
    // extract_tensor.  Auto-estimated grids are widened instead
    // (solve_missing_slices).
    if (const auto* explicit_grid = std::get_if<PDEGridConfig>(&pde_grid)) {
        const auto& m_axis = axes.grids[0];
        if (!m_axis.empty() &&
            (m_axis.front() < explicit_grid->grid_spec.x_min() ||
             m_axis.back() > explicit_grid->grid_spec.x_max())) {
            return std::unexpected(
                PriceTableError{PriceTableErrorCode::InvalidConfig});
        }
    }

    // On first iteration, set the initial tau grid; subsequent iterations
    // compare against it and clear cache only if tau actually changed.
    if (build_iteration == 0) {
        cache.set_tau_grid(tau_grid);
    } else {
        cache.invalidate_if_tau_changed(tau_grid);
    }
    build_iteration++;

    // Generate all (sigma,r) parameter combinations
    auto all_params = builder.make_batch(axes);

    // Extract (sigma,r) pairs from all_params
    std::vector<std::pair<double, double>> all_pairs;
    all_pairs.reserve(all_params.size());
    for (const auto& p : all_params) {
        double rate = get_zero_rate(p.rate, p.maturity);
        all_pairs.emplace_back(p.volatility, rate);
    }

    // Find which pairs are missing from cache
    auto missing_indices = cache.get_missing_indices(all_pairs);

    // Build batch of params for missing pairs only
    std::vector<PricingParams> missing_params;
    missing_params.reserve(missing_indices.size());
    for (size_t idx : missing_indices) {
        missing_params.push_back(all_params[idx]);
    }

    // Solve missing pairs
    BatchAmericanOptionResult fresh_results;
    if (!missing_params.empty()) {
        BatchAmericanOptionSolver batch_solver;
        batch_solver.set_snapshot_times(std::span{tau_grid});

        fresh_results = solve_missing_slices(
            batch_solver, missing_params, axes.grids[0], pde_grid);

        // Add fresh results to cache
        for (size_t i = 0; i < fresh_results.results.size(); ++i) {
            if (fresh_results.results[i].has_value()) {
                double sigma = missing_params[i].volatility;
                double rate = get_zero_rate(missing_params[i].rate, missing_params[i].maturity);
                auto result_ptr = std::make_shared<AmericanOptionResult>(
                    fresh_results.results[i].value().grid(),
                    missing_params[i]);
                cache.add(sigma, rate, std::move(result_ptr));
            }
        }
    } else {
        fresh_results.failed_count = 0;
    }

    // Merge cached + fresh results into full batch
    auto merged_results = merge_results(cache, all_params, missing_indices, fresh_results);

    // EEP transform: convert normalized prices to early exercise premium
    PriceTableBuilder::TensorTransformFn eep_transform =
        [K_ref, type, dividend_yield](PriceTensor& tensor, const PriceTableAxes& ax) {
            BSplineTensorAccessor accessor(tensor, ax, K_ref);
            eep_decompose(accessor, AnalyticalEEP(type, dividend_yield));
        };

    // Assemble surface: extract → repair → EEP → fit → build
    DividendSpec divs{.dividend_yield = dividend_yield, .discrete_dividends = {}};
    auto assembly = builder.assemble_surface(
        merged_results, axes, K_ref, divs, eep_transform);
    if (!assembly.has_value()) {
        return std::unexpected(assembly.error());
    }

    // Store for later extraction
    last_spline = assembly->spline;
    last_axes = axes;

    size_t pde_solves = missing_params.size();

    // Return a handle that queries the surface (reconstruct full American price)
    auto wrapper = make_bspline_surface(assembly->spline, K_ref, dividend_yield, type);
    if (!wrapper.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    // One wrapper shared by both callables: the round-trip inversion must
    // take its price and its slope from the same object.
    auto w = std::make_shared<BSplinePriceTable>(std::move(*wrapper));
    return SurfaceHandle{
        .price = [w](double query_spot, double strike, double tau,
                     double sigma, double rate) -> double {
            return w->price(query_spot, strike, tau, sigma, rate);
        },
        .vega = [w](double query_spot, double strike, double tau,
                    double sigma, double rate) -> double {
            return w->vega(query_spot, strike, tau, sigma, rate);
        },
        .pde_solves = pde_solves
    };
}

}  // anonymous namespace

// ============================================================================
// Public free functions
// ============================================================================

std::expected<BSplineAdaptiveResult, PriceTableError>
build_adaptive_bspline(const AdaptiveGridParams& params,
                       const OptionGrid& chain,
                       PDEGridSpec pde_grid,
                       OptionType type)
{
    // Create a fresh BSplinePDECache for this build
    BSplinePDECache cache;

    // Headroom scale comes from the expected seeded moneyness density
    // (spec D3), not from the raw strike count.
    auto domain = extract_chain_domain(
        chain, std::max(chain.strikes.size(), params.min_moneyness_points));
    if (!domain.has_value()) {
        return std::unexpected(domain.error());
    }
    auto ctx = std::move(*domain);
    ctx.option_type = type;

    // Shared state for the last spline built (so we can extract it after refinement)
    std::shared_ptr<const BSplineND<double, 4>> last_spline;
    PriceTableAxes last_axes;

    // Iteration counter for cache management (set_tau_grid vs invalidate_if_tau_changed)
    size_t build_iteration = 0;

    BuildFn build_fn = [&](std::span<const double> m_grid,
                           std::span<const double> tau_grid,
                           std::span<const double> v_grid,
                           std::span<const double> r_grid) {
        return build_cached_surface(
            params,
            cache,
            {m_grid.begin(), m_grid.end()},
            {tau_grid.begin(), tau_grid.end()},
            {v_grid.begin(), v_grid.end()},
            {r_grid.begin(), r_grid.end()},
            chain.spot, chain.dividend_yield,
            pde_grid, type,
            build_iteration, last_spline, last_axes);
    };

    // One counter for the whole build: the loop's fresh and holdout
    // preparations all draw from it, so `total_pde_solves` reports the
    // reference work once (spec D7).
    auto ref_counter = std::make_shared<ReferenceSolveCounter>();
    // A continuous surface describes a contract from now: no schedule to
    // roll, and no fixed expiry to roll it onto.
    const ReferenceOracle oracle{
        .dividend_yield = chain.dividend_yield,
        .option_type = type,
        .discrete_dividends = {},
        .reference_maturity = std::nullopt,
        .accuracy = make_grid_accuracy(kReferenceAccuracy),
    };
    auto prepare_refs_fn = make_stencil_refs_fn(params, oracle, ref_counter);
    auto score_fn = make_round_trip_score_fn(params, ctx, type);

    auto refine_fn = make_bspline_refine_fn(params);
    // No state hooks: the B-spline refiner's whole state is the grids (D6).
    auto grid_result = run_refinement(params, build_fn,
                                      refine_fn, ctx, prepare_refs_fn, score_fn,
                                      extract_initial_grids(chain),
                                      RefineStateHooks{});
    if (!grid_result.has_value()) {
        return std::unexpected(grid_result.error());
    }

    auto& grids = grid_result.value();

    BSplineAdaptiveResult result;
    result.spline = last_spline;
    result.axes = last_axes;
    result.K_ref = chain.spot;
    result.dividend_yield = chain.dividend_yield;
    result.iterations = std::move(grids.iterations);
    result.achieved_max_error = grids.achieved_max_error;
    result.achieved_avg_error = grids.achieved_avg_error;
    result.target_met = grids.target_met;
    result.diagnostics = std::move(grids.diagnostics);
    result.diagnostics.reference_solves_fine = ref_counter->fine_attempts.load();
    result.diagnostics.reference_solves_coarse = ref_counter->coarse_attempts.load();
    result.sample_bounds = ctx.sample_bounds;
    // `pde_solves_validation` counts *preparations*, not solves (spec D7);
    // the solves the stencil actually ran come from the one counter this
    // build owns, so they are added once rather than per iteration.
    result.total_pde_solves = result.diagnostics.reference_solves_fine
                            + result.diagnostics.reference_solves_coarse;
    for (const auto& it : result.iterations) {
        result.total_pde_solves += it.pde_solves_table;
    }

    return result;
}

// ============================================================================
// BSplineSegmentedBuilder
// ============================================================================

std::expected<BSplineSegmentedBuilder, PriceTableError>
BSplineSegmentedBuilder::create(const SegmentedAdaptiveConfig& config,
                                 const IVGrid& domain)
{
    auto K_refs = resolve_k_refs(config.kref_config, config.spot);
    if (!K_refs) return std::unexpected(K_refs.error());

    // Support domain: the user's ranges widened for the cumulative discrete
    // dividend spot shifts, so the fitted surface covers post-dividend
    // spots.  That widening is interpolation *support*, not something the
    // user asked to be able to query.
    auto support = expand_segmented_domain(
        domain, config.maturity, config.dividend_yield,
        config.discrete_dividends, K_refs->front());
    if (!support) return std::unexpected(support.error());

    // Sample (measurement) domain: the same construction *without* the
    // dividend widening (spec D2 -- accuracy is never measured in the
    // unqueryable support band).  With a 20%-of-spot dividend schedule the
    // two differ by more than a factor of two in strike, and measuring the
    // wider one condemns surfaces on strikes the user never asked for.
    auto sample = expand_segmented_domain(
        domain, config.maturity, config.dividend_yield, {}, K_refs->front());
    if (!sample) return std::unexpected(sample.error());

    // Support headroom is deliberately NOT applied here: its scale depends
    // on AdaptiveGridParams::min_moneyness_points (spec D3), which is only
    // available at build_adaptive() time.
    return BSplineSegmentedBuilder(config, std::move(*K_refs), *sample,
                                   *support, domain);
}

BSplineSegmentedBuilder::BSplineSegmentedBuilder(
    SegmentedAdaptiveConfig config,
    std::vector<double> K_refs,
    SurfaceBounds sample_domain,
    SurfaceBounds support_domain,
    IVGrid initial_grid)
    : config_(std::move(config))
    , K_refs_(std::move(K_refs))
    , sample_domain_(sample_domain)
    , support_domain_(support_domain)
    , initial_grid_(std::move(initial_grid))
{}

std::expected<BSplineMultiKRefInner, PriceTableError>
BSplineSegmentedBuilder::assemble(std::vector<BSplineSegmentedSurface> surfaces) const
{
    std::vector<BSplineMultiKRefEntry> entries;
    entries.reserve(K_refs_.size());
    for (size_t i = 0; i < K_refs_.size(); ++i) {
        entries.push_back({.K_ref = K_refs_[i], .surface = std::move(surfaces[i])});
    }
    return build_multi_kref_surface(std::move(entries));
}

std::expected<BSplineSegmentedAdaptiveResult, PriceTableError>
BSplineSegmentedBuilder::build_adaptive(const AdaptiveGridParams& params) const
{
    // 0. Derive the fit domain from the sample domain (spec D3): headroom
    //    scale is the expected seeded moneyness density, not the user's
    //    knot count.
    SurfaceBounds fit_domain = support_domain_;
    // The fitted table includes the exact payoff at expiry. Keep zero on
    // the refinement axis as well, rather than adding a nearly coincident
    // PDE row at the positive IV measurement floor.
    fit_domain.tau_min = 0.0;
    {
        double h = spline_support_headroom(
            sample_domain_.m_max - sample_domain_.m_min,
            std::max(initial_grid_.moneyness.size(),
                     params.min_moneyness_points));
        fit_domain.m_min -= h;
        fit_domain.m_max += h;
    }

    // 1. Select reference-strike probes, including the served range endpoints
    auto probes = select_probes(K_refs_, config_.spot);

    // The strike range the user can actually query (m = ln(spot/K)).
    const double user_k_lo = config_.spot * std::exp(-sample_domain_.m_max);
    const double user_k_hi = config_.spot * std::exp(-sample_domain_.m_min);
    // Endpoint K_refs can have empty served bands when callers provide
    // support beyond the query range. Include the references nearest the
    // queried strikes so refinement also sees the wings of that range.
    for (double strike : {user_k_lo, user_k_hi}) {
        const auto nearest = std::ranges::min_element(K_refs_, {},
            [strike](double k) { return std::abs(k - strike); });
        if (std::ranges::find(probes, *nearest) == probes.end()) probes.push_back(*nearest);
    }

    // The strike band a probe dominates in the assembled surface.  The
    // assembly blends in inverse strike, so a probe dominates between the
    // harmonic midpoints to its neighbours.  This scopes a sizing measurement, not a safety gate —
    // the assembled surface's own final validation queries the true blend.
    // The outermost bands run out to the user's strike range, and a single
    // K_ref serves all of it.
    const auto strike_band = [this, user_k_lo, user_k_hi](double k) {
        const size_t n = K_refs_.size();
        const size_t idx = static_cast<size_t>(
            std::ranges::lower_bound(K_refs_, k) - K_refs_.begin());
        double lo = (idx == 0)
            ? user_k_lo : 2.0 * K_refs_[idx - 1] / (1.0 + K_refs_[idx - 1] / K_refs_[idx]);
        double hi = (idx + 1 >= n)
            ? user_k_hi : 2.0 * K_refs_[idx] / (1.0 + K_refs_[idx] / K_refs_[idx + 1]);
        return std::pair{std::max(lo, user_k_lo), std::min(hi, user_k_hi)};
    };

    InitialGrids initial_grids;
    initial_grids.moneyness = initial_grid_.moneyness;
    initial_grids.vol = seed_grid(initial_grid_.vol, fit_domain.sigma_min,
                                  fit_domain.sigma_max, 7);
    // Raw cash-dividend prices can be nearly flat in volatility beside an
    // exercise boundary. A four-site cubic can undershoot that flat region;
    // resolve the seed before relying on randomly located refinement probes.
    const size_t vol_seed_count = std::min(params.max_points_per_dim, size_t{7});
    if (initial_grids.vol.size() < vol_seed_count) {
        const size_t missing = vol_seed_count - initial_grids.vol.size();
        initial_grids.vol = insert_largest_gap_midpoints(
            std::move(initial_grids.vol), missing, vol_seed_count);
    }
    initial_grids.rate = initial_grid_.rate;
    // Each dividend jump/projection starts another temporal boundary layer.
    // Cluster in sqrt(elapsed time) in every regime, retaining physical tau.
    std::vector<double> origins{fit_domain.tau_min, fit_domain.tau_max};
    for (const auto& div : filter_and_merge_dividends(
             config_.discrete_dividends, config_.maturity)) {
        const double t = config_.maturity - div.calendar_time;
        if (t > fit_domain.tau_min && t < fit_domain.tau_max) origins.push_back(t);
    }
    std::ranges::sort(origins);
    const size_t seed_cap = std::max(params.max_points_per_dim, size_t{4});
    const size_t per_regime = std::min(seed_cap, size_t{9});
    initial_grids.tau.push_back(origins.front());
    for (size_t j = 1; j < origins.size(); ++j) {
        for (size_t i = 1; i < per_regime; ++i) {
            const double u = static_cast<double>(i) / (per_regime - 1);
            const double t = i + 1 == per_regime ? origins[j]
                : std::lerp(origins[j - 1], origins[j], u * u);
            // Match the collocation solver's absolute separation floor.
            // Optional clustered sites must not make an otherwise usable
            // very short event interval unfittable.
            constexpr double min_spacing = 1e-14;
            if (i + 1 < per_regime &&
                (t - initial_grids.tau.back() < min_spacing ||
                 origins[j] - t < min_spacing)) continue;
            initial_grids.tau.push_back(t);
        }
    }
    // Bound seed cost for schedules with many events. The builder still
    // inserts event endpoints and the minimum cubic support in each regime.
    if (initial_grids.tau.size() > seed_cap) {
        std::vector<double> capped;
        for (size_t i = 0; i < seed_cap; ++i)
            capped.push_back(initial_grids.tau[i * (initial_grids.tau.size() - 1) / (seed_cap - 1)]);
        initial_grids.tau = std::move(capped);
    }

    // 2. Run adaptive refinement per probe, measured over its own band
    BuildDiagnostics diagnostics;

    // One counter for the whole build: every probe loop, the final
    // validation and the retry draw from it, so `total_pde_solves` reports
    // the reference work once (spec D7).
    auto ref_counter = std::make_shared<ReferenceSolveCounter>();
    // A segmented surface follows one fixed expiry: the schedule is anchored
    // to `config_.maturity` and rolled onto each query's remaining life.
    const ReferenceOracle oracle{
        .dividend_yield = config_.dividend_yield,
        .option_type = config_.option_type,
        .discrete_dividends = config_.discrete_dividends,
        .reference_maturity = config_.maturity,
        .accuracy = make_grid_accuracy(kReferenceAccuracy),
    };
    auto user_refs_fn = make_stencil_refs_fn(params, oracle, ref_counter);

    // Spec D4: an event neighborhood has no sampled representation, so a
    // sample inside a gap segment is excluded before any reference is drawn
    // -- it is neither a reference nor a defect of the candidate.
    const auto segments = compute_segment_boundaries(
        config_.discrete_dividends, config_.maturity,
        fit_domain.tau_min, fit_domain.tau_max);
    const std::function<bool(double)> maturity_is_supported =
        [b = segments.bounds, g = segments.is_gap](double tau) {
            for (size_t s = 0; s + 1 < b.size(); ++s)
                if (g[s] && tau > b[s] && tau < b[s + 1]) return false;
            return true;
        };

    std::vector<RefinementResult> probe_results;
    for (double probe_ref : probes) {
        // Measurement domain for this probe: the user's tau/vol/rate ranges,
        // moneyness restricted to the band this probe serves.
        SurfaceBounds probe_sample = sample_domain_;
        bool band_usable = false;
        if (auto [k_lo, k_hi] = strike_band(probe_ref);
            k_lo > 0.0 && k_hi > k_lo) {
            probe_sample.m_min = std::log(config_.spot / k_hi);
            probe_sample.m_max = std::log(config_.spot / k_lo);
            // A band too thin for the loop's non-degeneracy check is widened
            // about its midpoint, never past the user's own range.
            constexpr double kMinBandWidth = 1e-3;
            if (probe_sample.m_max - probe_sample.m_min < kMinBandWidth) {
                const double mid =
                    0.5 * (probe_sample.m_min + probe_sample.m_max);
                probe_sample.m_min = std::max(sample_domain_.m_min,
                                              mid - 0.5 * kMinBandWidth);
                probe_sample.m_max = std::min(sample_domain_.m_max,
                                              mid + 0.5 * kMinBandWidth);
            }
            band_usable = probe_sample.m_max > probe_sample.m_min;
        }

        if (!band_usable) {
            // Nothing measurable: this probe serves no strike the user can
            // query.  It still contributes its seed sizes to the aggregate.
            RefinementContext seed_ctx{
                .spot = config_.spot,
                .dividend_yield = config_.dividend_yield,
                .option_type = config_.option_type,
                .bounds = fit_domain,
                .sample_bounds = sample_domain_,
            };
            auto seeded = seed_refinement_grids(params, seed_ctx,
                                                initial_grids);
            RefinementResult skipped;
            skipped.tau_points = static_cast<int>(seeded.tau.size());
            IterationStats stats;
            stats.refined_dim = -3;  // marker: probe skipped, empty band
            stats.grid_sizes = {seeded.moneyness.size(), seeded.tau.size(),
                                seeded.vol.size(), seeded.rate.size()};
            skipped.iterations.push_back(stats);
            skipped.moneyness = std::move(seeded.moneyness);
            skipped.tau = std::move(seeded.tau);
            skipped.vol = std::move(seeded.vol);
            skipped.rate = std::move(seeded.rate);
            probe_results.push_back(std::move(skipped));
            continue;
        }

        BuildFn build_fn = [this, probe_ref, &diagnostics](
            std::span<const double> m_grid,
            std::span<const double> tau_grid,
            std::span<const double> v_grid,
            std::span<const double> r_grid)
            -> std::expected<SurfaceHandle, PriceTableError>
        {
            std::vector<double> m_vec(m_grid.begin(), m_grid.end());
            std::vector<double> v_vec(v_grid.begin(), v_grid.end());
            std::vector<double> r_vec(r_grid.begin(), r_grid.end());
            auto seg_cfg = make_seg_config(config_, m_vec, v_vec, r_vec, tau_grid);
            seg_cfg.K_ref = probe_ref;
            auto result = SegmentedPriceTableBuilder::build_with_diagnostics(seg_cfg);
            if (!result) return std::unexpected(result.error());
            record_sampling(diagnostics, *result);
            auto shared = std::make_shared<BSplineSegmentedSurface>(std::move(result->surface));
            return SurfaceHandle{
                // A probe surface is a single-K_ref object: TauSegmentSplit
                // *discards* the query strike and prices at K_ref, so calling
                // it with the validation strike would compare a K_ref-struck
                // price against a K-struck reference (errors of several IV
                // points on a healthy surface).  Map the query onto the
                // probe's own K_ref problem instead -- and measure it against
                // a reference solved at the *same* scaled coordinates (see
                // the PrepareRefsFn below), so what the loop scores is this
                // probe's interpolation error and nothing else.
                .price = [shared, probe_ref](double spot, double strike,
                                             double tau, double sigma,
                                             double rate) -> double {
                    const double scale =
                        (strike > 0.0) ? strike / probe_ref : 1.0;
                    return scale * shared->price(spot / scale, probe_ref,
                                                 tau, sigma, rate);
                },
                // The same map, so the round-trip inversion takes its slope
                // from the same probe problem as its price.
                .vega = [shared, probe_ref](double spot, double strike,
                                            double tau, double sigma,
                                            double rate) -> double {
                    const double scale =
                        (strike > 0.0) ? strike / probe_ref : 1.0;
                    return scale * shared->vega(spot / scale, probe_ref,
                                                tau, sigma, rate);
                },
                .pde_solves = result->pde_solves
            };
        };

        // The probe's references live on the probe's own problem.  A query
        // (S, K) reaches the surface as scale * probe(S/scale, K_ref) with
        // scale = K/K_ref, so the reference is the FD solve at
        // (S/scale, K_ref) under the same dividend schedule, scaled the same
        // way.  Rescaling the *option* rather than the query -- pricing
        // (S, K) and comparing against a K_ref-struck surface, or leaning on
        // P(lambda S, lambda K) homogeneity -- does not hold here: absolute
        // discrete dividends are not scaled by lambda, so
        // scale * P(S/scale, K_ref; D) is P(S, K; scale * D), and the
        // (scale - 1) * D * dP/dD residual would be scored as interpolation
        // error.  Every monetary term of the stencil scales together, so the
        // error the loop sees is unaffected by the scaling itself.
        PrepareRefsFn prepare_refs_fn =
            [user_refs_fn, probe_ref](double spot, double strike, double tau,
                                      double sigma, double rate)
            -> std::expected<ErrorRefs, SolverError> {
            const double scale = (strike > 0.0) ? strike / probe_ref : 1.0;
            auto refs = user_refs_fn(spot / scale, probe_ref, tau, sigma, rate);
            if (!refs) return std::unexpected(refs.error());
            // Spec L6: every monetary quantity of the probe's stencil
            // scales alike; the sigma coordinates and `resolved` do not.
            ErrorRefs scaled = *refs;
            scaled.ref_price = scale * refs->ref_price;
            scaled.bracket_lo_price = scale * refs->bracket_lo_price;
            scaled.bracket_hi_price = scale * refs->bracket_hi_price;
            scaled.delta = scale * refs->delta;
            scaled.delta_lo = scale * refs->delta_lo;
            scaled.delta_hi = scale * refs->delta_hi;
            return scaled;
        };

        // Grids still span the whole fit domain; only the *measurement* is
        // band-scoped (spec D2: measure where the surface is used).
        RefinementContext ctx{
            .spot = config_.spot,
            .dividend_yield = config_.dividend_yield,
            .option_type = config_.option_type,
            .bounds = fit_domain,
            .sample_bounds = probe_sample,
            .maturity_is_supported = maturity_is_supported,
        };
        auto score_fn = make_round_trip_score_fn(params, ctx,
                                                 config_.option_type);

        auto refine_fn = make_bspline_refine_fn(params);
        // No state hooks: the B-spline refiner's whole state is the grids.
        auto sizes = run_refinement(params, build_fn,
                                    refine_fn, ctx,
                                    prepare_refs_fn, score_fn, initial_grids,
                                    RefineStateHooks{});
        if (!sizes) return std::unexpected(sizes.error());
        probe_results.push_back(std::move(*sizes));
    }

    // 3. Merge the probes' refined knot positions (issue #461) and gather
    //    convergence stats across probes
    auto agg = aggregate_probe_grids(probe_results, params.max_points_per_dim);

    // Worst-case convergence stats across probes
    std::vector<IterationStats> all_iterations;
    size_t total_pde = 0;
    for (const auto& pr : probe_results) {
        for (const auto& it : pr.iterations) {
            all_iterations.push_back(it);
            // `pde_solves_validation` counts *preparations*, not solves
            // (spec D7); the solves the stencil actually ran come from the
            // one counter this build owns and are added once, below.
            total_pde += it.pde_solves_table;
        }
    }

    // 4. Build all surfaces on the merged grids.  These are the positions the
    //    probe loops actually chose; re-spacing them uniformly at the same
    //    sizes was the defect behind issue #461.

    auto seg_template = make_seg_config(config_, agg.moneyness, agg.vol, agg.rate, agg.tau);
    auto seg_surfaces = build_segmented_surfaces(seg_template, K_refs_, total_pde, diagnostics);
    if (!seg_surfaces) return std::unexpected(seg_surfaces.error());

    // 5. Assemble multi-K_ref surface
    auto surface = assemble(std::move(*seg_surfaces));
    if (!surface) return std::unexpected(surface.error());

    // 6. Final multi-K_ref validation at arbitrary strikes (spec D9).
    //
    // The probe loops measured single-K_ref surfaces on their own bands; the
    // object the caller receives is the blend of *all* K_refs on the uniform
    // aggregated grids, so it gets its own references and its own gate.
    RefinementContext final_ctx{
        .spot = config_.spot,
        .dividend_yield = config_.dividend_yield,
        .option_type = config_.option_type,
        .bounds = fit_domain,
        // Final validation measures the user-facing domain (spec D2), not
        // the interpolation support band.
        .sample_bounds = sample_domain_,
        .maturity_is_supported = maturity_is_supported,
    };

    auto final_score_fn = make_round_trip_score_fn(params, final_ctx,
                                                   config_.option_type);

    // References are computed ONCE here and reused for the retry, so the two
    // assembled surfaces are compared on identical coordinates.  The
    // assembled surface answers on the user's own contract, so it takes the
    // unscaled stencil -- the probe adapter above belongs to the sizing
    // loops alone.
    auto validation = detail::prepare_final_validation(
        params, final_ctx, user_refs_fn, params.lhs_seed + 999);
    if (!validation) return std::unexpected(validation.error());

    // The references are not free: the stencil runs up to six solves per
    // prepared point across the probe loops, the final validation and the
    // retry, and the caller's PDE budget should say so (spec D7).
    total_pde += ref_counter->fine_attempts.load()
               + ref_counter->coarse_attempts.load();

    // The lambda captures the surface by pointer, not by reference to the
    // parameter: a reference capture would dangle the moment `handle_for`
    // returns, even though the referent outlives every use.
    const auto handle_for = [](const BSplineMultiKRefInner& s) {
        return SurfaceHandle{
            .price = [p = &s](double query_spot, double strike, double tau,
                              double sigma, double rate) -> double {
                return p->price(query_spot, strike, tau, sigma, rate);
            },
            .vega = [p = &s](double query_spot, double strike, double tau,
                             double sigma, double rate) -> double {
                return p->vega(query_spot, strike, tau, sigma, rate);
            },
            .pde_solves = 0,
        };
    };

    // `orig_handle` points into `*surface`, which is moved from below when
    // the original is the pick.  It must not be used past that move: the
    // scoring here and the retry comparison are its only uses, and the
    // monotonicity scan deliberately re-derives a handle from
    // `picked_surface` rather than reusing this one.
    const SurfaceHandle orig_handle = handle_for(*surface);
    const auto orig_score = detail::score_final_surface(
        validation->points, orig_handle, final_score_fn, final_ctx);

    // 7. Optional retry with bumped grids -- triggered when the original
    //    misses the target OR is not viable at all (spec D9 step 2).  The
    //    bump inserts midpoints into the merged grids' largest gaps so the
    //    retained positions survive it.
    std::optional<BSplineMultiKRefInner> retry_surface;
    std::optional<detail::FinalScore> retry_score;
    IVGrid retry_grid;
    std::vector<double> retry_tau_grid;

    if (detail::needs_final_retry(orig_score, params.target_iv_error)) {
        const size_t cap = params.max_points_per_dim;
        auto retry_tau = insert_largest_gap_midpoints(agg.tau, 2, cap);

        auto retry_m = insert_largest_gap_midpoints(agg.moneyness, 2, cap);
        auto retry_v = insert_largest_gap_midpoints(agg.vol, 1, cap);
        auto retry_r = insert_largest_gap_midpoints(agg.rate, 1, cap);

        auto retry_template = make_seg_config(config_, retry_m, retry_v, retry_r, retry_tau);
        auto retry_segs = build_segmented_surfaces(retry_template, K_refs_, total_pde, diagnostics);
        if (retry_segs) {
            auto assembled = assemble(std::move(*retry_segs));
            if (assembled) {
                retry_surface = std::move(*assembled);
                // Scored on the SAME cached refs -- no second reference pass.
                retry_score = detail::score_final_surface(
                    validation->points, handle_for(*retry_surface),
                    final_score_fn, final_ctx);
                retry_grid = retry_template.grid;
                retry_tau_grid = std::move(retry_tau);
            }
        }
    }

    // 8. Return the lower-error viable surface; neither viable => refuse.
    const auto pick = detail::select_final_surface(orig_score, retry_score);
    if (pick == detail::FinalPick::None) {
        // The returned error has no room for the outcomes that produced the
        // refusal, so the probe carries them, summed over both assembled
        // surfaces the gate considered (spec D7).
        PointStatusCounts totals = orig_score.status_counts;
        size_t rescues = orig_score.edge_band_rescues;
        if (retry_score) {
            for (size_t i = 0; i < totals.size(); ++i) {
                totals[i] += retry_score->status_counts[i];
            }
            rescues += retry_score->edge_band_rescues;
        }
        MANGO_TRACE_ADAPTIVE_NO_VIABLE_SURFACE(
            ADAPTIVE_STAGE_FINAL, retry_score ? 2u : 1u,
            totals[static_cast<size_t>(PointStatus::SurfaceNoRoot)],
            totals[static_cast<size_t>(PointStatus::SurfaceAmbiguous)],
            totals[static_cast<size_t>(PointStatus::SurfaceNonConvergent)],
            totals[static_cast<size_t>(PointStatus::SurfaceNonFinite)],
            totals[static_cast<size_t>(PointStatus::SurfaceVegaTooSmall)],
            rescues);
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::NoViableSurface});
    }

    const bool use_retry = (pick == detail::FinalPick::Retry);
    const detail::FinalScore& final_score = use_retry ? *retry_score : orig_score;
    BSplineMultiKRefInner picked_surface =
        use_retry ? std::move(*retry_surface) : std::move(*surface);

    diagnostics.target_met =
        final_score.viable() &&
        final_score.max_error <= params.target_iv_error;
    diagnostics.achieved_max_error = final_score.max_error;
    diagnostics.achieved_avg_error = final_score.avg_error;
    // Iterations actually built across the probe loops: the retention final
    // rebuild (-2) and the skipped-probe marker (-3) are not builds charged
    // to a budget (spec D7).
    diagnostics.total_iterations = static_cast<size_t>(std::ranges::count_if(
        all_iterations,
        [](const IterationStats& it) { return it.refined_dim >= -1; }));
    // Same meaning as the loop's (spec D7): `holdout_points` is the usable
    // reference set, `holdout_points_measured` how much of it actually scored
    // the returned surface -- the difference is the unresolved references and
    // the points where the shipped inversion failed on the surface's price.
    diagnostics.holdout_points = validation->points.size();
    diagnostics.holdout_points_measured = final_score.measured;
    diagnostics.holdout_points_invalid = validation->invalid + final_score.skipped;
    diagnostics.holdout_points_unresolved = final_score.unresolved;
    diagnostics.surface_failures = final_score.surface_failures;
    diagnostics.edge_band_rescues = final_score.edge_band_rescues;
    diagnostics.max_price_residual = final_score.max_price_residual;
    // An estimate of how far the references themselves could be off, never a
    // certificate that they are not further.
    diagnostics.reference_uncertainty_max = final_score.max_delta;
    diagnostics.reference_solves_fine = ref_counter->fine_attempts.load();
    diagnostics.reference_solves_coarse = ref_counter->coarse_attempts.load();
    for (const auto& pr : probe_results) {
        diagnostics.build_failure_fallback |=
            pr.diagnostics.build_failure_fallback;
    }
    detail::scan_monotonicity(validation->points, handle_for(picked_surface),
                              final_ctx, params.target_iv_error, diagnostics);
    diagnostics.iterations = all_iterations;

    size_t max_tau_points = 0;
    for (const auto& piece : picked_surface.pieces().front().pieces()) {
        max_tau_points = std::max(max_tau_points, piece.interpolant().get().grid(1).size());
    }
    return BSplineSegmentedAdaptiveResult{
        .surface = std::move(picked_surface),
        .grid = use_retry ? retry_grid : seg_template.grid,
        .tau_points_per_segment = static_cast<int>(max_tau_points),
        .iterations = std::move(all_iterations),
        .achieved_max_error = final_score.max_error,
        .achieved_avg_error = final_score.avg_error,
        .target_met = diagnostics.target_met,
        .total_pde_solves = total_pde,
        .used_retry = use_retry,
        .diagnostics = std::move(diagnostics),
        .sample_bounds = sample_domain_,
        .tau_grid = use_retry ? std::move(retry_tau_grid) : std::move(agg.tau),
    };
}

std::expected<BSplineSegmentedAdaptiveResult, PriceTableError>
build_adaptive_bspline_segmented(const AdaptiveGridParams& params,
                                 const SegmentedAdaptiveConfig& config,
                                 const IVGrid& domain)
{
    auto builder = BSplineSegmentedBuilder::create(config, domain);
    if (!builder) return std::unexpected(builder.error());
    return builder->build_adaptive(params);
}

}  // namespace mango
