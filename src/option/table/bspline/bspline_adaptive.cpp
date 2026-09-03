// SPDX-License-Identifier: MIT
#include "mango/option/table/bspline/bspline_adaptive.hpp"
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/adaptive_metrics.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
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
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
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
    int tau_pts)
{
    return {
        .K_ref = 0.0,
        .option_type = config.option_type,
        .dividends = {.dividend_yield = config.dividend_yield,
                      .discrete_dividends = config.discrete_dividends},
        .grid = {.moneyness = m_grid, .vol = v_grid, .rate = r_grid},
        .maturity = config.maturity,
        .tau_points_per_segment = tau_pts,
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

/// Build a SegmentedSurface for each K_ref in the list.
/// Takes a Config template with K_ref set per iteration.
std::expected<std::vector<BSplineSegmentedSurface>, PriceTableError>
build_segmented_surfaces(
    SegmentedPriceTableBuilder::Config base_config,
    const std::vector<double>& ref_values)
{
    std::vector<BSplineSegmentedSurface> surfaces;
    surfaces.reserve(ref_values.size());

    for (double ref : ref_values) {
        base_config.K_ref = ref;
        auto surface = SegmentedPriceTableBuilder::build(base_config);
        if (!surface.has_value()) {
            return std::unexpected(surface.error());
        }
        surfaces.push_back(std::move(*surface));
    }

    return surfaces;
}


/// Solve missing PDE slices, dispatching on PDEGridSpec variant.
BatchAmericanOptionResult solve_missing_slices(
    BatchAmericanOptionSolver& batch_solver,
    const std::vector<PricingParams>& missing_params,
    const std::vector<PricingParams>& all_params,
    std::span<const double> m_grid,
    const PDEGridSpec& pde_grid,
    const std::vector<double>& tau_grid)
{
    // Precondition: the only caller already skips this call when
    // missing_params is empty, but std::ranges::max below is UB over an
    // empty range, so guard the file-local helper directly too.
    if (missing_params.empty()) {
        return {};
    }

    if (const auto* explicit_grid = std::get_if<PDEGridConfig>(&pde_grid)) {
        const auto& grid_spec = explicit_grid->grid_spec;
        const size_t n_time = explicit_grid->n_time;

        constexpr double MAX_WIDTH = 5.8;
        constexpr double MAX_DX = 0.05;

        const double grid_width = grid_spec.x_max() - grid_spec.x_min();

        double max_dx;
        if (grid_spec.type() == GridSpec<double>::Type::Uniform) {
            max_dx = grid_width / static_cast<double>(grid_spec.n_points() - 1);
        } else {
            auto grid_buffer = grid_spec.generate();
            auto spacings = grid_buffer.span() | std::views::pairwise
                                               | std::views::transform([](auto pair) {
                                                     auto [a, b] = pair;
                                                     return b - a;
                                                 });
            max_dx = std::ranges::max(spacings);
        }

        auto sigma_sqrt_tau = [](const PricingParams& p) {
            return p.volatility * std::sqrt(p.maturity);
        };
        const double max_sigma_sqrt_tau = std::ranges::max(
            all_params | std::views::transform(sigma_sqrt_tau));
        const double min_required_width = 6.0 * max_sigma_sqrt_tau;

        const bool grid_meets_constraints =
            (grid_width <= MAX_WIDTH) &&
            (max_dx <= MAX_DX) &&
            (grid_width >= min_required_width);

        if (grid_meets_constraints) {
            const double max_maturity = tau_grid.back();
            TimeDomain time_domain = TimeDomain::from_n_steps(0.0, max_maturity, n_time);
            PDEGridSpec custom_grid{PDEGridConfig{grid_spec, time_domain.n_steps(), std::vector<double>{}}};
            return batch_solver.solve_batch(missing_params, true, nullptr, custom_grid);
        }

        GridAccuracyParams accuracy;
        const size_t n_points = grid_spec.n_points();
        const size_t clamped = std::clamp(n_points, size_t{100}, size_t{1200});
        accuracy.min_spatial_points = clamped;
        accuracy.max_spatial_points = clamped;
        accuracy.max_time_steps = n_time;

        if (grid_spec.type() == GridSpec<double>::Type::SinhSpaced) {
            accuracy.alpha = grid_spec.concentration();
        }

        const double x_min = grid_spec.x_min();
        const double x_max = grid_spec.x_max();
        const double max_abs_x = std::max(std::abs(x_min), std::abs(x_max));
        constexpr double DOMAIN_MARGIN_FACTOR = 1.1;

        const double missing_max_sigma_sqrt_tau = std::ranges::max(
            missing_params | std::views::transform(sigma_sqrt_tau));

        if (missing_max_sigma_sqrt_tau >= 1e-10) {
            double required_n_sigma =
                (max_abs_x / missing_max_sigma_sqrt_tau) * DOMAIN_MARGIN_FACTOR;
            accuracy.n_sigma = std::max(5.0, required_n_sigma);
        }

        // Every moneyness node is read from the batch solutions, so the
        // solver must resolve the whole node span (spec D12).
        accuracy.log_moneyness_coverage = LogMoneynessRange::of(m_grid);
        // One shared grid per cohort (spec D13): keeps every cached slice on
        // the same x grid and the branch's numbers unchanged.
        return batch_solver.solve_batch(
            missing_params, true, nullptr,
            estimate_batch_pde_grid_config(missing_params, accuracy));
    }

    if (const auto* accuracy_grid = std::get_if<GridAccuracyParams>(&pde_grid)) {
        GridAccuracyParams accuracy = *accuracy_grid;
        // Every moneyness node is read from the batch solutions, so the
        // solver must resolve the whole node span (spec D12).
        accuracy.log_moneyness_coverage = LogMoneynessRange::of(m_grid);
        // One shared grid per cohort (spec D13): keeps every cached slice on
        // the same x grid and the branch's numbers unchanged.
        return batch_solver.solve_batch(
            missing_params, true, nullptr,
            estimate_batch_pde_grid_config(missing_params, accuracy));
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
            batch_solver, missing_params, all_params, axes.grids[0], pde_grid,
            tau_grid);

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

    return SurfaceHandle{
        .price = [w = std::move(*wrapper)](double query_spot, double strike, double tau,
                                           double sigma, double rate) -> double {
            return w.price(query_spot, strike, tau, sigma, rate);
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

    auto validate_fn = make_validate_fn(chain.dividend_yield, type);

    auto prepare_refs_fn = make_fd_vega_refs_fn(params, validate_fn);
    auto score_fn = make_iv_score_fn(params, type);

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
    result.sample_bounds = ctx.sample_bounds;
    result.total_pde_solves = 0;
    for (auto& it : result.iterations) {
        // Standard path uses FD American vega: 1 base solve + 2 vega bump solves = 3x
        it.pde_solves_validation *= 3;
        result.total_pde_solves += it.pde_solves_table + it.pde_solves_validation;
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
    {
        double h = spline_support_headroom(
            sample_domain_.m_max - sample_domain_.m_min,
            std::max(initial_grid_.moneyness.size(),
                     params.min_moneyness_points));
        fit_domain.m_min -= h;
        fit_domain.m_max += h;
    }

    // 1. Select probe values (up to 3: front, back, nearest ATM)
    auto probes = select_probes(K_refs_, config_.spot);

    // The strike range the user can actually query (m = ln(spot/K)).
    const double user_k_lo = config_.spot * std::exp(-sample_domain_.m_max);
    const double user_k_hi = config_.spot * std::exp(-sample_domain_.m_min);

    // The strike band a probe dominates in the assembled surface.  The
    // assembly blends the two K_refs bracketing a query's strike linearly
    // (MultiKRefSplit::bracket), so a probe's weight is largest between the
    // midpoints to its neighbours; we take geometric midpoints since K_refs
    // are log-spaced.  This scopes a sizing measurement, not a safety gate —
    // the assembled surface's own final validation queries the true blend.
    // The outermost bands run out to the user's strike range, and a single
    // K_ref serves all of it.
    const auto strike_band = [this, user_k_lo, user_k_hi](double k) {
        const size_t n = K_refs_.size();
        const size_t idx = static_cast<size_t>(
            std::ranges::lower_bound(K_refs_, k) - K_refs_.begin());
        double lo = (idx == 0)
            ? user_k_lo : std::sqrt(K_refs_[idx - 1] * K_refs_[idx]);
        double hi = (idx + 1 >= n)
            ? user_k_hi : std::sqrt(K_refs_[idx] * K_refs_[idx + 1]);
        return std::pair{std::max(lo, user_k_lo), std::min(hi, user_k_hi)};
    };

    InitialGrids initial_grids;
    initial_grids.moneyness = initial_grid_.moneyness;
    initial_grids.vol = initial_grid_.vol;
    initial_grids.rate = initial_grid_.rate;

    // 2. Run adaptive refinement per probe, measured over its own band
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

        BuildFn build_fn = [this, probe_ref](
            std::span<const double> m_grid,
            std::span<const double> tau_grid,
            std::span<const double> v_grid,
            std::span<const double> r_grid)
            -> std::expected<SurfaceHandle, PriceTableError>
        {
            int tau_pts = static_cast<int>(tau_grid.size());
            std::vector<double> m_vec(m_grid.begin(), m_grid.end());
            std::vector<double> v_vec(v_grid.begin(), v_grid.end());
            std::vector<double> r_vec(r_grid.begin(), r_grid.end());
            auto seg_cfg = make_seg_config(config_, m_vec, v_vec, r_vec, tau_pts);
            seg_cfg.K_ref = probe_ref;
            auto surface = SegmentedPriceTableBuilder::build(seg_cfg);
            if (!surface) return std::unexpected(surface.error());
            auto shared = std::make_shared<BSplineSegmentedSurface>(std::move(*surface));
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
                .pde_solves = 0
            };
        };

        auto validate_fn = make_validate_fn(
            config_.dividend_yield, config_.option_type,
            config_.discrete_dividends);

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
        // error.  Price and vega scale together, so the IV error the loop
        // sees is unaffected by the scaling itself.
        auto base_refs_fn = make_fd_vega_refs_fn(params, validate_fn);
        PrepareRefsFn prepare_refs_fn =
            [base_refs_fn, probe_ref](double spot, double strike, double tau,
                                      double sigma, double rate)
            -> std::expected<ErrorRefs, SolverError> {
            const double scale = (strike > 0.0) ? strike / probe_ref : 1.0;
            auto refs = base_refs_fn(spot / scale, probe_ref, tau, sigma, rate);
            if (!refs) return std::unexpected(refs.error());
            return ErrorRefs{.ref_price = scale * refs->ref_price,
                             .vega = scale * refs->vega};
        };
        auto score_fn = make_iv_score_fn(params, config_.option_type);

        // Grids still span the whole fit domain; only the *measurement* is
        // band-scoped (spec D2: measure where the surface is used).
        RefinementContext ctx{
            .spot = config_.spot,
            .dividend_yield = config_.dividend_yield,
            .option_type = config_.option_type,
            .bounds = fit_domain,
            .sample_bounds = probe_sample,
        };

        auto refine_fn = make_bspline_refine_fn(params);
        // No state hooks: the B-spline refiner's whole state is the grids.
        auto sizes = run_refinement(params, build_fn,
                                    refine_fn, ctx,
                                    prepare_refs_fn, score_fn, initial_grids,
                                    RefineStateHooks{});
        if (!sizes) return std::unexpected(sizes.error());
        probe_results.push_back(std::move(*sizes));
    }

    // 3. Aggregate max grid sizes and convergence stats across probes
    auto gsz = aggregate_max_sizes(probe_results);

    // Worst-case convergence stats across probes
    std::vector<IterationStats> all_iterations;
    size_t total_pde = 0;
    for (const auto& pr : probe_results) {
        for (const auto& it : pr.iterations) {
            all_iterations.push_back(it);
            total_pde += it.pde_solves_table + it.pde_solves_validation;
        }
    }

    // 4. Build final uniform grids and all surfaces
    auto final_m = linspace(fit_domain.m_min, fit_domain.m_max, gsz.moneyness);
    auto final_v = linspace(fit_domain.sigma_min, fit_domain.sigma_max, gsz.vol);
    auto final_r = linspace(fit_domain.rate_min, fit_domain.rate_max, gsz.rate);
    int max_tau_pts = gsz.tau_points;

    auto seg_template = make_seg_config(config_, final_m, final_v, final_r, max_tau_pts);
    auto seg_surfaces = build_segmented_surfaces(seg_template, K_refs_);
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
    };

    auto final_validate_fn = make_validate_fn(
        config_.dividend_yield, config_.option_type,
        config_.discrete_dividends);
    auto final_prepare_refs_fn = make_fd_vega_refs_fn(params, final_validate_fn);
    auto final_score_fn = make_iv_score_fn(params, config_.option_type);

    // References are computed ONCE here and reused for the retry, so the two
    // assembled surfaces are compared on identical coordinates.
    auto validation = detail::prepare_final_validation(
        params, final_ctx, final_prepare_refs_fn, params.lhs_seed + 999);
    if (!validation) return std::unexpected(validation.error());

    // The final validation is not free: every reference is a base solve plus
    // two sigma bumps, and the caller's PDE budget should say so.
    total_pde += validation->ref_attempts * 3;

    // The lambda captures the surface by pointer, not by reference to the
    // parameter: a reference capture would dangle the moment `handle_for`
    // returns, even though the referent outlives every use.
    const auto handle_for = [](const BSplineMultiKRefInner& s) {
        return SurfaceHandle{
            .price = [p = &s](double query_spot, double strike, double tau,
                              double sigma, double rate) -> double {
                return p->price(query_spot, strike, tau, sigma, rate);
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
    //    misses the target OR is not viable at all (spec D9 step 2).
    std::optional<BSplineMultiKRefInner> retry_surface;
    std::optional<detail::FinalScore> retry_score;
    IVGrid retry_grid;
    int retry_tau_pts = 0;

    if (detail::needs_final_retry(orig_score, params.target_iv_error)) {
        size_t bumped_m = std::min(gsz.moneyness + 2, params.max_points_per_dim);
        size_t bumped_v = std::min(gsz.vol + 1, params.max_points_per_dim);
        size_t bumped_r = std::min(gsz.rate + 1, params.max_points_per_dim);
        int bumped_tau = std::min(gsz.tau_points + 2,
            static_cast<int>(params.max_points_per_dim));

        auto retry_m = linspace(fit_domain.m_min, fit_domain.m_max, bumped_m);
        auto retry_v = linspace(fit_domain.sigma_min, fit_domain.sigma_max, bumped_v);
        auto retry_r = linspace(fit_domain.rate_min, fit_domain.rate_max, bumped_r);

        auto retry_template = make_seg_config(config_, retry_m, retry_v, retry_r, bumped_tau);
        auto retry_segs = build_segmented_surfaces(retry_template, K_refs_);
        if (retry_segs) {
            auto assembled = assemble(std::move(*retry_segs));
            if (assembled) {
                retry_surface = std::move(*assembled);
                // Scored on the SAME cached refs -- no second reference pass.
                retry_score = detail::score_final_surface(
                    validation->points, handle_for(*retry_surface),
                    final_score_fn, final_ctx);
                retry_grid = retry_template.grid;
                retry_tau_pts = bumped_tau;
            }
        }
    }

    // 8. Return the lower-error viable surface; neither viable => refuse.
    const auto pick = detail::select_final_surface(orig_score, retry_score);
    if (pick == detail::FinalPick::None) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::NoViableSurface});
    }

    const bool use_retry = (pick == detail::FinalPick::Retry);
    const detail::FinalScore& final_score = use_retry ? *retry_score : orig_score;
    BSplineMultiKRefInner picked_surface =
        use_retry ? std::move(*retry_surface) : std::move(*surface);

    BuildDiagnostics diagnostics;
    diagnostics.target_met =
        final_score.measured > 0 &&
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
    // the returned surface -- the difference is what the score fn filtered.
    diagnostics.holdout_points = validation->points.size();
    diagnostics.holdout_points_measured = final_score.measured;
    diagnostics.holdout_points_invalid = validation->invalid + final_score.skipped;
    for (const auto& pr : probe_results) {
        diagnostics.build_failure_fallback |=
            pr.diagnostics.build_failure_fallback;
    }
    detail::scan_monotonicity(validation->points, handle_for(picked_surface),
                              final_ctx, params.target_iv_error,
                              params.vega_floor, diagnostics);
    diagnostics.iterations = all_iterations;

    return BSplineSegmentedAdaptiveResult{
        .surface = std::move(picked_surface),
        .grid = use_retry ? retry_grid : seg_template.grid,
        .tau_points_per_segment = use_retry ? retry_tau_pts : max_tau_pts,
        .iterations = std::move(all_iterations),
        .achieved_max_error = final_score.max_error,
        .achieved_avg_error = final_score.avg_error,
        .target_met = diagnostics.target_met,
        .total_pde_solves = total_pde,
        .used_retry = use_retry,
        .diagnostics = std::move(diagnostics),
        .sample_bounds = sample_domain_,
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
