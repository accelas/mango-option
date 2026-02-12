// SPDX-License-Identifier: MIT
#include "mango/option/table/bspline/bspline_adaptive.hpp"
#include "mango/option/table/adaptive_grid_types.hpp"
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
#include "mango/math/latin_hypercube.hpp"
#include "mango/pde/core/time_domain.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
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

// ============================================================================
// B-spline refinement strategy
// ============================================================================

/// Create a RefineFn that does B-spline midpoint insertion in problematic bins.
static RefineFn make_bspline_refine_fn(const AdaptiveGridParams& params) {
    return [&params](size_t worst_dim, const ErrorBins& error_bins,
                     std::vector<double>& moneyness,
                     std::vector<double>& tau,
                     std::vector<double>& vol,
                     std::vector<double>& rate) -> bool
    {
        auto problematic = error_bins.problematic_bins(worst_dim);

        auto refine_grid_targeted = [&params, &problematic](
            std::vector<double>& grid, double lo, double hi)
        {
            size_t target_size = std::min(
                static_cast<size_t>(grid.size() * params.refinement_factor),
                params.max_points_per_dim
            );

            // Already at or beyond the limit - no refinement possible
            if (target_size <= grid.size()) return;

            size_t max_new_points = target_size - grid.size();

            // Build set of intervals to refine based on problematic bins
            std::vector<std::pair<double, double>> refine_intervals;
            if (problematic.empty()) {
                // No concentrated errors - uniform refinement
                refine_intervals.push_back({lo, hi});
            } else {
                // Only refine within problematic bins
                constexpr double N_BINS = static_cast<double>(ErrorBins::N_BINS);
                for (size_t bin : problematic) {
                    double bin_lo = lo + (hi - lo) * bin / N_BINS;
                    double bin_hi = lo + (hi - lo) * (bin + 1) / N_BINS;
                    refine_intervals.push_back({bin_lo, bin_hi});
                }
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

            std::sort(new_grid.begin(), new_grid.end());
            new_grid.erase(std::unique(new_grid.begin(), new_grid.end()),
                           new_grid.end());
            grid = std::move(new_grid);
        };

        // Need domain bounds for targeted refinement
        double m_lo = moneyness.front(), m_hi = moneyness.back();
        double t_lo = tau.front(), t_hi = tau.back();
        double v_lo = vol.front(), v_hi = vol.back();
        double r_lo = rate.front(), r_hi = rate.back();

        switch (worst_dim) {
            case 0: refine_grid_targeted(moneyness, m_lo, m_hi); break;
            case 1: refine_grid_targeted(tau, t_lo, t_hi); break;
            case 2: refine_grid_targeted(vol, v_lo, v_hi); break;
            case 3: refine_grid_targeted(rate, r_lo, r_hi); break;
        }
        return true;
    };
}

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
    const PDEGridSpec& pde_grid,
    const std::vector<double>& tau_grid)
{
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

        if (max_sigma_sqrt_tau >= 1e-10) {
            double required_n_sigma = (max_abs_x / max_sigma_sqrt_tau) * DOMAIN_MARGIN_FACTOR;
            accuracy.n_sigma = std::max(5.0, required_n_sigma);
        }

        batch_solver.set_grid_accuracy(accuracy);
        return batch_solver.solve_batch(missing_params, true);
    }

    if (const auto* accuracy_grid = std::get_if<GridAccuracyParams>(&pde_grid)) {
        batch_solver.set_grid_accuracy(*accuracy_grid);
        return batch_solver.solve_batch(missing_params, true);
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
            batch_solver, missing_params, all_params, pde_grid, tau_grid);

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

    auto domain = extract_chain_domain(chain);
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

    auto compute_error_fn = make_fd_vega_error_fn(params, validate_fn, type);

    auto refine_fn = make_bspline_refine_fn(params);
    auto grid_result = run_refinement(params, build_fn, validate_fn,
                                      refine_fn, ctx, compute_error_fn,
                                      extract_initial_grids(chain));
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

    auto dom = expand_segmented_domain(
        domain, config.maturity, config.dividend_yield,
        config.discrete_dividends, K_refs->front());
    if (!dom) return std::unexpected(dom.error());

    // B-spline support headroom on moneyness
    double h = spline_support_headroom(dom->max_m - dom->min_m, domain.moneyness.size());
    dom->min_m -= h;
    dom->max_m += h;

    return BSplineSegmentedBuilder(config, std::move(*K_refs), *dom, domain);
}

BSplineSegmentedBuilder::BSplineSegmentedBuilder(
    SegmentedAdaptiveConfig config,
    std::vector<double> K_refs,
    DomainBounds domain,
    IVGrid initial_grid)
    : config_(std::move(config))
    , K_refs_(std::move(K_refs))
    , domain_(domain)
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
    // 1. Select probe values (up to 3: front, back, nearest ATM)
    auto probes = select_probes(K_refs_, config_.spot);

    // 2. Run adaptive refinement per probe
    std::vector<RefinementResult> probe_results;
    for (double probe_ref : probes) {
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
                .price = [shared](double spot, double strike,
                                  double tau, double sigma, double rate) -> double {
                    return shared->price(spot, strike, tau, sigma, rate);
                },
                .pde_solves = 0
            };
        };

        auto validate_fn = make_validate_fn(
            config_.dividend_yield, config_.option_type,
            config_.discrete_dividends);

        auto compute_error_fn = make_fd_vega_error_fn(
            params, validate_fn, config_.option_type);

        RefinementContext ctx{
            .spot = config_.spot,
            .dividend_yield = config_.dividend_yield,
            .option_type = config_.option_type,
            .min_moneyness = domain_.min_m,
            .max_moneyness = domain_.max_m,
            .min_tau = domain_.min_tau,
            .max_tau = domain_.max_tau,
            .min_vol = domain_.min_vol,
            .max_vol = domain_.max_vol,
            .min_rate = domain_.min_rate,
            .max_rate = domain_.max_rate,
        };

        InitialGrids initial_grids;
        initial_grids.moneyness = initial_grid_.moneyness;
        initial_grids.vol = initial_grid_.vol;
        initial_grids.rate = initial_grid_.rate;

        auto refine_fn = make_bspline_refine_fn(params);
        auto sizes = run_refinement(params, build_fn, validate_fn,
                                    refine_fn, ctx,
                                    compute_error_fn, initial_grids);
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
    auto final_m = linspace(domain_.min_m, domain_.max_m, gsz.moneyness);
    auto final_v = linspace(domain_.min_vol, domain_.max_vol, gsz.vol);
    auto final_r = linspace(domain_.min_rate, domain_.max_rate, gsz.rate);
    int max_tau_pts = gsz.tau_points;

    auto seg_template = make_seg_config(config_, final_m, final_v, final_r, max_tau_pts);
    auto seg_surfaces = build_segmented_surfaces(seg_template, K_refs_);
    if (!seg_surfaces) return std::unexpected(seg_surfaces.error());

    // 5. Assemble multi-K_ref surface
    auto surface = assemble(std::move(*seg_surfaces));
    if (!surface) return std::unexpected(surface.error());

    // 6. Final multi-K_ref validation at arbitrary strikes
    auto final_validate_fn = make_validate_fn(
        config_.dividend_yield, config_.option_type,
        config_.discrete_dividends);
    auto final_error_fn = make_fd_vega_error_fn(
        params, final_validate_fn, config_.option_type);

    auto final_samples = latin_hypercube_4d(
        params.validation_samples, params.lhs_seed + 999);

    std::array<std::pair<double, double>, 4> final_bounds = {{
        {domain_.min_m, domain_.max_m},
        {domain_.min_tau, domain_.max_tau},
        {domain_.min_vol, domain_.max_vol},
        {domain_.min_rate, domain_.max_rate},
    }};
    auto scaled = scale_lhs_samples(final_samples, final_bounds);

    double final_max_error = 0.0;
    double final_sum_error = 0.0;
    size_t valid = 0;

    for (const auto& sample : scaled) {
        double m = sample[0], tau = sample[1], sigma = sample[2], rate = sample[3];
        double strike = config_.spot * std::exp(-m);

        double interp = surface->price(config_.spot, strike, tau, sigma, rate);

        auto fd = final_validate_fn(config_.spot, strike, tau, sigma, rate);
        if (!fd.has_value()) continue;

        double ref_price = fd.value();
        double err = final_error_fn(
            interp, ref_price,
            config_.spot, strike, tau, sigma, rate,
            config_.dividend_yield);

        final_max_error = std::max(final_max_error, err);
        final_sum_error += err;
        if (err > 0.0) valid++;
    }
    double final_avg_error = valid > 0 ? final_sum_error / static_cast<double>(valid) : 0.0;

    // 7. Optional retry with bumped grids
    if (valid > 0 && final_max_error > params.target_iv_error) {
        size_t bumped_m = std::min(gsz.moneyness + 2, params.max_points_per_dim);
        size_t bumped_v = std::min(gsz.vol + 1, params.max_points_per_dim);
        size_t bumped_r = std::min(gsz.rate + 1, params.max_points_per_dim);
        int bumped_tau = std::min(gsz.tau_points + 2,
            static_cast<int>(params.max_points_per_dim));

        auto retry_m = linspace(domain_.min_m, domain_.max_m, bumped_m);
        auto retry_v = linspace(domain_.min_vol, domain_.max_vol, bumped_v);
        auto retry_r = linspace(domain_.min_rate, domain_.max_rate, bumped_r);

        auto retry_template = make_seg_config(config_, retry_m, retry_v, retry_r, bumped_tau);
        auto retry_segs = build_segmented_surfaces(retry_template, K_refs_);
        if (retry_segs) {
            auto retry_surface = assemble(std::move(*retry_segs));
            if (retry_surface) {
                return BSplineSegmentedAdaptiveResult{
                    .surface = std::move(*retry_surface),
                    .grid = {.moneyness = retry_m, .vol = retry_v, .rate = retry_r},
                    .tau_points_per_segment = bumped_tau,
                    .iterations = std::move(all_iterations),
                    .achieved_max_error = final_max_error,
                    .achieved_avg_error = final_avg_error,
                    .target_met = false,  // retry means target wasn't met
                    .total_pde_solves = total_pde,
                    .used_retry = true,
                };
            }
        }
    }

    bool met = (valid == 0) || (final_max_error <= params.target_iv_error);
    return BSplineSegmentedAdaptiveResult{
        .surface = std::move(*surface),
        .grid = seg_template.grid,
        .tau_points_per_segment = max_tau_pts,
        .iterations = std::move(all_iterations),
        .achieved_max_error = final_max_error,
        .achieved_avg_error = final_avg_error,
        .target_met = met,
        .total_pde_solves = total_pde,
        .used_retry = false,
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
