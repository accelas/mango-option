// SPDX-License-Identifier: MIT
#include "mango/option/table/bspline/bspline_adaptive.hpp"
#include "mango/option/table/adaptive_grid_builder.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_tensor_accessor.hpp"
#include "mango/option/table/bspline/bspline_slice_cache.hpp"
#include "mango/option/table/bspline/bspline_segmented_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/eep/eep_decomposer.hpp"
#include "mango/option/table/split_surface.hpp"
#include "mango/option/table/splits/multi_kref.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/dividend_utils.hpp"
#include "mango/math/black_scholes_analytics.hpp"
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

constexpr double kMinPositive = 1e-6;

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

/// Result of the shared segmented probe-and-build pipeline.
struct SegmentedBuildResult {
    std::vector<BSplineSegmentedSurface> surfaces;
    SegmentedPriceTableBuilder::Config seg_template;
    MaxGridSizes gsz;
    // Domain bounds (needed for validation/retry)
    double min_m, max_m;
    double min_vol, max_vol;
    double min_rate, max_rate;
    double min_tau, max_tau;
};

/// Shared probe-and-build pipeline for build_segmented.
/// Selects representative probes, runs adaptive refinement per probe, aggregates
/// grid sizes, then builds one SegmentedSurface per ref_value.
static std::expected<SegmentedBuildResult, PriceTableError>
probe_and_build(
    const AdaptiveGridParams& params,
    const SegmentedAdaptiveConfig& config,
    const std::vector<double>& ref_values,
    const IVGrid& domain)
{
    // 1. Select probe values (up to 3: front, back, nearest ATM)
    auto probes = select_probes(ref_values, config.spot);

    // 2. Expand domain bounds
    if (domain.moneyness.empty() || domain.vol.empty() || domain.rate.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    // domain.moneyness is log(S/K) -- callers convert from S/K at the
    // user API boundary (see interpolated_iv_solver.cpp).
    double min_m = domain.moneyness.front();
    double max_m = domain.moneyness.back();

    // Expand lower bound in moneyness-space to account for cumulative
    // discrete-dividend spot shifts, then map back to log-moneyness.
    double total_div = total_discrete_dividends(
        config.discrete_dividends, config.maturity);
    double ref_min = ref_values.front();
    double expansion = (ref_min > 0.0) ? total_div / ref_min : 0.0;
    if (expansion > 0.0) {
        double m_min_money = std::exp(min_m);
        double expanded = std::max(m_min_money - expansion, 0.01);
        min_m = std::log(expanded);
    }

    double min_vol = domain.vol.front();
    double max_vol = domain.vol.back();
    double min_rate = domain.rate.front();
    double max_rate = domain.rate.back();

    expand_domain_bounds(min_m, max_m, 0.10);
    double h = spline_support_headroom(max_m - min_m, domain.moneyness.size());
    min_m -= h;
    max_m += h;

    expand_domain_bounds(min_vol, max_vol, 0.10, kMinPositive);
    expand_domain_bounds(min_rate, max_rate, 0.04);

    double min_tau = std::min(0.01, config.maturity * 0.5);
    double max_tau = config.maturity;
    expand_domain_bounds(min_tau, max_tau, 0.1, kMinPositive);
    max_tau = std::min(max_tau, config.maturity);

    // 3. Run adaptive refinement per probe
    std::vector<RefinementResult> probe_results;
    for (double probe_ref : probes) {
        BuildFn build_fn = [&config, probe_ref](
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
            auto seg_cfg = make_seg_config(config, m_vec, v_vec, r_vec, tau_pts);
            seg_cfg.K_ref = probe_ref;
            auto surface = SegmentedPriceTableBuilder::build(seg_cfg);
            if (!surface.has_value()) {
                return std::unexpected(surface.error());
            }
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
            config.dividend_yield, config.option_type,
            config.discrete_dividends);

        auto compute_error_fn = make_bs_vega_error_fn(params);

        RefinementContext ctx{
            .spot = config.spot,
            .dividend_yield = config.dividend_yield,
            .option_type = config.option_type,
            .min_moneyness = min_m,
            .max_moneyness = max_m,
            .min_tau = min_tau,
            .max_tau = max_tau,
            .min_vol = min_vol,
            .max_vol = max_vol,
            .min_rate = min_rate,
            .max_rate = max_rate,
        };

        InitialGrids initial_grids;
        initial_grids.moneyness = domain.moneyness;
        initial_grids.vol = domain.vol;
        initial_grids.rate = domain.rate;

        auto refine_fn = make_bspline_refine_fn(params);
        auto sizes = run_refinement(params, build_fn, validate_fn,
                                    refine_fn, ctx,
                                    compute_error_fn, initial_grids);
        if (!sizes.has_value()) {
            return std::unexpected(sizes.error());
        }
        probe_results.push_back(std::move(*sizes));
    }

    // 4. Aggregate max grid sizes across probes
    auto gsz = aggregate_max_sizes(probe_results);

    // 5. Build final uniform grids and all surfaces
    auto final_m = linspace(min_m, max_m, gsz.moneyness);
    auto final_v = linspace(min_vol, max_vol, gsz.vol);
    auto final_r = linspace(min_rate, max_rate, gsz.rate);
    int max_tau_pts = gsz.tau_points;

    auto seg_template = make_seg_config(config, final_m, final_v, final_r, max_tau_pts);

    auto seg_surfaces = build_segmented_surfaces(seg_template, ref_values);
    if (!seg_surfaces.has_value()) {
        return std::unexpected(seg_surfaces.error());
    }

    return SegmentedBuildResult{
        .surfaces = std::move(*seg_surfaces),
        .seg_template = std::move(seg_template),
        .gsz = gsz,
        .min_m = min_m,
        .max_m = max_m,
        .min_vol = min_vol,
        .max_vol = max_vol,
        .min_rate = min_rate,
        .max_rate = max_rate,
        .min_tau = min_tau,
        .max_tau = max_tau,
    };
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

/// Compute IV error metric from price error and vega
static double compute_error_metric(const AdaptiveGridParams& params,
                                   double price_error, double vega) {
    return compute_iv_error(price_error, vega,
                            params.vega_floor, params.target_iv_error);
}

static BatchAmericanOptionResult merge_results(
    const SliceCache& cache,
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
    SliceCache& cache,
    const std::vector<double>& m_grid,
    const std::vector<double>& tau_grid,
    const std::vector<double>& v_grid,
    const std::vector<double>& r_grid,
    double K_ref,
    double dividend_yield,
    const PDEGridSpec& pde_grid,
    OptionType type,
    size_t& build_iteration,
    std::shared_ptr<const PriceTableSurface>& last_surface,
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

    // Extract tensor from merged results
    auto tensor_result = builder.extract_tensor(merged_results, axes);
    if (!tensor_result.has_value()) {
        return std::unexpected(tensor_result.error());
    }

    auto& extraction = tensor_result.value();

    // Repair failed slices if needed
    if (!extraction.failed_pde.empty() || !extraction.failed_spline.empty()) {
        auto repair_result = builder.repair_failed_slices(
            extraction.tensor, extraction.failed_pde,
            extraction.failed_spline, axes);
        if (!repair_result.has_value()) {
            return std::unexpected(repair_result.error());
        }
    }

    // EEP decomposition: convert normalized prices to early exercise premium
    BSplineTensorAccessor accessor(extraction.tensor, axes, K_ref);
    eep_decompose(accessor, AnalyticalEEP(type, dividend_yield));

    // Fit coefficients
    auto fit_result = builder.fit_coeffs(extraction.tensor, axes);
    if (!fit_result.has_value()) {
        return std::unexpected(fit_result.error());
    }

    // Build surface
    auto surface = PriceTableSurface::build(
        axes, std::move(fit_result->coefficients), K_ref,
        DividendSpec{.dividend_yield = dividend_yield, .discrete_dividends = {}});
    if (!surface.has_value()) {
        return std::unexpected(surface.error());
    }

    // Store for later extraction
    last_surface = surface.value();
    last_axes = axes;

    size_t pde_solves = missing_params.size();

    // Return a handle that queries the surface (reconstruct full American price)
    auto surface_ptr = surface.value();
    auto wrapper = make_bspline_surface(surface_ptr, type);
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
    // Create a fresh SliceCache for this build
    SliceCache cache;

    auto domain = extract_chain_domain(chain);
    if (!domain.has_value()) {
        return std::unexpected(domain.error());
    }
    auto ctx = std::move(*domain);
    ctx.option_type = type;

    // Shared state for the last surface built (so we can extract it after refinement)
    std::shared_ptr<const PriceTableSurface> last_surface;
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
            build_iteration, last_surface, last_axes);
    };

    auto validate_fn = make_validate_fn(chain.dividend_yield, type);

    // compute_error: FD American vega for standard path
    auto compute_error_fn = [&params, &validate_fn](
        double interp_price, double ref_price,
        double spot, double strike, double tau,
        double sigma, double rate,
        double /*dividend_yield*/) -> double
    {
        double price_error = std::abs(interp_price - ref_price);

        // Compute American vega via central finite difference
        double eps = std::max(1e-4, 0.01 * sigma);

        // Ensure sigma - eps stays positive (minimum 1e-4)
        double sigma_dn = std::max(1e-4, sigma - eps);
        double sigma_up = sigma + eps;

        // Adjust eps if we had to clamp sigma_dn
        double effective_eps = (sigma_up - sigma_dn) / 2.0;

        auto fd_up = validate_fn(spot, strike, tau, sigma_up, rate);
        auto fd_dn = validate_fn(spot, strike, tau, sigma_dn, rate);

        double vega;
        if (fd_up.has_value() && fd_dn.has_value() && effective_eps > 1e-6) {
            vega = (fd_up.value() - fd_dn.value()) / (2.0 * effective_eps);
        } else {
            // FD bump failed or eps too small -- clamp to floor inside compute_error_metric
            vega = 0.0;
        }
        return compute_error_metric(params, price_error, vega);
    };

    auto refine_fn = make_bspline_refine_fn(params);
    auto grid_result = run_refinement(params, build_fn, validate_fn,
                                      refine_fn, ctx, compute_error_fn,
                                      extract_initial_grids(chain));
    if (!grid_result.has_value()) {
        return std::unexpected(grid_result.error());
    }

    auto& grids = grid_result.value();

    BSplineAdaptiveResult result;
    result.surface = last_surface;
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

std::expected<BSplineSegmentedAdaptiveResult, PriceTableError>
build_adaptive_bspline_segmented(const AdaptiveGridParams& params,
                                 const SegmentedAdaptiveConfig& config,
                                 const IVGrid& domain)
{
    // 1. Determine full K_ref list
    std::vector<double> K_refs = config.kref_config.K_refs;
    if (K_refs.empty()) {
        const int count = config.kref_config.K_ref_count;
        const double span = config.kref_config.K_ref_span;
        if (count < 1 || span <= 0.0) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
        }
        const double log_lo = std::log(1.0 - span);
        const double log_hi = std::log(1.0 + span);
        K_refs.reserve(static_cast<size_t>(count));
        if (count == 1) {
            K_refs.push_back(config.spot);
        } else {
            for (int i = 0; i < count; ++i) {
                double t = static_cast<double>(i) / static_cast<double>(count - 1);
                K_refs.push_back(config.spot * std::exp(log_lo + t * (log_hi - log_lo)));
            }
        }
    }
    std::sort(K_refs.begin(), K_refs.end());

    // 2. Probe, refine, and build all surfaces
    auto result = probe_and_build(params, config, K_refs, domain);
    if (!result.has_value()) {
        return std::unexpected(result.error());
    }
    auto& build = *result;

    // 3. Assemble MultiKRefSurface
    std::vector<BSplineMultiKRefEntry> entries;
    for (size_t i = 0; i < K_refs.size(); ++i) {
        entries.push_back({.K_ref = K_refs[i], .surface = std::move(build.surfaces[i])});
    }
    auto surface = build_multi_kref_surface(std::move(entries));
    if (!surface.has_value()) {
        return std::unexpected(surface.error());
    }

    // 4. Final multi-K_ref validation at arbitrary strikes
    auto final_samples = latin_hypercube_4d(
        params.validation_samples, params.lhs_seed + 999);

    std::array<std::pair<double, double>, 4> final_bounds = {{
        {build.min_m, build.max_m},
        {build.min_tau, build.max_tau},
        {build.min_vol, build.max_vol},
        {build.min_rate, build.max_rate},
    }};
    auto scaled = scale_lhs_samples(final_samples, final_bounds);

    double final_max_error = 0.0;
    size_t valid = 0;

    for (const auto& sample : scaled) {
        double m = sample[0], tau = sample[1], sigma = sample[2], rate = sample[3];
        double strike = config.spot * std::exp(-m);

        double interp = surface->price(config.spot, strike, tau, sigma, rate);

        PricingParams pp;
        pp.spot = config.spot;
        pp.strike = strike;
        pp.maturity = tau;
        pp.rate = rate;
        pp.dividend_yield = config.dividend_yield;
        pp.option_type = config.option_type;
        pp.volatility = sigma;
        pp.discrete_dividends = config.discrete_dividends;

        auto fd = solve_american_option(pp);
        if (!fd.has_value()) continue;

        double price_error = std::abs(interp - fd->value());
        double vega = bs_vega(config.spot, strike, tau, sigma, rate, config.dividend_yield);
        double err = compute_error_metric(params, price_error, vega);

        final_max_error = std::max(final_max_error, err);
        valid++;
    }

    if (valid > 0 && final_max_error > params.target_iv_error) {
        // Bump grids by one refinement step and rebuild (one retry)
        size_t bumped_m = std::min(build.gsz.moneyness + 2, params.max_points_per_dim);
        size_t bumped_v = std::min(build.gsz.vol + 1, params.max_points_per_dim);
        size_t bumped_r = std::min(build.gsz.rate + 1, params.max_points_per_dim);
        int bumped_tau = std::min(build.gsz.tau_points + 2,
            static_cast<int>(params.max_points_per_dim));

        auto retry_m = linspace(build.min_m, build.max_m, bumped_m);
        auto retry_v = linspace(build.min_vol, build.max_vol, bumped_v);
        auto retry_r = linspace(build.min_rate, build.max_rate, bumped_r);

        build.seg_template.grid = {.moneyness = retry_m, .vol = retry_v, .rate = retry_r};
        build.seg_template.tau_points_per_segment = bumped_tau;
        auto retry_segs = build_segmented_surfaces(build.seg_template, K_refs);
        if (retry_segs.has_value()) {
            std::vector<BSplineMultiKRefEntry> retry_entries;
            for (size_t i = 0; i < K_refs.size(); ++i) {
                retry_entries.push_back({.K_ref = K_refs[i], .surface = std::move((*retry_segs)[i])});
            }
            auto retry_surface = build_multi_kref_surface(std::move(retry_entries));
            if (retry_surface.has_value()) {
                return BSplineSegmentedAdaptiveResult{
                    .surface = std::move(*retry_surface),
                    .grid = {.moneyness = retry_m, .vol = retry_v, .rate = retry_r},
                    .tau_points_per_segment = bumped_tau,
                };
            }
        }
    }

    return BSplineSegmentedAdaptiveResult{
        .surface = std::move(*surface),
        .grid = build.seg_template.grid,
        .tau_points_per_segment = build.seg_template.tau_points_per_segment,
    };
}

}  // namespace mango
