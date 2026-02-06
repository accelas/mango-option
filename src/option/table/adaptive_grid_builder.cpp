// SPDX-License-Identifier: MIT
#include "mango/option/table/adaptive_grid_builder.hpp"
#include "mango/math/black_scholes_analytics.hpp"
#include "mango/math/latin_hypercube.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/table/american_price_surface.hpp"
#include "mango/option/table/price_table_surface.hpp"
#include "mango/option/table/segmented_price_table_builder.hpp"
#include "mango/option/table/spliced_surface_builder.hpp"
#include "mango/pde/core/time_domain.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <map>
#include <random>
#include <ranges>

namespace mango {

namespace {

// ============================================================================
// Shared helpers
// ============================================================================

constexpr double kMinPositive = 1e-6;

/// Expand [lo, hi] to at least min_spread wide, keeping lo >= kMinPositive.
void expand_bounds_positive(double& lo, double& hi, double min_spread) {
    if (hi - lo < min_spread) {
        double mid = (lo + hi) / 2.0;
        lo = mid - min_spread / 2.0;
        hi = mid + min_spread / 2.0;
    }
    if (lo < kMinPositive) {
        double shift = kMinPositive - lo;
        lo = kMinPositive;
        hi += shift;
    }
}

/// Expand [lo, hi] to at least min_spread wide (no positivity constraint).
void expand_bounds(double& lo, double& hi, double min_spread) {
    if (hi - lo < min_spread) {
        double mid = (lo + hi) / 2.0;
        lo = mid - min_spread / 2.0;
        hi = mid + min_spread / 2.0;
    }
}

/// Select up to 3 probes from a sorted vector: front, back, and nearest to
/// reference_value. Returns all items if size <= 3.
std::vector<double> select_probes(const std::vector<double>& items,
                                  double reference_value) {
    if (items.size() <= 3) return items;
    std::vector<double> probes;
    probes.push_back(items.front());
    probes.push_back(items.back());
    auto atm_it = std::min_element(items.begin(), items.end(),
        [&](double a, double b) {
            return std::abs(a - reference_value) < std::abs(b - reference_value);
        });
    if (*atm_it != items.front() && *atm_it != items.back()) {
        probes.push_back(*atm_it);
    }
    return probes;
}

/// Build a SegmentedPriceTableBuilder::Config from a SegmentedAdaptiveConfig.
/// K_ref is set to 0 — caller must set it or use build_segmented_surfaces().
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
        .skip_moneyness_expansion = true,
    };
}

/// Sum discrete dividends strictly inside (0, maturity) with positive amount.
double total_discrete_dividends(const std::vector<Dividend>& dividends,
                                double maturity) {
    double total = 0.0;
    for (const auto& div : dividends) {
        if (div.calendar_time > 0.0 && div.calendar_time < maturity &&
            div.amount > 0.0) {
            total += div.amount;
        }
    }
    return total;
}

// ============================================================================
// Types for the callback-based refinement loop
// ============================================================================

/// Type-erased surface for validation queries
struct SurfaceHandle {
    std::function<double(double spot, double strike, double tau,
                         double sigma, double rate)> price;
    size_t pde_solves = 0;  ///< Number of PDE solves used to build this surface
};

/// Builds a surface from current grids, returns handle for querying
using BuildFn = std::function<std::expected<SurfaceHandle, PriceTableError>(
    const std::vector<double>& moneyness,
    const std::vector<double>& tau_grid,
    const std::vector<double>& vol,
    const std::vector<double>& rate)>;

/// Produces a fresh FD reference price for one validation point
using ValidateFn = std::function<std::expected<double, SolverError>(
    double spot, double strike, double tau,
    double sigma, double rate)>;

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
struct GridSizes {
    std::vector<double> moneyness;
    std::vector<double> tau;
    std::vector<double> vol;
    std::vector<double> rate;
    int tau_points;
    double achieved_max_error;
    double achieved_avg_error;
    bool target_met;
    std::vector<IterationStats> iterations;
};

/// Aggregate max grid sizes across probe results.
struct MaxGridSizes {
    size_t moneyness = 0, vol = 0, rate = 0;
    int tau_points = 0;
};

MaxGridSizes aggregate_max_sizes(const std::vector<GridSizes>& probe_results) {
    MaxGridSizes s;
    for (const auto& pr : probe_results) {
        s.moneyness = std::max(s.moneyness, pr.moneyness.size());
        s.vol = std::max(s.vol, pr.vol.size());
        s.rate = std::max(s.rate, pr.rate.size());
        s.tau_points = std::max(s.tau_points, pr.tau_points);
    }
    return s;
}

/// Helper to create evenly spaced grid
/// Requires n >= 2 to avoid divide-by-zero; returns {lo, hi} if n < 2
std::vector<double> linspace(double lo, double hi, size_t n) {
    if (n < 2) {
        return {lo, hi};  // Minimum valid grid
    }
    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = lo + (hi - lo) * i / (n - 1);
    }
    return v;
}

/// Seed a grid from user-provided knots, or fall back to linspace.
/// Ensures domain endpoints are included and minimum 4 points for B-spline.
std::vector<double> seed_grid(const std::vector<double>& user_knots,
                               double lo, double hi, size_t fallback_n = 5) {
    std::vector<double> grid;

    if (!user_knots.empty()) {
        // Filter knots to domain bounds
        for (double v : user_knots) {
            if (v >= lo && v <= hi) {
                grid.push_back(v);
            }
        }
        // Ensure domain endpoints are included
        if (grid.empty() || grid.front() > lo + 1e-12) {
            grid.insert(grid.begin(), lo);
        }
        if (grid.back() < hi - 1e-12) {
            grid.push_back(hi);
        }
        std::sort(grid.begin(), grid.end());
        grid.erase(std::unique(grid.begin(), grid.end()), grid.end());

        // Need minimum 4 points for cubic B-spline
        while (grid.size() < 4) {
            // Insert midpoint in largest gap
            double max_gap = 0.0;
            size_t max_idx = 0;
            for (size_t i = 0; i + 1 < grid.size(); ++i) {
                double gap = grid[i + 1] - grid[i];
                if (gap > max_gap) { max_gap = gap; max_idx = i; }
            }
            grid.push_back((grid[max_idx] + grid[max_idx + 1]) / 2.0);
            std::sort(grid.begin(), grid.end());
        }
    } else {
        grid = linspace(lo, hi, fallback_n);
    }

    return grid;
}

// ============================================================================
// run_refinement: the extracted iterative refinement loop
// ============================================================================

/// Initial grids for seeding the refinement loop (optional for each dimension)
struct InitialGrids {
    std::vector<double> moneyness;
    std::vector<double> tau;
    std::vector<double> vol;
    std::vector<double> rate;
};

static std::expected<GridSizes, PriceTableError> run_refinement(
    const AdaptiveGridParams& params,
    BuildFn build_fn,
    ValidateFn validate_fn,
    const RefinementContext& ctx,
    const std::function<double(double, double, double, double,
                               double, double, double, double)>& compute_error,
    const InitialGrids& initial_grids = {})
{
    // Validation requires at least one sample per iteration
    if (params.validation_samples == 0) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig
        });
    }

    // B-spline requires minimum 4 control points per dimension
    if (params.min_moneyness_points < 4) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig
        });
    }

    const double min_moneyness = ctx.min_moneyness;
    const double max_moneyness = ctx.max_moneyness;
    const double min_tau = ctx.min_tau;
    const double max_tau = ctx.max_tau;
    const double min_vol = ctx.min_vol;
    const double max_vol = ctx.max_vol;
    const double min_rate = ctx.min_rate;
    const double max_rate = ctx.max_rate;

    // Seed grids from user-provided knots (or linspace fallback)
    // This ensures user-specified knots (e.g., benchmark vols) are always grid points
    auto moneyness_grid = seed_grid(initial_grids.moneyness, min_moneyness, max_moneyness,
                                    params.min_moneyness_points);
    auto maturity_grid = seed_grid(initial_grids.tau, min_tau, max_tau, 5);
    auto vol_grid = seed_grid(initial_grids.vol, min_vol, max_vol, 5);
    auto rate_grid = seed_grid(initial_grids.rate, min_rate, max_rate, 4);

    // Ensure moneyness grid meets minimum density requirement
    // Moneyness needs higher density due to exercise boundary curvature
    while (moneyness_grid.size() < params.min_moneyness_points) {
        // Insert midpoints in largest gaps until we reach minimum
        double max_gap = 0.0;
        size_t max_idx = 0;
        for (size_t i = 0; i + 1 < moneyness_grid.size(); ++i) {
            double gap = moneyness_grid[i + 1] - moneyness_grid[i];
            if (gap > max_gap) { max_gap = gap; max_idx = i; }
        }
        moneyness_grid.push_back(
            (moneyness_grid[max_idx] + moneyness_grid[max_idx + 1]) / 2.0);
        std::sort(moneyness_grid.begin(), moneyness_grid.end());
    }

    GridSizes result;
    result.iterations.reserve(params.max_iter);

    std::array<std::vector<size_t>, 4> focus_bins;
    bool focus_active = false;

    for (size_t iteration = 0; iteration < params.max_iter; ++iteration) {
        auto iter_start = std::chrono::steady_clock::now();

        IterationStats stats;
        stats.iteration = iteration;
        stats.grid_sizes = {
            moneyness_grid.size(),
            maturity_grid.size(),
            vol_grid.size(),
            rate_grid.size()
        };

        // a. BUILD/UPDATE TABLE via callback
        auto surface_result = build_fn(moneyness_grid, maturity_grid, vol_grid, rate_grid);
        if (!surface_result.has_value()) {
            return std::unexpected(surface_result.error());
        }
        auto& handle = surface_result.value();

        stats.pde_solves_table = handle.pde_solves;

        // b. GENERATE VALIDATION SAMPLE
        const size_t total_samples = params.validation_samples;
        bool has_focus_bins = focus_active;
        if (focus_active) {
            has_focus_bins = std::any_of(focus_bins.begin(), focus_bins.end(),
                                         [](const std::vector<size_t>& bins) { return !bins.empty(); });
        }

        size_t base_samples = (has_focus_bins && total_samples > 1)
            ? std::max<size_t>(total_samples / 2, 1)
            : total_samples;
        size_t targeted_samples = has_focus_bins ? total_samples - base_samples : 0;

        auto base_unit_samples = latin_hypercube_4d(base_samples,
                                                    params.lhs_seed + iteration);

        std::array<std::pair<double, double>, 4> bounds = {{
            {min_moneyness, max_moneyness},
            {min_tau, max_tau},
            {min_vol, max_vol},
            {min_rate, max_rate}
        }};
        std::vector<std::array<double, 4>> samples = scale_lhs_samples(base_unit_samples, bounds);

        if (targeted_samples > 0 && has_focus_bins) {
            std::mt19937_64 targeted_rng(params.lhs_seed ^ (iteration * 1315423911ULL + 0x9e3779b97f4a7c15ULL));
            std::uniform_real_distribution<double> uniform(0.0, 1.0);

            std::vector<std::array<double, 4>> targeted_unit;
            targeted_unit.reserve(targeted_samples);

            for (size_t i = 0; i < targeted_samples; ++i) {
                std::array<double, 4> point{};
                for (size_t d = 0; d < 4; ++d) {
                    double u = uniform(targeted_rng);
                    if (!focus_bins[d].empty()) {
                        const auto& dim_bins = focus_bins[d];
                        size_t bin = dim_bins[i % dim_bins.size()];
                        double bin_lo = static_cast<double>(bin) / ErrorBins::N_BINS;
                        double bin_hi = static_cast<double>(bin + 1) / ErrorBins::N_BINS;
                        double span = bin_hi - bin_lo;
                        point[d] = bin_lo + u * span;
                    } else {
                        point[d] = u;
                    }
                }
                targeted_unit.push_back(point);
            }

            auto targeted_scaled = scale_lhs_samples(targeted_unit, bounds);
            samples.insert(samples.end(), targeted_scaled.begin(), targeted_scaled.end());
        }

        // c. VALIDATE AGAINST FRESH FD SOLVES
        double max_error = 0.0;
        double sum_error = 0.0;
        size_t valid_samples = 0;
        ErrorBins error_bins;

        for (const auto& sample : samples) {
            double m = sample[0];
            double tau = sample[1];
            double sigma = sample[2];
            double rate = sample[3];

            // Interpolated price from surface via callback
            double strike = ctx.spot / m;
            double interp_price = handle.price(ctx.spot, strike, tau, sigma, rate);

            // Fresh FD solve for reference via callback
            auto fd_result = validate_fn(ctx.spot, strike, tau, sigma, rate);

            if (!fd_result.has_value()) {
                continue;  // Skip failed solves
            }

            stats.pde_solves_validation++;

            double ref_price = fd_result.value();
            double iv_error = compute_error(
                interp_price, ref_price,
                ctx.spot, strike, tau, sigma, rate,
                ctx.dividend_yield);
            max_error = std::max(max_error, iv_error);
            sum_error += iv_error;
            valid_samples++;

            // Normalize position for error bins
            std::array<double, 4> norm_pos = {{
                (m - min_moneyness) / (max_moneyness - min_moneyness),
                (tau - min_tau) / (max_tau - min_tau),
                (sigma - min_vol) / (max_vol - min_vol),
                (rate - min_rate) / (max_rate - min_rate)
            }};
            error_bins.record_error(norm_pos, iv_error, params.target_iv_error);
        }

        if (valid_samples == 0) {
            return std::unexpected(PriceTableError{
                PriceTableErrorCode::ExtractionFailed, /*axis=*/0, /*detail=*/0
            });
        }

        double avg_error = sum_error / valid_samples;

        stats.max_error = max_error;
        stats.avg_error = avg_error;

        auto iter_end = std::chrono::steady_clock::now();
        stats.elapsed_seconds = std::chrono::duration<double>(iter_end - iter_start).count();

        // d. CHECK CONVERGENCE
        bool converged = (max_error <= params.target_iv_error);

        if (converged || iteration == params.max_iter - 1) {
            // Final iteration - save results
            stats.refined_dim = -1;  // No refinement on final iteration
            result.iterations.push_back(stats);
            result.moneyness = moneyness_grid;
            result.tau = maturity_grid;
            result.vol = vol_grid;
            result.rate = rate_grid;
            result.tau_points = static_cast<int>(maturity_grid.size());
            result.achieved_max_error = max_error;
            result.achieved_avg_error = avg_error;
            result.target_met = converged;
            break;
        }

        // e. DIAGNOSE & REFINE
        size_t worst_dim = error_bins.worst_dimension();
        auto problematic = error_bins.problematic_bins(worst_dim);

        // Refine the worst dimension, focusing on problematic bins
        auto refine_grid_targeted = [&params, &problematic](std::vector<double>& grid,
                                                            double lo, double hi) {
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

        switch (worst_dim) {
            case 0: refine_grid_targeted(moneyness_grid, min_moneyness, max_moneyness); break;
            case 1: refine_grid_targeted(maturity_grid, min_tau, max_tau); break;
            case 2: refine_grid_targeted(vol_grid, min_vol, max_vol); break;
            case 3: refine_grid_targeted(rate_grid, min_rate, max_rate); break;
        }

        focus_active = false;
        for (size_t d = 0; d < focus_bins.size(); ++d) {
            focus_bins[d] = error_bins.problematic_bins(d);
            if (!focus_bins[d].empty()) {
                focus_active = true;
            }
        }

        stats.refined_dim = static_cast<int>(worst_dim);
        result.iterations.push_back(stats);
    }

    return result;
}

/// Build a SegmentedSurface for each K_ref in the list.
/// Takes a Config template with K_ref set per iteration.
std::expected<std::vector<SegmentedSurface<>>, PriceTableError>
build_segmented_surfaces(
    SegmentedPriceTableBuilder::Config base_config,
    const std::vector<double>& ref_values)
{
    std::vector<SegmentedSurface<>> surfaces;
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
    std::vector<SegmentedSurface<>> surfaces;
    SegmentedPriceTableBuilder::Config seg_template;
    MaxGridSizes gsz;
    // Domain bounds (needed for validation/retry)
    double expanded_min_m, max_m;
    double min_vol, max_vol;
    double min_rate, max_rate;
    double min_tau, max_tau;
};

/// Shared probe-and-build pipeline for build_segmented and build_segmented_strike.
/// Selects representative probes, runs adaptive refinement per probe, aggregates
/// grid sizes, then builds one SegmentedSurface per ref_value.
static std::expected<SegmentedBuildResult, PriceTableError>
probe_and_build(
    const AdaptiveGridParams& params,
    const SegmentedAdaptiveConfig& config,
    const std::vector<double>& ref_values,
    const ManualGrid& domain)
{
    // 1. Select probe values (up to 3: front, back, nearest ATM)
    auto probes = select_probes(ref_values, config.spot);

    // 2. Expand domain bounds
    double total_div = total_discrete_dividends(config.discrete_dividends, config.maturity);
    double ref_min = ref_values.front();
    double expansion = (ref_min > 0.0) ? total_div / ref_min : 0.0;

    if (domain.moneyness.empty() || domain.vol.empty() || domain.rate.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    double min_m = domain.moneyness.front();
    double max_m = domain.moneyness.back();
    double expanded_min_m = std::max(min_m - expansion, 0.01);

    double min_vol = domain.vol.front();
    double max_vol = domain.vol.back();
    double min_rate = domain.rate.front();
    double max_rate = domain.rate.back();

    expand_bounds_positive(expanded_min_m, max_m, 0.10);
    expand_bounds_positive(min_vol, max_vol, 0.10);
    expand_bounds(min_rate, max_rate, 0.04);

    double min_tau = std::min(0.01, config.maturity * 0.5);
    double max_tau = config.maturity;
    expand_bounds_positive(min_tau, max_tau, 0.1);
    max_tau = std::min(max_tau, config.maturity);

    // 3. Run adaptive refinement per probe
    std::vector<GridSizes> probe_results;
    for (double probe_ref : probes) {
        BuildFn build_fn = [&config, probe_ref](
            const std::vector<double>& m_grid,
            const std::vector<double>& tau_grid,
            const std::vector<double>& v_grid,
            const std::vector<double>& r_grid)
            -> std::expected<SurfaceHandle, PriceTableError>
        {
            int tau_pts = static_cast<int>(tau_grid.size());
            auto seg_cfg = make_seg_config(config, m_grid, v_grid, r_grid, tau_pts);
            seg_cfg.K_ref = probe_ref;
            auto surface = SegmentedPriceTableBuilder::build(seg_cfg);
            if (!surface.has_value()) {
                return std::unexpected(surface.error());
            }
            auto shared = std::make_shared<SegmentedSurface<>>(std::move(*surface));
            double spot = config.spot;
            return SurfaceHandle{
                .price = [shared, spot](double /*spot_arg*/, double strike,
                                        double tau, double sigma, double rate) -> double {
                    return shared->price(PriceQuery{spot, strike, tau, sigma, rate});
                },
                .pde_solves = 0
            };
        };

        ValidateFn validate_fn = [&config](
            double spot, double strike, double tau,
            double sigma, double rate)
            -> std::expected<double, SolverError>
        {
            PricingParams p;
            p.spot = spot;
            p.strike = strike;
            p.maturity = tau;
            p.rate = rate;
            p.dividend_yield = config.dividend_yield;
            p.option_type = config.option_type;
            p.volatility = sigma;
            p.discrete_dividends = config.discrete_dividends;
            auto fd = solve_american_option(p);
            if (!fd.has_value()) return std::unexpected(fd.error());
            return fd->value();
        };

        auto compute_error_fn = [&params](
            double interp, double ref_price,
            double spot, double strike, double tau,
            double sigma, double rate,
            double div_yield) -> double
        {
            double price_error = std::abs(interp - ref_price);
            double vega = bs_vega(spot, strike, tau, sigma, rate, div_yield);
            double vega_clamped = std::max(std::abs(vega), params.vega_floor);
            double iv_error = price_error / vega_clamped;
            double price_tol = params.target_iv_error * params.vega_floor;
            if (price_error <= price_tol) {
                iv_error = std::min(iv_error, params.target_iv_error);
            }
            return iv_error;
        };

        RefinementContext ctx{
            .spot = config.spot,
            .dividend_yield = config.dividend_yield,
            .option_type = config.option_type,
            .min_moneyness = expanded_min_m,
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

        auto sizes = run_refinement(params, build_fn, validate_fn, ctx,
                                    compute_error_fn, initial_grids);
        if (!sizes.has_value()) {
            return std::unexpected(sizes.error());
        }
        probe_results.push_back(std::move(*sizes));
    }

    // 4. Aggregate max grid sizes across probes
    auto gsz = aggregate_max_sizes(probe_results);

    // 5. Build final uniform grids and all surfaces
    auto final_m = linspace(expanded_min_m, max_m, gsz.moneyness);
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
        .expanded_min_m = expanded_min_m,
        .max_m = max_m,
        .min_vol = min_vol,
        .max_vol = max_vol,
        .min_rate = min_rate,
        .max_rate = max_rate,
        .min_tau = min_tau,
        .max_tau = max_tau,
    };
}

}  // anonymous namespace

// ============================================================================
// AdaptiveGridBuilder implementation
// ============================================================================

AdaptiveGridBuilder::AdaptiveGridBuilder(AdaptiveGridParams params)
    : params_(std::move(params))
{}

std::expected<AdaptiveResult, PriceTableError>
AdaptiveGridBuilder::build(const OptionGrid& chain,
                           PDEGridSpec pde_grid,
                           OptionType type)
{
    // ========================================================================
    // 1. SEED ESTIMATE - Extract grid bounds from chain
    // ========================================================================

    // Clear cache to prevent stale slices from previous builds leaking into this one
    // (each build may have different spot/dividend parameters)
    cache_.clear();

    if (chain.strikes.empty() || chain.maturities.empty()) {
        return std::unexpected(PriceTableError(
            PriceTableErrorCode::InvalidConfig
        ));
    }

    // Bug fix: Check implied_vols and rates are non-empty before dereferencing
    if (chain.implied_vols.empty() || chain.rates.empty()) {
        return std::unexpected(PriceTableError(
            PriceTableErrorCode::InvalidConfig
        ));
    }

    double min_moneyness = std::numeric_limits<double>::max();
    double max_moneyness = std::numeric_limits<double>::lowest();

    for (double strike : chain.strikes) {
        double m = chain.spot / strike;
        min_moneyness = std::min(min_moneyness, m);
        max_moneyness = std::max(max_moneyness, m);
    }

    double min_tau = *std::min_element(chain.maturities.begin(), chain.maturities.end());
    double max_tau = *std::max_element(chain.maturities.begin(), chain.maturities.end());

    double min_vol = *std::min_element(chain.implied_vols.begin(), chain.implied_vols.end());
    double max_vol = *std::max_element(chain.implied_vols.begin(), chain.implied_vols.end());

    double min_rate = *std::min_element(chain.rates.begin(), chain.rates.end());
    double max_rate = *std::max_element(chain.rates.begin(), chain.rates.end());

    expand_bounds_positive(min_moneyness, max_moneyness, 0.10);
    expand_bounds_positive(min_tau, max_tau, 0.5);
    expand_bounds_positive(min_vol, max_vol, 0.10);
    expand_bounds(min_rate, max_rate, 0.04);

    // ========================================================================
    // 2. Create callbacks for run_refinement
    // ========================================================================

    // Shared state for the last surface built (so we can extract it after refinement)
    std::shared_ptr<const PriceTableSurface<4>> last_surface;
    PriceTableAxes<4> last_axes;

    // Iteration counter for cache management (set_tau_grid vs invalidate_if_tau_changed)
    size_t build_iteration = 0;

    // BuildFn: builds price table surface from grids, returns SurfaceHandle
    BuildFn build_fn = [&](const std::vector<double>& m_grid,
                           const std::vector<double>& tau_grid,
                           const std::vector<double>& v_grid,
                           const std::vector<double>& r_grid)
        -> std::expected<SurfaceHandle, PriceTableError>
    {
        auto builder_result = PriceTableBuilder<4>::from_vectors(
            m_grid, tau_grid, v_grid, r_grid,
            chain.spot,
            pde_grid, type, chain.dividend_yield,
            params_.max_failure_rate);

        if (!builder_result.has_value()) {
            return std::unexpected(builder_result.error());
        }

        auto& [builder, axes] = builder_result.value();

        // On first iteration, set the initial tau grid; subsequent iterations
        // compare against it and clear cache only if tau actually changed.
        if (build_iteration == 0) {
            cache_.set_tau_grid(tau_grid);
        } else {
            cache_.invalidate_if_tau_changed(tau_grid);
        }
        build_iteration++;

        // Generate all (σ,r) parameter combinations
        auto all_params = builder.make_batch(axes);

        // Extract (σ,r) pairs from all_params
        std::vector<std::pair<double, double>> all_pairs;
        all_pairs.reserve(all_params.size());
        for (const auto& p : all_params) {
            double rate = get_zero_rate(p.rate, p.maturity);
            all_pairs.emplace_back(p.volatility, rate);
        }

        // Find which pairs are missing from cache
        auto missing_indices = cache_.get_missing_indices(all_pairs);

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

            // Handle PDEGridSpec variant: either explicit config or auto-estimated
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
                    fresh_results = batch_solver.solve_batch(missing_params, true, nullptr, custom_grid);
                } else {
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
                    fresh_results = batch_solver.solve_batch(missing_params, true);
                }
            } else if (const auto* accuracy_grid = std::get_if<GridAccuracyParams>(&pde_grid)) {
                // Auto-estimated grid: use accuracy params directly
                batch_solver.set_grid_accuracy(*accuracy_grid);
                fresh_results = batch_solver.solve_batch(missing_params, true);
            }

            // Add fresh results to cache
            for (size_t i = 0; i < fresh_results.results.size(); ++i) {
                if (fresh_results.results[i].has_value()) {
                    double sigma = missing_params[i].volatility;
                    double rate = get_zero_rate(missing_params[i].rate, missing_params[i].maturity);
                    auto result_ptr = std::make_shared<AmericanOptionResult>(
                        fresh_results.results[i].value().grid(),
                        missing_params[i]);
                    cache_.add(sigma, rate, std::move(result_ptr));
                }
            }
        } else {
            fresh_results.failed_count = 0;
        }

        // Merge cached + fresh results into full batch
        auto merged_results = merge_results(all_params, missing_indices, fresh_results);

        // Extract tensor from merged results
        auto tensor_result = builder.extract_tensor(merged_results, axes);
        if (!tensor_result.has_value()) {
            return std::unexpected(tensor_result.error());
        }

        auto& extraction = tensor_result.value();

        // Repair failed slices if needed
        RepairStats repair_stats{0, 0};
        if (!extraction.failed_pde.empty() || !extraction.failed_spline.empty()) {
            auto repair_result = builder.repair_failed_slices(
                extraction.tensor, extraction.failed_pde,
                extraction.failed_spline, axes);
            if (!repair_result.has_value()) {
                return std::unexpected(repair_result.error());
            }
            repair_stats = repair_result.value();
        }

        // Fit coefficients
        auto fit_result = builder.fit_coeffs(extraction.tensor, axes);
        if (!fit_result.has_value()) {
            return std::unexpected(fit_result.error());
        }
        auto coeffs_result = std::move(fit_result->coefficients);

        // Build metadata
        PriceTableMetadata metadata;
        metadata.K_ref = chain.spot;
        metadata.dividends.dividend_yield = chain.dividend_yield;
        metadata.m_min = m_grid.front();
        metadata.m_max = m_grid.back();
        metadata.dividends.discrete_dividends = {};
        metadata.content = SurfaceContent::EarlyExercisePremium;

        // Build surface
        auto surface = PriceTableSurface<4>::build(axes, coeffs_result, metadata);
        if (!surface.has_value()) {
            return std::unexpected(surface.error());
        }

        // Store for later extraction
        last_surface = surface.value();
        last_axes = axes;

        size_t pde_solves = missing_params.size();

        // Return a handle that queries the surface (reconstruct full American price)
        auto surface_ptr = surface.value();
        auto aps = AmericanPriceSurface::create(surface_ptr, type);
        if (!aps.has_value()) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
        }

        return SurfaceHandle{
            .price = [aps = std::move(*aps)](double query_spot, double strike, double tau,
                                             double sigma, double rate) -> double {
                return aps.price(query_spot, strike, tau, sigma, rate);
            },
            .pde_solves = pde_solves
        };
    };

    // ValidateFn: fresh FD solve for a single validation point
    ValidateFn validate_fn = [&](double spot, double strike, double tau,
                                  double sigma, double rate)
        -> std::expected<double, SolverError>
    {
        PricingParams params;
        params.spot = spot;
        params.strike = strike;
        params.maturity = tau;
        params.rate = rate;
        params.dividend_yield = chain.dividend_yield;
        params.option_type = type;
        params.volatility = sigma;

        auto fd_result = solve_american_option(params);
        if (!fd_result.has_value()) {
            return std::unexpected(fd_result.error());
        }
        return fd_result->value();
    };

    // compute_error: FD American vega for standard path
    auto compute_error_fn = [this, &validate_fn](
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
            // FD bump failed or eps too small — clamp to floor inside compute_error_metric
            vega = 0.0;
        }
        return compute_error_metric(price_error, vega);
    };

    RefinementContext ctx{
        .spot = chain.spot,
        .dividend_yield = chain.dividend_yield,
        .option_type = type,
        .min_moneyness = min_moneyness,
        .max_moneyness = max_moneyness,
        .min_tau = min_tau,
        .max_tau = max_tau,
        .min_vol = min_vol,
        .max_vol = max_vol,
        .min_rate = min_rate,
        .max_rate = max_rate
    };

    // ========================================================================
    // 3. Run refinement and assemble result
    // ========================================================================

    // Build initial grids from chain data (ensures user-specified knots are grid points)
    InitialGrids initial_grids;
    initial_grids.moneyness.reserve(chain.strikes.size());
    for (double strike : chain.strikes) {
        initial_grids.moneyness.push_back(chain.spot / strike);
    }
    initial_grids.tau = chain.maturities;
    initial_grids.vol = chain.implied_vols;
    initial_grids.rate = chain.rates;

    auto grid_result = run_refinement(params_, build_fn, validate_fn, ctx,
                                      compute_error_fn, initial_grids);
    if (!grid_result.has_value()) {
        return std::unexpected(grid_result.error());
    }

    auto& grids = grid_result.value();

    AdaptiveResult result;
    result.surface = last_surface;
    result.axes = last_axes;
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

double AdaptiveGridBuilder::compute_error_metric(
    double price_error, double vega) const
{
    double vega_clamped = std::max(std::abs(vega), params_.vega_floor);
    double iv_error = price_error / vega_clamped;

    // Cap when price is already within tolerance.
    // Prevents FD noise from driving runaway refinement.
    double price_tol = params_.target_iv_error * params_.vega_floor;
    if (price_error <= price_tol) {
        iv_error = std::min(iv_error, params_.target_iv_error);
    }
    return iv_error;
}

BatchAmericanOptionResult AdaptiveGridBuilder::merge_results(
    const std::vector<PricingParams>& all_params,
    const std::vector<size_t>& fresh_indices,
    const BatchAmericanOptionResult& fresh_results) const
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
            auto cached = cache_.get(sigma, rate);
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


std::expected<SegmentedAdaptiveResult, PriceTableError>
AdaptiveGridBuilder::build_segmented(
    const SegmentedAdaptiveConfig& config,
    const ManualGrid& domain)
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
    auto result = probe_and_build(params_, config, K_refs, domain);
    if (!result.has_value()) {
        return std::unexpected(result.error());
    }
    auto& build = *result;

    // 3. Assemble MultiKRefSurface
    std::vector<MultiKRefEntry> entries;
    for (size_t i = 0; i < K_refs.size(); ++i) {
        entries.push_back({.K_ref = K_refs[i], .surface = std::move(build.surfaces[i])});
    }
    auto surface = build_multi_kref_surface(std::move(entries));
    if (!surface.has_value()) {
        return std::unexpected(surface.error());
    }

    // 4. Final multi-K_ref validation at arbitrary strikes
    auto final_samples = latin_hypercube_4d(
        params_.validation_samples, params_.lhs_seed + 999);

    std::array<std::pair<double, double>, 4> final_bounds = {{
        {build.expanded_min_m, build.max_m},
        {build.min_tau, build.max_tau},
        {build.min_vol, build.max_vol},
        {build.min_rate, build.max_rate},
    }};
    auto scaled = scale_lhs_samples(final_samples, final_bounds);

    double final_max_error = 0.0;
    size_t valid = 0;

    for (const auto& sample : scaled) {
        double m = sample[0], tau = sample[1], sigma = sample[2], rate = sample[3];
        double strike = config.spot / m;

        PriceQuery query{
            .spot = config.spot,
            .strike = strike,
            .tau = tau,
            .sigma = sigma,
            .rate = rate,
        };
        double interp = surface->price(query);

        PricingParams params;
        params.spot = config.spot;
        params.strike = strike;
        params.maturity = tau;
        params.rate = rate;
        params.dividend_yield = config.dividend_yield;
        params.option_type = config.option_type;
        params.volatility = sigma;
        params.discrete_dividends = config.discrete_dividends;

        auto fd = solve_american_option(params);
        if (!fd.has_value()) continue;

        double price_error = std::abs(interp - fd->value());
        double vega = bs_vega(config.spot, strike, tau, sigma, rate, config.dividend_yield);
        double err = compute_error_metric(price_error, vega);

        final_max_error = std::max(final_max_error, err);
        valid++;
    }

    if (valid > 0 && final_max_error > params_.target_iv_error) {
        // Bump grids by one refinement step and rebuild (one retry)
        size_t bumped_m = std::min(build.gsz.moneyness + 2, params_.max_points_per_dim);
        size_t bumped_v = std::min(build.gsz.vol + 1, params_.max_points_per_dim);
        size_t bumped_r = std::min(build.gsz.rate + 1, params_.max_points_per_dim);
        int bumped_tau = std::min(build.gsz.tau_points + 2,
            static_cast<int>(params_.max_points_per_dim));

        auto retry_m = linspace(build.expanded_min_m, build.max_m, bumped_m);
        auto retry_v = linspace(build.min_vol, build.max_vol, bumped_v);
        auto retry_r = linspace(build.min_rate, build.max_rate, bumped_r);

        build.seg_template.grid = {.moneyness = retry_m, .vol = retry_v, .rate = retry_r};
        build.seg_template.tau_points_per_segment = bumped_tau;
        auto retry_segs = build_segmented_surfaces(build.seg_template, K_refs);
        if (retry_segs.has_value()) {
            std::vector<MultiKRefEntry> retry_entries;
            for (size_t i = 0; i < K_refs.size(); ++i) {
                retry_entries.push_back({.K_ref = K_refs[i], .surface = std::move((*retry_segs)[i])});
            }
            auto retry_surface = build_multi_kref_surface(std::move(retry_entries));
            if (retry_surface.has_value()) {
                return SegmentedAdaptiveResult{
                    .surface = std::move(*retry_surface),
                    .grid = {.moneyness = retry_m, .vol = retry_v, .rate = retry_r},
                    .tau_points_per_segment = bumped_tau,
                };
            }
        }
    }

    return SegmentedAdaptiveResult{
        .surface = std::move(*surface),
        .grid = build.seg_template.grid,
        .tau_points_per_segment = build.seg_template.tau_points_per_segment,
    };
}

std::expected<StrikeAdaptiveResult, PriceTableError>
AdaptiveGridBuilder::build_segmented_strike(
    const SegmentedAdaptiveConfig& config,
    const std::vector<double>& strike_grid,
    const ManualGrid& domain)
{
    std::vector<double> strikes = strike_grid;
    if (strikes.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }
    std::sort(strikes.begin(), strikes.end());
    strikes.erase(std::unique(strikes.begin(), strikes.end()), strikes.end());

    auto result = probe_and_build(params_, config, strikes, domain);
    if (!result.has_value()) {
        return std::unexpected(result.error());
    }

    std::vector<StrikeEntry> entries;
    for (size_t i = 0; i < strikes.size(); ++i) {
        entries.push_back({.strike = strikes[i], .surface = std::move(result->surfaces[i])});
    }
    auto surface = build_strike_surface(std::move(entries), /*use_nearest=*/true);
    if (!surface.has_value()) {
        return std::unexpected(surface.error());
    }
    return StrikeAdaptiveResult{
        .surface = std::move(*surface),
        .grid = result->seg_template.grid,
        .tau_points_per_segment = result->seg_template.tau_points_per_segment,
    };
}

// Backward-compatible overload: delegates to PDEGridSpec version
std::expected<AdaptiveResult, PriceTableError>
AdaptiveGridBuilder::build(const OptionGrid& chain,
                           GridSpec<double> grid_spec,
                           size_t n_time,
                           OptionType type)
{
    return build(chain, PDEGridConfig{grid_spec, n_time, {}}, type);
}

}  // namespace mango
