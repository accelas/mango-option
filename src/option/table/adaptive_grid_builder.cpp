// SPDX-License-Identifier: MIT
#include "src/option/table/adaptive_grid_builder.hpp"
#include "src/math/black_scholes_analytics.hpp"
#include "src/math/latin_hypercube.hpp"
#include "src/option/american_option_batch.hpp"
#include "src/option/table/price_table_surface.hpp"
#include "src/pde/core/time_domain.hpp"
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

/// Helper to create evenly spaced grid
std::vector<double> linspace(double lo, double hi, size_t n) {
    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = lo + (hi - lo) * i / (n - 1);
    }
    return v;
}

// ============================================================================
// run_refinement: the extracted iterative refinement loop
// ============================================================================

static std::expected<GridSizes, PriceTableError> run_refinement(
    const AdaptiveGridParams& params,
    BuildFn build_fn,
    ValidateFn validate_fn,
    const RefinementContext& ctx,
    const std::function<std::optional<double>(double, double, double, double,
                                              double, double, double, double)>& compute_error)
{
    // Validation requires at least one sample per iteration
    if (params.validation_samples == 0) {
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

    // Start with ~5-7 points per dimension (minimum 4 for B-splines)
    std::vector<double> moneyness_grid = linspace(min_moneyness, max_moneyness, 5);
    std::vector<double> maturity_grid = linspace(min_tau, max_tau, 5);
    std::vector<double> vol_grid = linspace(min_vol, max_vol, 5);
    std::vector<double> rate_grid = linspace(min_rate, max_rate, 4);  // Need at least 4

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
            auto iv_error_opt = compute_error(
                interp_price, ref_price,
                ctx.spot, strike, tau, sigma, rate,
                ctx.dividend_yield);

            // Skip low-vega samples where error metric is undefined
            if (!iv_error_opt.has_value()) {
                continue;
            }

            double iv_error = iv_error_opt.value();
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

}  // anonymous namespace

// ============================================================================
// AdaptiveGridBuilder implementation
// ============================================================================

AdaptiveGridBuilder::AdaptiveGridBuilder(AdaptiveGridParams params)
    : params_(std::move(params))
{}

std::expected<AdaptiveResult, PriceTableError>
AdaptiveGridBuilder::build(const OptionGrid& chain,
                           GridSpec<double> grid_spec,
                           size_t n_time,
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

    // Expand bounds when min == max (or close) to ensure linspace produces distinct points
    constexpr double kMinPositive = 1e-6;

    auto expand_bounds_positive = [kMinPositive](double& lo, double& hi, double min_spread) {
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
    };

    auto expand_bounds = [](double& lo, double& hi, double min_spread) {
        if (hi - lo < min_spread) {
            double mid = (lo + hi) / 2.0;
            lo = mid - min_spread / 2.0;
            hi = mid + min_spread / 2.0;
        }
    };

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
            PDEGridConfig{grid_spec, n_time}, type, chain.dividend_yield,
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

            BatchAmericanOptionSolver batch_solver;
            batch_solver.set_snapshot_times(std::span{tau_grid});

            if (grid_meets_constraints) {
                const double max_maturity = tau_grid.back();
                TimeDomain time_domain = TimeDomain::from_n_steps(0.0, max_maturity, n_time);
                PDEGridSpec custom_grid = PDEGridConfig{grid_spec, time_domain.n_steps(), {}};
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

        // Build surface
        auto surface = PriceTableSurface<4>::build(axes, coeffs_result, metadata);
        if (!surface.has_value()) {
            return std::unexpected(surface.error());
        }

        // Store for later extraction
        last_surface = surface.value();
        last_axes = axes;

        size_t pde_solves = missing_params.size();

        // Return a handle that queries the surface
        auto surface_ptr = surface.value();
        double spot = chain.spot;
        return SurfaceHandle{
            .price = [surface_ptr, spot](double /*query_spot*/, double strike, double tau,
                                         double sigma, double rate) -> double {
                double m = spot / strike;
                return surface_ptr->value({m, tau, sigma, rate});
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

    // compute_error: wraps compute_error_metric
    auto compute_error_fn = [this](double interp_price, double ref_price,
                                    double spot, double strike, double tau,
                                    double sigma, double rate,
                                    double dividend_yield) -> std::optional<double> {
        return compute_error_metric(interp_price, ref_price,
                                    spot, strike, tau, sigma, rate,
                                    dividend_yield);
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

    auto grid_result = run_refinement(params_, build_fn, validate_fn, ctx, compute_error_fn);
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
    for (const auto& it : result.iterations) {
        result.total_pde_solves += it.pde_solves_table + it.pde_solves_validation;
    }

    return result;
}

std::optional<double> AdaptiveGridBuilder::compute_error_metric(
    double interpolated_price, double reference_price,
    double spot, double strike, double tau, double sigma, double rate,
    double dividend_yield) const
{
    double price_error = std::abs(interpolated_price - reference_price);

    // Use European Black-Scholes vega as an approximation for American vega.
    //
    // Limitation: For deep ITM American puts, true American vega is smaller
    // than European vega because part of the option value is intrinsic (not
    // sensitive to vol). Using European vega here underestimates the IV error,
    // which may cause under-refinement in those regions.
    //
    // This is acceptable because:
    // 1. For ATM/OTM options (most common), the vegas are nearly identical
    // 2. Deep ITM puts have small vega anyway, so the vega_floor kicks in
    // 3. Computing true American vega would require 2 extra PDE solves per
    //    validation point, which is expensive
    //
    // Future: Could use Barone-Adesi-Whaley or finite-difference vega if
    // higher accuracy is needed for deep ITM regions.
    double vega = bs_vega(spot, strike, tau, sigma, rate, dividend_yield);

    if (vega >= params_.vega_floor) {
        // ΔIV ≈ ΔP / vega (first-order Taylor approximation)
        return price_error / vega;
    } else {
        // Vega too small (deep ITM/OTM or very short τ): IV is ill-defined.
        // If price error is within tolerance, skip this sample entirely.
        // Otherwise, use vega_floor to get a bounded error estimate.
        double price_tol = params_.target_iv_error * params_.vega_floor;
        if (price_error <= price_tol) {
            return std::nullopt;  // Price is accurate enough - skip this sample
        }
        return price_error / params_.vega_floor;
    }
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

}  // namespace mango
