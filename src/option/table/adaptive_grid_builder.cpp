#include "src/option/table/adaptive_grid_builder.hpp"
#include "src/math/black_scholes_analytics.hpp"
#include "src/math/latin_hypercube.hpp"
#include "src/option/american_option_batch.hpp"
#include "src/option/table/price_table_surface.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <map>

namespace mango {

AdaptiveGridBuilder::AdaptiveGridBuilder(AdaptiveGridParams params)
    : params_(std::move(params))
{}

std::expected<AdaptiveResult, PriceTableError>
AdaptiveGridBuilder::build(const OptionChain& chain,
                           GridSpec<double> grid_spec,
                           size_t n_time,
                           OptionType type)
{
    // ========================================================================
    // 1. SEED ESTIMATE - Extract grid bounds from chain
    // ========================================================================

    if (chain.strikes.empty() || chain.maturities.empty()) {
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

    // Helper to create evenly spaced grid
    auto linspace = [](double lo, double hi, size_t n) {
        std::vector<double> v(n);
        for (size_t i = 0; i < n; ++i) {
            v[i] = lo + (hi - lo) * i / (n - 1);
        }
        return v;
    };

    // Start with ~5-7 points per dimension (minimum 4 for B-splines)
    std::vector<double> moneyness_grid = linspace(min_moneyness, max_moneyness, 5);
    std::vector<double> maturity_grid = linspace(min_tau, max_tau, 5);
    std::vector<double> vol_grid = linspace(min_vol, max_vol, 5);
    std::vector<double> rate_grid = linspace(min_rate, max_rate, 4);  // Need at least 4

    AdaptiveResult result;
    result.iterations.reserve(params_.max_iterations);

    // ========================================================================
    // 2. MAIN LOOP - Iterative refinement
    // ========================================================================

    for (size_t iteration = 0; iteration < params_.max_iterations; ++iteration) {
        auto iter_start = std::chrono::steady_clock::now();

        IterationStats stats;
        stats.iteration = iteration;
        stats.grid_sizes = {
            moneyness_grid.size(),
            maturity_grid.size(),
            vol_grid.size(),
            rate_grid.size()
        };

        // a. BUILD/UPDATE TABLE
        auto builder_result = PriceTableBuilder<4>::from_vectors(
            moneyness_grid, maturity_grid, vol_grid, rate_grid,
            chain.spot,  // K_ref = spot as reference strike
            grid_spec, n_time, type, chain.dividend_yield,
            params_.max_failure_rate);

        if (!builder_result.has_value()) {
            return std::unexpected(builder_result.error());
        }

        auto& [builder, axes] = builder_result.value();

        // Check if τ grid changed - if so, invalidate cache
        cache_.invalidate_if_tau_changed(maturity_grid);

        // Generate all (σ,r) parameter combinations
        auto all_params = builder.make_batch_internal(axes);

        // Extract (σ,r) pairs from all_params
        std::vector<std::pair<double, double>> all_pairs;
        all_pairs.reserve(all_params.size());
        for (const auto& p : all_params) {
            // Extract scalar rate from RateSpec
            double rate = get_zero_rate(p.rate, p.maturity);
            all_pairs.emplace_back(p.volatility, rate);
        }

        // Find which pairs are missing from cache
        auto missing_indices = cache_.get_missing_indices(all_pairs);

        // Build batch of params for missing pairs only
        std::vector<AmericanOptionParams> missing_params;
        missing_params.reserve(missing_indices.size());
        for (size_t idx : missing_indices) {
            missing_params.push_back(all_params[idx]);
        }

        // Solve missing pairs
        BatchAmericanOptionResult fresh_results;
        if (!missing_params.empty()) {
            BatchAmericanOptionSolver batch_solver;
            batch_solver.set_grid_accuracy(GridAccuracyParams{})
                        .set_snapshot_times(std::span{maturity_grid});
            fresh_results = batch_solver.solve_batch(missing_params, true);

            // Add fresh results to cache (share grids, don't move results)
            for (size_t i = 0; i < fresh_results.results.size(); ++i) {
                if (fresh_results.results[i].has_value()) {
                    double sigma = missing_params[i].volatility;
                    double rate = get_zero_rate(missing_params[i].rate, missing_params[i].maturity);
                    // Store a new AmericanOptionResult that shares the grid
                    auto result_ptr = std::make_shared<AmericanOptionResult>(
                        fresh_results.results[i].value().grid(),
                        missing_params[i]);
                    cache_.add(sigma, rate, std::move(result_ptr));
                }
            }
        } else {
            // All results cached - no fresh solves needed
            fresh_results.failed_count = 0;
        }

        // Merge cached + fresh results into full batch
        auto merged_results = merge_results(all_params, missing_indices, fresh_results);

        // Extract tensor from merged results
        auto tensor_result = builder.extract_tensor_internal(merged_results, axes);
        if (!tensor_result.has_value()) {
            return std::unexpected(tensor_result.error());
        }

        auto& extraction = tensor_result.value();

        // Repair failed slices if needed
        RepairStats repair_stats{0, 0};
        if (!extraction.failed_pde.empty() || !extraction.failed_spline.empty()) {
            auto repair_result = builder.repair_failed_slices_internal(
                extraction.tensor, extraction.failed_pde,
                extraction.failed_spline, axes);
            if (!repair_result.has_value()) {
                return std::unexpected(repair_result.error());
            }
            repair_stats = repair_result.value();
        }

        // Fit coefficients
        auto coeffs_result = builder.fit_coeffs_internal(extraction.tensor, axes);
        if (!coeffs_result.has_value()) {
            return std::unexpected(coeffs_result.error());
        }

        // Build metadata
        PriceTableMetadata metadata;
        metadata.K_ref = chain.spot;
        metadata.dividend_yield = chain.dividend_yield;
        metadata.m_min = moneyness_grid.front();
        metadata.m_max = moneyness_grid.back();
        metadata.discrete_dividends = {};

        // Build surface
        auto surface = PriceTableSurface<4>::build(axes, coeffs_result.value(), metadata);
        if (!surface.has_value()) {
            return std::unexpected(surface.error());
        }

        stats.pde_solves_table = missing_params.size();  // Only count fresh solves

        // b. GENERATE VALIDATION SAMPLE
        auto unit_samples = latin_hypercube_4d(params_.validation_samples,
                                              params_.lhs_seed + iteration);

        std::array<std::pair<double, double>, 4> bounds = {{
            {min_moneyness, max_moneyness},
            {min_tau, max_tau},
            {min_vol, max_vol},
            {min_rate, max_rate}
        }};
        auto samples = scale_lhs_samples(unit_samples, bounds);

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

            // Interpolated price from surface
            double interp_price = surface.value()->value({m, tau, sigma, rate});

            // Fresh FD solve for reference
            double strike = chain.spot / m;

            // Create PricingParams using constructor
            PricingParams params;
            params.spot = chain.spot;
            params.strike = strike;
            params.maturity = tau;
            params.rate = rate;
            params.dividend_yield = chain.dividend_yield;
            params.type = type;
            params.volatility = sigma;

            auto fd_result = solve_american_option_auto(params);

            if (!fd_result.has_value()) {
                continue;  // Skip failed solves
            }

            stats.pde_solves_validation++;

            double ref_price = fd_result->value();
            double iv_error = compute_error_metric(
                interp_price, ref_price,
                chain.spot, strike, tau, sigma, rate,
                chain.dividend_yield);

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
            error_bins.record_error(norm_pos, iv_error, params_.target_iv_error);
        }

        double avg_error = valid_samples > 0 ? sum_error / valid_samples : 0.0;

        stats.max_error = max_error;
        stats.avg_error = avg_error;

        auto iter_end = std::chrono::steady_clock::now();
        stats.elapsed_seconds = std::chrono::duration<double>(iter_end - iter_start).count();

        // d. CHECK CONVERGENCE
        bool converged = (max_error <= params_.target_iv_error);

        if (converged || iteration == params_.max_iterations - 1) {
            // Final iteration - save results
            stats.refined_dim = -1;  // No refinement on final iteration
            result.iterations.push_back(stats);
            result.surface = surface.value();
            result.axes = axes;
            result.achieved_max_error = max_error;
            result.achieved_avg_error = avg_error;
            result.target_met = converged;
            result.total_pde_solves = 0;
            for (const auto& it : result.iterations) {
                result.total_pde_solves += it.pde_solves_table + it.pde_solves_validation;
            }
            break;
        }

        // e. DIAGNOSE & REFINE
        size_t worst_dim = error_bins.worst_dimension();
        auto problematic = error_bins.problematic_bins(worst_dim);

        // Refine the worst dimension
        auto refine_grid = [this](std::vector<double>& grid) {
            size_t new_size = std::min(
                static_cast<size_t>(grid.size() * params_.refinement_factor),
                params_.max_points_per_dim
            );
            if (new_size > grid.size()) {
                // Insert midpoints to increase density
                std::vector<double> new_grid;
                new_grid.reserve(new_size);
                for (size_t i = 0; i < grid.size() - 1; ++i) {
                    new_grid.push_back(grid[i]);
                    if (new_grid.size() < new_size) {
                        new_grid.push_back((grid[i] + grid[i+1]) / 2.0);
                    }
                }
                new_grid.push_back(grid.back());
                std::sort(new_grid.begin(), new_grid.end());
                new_grid.erase(std::unique(new_grid.begin(), new_grid.end()),
                              new_grid.end());
                grid = std::move(new_grid);
            }
        };

        switch (worst_dim) {
            case 0: refine_grid(moneyness_grid); break;
            case 1: refine_grid(maturity_grid); break;
            case 2: refine_grid(vol_grid); break;
            case 3: refine_grid(rate_grid); break;
        }

        stats.refined_dim = static_cast<int>(worst_dim);
        result.iterations.push_back(stats);
    }

    return result;
}

double AdaptiveGridBuilder::compute_error_metric(
    double interpolated_price, double reference_price,
    double spot, double strike, double tau, double sigma, double rate,
    double dividend_yield) const
{
    double price_error = std::abs(interpolated_price - reference_price);
    double vega = bs_vega(spot, strike, tau, sigma, rate, dividend_yield);

    if (vega >= params_.vega_floor) {
        return price_error / vega;
    } else {
        // Fallback: treat vega_floor as minimum vega
        return price_error / params_.vega_floor;
    }
}

BatchAmericanOptionResult AdaptiveGridBuilder::merge_results(
    const std::vector<AmericanOptionParams>& all_params,
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
                // Cache stores shared_ptr<AmericanOptionResult>
                // We can create a new AmericanOptionResult that shares the same grid
                // AmericanOptionParams is an alias for PricingParams, so we can pass it directly
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
