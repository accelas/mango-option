// SPDX-License-Identifier: MIT
/**
 * @file market_iv_e2e_benchmark.cc
 * @brief End-to-end API usage example with realistic option market data
 *
 * **Purpose:** This benchmark serves as both a performance test and a comprehensive
 * API usage example, demonstrating the recommended workflow for:
 * 1. Building price tables with the normalized chain solver
 * 2. Computing implied volatility for entire option surfaces
 * 3. Validating results against market-implied volatilities
 *
 * **API Design Validation:**
 * - Tests ease of integration with market-like data
 * - Identifies pain points in API ergonomics
 * - Documents recommended usage patterns
 * - Measures end-to-end performance on production-like workloads
 *
 * **Workflow Example:**
 * ```cpp
 * // Step 1: Define option surface grid (from market data)
 * auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, 2.0).value();
 * auto [builder, axes] = PriceTableBuilder<4>::from_vectors(
 *     {0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2},  // moneyness
 *     {0.1, 0.25, 0.5, 1.0, 2.0},              // maturity
 *     {0.15, 0.20, 0.25, 0.30, 0.40},          // volatility
 *     {0.02, 0.03, 0.04, 0.05},                // rate
 *     100.0,                                    // K_ref
 *     ExplicitPDEGrid{grid_spec, 1000},
 *     OptionType::PUT,
 *     dividend).value();
 *
 * // Step 2: Build price table (one-time precomputation)
 * auto result = builder.build(axes);
 *
 * // Step 3: Create IV solver from surface
 * auto aps = AmericanPriceSurface::create(result.value().surface, OptionType::PUT);
 * auto solver_result = IVSolverInterpolatedStandard::create(std::move(*aps));
 * const auto& iv_solver = solver_result.value();
 *
 * // Step 4: Solve for IV at any (S, K, T, r)
 * IVQuery query{spot, strike, maturity, rate, dividend, OptionType::PUT, market_price};
 * auto iv_result = iv_solver.solve(query);
 * ```
 *
 * **Usage:**
 * ```bash
 * bazel run -c opt //benchmarks:market_iv_e2e_benchmark
 * ```
 */

#include "src/option/table/price_table_builder.hpp"
#include "src/option/table/price_table_surface.hpp"
#include "src/option/table/american_price_surface.hpp"
#include "src/option/iv_solver_interpolated.hpp"
#include "src/math/bspline_nd_separable.hpp"
#include <benchmark/benchmark.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

using namespace mango;

namespace {

/// Market-representative grid configuration
struct MarketGrid {
    std::vector<double> moneyness;     // S/K ratios
    std::vector<double> maturities;    // Years
    std::vector<double> volatilities;  // Annual vols
    std::vector<double> rates;         // Risk-free rates
    double K_ref;                      // Reference strike
    double spot;                       // Current underlying price
    double dividend;                   // Continuous dividend yield
};

/// Generate realistic market grid (based on SPY options)
MarketGrid generate_market_grid() {
    MarketGrid grid;

    // SPY-like parameters
    grid.spot = 450.0;
    grid.K_ref = 450.0;  // ATM strike as reference
    grid.dividend = 0.015;  // ~1.5% dividend yield

    // Moneyness grid: 0.85 to 1.15 (15% OTM to 15% ITM)
    // Denser near ATM (where most trading occurs)
    grid.moneyness = {
        0.85, 0.90, 0.93, 0.95, 0.97, 0.99,  // OTM puts
        1.00,                                  // ATM
        1.01, 1.03, 1.05, 1.07, 1.10, 1.15   // ITM puts
    };

    // Maturities: weekly to 2 years
    grid.maturities = {
        7.0/365,    // 1 week
        14.0/365,   // 2 weeks
        30.0/365,   // 1 month
        60.0/365,   // 2 months
        90.0/365,   // 3 months
        180.0/365,  // 6 months
        1.0,        // 1 year
        2.0         // 2 years
    };

    // Volatility grid: 10% to 50% (typical equity range)
    grid.volatilities = {
        0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50
    };

    // Rate grid: 2% to 5% (current environment)
    grid.rates = {0.02, 0.03, 0.04, 0.05};

    return grid;
}

/// Simulate market option prices at specific parameters
struct MarketObservation {
    double spot;
    double strike;
    double maturity;
    double rate;
    double true_vol;        // "True" market IV
    double market_price;    // Observed price (with noise)
    OptionType type;
};

/// Generate sample market observations for testing
std::vector<MarketObservation> generate_market_observations(
    const MarketGrid& grid,
    size_t num_obs = 100)
{
    std::vector<MarketObservation> observations;
    std::mt19937 rng(42);  // Fixed seed for reproducibility

    // Sample random points from grid
    std::uniform_int_distribution<size_t> m_dist(0, grid.moneyness.size() - 1);
    std::uniform_int_distribution<size_t> t_dist(0, grid.maturities.size() - 1);
    std::uniform_int_distribution<size_t> v_dist(0, grid.volatilities.size() - 1);
    std::uniform_int_distribution<size_t> r_dist(0, grid.rates.size() - 1);

    for (size_t i = 0; i < num_obs; ++i) {
        double m = grid.moneyness[m_dist(rng)];
        double tau = grid.maturities[t_dist(rng)];
        double vol = grid.volatilities[v_dist(rng)];
        double rate = grid.rates[r_dist(rng)];

        double strike = grid.spot / m;

        // Generate "market" price (we'll use this to test IV recovery)
        // In real usage, this would come from exchange data
        observations.push_back(MarketObservation{
            grid.spot,
            strike,
            tau,
            rate,
            vol,
            0.0,  // Will be filled after price table build
            OptionType::PUT
        });
    }

    return observations;
}

} // namespace

// ============================================================================
// Benchmark: API Usage - Building Price Table
// ============================================================================

static void BM_API_BuildPriceTable(benchmark::State& state) {
    MarketGrid grid = generate_market_grid();

    for (auto _ : state) {
        // API STEP 1: Create grid spec and builder with market grids
        auto grid_spec_result = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0);
        if (!grid_spec_result) {
            state.SkipWithError("GridSpec creation failed");
            return;
        }
        auto grid_spec = grid_spec_result.value();

        auto builder_axes_result = PriceTableBuilder<4>::from_vectors(
            grid.moneyness,
            grid.maturities,
            grid.volatilities,
            grid.rates,
            grid.K_ref,
            ExplicitPDEGrid{grid_spec, 500},
            OptionType::PUT,
            grid.dividend);

        if (!builder_axes_result) {
            state.SkipWithError("PriceTableBuilder::from_vectors failed");
            return;
        }
        auto [builder, axes] = std::move(builder_axes_result.value());

        // API STEP 2: Precompute all prices (one PDE solve per σ,r pair)
        auto result = builder.build(axes);

        if (!result) {
            state.SkipWithError("PriceTableBuilder::build failed");
            return;
        }

        benchmark::DoNotOptimize(result.value());
    }

    // Report grid size
    state.counters["moneyness_pts"] = grid.moneyness.size();
    state.counters["maturity_pts"] = grid.maturities.size();
    state.counters["vol_pts"] = grid.volatilities.size();
    state.counters["rate_pts"] = grid.rates.size();
    state.counters["total_points"] = grid.moneyness.size() * grid.maturities.size()
                                   * grid.volatilities.size() * grid.rates.size();
}

BENCHMARK(BM_API_BuildPriceTable)->Unit(benchmark::kMillisecond);

// ============================================================================
// Benchmark: API Usage - Computing IV for Option Surface
// ============================================================================

static void BM_API_ComputeIVSurface(benchmark::State& state) {
    MarketGrid grid = generate_market_grid();

    // Build price table once (setup, not timed)
    auto grid_spec_result = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0);
    if (!grid_spec_result) {
        state.SkipWithError("GridSpec creation failed");
        return;
    }
    auto grid_spec = grid_spec_result.value();

    auto builder_axes_result = PriceTableBuilder<4>::from_vectors(
        grid.moneyness,
        grid.maturities,
        grid.volatilities,
        grid.rates,
        grid.K_ref,
        ExplicitPDEGrid{grid_spec, 500},
        OptionType::PUT,
        grid.dividend);

    if (!builder_axes_result) {
        state.SkipWithError("PriceTableBuilder::from_vectors failed");
        return;
    }
    auto [builder, axes] = std::move(builder_axes_result.value());
    auto price_table_result = builder.build(axes);

    if (!price_table_result) {
        state.SkipWithError("PriceTableBuilder::build failed");
        return;
    }

    const auto& price_table = price_table_result.value();
    const auto& surface = price_table.surface;

    // Generate market observations
    auto observations = generate_market_observations(grid, 100);

    // Fill in "market prices" using our price table (simulates real market data)
    for (auto& obs : observations) {
        double m = obs.spot / obs.strike;
        obs.market_price = surface->value({m, obs.maturity, obs.true_vol, obs.rate});
    }

    // API STEP 3: Create IV solver
    IVSolverInterpolatedConfig solver_config;
    solver_config.max_iter = 50;
    solver_config.tolerance = 1e-6;

    auto aps = AmericanPriceSurface::create(surface, OptionType::PUT);
    if (!aps) {
        state.SkipWithError("AmericanPriceSurface::create failed");
        return;
    }
    auto iv_solver_result = IVSolverInterpolatedStandard::create(std::move(*aps), solver_config);
    if (!iv_solver_result) {
        auto err = iv_solver_result.error();
        std::string error_msg = "Validation error code " + std::to_string(static_cast<int>(err.code));
        state.SkipWithError(error_msg.c_str());
        return;
    }
    const auto& iv_solver = iv_solver_result.value();

    // Benchmark: Compute IV for all observations
    size_t converged = 0;
    double total_error = 0.0;

    for (auto _ : state) {
        converged = 0;
        total_error = 0.0;

        for (const auto& obs : observations) {
            // API STEP 4: Solve for IV
            IVQuery query;
            query.spot = obs.spot;
            query.strike = obs.strike;
            query.maturity = obs.maturity;
            query.rate = obs.rate;
            query.dividend_yield = grid.dividend;
            query.type = obs.type;
            query.market_price = obs.market_price;

            auto result = iv_solver.solve(query);

            if (result.has_value()) {
                converged++;
                total_error += std::abs(result->implied_vol - obs.true_vol);
            }
        }

        benchmark::DoNotOptimize(converged);
        benchmark::DoNotOptimize(total_error);
    }

    // Report statistics
    state.counters["observations"] = observations.size();
    state.counters["converged"] = converged;
    state.counters["convergence_rate"] = (double)converged / observations.size();
    state.counters["mean_abs_error_pct"] = (total_error / converged) * 100;
    state.counters["ivs_per_sec"] = benchmark::Counter(
        observations.size(), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_API_ComputeIVSurface)->Unit(benchmark::kMillisecond);

// ============================================================================
// Benchmark: End-to-End Workflow (Table Build + IV Computation)
// ============================================================================

static void BM_API_EndToEnd(benchmark::State& state) {
    MarketGrid grid = generate_market_grid();
    auto observations = generate_market_observations(grid, 100);

    for (auto _ : state) {
        // FULL WORKFLOW: Build table → Solve IVs

        // Step 1-2: Build price table
        auto grid_spec_result = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0);
        if (!grid_spec_result) {
            state.SkipWithError("GridSpec creation failed");
            return;
        }
        auto grid_spec = grid_spec_result.value();

        auto builder_axes_result = PriceTableBuilder<4>::from_vectors(
            grid.moneyness,
            grid.maturities,
            grid.volatilities,
            grid.rates,
            grid.K_ref,
            ExplicitPDEGrid{grid_spec, 500},
            OptionType::PUT,
            grid.dividend);

        if (!builder_axes_result) {
            state.SkipWithError("PriceTableBuilder::from_vectors failed");
            return;
        }
        auto [builder, axes] = std::move(builder_axes_result.value());
        auto price_table_result = builder.build(axes);

        if (!price_table_result) {
            state.SkipWithError("PriceTableBuilder::build failed");
            return;
        }
        auto price_table = std::move(price_table_result.value());
        const auto& surface = price_table.surface;

        // Fill market prices
        for (auto& obs : observations) {
            double m = obs.spot / obs.strike;
            obs.market_price = surface->value({m, obs.maturity, obs.true_vol, obs.rate});
        }

        // Step 3-4: Compute IVs
        auto aps = AmericanPriceSurface::create(surface, OptionType::PUT);
        if (!aps) {
            state.SkipWithError("AmericanPriceSurface::create failed");
            return;
        }
        auto iv_solver_result = IVSolverInterpolatedStandard::create(std::move(*aps));
        if (!iv_solver_result) {
            auto err = iv_solver_result.error();
            std::string error_msg = "Validation error code " + std::to_string(static_cast<int>(err.code));
            state.SkipWithError(error_msg.c_str());
            return;
        }
        auto& iv_solver = iv_solver_result.value();

        size_t converged = 0;
        for (const auto& obs : observations) {
            IVQuery query;
            query.spot = obs.spot;
            query.strike = obs.strike;
            query.maturity = obs.maturity;
            query.rate = obs.rate;
            query.dividend_yield = 0.0;
            query.type = obs.type;
            query.market_price = obs.market_price;
            if (iv_solver.solve(query).has_value()) {
                converged++;
            }
        }

        benchmark::DoNotOptimize(converged);
    }

    state.counters["total_operations"] = 2;  // Build + Solve
}

BENCHMARK(BM_API_EndToEnd)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
