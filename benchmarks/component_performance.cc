/**
 * @file component_performance.cc
 * @brief Component-level performance benchmarks for C++20 implementation
 *
 * Benchmarks individual components to understand performance characteristics:
 * - American option pricing (single calculation)
 * - Implied volatility solving (single calculation)
 * - Batch operations
 *
 * Run with: bazel run //benchmarks:component_performance
 */

#include "src/american_option.hpp"
#include "src/iv_solver.hpp"
#include <benchmark/benchmark.h>
#include <cmath>

using namespace mango;

// ============================================================================
// American Option Pricing Benchmarks
// ============================================================================

static void BM_AmericanPut_ATM_1Y(benchmark::State& state) {
    AmericanOptionParams params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .discrete_dividends = {}
    };

    AmericanOptionGrid grid;
    grid.n_space = state.range(0);
    grid.n_time = 1000;

    for (auto _ : state) {
        AmericanOptionSolver solver(params, grid);
        auto result = solver.solve();
        benchmark::DoNotOptimize(result);
    }

    state.SetLabel("ATM Put, T=1Y, σ=0.20");
}
BENCHMARK(BM_AmericanPut_ATM_1Y)->Arg(101)->Arg(201)->Arg(501);

static void BM_AmericanPut_OTM_3M(benchmark::State& state) {
    AmericanOptionParams params{
        .strike = 100.0,
        .spot = 110.0,  // OTM put
        .maturity = 0.25,
        .volatility = 0.30,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .discrete_dividends = {}
    };

    AmericanOptionGrid grid;
    grid.n_space = 101;
    grid.n_time = state.range(0);

    for (auto _ : state) {
        AmericanOptionSolver solver(params, grid);
        auto result = solver.solve();
        benchmark::DoNotOptimize(result);
    }

    state.SetLabel("OTM Put, T=3M, σ=0.30");
}
BENCHMARK(BM_AmericanPut_OTM_3M)->Arg(500)->Arg(1000)->Arg(2000);

static void BM_AmericanPut_ITM_2Y(benchmark::State& state) {
    AmericanOptionParams params{
        .strike = 100.0,
        .spot = 90.0,  // ITM put
        .maturity = 2.0,
        .volatility = 0.25,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .discrete_dividends = {}
    };

    AmericanOptionGrid grid;
    grid.n_space = 101;
    grid.n_time = 1000;

    for (auto _ : state) {
        AmericanOptionSolver solver(params, grid);
        auto result = solver.solve();
        benchmark::DoNotOptimize(result);
    }

    state.SetLabel("ITM Put, T=2Y, σ=0.25");
}
BENCHMARK(BM_AmericanPut_ITM_2Y);

static void BM_AmericanCall_WithDividends(benchmark::State& state) {
    AmericanOptionParams params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::CALL,
        .discrete_dividends = {{0.25, 2.0}, {0.5, 2.0}, {0.75, 2.0}}
    };

    AmericanOptionGrid grid;
    grid.n_space = 101;
    grid.n_time = 1000;

    for (auto _ : state) {
        AmericanOptionSolver solver(params, grid);
        auto result = solver.solve();
        benchmark::DoNotOptimize(result);
    }

    state.SetLabel("Call with 3 discrete dividends");
}
BENCHMARK(BM_AmericanCall_WithDividends);

// ============================================================================
// Implied Volatility Benchmarks
// ============================================================================

static void BM_ImpliedVol_ATM_Put(benchmark::State& state) {
    IVParams params{
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .market_price = 6.0,  // Approximate for σ=0.20
        .is_call = false
    };

    IVConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    config.grid_n_space = 101;
    config.grid_n_time = 1000;

    for (auto _ : state) {
        IVSolver solver(params, config);
        auto result = solver.solve();
        benchmark::DoNotOptimize(result);
    }

    state.SetLabel("ATM Put, T=1Y");
}
BENCHMARK(BM_ImpliedVol_ATM_Put);

static void BM_ImpliedVol_OTM_Put(benchmark::State& state) {
    IVParams params{
        .spot_price = 110.0,
        .strike = 100.0,
        .time_to_maturity = 0.25,
        .risk_free_rate = 0.05,
        .market_price = 0.80,  // OTM put
        .is_call = false
    };

    IVConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    config.grid_n_space = 101;
    config.grid_n_time = 1000;

    for (auto _ : state) {
        IVSolver solver(params, config);
        auto result = solver.solve();
        benchmark::DoNotOptimize(result);
    }

    state.SetLabel("OTM Put, T=3M");
}
BENCHMARK(BM_ImpliedVol_OTM_Put);

static void BM_ImpliedVol_ITM_Put(benchmark::State& state) {
    IVParams params{
        .spot_price = 90.0,
        .strike = 100.0,
        .time_to_maturity = 2.0,
        .risk_free_rate = 0.05,
        .market_price = 15.0,  // ITM put
        .is_call = false
    };

    IVConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    config.grid_n_space = 101;
    config.grid_n_time = 1000;

    for (auto _ : state) {
        IVSolver solver(params, config);
        auto result = solver.solve();
        benchmark::DoNotOptimize(result);
    }

    state.SetLabel("ITM Put, T=2Y");
}
BENCHMARK(BM_ImpliedVol_ITM_Put);

// ============================================================================
// Grid Resolution Impact
// ============================================================================

static void BM_AmericanPut_GridResolution(benchmark::State& state) {
    size_t n_space = state.range(0);
    size_t n_time = state.range(1);

    AmericanOptionParams params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .discrete_dividends = {}
    };

    AmericanOptionGrid grid;
    grid.n_space = n_space;
    grid.n_time = n_time;

    for (auto _ : state) {
        AmericanOptionSolver solver(params, grid);
        auto result = solver.solve();
        benchmark::DoNotOptimize(result);
    }

    state.SetLabel("Grid: " + std::to_string(n_space) + "x" + std::to_string(n_time));
}
BENCHMARK(BM_AmericanPut_GridResolution)
    ->Args({51, 500})
    ->Args({101, 1000})
    ->Args({201, 2000})
    ->Args({501, 5000});

// ============================================================================
// Batch Processing Benchmarks
// ============================================================================

static void BM_AmericanPut_Batch(benchmark::State& state) {
    size_t batch_size = state.range(0);

    // Generate batch of strike prices around ATM
    std::vector<AmericanOptionParams> batch;
    batch.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        double strike = 90.0 + i * 0.5;  // Strikes from 90 to 90 + batch_size*0.5
        batch.push_back(AmericanOptionParams{
            .strike = strike,
            .spot = 100.0,
            .maturity = 1.0,
            .volatility = 0.20,
            .rate = 0.05,
            .continuous_dividend_yield = 0.02,
            .option_type = OptionType::PUT,
            .discrete_dividends = {}
        });
    }

    AmericanOptionGrid grid;
    grid.n_space = 101;
    grid.n_time = 1000;

    for (auto _ : state) {
        #pragma omp parallel for
        for (size_t i = 0; i < batch.size(); ++i) {
            AmericanOptionSolver solver(batch[i], grid);
            auto result = solver.solve();
            benchmark::DoNotOptimize(result);
        }
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
    state.SetLabel("Parallel batch: " + std::to_string(batch_size) + " options");
}
BENCHMARK(BM_AmericanPut_Batch)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100);

static void BM_ImpliedVol_Batch(benchmark::State& state) {
    size_t batch_size = state.range(0);

    // Generate batch of market prices
    std::vector<IVParams> batch;
    batch.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        double market_price = 5.0 + i * 0.1;  // Different prices
        batch.push_back(IVParams{
            .spot_price = 100.0,
            .strike = 100.0,
            .time_to_maturity = 1.0,
            .risk_free_rate = 0.05,
            .market_price = market_price,
            .is_call = false
        });
    }

    IVConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    config.grid_n_space = 101;
    config.grid_n_time = 1000;

    for (auto _ : state) {
        #pragma omp parallel for
        for (size_t i = 0; i < batch.size(); ++i) {
            IVSolver solver(batch[i], config);
            auto result = solver.solve();
            benchmark::DoNotOptimize(result);
        }
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
    state.SetLabel("Parallel batch: " + std::to_string(batch_size) + " IVs");
}
BENCHMARK(BM_ImpliedVol_Batch)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100);

BENCHMARK_MAIN();
