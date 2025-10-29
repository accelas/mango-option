#include <benchmark/benchmark.h>
#include <vector>
#include <cmath>

extern "C" {
#include "../src/american_option.h"
#include "../src/implied_volatility.h"
#include "../src/european_option.h"
}

// Benchmark: Sequential American option pricing
static void BM_AmericanOption_Sequential(benchmark::State& state) {
    const size_t n_options = state.range(0);

    // Setup options
    std::vector<OptionData> options(n_options);
    for (size_t i = 0; i < n_options; i++) {
        options[i] = (OptionData){
            .strike = 95.0 + i * 0.5,
            .volatility = 0.2 + i * 0.005,
            .risk_free_rate = 0.05,
            .time_to_maturity = 1.0,
            .option_type = (i % 2 == 0) ? OPTION_PUT : OPTION_CALL,
            .n_dividends = 0,
            .dividend_times = nullptr,
            .dividend_amounts = nullptr
        };
    }

    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 101,
        .dt = 0.001,
        .n_steps = 500
    };

    // Benchmark loop
    for (auto _ : state) {
        for (size_t i = 0; i < n_options; i++) {
            AmericanOptionResult result = american_option_price(&options[i], &grid);
            if (result.status == 0 && result.solver != nullptr) {
                // Prevent optimization
                benchmark::DoNotOptimize(result.solver);
                american_option_free_result(&result);
            }
        }
    }

    state.SetItemsProcessed(state.iterations() * n_options);
    state.SetLabel("sequential");
}

// Benchmark: Batch American option pricing (OpenMP parallel)
static void BM_AmericanOption_Batch(benchmark::State& state) {
    const size_t n_options = state.range(0);

    // Setup options
    std::vector<OptionData> options(n_options);
    for (size_t i = 0; i < n_options; i++) {
        options[i] = (OptionData){
            .strike = 95.0 + i * 0.5,
            .volatility = 0.2 + i * 0.005,
            .risk_free_rate = 0.05,
            .time_to_maturity = 1.0,
            .option_type = (i % 2 == 0) ? OPTION_PUT : OPTION_CALL,
            .n_dividends = 0,
            .dividend_times = nullptr,
            .dividend_amounts = nullptr
        };
    }

    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 101,
        .dt = 0.001,
        .n_steps = 500
    };

    std::vector<AmericanOptionResult> results(n_options);

    // Benchmark loop
    for (auto _ : state) {
        int status = american_option_price_batch(options.data(), &grid, n_options, results.data());
        benchmark::DoNotOptimize(status);

        // Cleanup
        for (size_t i = 0; i < n_options; i++) {
            if (results[i].status == 0 && results[i].solver != nullptr) {
                pde_solver_destroy(results[i].solver);
            }
        }
    }

    state.SetItemsProcessed(state.iterations() * n_options);
    state.SetLabel("batch_parallel");
}

// Benchmark: Implied volatility recovery (sequential)
static void BM_ImpliedVol_Sequential(benchmark::State& state) {
    const size_t n_options = state.range(0);

    // Generate market prices using known volatilities
    std::vector<double> market_prices(n_options);
    std::vector<IVParams> params(n_options);

    for (size_t i = 0; i < n_options; i++) {
        double strike = 95.0 + i * 0.5;
        double vol = 0.2 + i * 0.005;

        // Generate "market" price using Black-Scholes
        market_prices[i] = black_scholes_price(
            100.0,  // spot
            strike,
            1.0,    // time_to_maturity
            0.05,   // risk_free_rate
            vol,
            true    // is_call
        );

        params[i] = (IVParams){
            .spot_price = 100.0,
            .strike = strike,
            .time_to_maturity = 1.0,
            .risk_free_rate = 0.05,
            .market_price = market_prices[i],
            .is_call = true
        };
    }

    // Benchmark loop
    for (auto _ : state) {
        for (size_t i = 0; i < n_options; i++) {
            IVResult result = implied_volatility_calculate_simple(&params[i]);
            benchmark::DoNotOptimize(result.implied_vol);
        }
    }

    state.SetItemsProcessed(state.iterations() * n_options);
    state.SetLabel("sequential");
}

// Benchmark: Compare grid resolutions (sequential)
static void BM_AmericanOption_GridResolution(benchmark::State& state) {
    const size_t n_points = state.range(0);

    OptionData option = {
        .strike = 100.0,
        .volatility = 0.25,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = n_points,
        .dt = 0.001,
        .n_steps = 1000
    };

    for (auto _ : state) {
        AmericanOptionResult result = american_option_price(&option, &grid);
        if (result.status == 0 && result.solver != nullptr) {
            benchmark::DoNotOptimize(result.solver);
            american_option_free_result(&result);
        }
    }

    state.SetLabel(std::to_string(n_points) + "_points");
}

// Benchmark: Batch with varying batch sizes
static void BM_AmericanOption_BatchScaling(benchmark::State& state) {
    const size_t n_options = state.range(0);

    std::vector<OptionData> options(n_options);
    for (size_t i = 0; i < n_options; i++) {
        options[i] = (OptionData){
            .strike = 100.0,
            .volatility = 0.25,
            .risk_free_rate = 0.05,
            .time_to_maturity = 1.0,
            .option_type = OPTION_PUT,
            .n_dividends = 0,
            .dividend_times = nullptr,
            .dividend_amounts = nullptr
        };
    }

    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 101,
        .dt = 0.001,
        .n_steps = 500
    };

    std::vector<AmericanOptionResult> results(n_options);

    for (auto _ : state) {
        int status = american_option_price_batch(options.data(), &grid, n_options, results.data());
        benchmark::DoNotOptimize(status);

        for (size_t i = 0; i < n_options; i++) {
            if (results[i].status == 0 && results[i].solver != nullptr) {
                pde_solver_destroy(results[i].solver);
            }
        }
    }

    state.SetItemsProcessed(state.iterations() * n_options);
}

// Benchmark: Time step impact on single option
static void BM_AmericanOption_TimeSteps(benchmark::State& state) {
    const size_t n_steps = state.range(0);

    OptionData option = {
        .strike = 100.0,
        .volatility = 0.25,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 101,
        .dt = 1.0 / n_steps,
        .n_steps = n_steps
    };

    for (auto _ : state) {
        AmericanOptionResult result = american_option_price(&option, &grid);
        if (result.status == 0 && result.solver != nullptr) {
            benchmark::DoNotOptimize(result.solver);
            american_option_free_result(&result);
        }
    }

    state.SetLabel(std::to_string(n_steps) + "_steps");
}

// Register benchmarks with various batch sizes

// Sequential vs Batch comparison (10, 25, 50, 100 options)
BENCHMARK(BM_AmericanOption_Sequential)->Arg(10)->Arg(25)->Arg(50)->Arg(100)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AmericanOption_Batch)->Arg(10)->Arg(25)->Arg(50)->Arg(100)->Unit(benchmark::kMillisecond);

// Implied volatility sequential
BENCHMARK(BM_ImpliedVol_Sequential)->Arg(10)->Arg(50)->Arg(100)->Unit(benchmark::kMillisecond);

// Grid resolution impact (51, 101, 201 points)
BENCHMARK(BM_AmericanOption_GridResolution)->Arg(51)->Arg(101)->Arg(201)->Unit(benchmark::kMillisecond);

// Batch scaling (5, 10, 20, 50, 100, 200 options)
BENCHMARK(BM_AmericanOption_BatchScaling)
    ->RangeMultiplier(2)
    ->Range(5, 200)
    ->Unit(benchmark::kMillisecond);

// Time step impact (250, 500, 1000, 2000 steps)
BENCHMARK(BM_AmericanOption_TimeSteps)
    ->Arg(250)->Arg(500)->Arg(1000)->Arg(2000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
