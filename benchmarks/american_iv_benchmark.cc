#include <benchmark/benchmark.h>
#include <vector>
#include <cmath>

extern "C" {
#include "../src/implied_volatility.h"
#include "../src/american_option.h"
#include "../src/lets_be_rational.h"
}

// Benchmark: Let's Be Rational European IV (for comparison)
static void BM_LetsBeRational(benchmark::State& state) {
    double spot = 100.0;
    double strike = 100.0;
    double time_to_maturity = 1.0;
    double risk_free_rate = 0.05;
    double market_price = 10.45;
    bool is_call = true;

    for (auto _ : state) {
        LBRResult result = lbr_implied_volatility(spot, strike, time_to_maturity,
                                                   risk_free_rate, market_price, is_call);
        benchmark::DoNotOptimize(result.implied_vol);
    }

    state.SetLabel("Let's Be Rational (European IV estimate)");
}

// Benchmark: Single American IV calculation
static void BM_AmericanIV_Single(benchmark::State& state) {
    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .market_price = 6.08,
        .is_call = false
    };

    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 141,
        .dt = 0.001,
        .n_steps = 1000
    };

    for (auto _ : state) {
        IVResult result = calculate_iv(&params, &grid, 1e-6, 100);
        benchmark::DoNotOptimize(result.implied_vol);
    }

    state.SetLabel("FDM-based American IV (141 pts, 1000 steps)");
}

// Benchmark: American IV with different grid sizes
static void BM_AmericanIV_GridSize(benchmark::State& state) {
    const size_t n_points = state.range(0);

    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .market_price = 6.08,
        .is_call = false
    };

    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = static_cast<int>(n_points),
        .dt = 0.001,
        .n_steps = 1000
    };

    for (auto _ : state) {
        IVResult result = calculate_iv(&params, &grid, 1e-6, 100);
        benchmark::DoNotOptimize(result.implied_vol);
    }

    state.SetLabel(std::to_string(n_points) + " spatial points");
}

// Benchmark: American IV with different maturities
static void BM_AmericanIV_Maturity(benchmark::State& state) {
    const double maturity = state.range(0) / 12.0;  // Convert months to years

    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = maturity,
        .risk_free_rate = 0.05,
        .market_price = maturity < 0.5 ? 3.0 : 6.08,
        .is_call = false
    };

    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 141,
        .dt = 0.001,
        .n_steps = 1000
    };

    for (auto _ : state) {
        IVResult result = calculate_iv(&params, &grid, 1e-6, 100);
        benchmark::DoNotOptimize(result.implied_vol);
    }

    char label[64];
    snprintf(label, sizeof(label), "%.1f months", maturity * 12.0);
    state.SetLabel(label);
}

// Benchmark: American IV - ATM vs OTM vs ITM
static void BM_AmericanIV_Moneyness(benchmark::State& state) {
    const int moneyness_case = state.range(0);
    double strike, market_price;

    switch (moneyness_case) {
        case 0:  // ATM
            strike = 100.0;
            market_price = 6.08;
            break;
        case 1:  // OTM (strike > spot)
            strike = 110.0;
            market_price = 11.0;
            break;
        case 2:  // ITM (strike < spot)
            strike = 90.0;
            market_price = 3.0;
            break;
        default:
            strike = 100.0;
            market_price = 6.08;
    }

    IVParams params = {
        .spot_price = 100.0,
        .strike = strike,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .market_price = market_price,
        .is_call = false
    };

    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 141,
        .dt = 0.001,
        .n_steps = 1000
    };

    for (auto _ : state) {
        IVResult result = calculate_iv(&params, &grid, 1e-6, 100);
        benchmark::DoNotOptimize(result.implied_vol);
    }

    const char* labels[] = {"ATM", "OTM", "ITM"};
    state.SetLabel(labels[moneyness_case]);
}

// Register benchmarks
BENCHMARK(BM_LetsBeRational);
BENCHMARK(BM_AmericanIV_Single);
BENCHMARK(BM_AmericanIV_GridSize)->Arg(71)->Arg(141)->Arg(201);
BENCHMARK(BM_AmericanIV_Maturity)->Arg(1)->Arg(3)->Arg(6)->Arg(12);
BENCHMARK(BM_AmericanIV_Moneyness)->Arg(0)->Arg(1)->Arg(2);

BENCHMARK_MAIN();
