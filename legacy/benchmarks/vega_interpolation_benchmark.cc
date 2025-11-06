#include "benchmark/benchmark.h"
#include "../src/price_table.h"
#include <vector>
#include <cmath>

static OptionPriceTable* g_table = nullptr;

static void SetupTable(const benchmark::State& state) {
    if (g_table) return;  // Already set up

    // Create 4D table with reasonable grid
    std::vector<double> m(30);
    std::vector<double> tau(25);
    std::vector<double> sigma(15);
    std::vector<double> r(10);

    // Log-spaced moneyness
    for (size_t i = 0; i < 30; i++) {
        double t = (double)i / 29.0;
        m[i] = 0.7 * exp(t * log(1.5 / 0.7));
    }

    // Linear maturity
    for (size_t i = 0; i < 25; i++) {
        tau[i] = 0.027 + i * (2.5 - 0.027) / 24.0;
    }

    // Linear volatility
    for (size_t i = 0; i < 15; i++) {
        sigma[i] = 0.10 + i * (0.60 - 0.10) / 14.0;
    }

    // Linear rate
    for (size_t i = 0; i < 10; i++) {
        r[i] = 0.0 + i * (0.10 - 0.0) / 9.0;
    }

    g_table = price_table_create_ex(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_PUT, AMERICAN,
        COORD_LOG_SQRT, LAYOUT_M_INNER);

    // Precompute
    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 101, .dt = 0.001, .n_steps = 1000
    };

    price_table_precompute(g_table, &grid);
    price_table_build_interpolation(g_table);
}

static void BM_VegaInterpolation4D(benchmark::State& state) {
    SetupTable(state);

    // Query at off-grid point
    const double m = 1.05;
    const double tau = 0.5;
    const double sigma = 0.25;
    const double r = 0.05;

    for (auto _ : state) {
        double vega = price_table_interpolate_vega_4d(g_table, m, tau, sigma, r);
        benchmark::DoNotOptimize(vega);
    }
}
BENCHMARK(BM_VegaInterpolation4D);

static void BM_PriceInterpolation4D(benchmark::State& state) {
    SetupTable(state);

    const double m = 1.05;
    const double tau = 0.5;
    const double sigma = 0.25;
    const double r = 0.05;

    for (auto _ : state) {
        double price = price_table_interpolate_4d(g_table, m, tau, sigma, r);
        benchmark::DoNotOptimize(price);
    }
}
BENCHMARK(BM_PriceInterpolation4D);

BENCHMARK_MAIN();
