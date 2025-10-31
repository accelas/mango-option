#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <chrono>

extern "C" {
#include "../src/price_table.h"
#include "../src/american_option.h"
}

// Global state for precomputed table (shared across benchmarks)
static OptionPriceTable* g_precomputed_table = nullptr;
static std::vector<double> g_query_moneyness;
static std::vector<double> g_query_maturity;
static std::vector<double> g_query_volatility;
static std::vector<double> g_query_rate;

// Setup: Create and precompute price table (runs once)
static void SetupPrecomputedTable(const benchmark::State& state) {
    if (g_precomputed_table != nullptr) {
        return; // Already initialized
    }

    std::cout << "\n========================================\n";
    std::cout << "Precomputing Price Table (One-Time Setup)\n";
    std::cout << "========================================\n";

    // Create 4D grid (smaller for reasonable precompute time)
    const size_t n_m = 20;      // Moneyness: 20 points
    const size_t n_tau = 15;    // Maturity: 15 points
    const size_t n_sigma = 10;  // Volatility: 10 points
    const size_t n_r = 5;       // Rate: 5 points

    std::vector<double> moneyness(n_m);
    std::vector<double> maturity(n_tau);
    std::vector<double> volatility(n_sigma);
    std::vector<double> rate(n_r);

    // Log-spaced moneyness
    for (size_t i = 0; i < n_m; i++) {
        double t = static_cast<double>(i) / (n_m - 1);
        moneyness[i] = 0.8 * exp(t * log(1.3 / 0.8));
    }

    // Linear maturity
    for (size_t i = 0; i < n_tau; i++) {
        maturity[i] = 0.027 + i * (2.0 - 0.027) / (n_tau - 1);
    }

    // Linear volatility
    for (size_t i = 0; i < n_sigma; i++) {
        volatility[i] = 0.10 + i * (0.50 - 0.10) / (n_sigma - 1);
    }

    // Linear rate
    for (size_t i = 0; i < n_r; i++) {
        rate[i] = 0.0 + i * (0.08 - 0.0) / (n_r - 1);
    }

    // Create table
    g_precomputed_table = price_table_create(
        moneyness.data(), n_m,
        maturity.data(), n_tau,
        volatility.data(), n_sigma,
        rate.data(), n_r,
        nullptr, 0,
        OPTION_PUT, AMERICAN);

    if (!g_precomputed_table) {
        std::cerr << "Failed to create price table\n";
        return;
    }

    price_table_set_underlying(g_precomputed_table, "TEST");

    size_t total_points = n_m * n_tau * n_sigma * n_r;
    std::cout << "Grid: " << n_m << "×" << n_tau << "×" << n_sigma << "×" << n_r
              << " = " << total_points << " points\n";

    // Configure FDM grid (smaller for faster precompute)
    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 101,
        .dt = 0.001,
        .n_steps = 500
    };

    std::cout << "FDM grid: " << grid.n_points << " points × " << grid.n_steps << " steps\n";
    std::cout << "Starting precomputation...\n";

    auto start = std::chrono::high_resolution_clock::now();
    int status = price_table_precompute(g_precomputed_table, &grid);
    auto end = std::chrono::high_resolution_clock::now();

    if (status != 0) {
        std::cerr << "Precomputation failed!\n";
        return;
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Precomputation complete: " << duration.count() << " ms\n";
    std::cout << "Throughput: " << (total_points * 1000.0 / duration.count()) << " opts/sec\n";
    std::cout << "========================================\n\n";

    // Generate random query points (1000 samples)
    std::mt19937 gen(42);
    std::uniform_real_distribution<> m_dist(0.85, 1.25);
    std::uniform_real_distribution<> tau_dist(0.1, 1.5);
    std::uniform_real_distribution<> sigma_dist(0.15, 0.40);
    std::uniform_real_distribution<> r_dist(0.01, 0.06);

    g_query_moneyness.resize(1000);
    g_query_maturity.resize(1000);
    g_query_volatility.resize(1000);
    g_query_rate.resize(1000);

    for (size_t i = 0; i < 1000; i++) {
        g_query_moneyness[i] = m_dist(gen);
        g_query_maturity[i] = tau_dist(gen);
        g_query_volatility[i] = sigma_dist(gen);
        g_query_rate[i] = r_dist(gen);
    }
}

// Benchmark: Interpolation from precomputed table
static void BM_Interpolation_4D(benchmark::State& state) {
    SetupPrecomputedTable(state);

    if (!g_precomputed_table) {
        state.SkipWithError("Failed to setup precomputed table");
        return;
    }

    size_t idx = 0;
    for (auto _ : state) {
        double price = price_table_interpolate_4d(
            g_precomputed_table,
            g_query_moneyness[idx],
            g_query_maturity[idx],
            g_query_volatility[idx],
            g_query_rate[idx]);

        benchmark::DoNotOptimize(price);
        idx = (idx + 1) % 1000;
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("table_interpolation");
}

// Benchmark: Direct FDM solve
static void BM_DirectFDM_4D(benchmark::State& state) {
    SetupPrecomputedTable(state);

    // FDM grid configuration (same as precompute)
    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 101,
        .dt = 0.001,
        .n_steps = 500
    };

    const double strike = 100.0;
    size_t idx = 0;

    for (auto _ : state) {
        // Convert query parameters to option
        double spot = g_query_moneyness[idx] * strike;

        OptionData option = {
            .strike = strike,
            .volatility = g_query_volatility[idx],
            .risk_free_rate = g_query_rate[idx],
            .time_to_maturity = g_query_maturity[idx],
            .option_type = OPTION_PUT,
            .n_dividends = 0,
            .dividend_times = nullptr,
            .dividend_amounts = nullptr
        };

        AmericanOptionResult result = american_option_price(&option, &grid);

        if (result.status == 0 && result.solver != nullptr) {
            double price = american_option_get_value_at_spot(result.solver, spot, strike);
            benchmark::DoNotOptimize(price);
            american_option_free_result(&result);
        }

        idx = (idx + 1) % 1000;
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("direct_fdm");
}

// Benchmark: Batch comparison (10 queries)
static void BM_Interpolation_Batch10(benchmark::State& state) {
    SetupPrecomputedTable(state);

    if (!g_precomputed_table) {
        state.SkipWithError("Failed to setup precomputed table");
        return;
    }

    for (auto _ : state) {
        for (size_t i = 0; i < 10; i++) {
            double price = price_table_interpolate_4d(
                g_precomputed_table,
                g_query_moneyness[i],
                g_query_maturity[i],
                g_query_volatility[i],
                g_query_rate[i]);
            benchmark::DoNotOptimize(price);
        }
    }

    state.SetItemsProcessed(state.iterations() * 10);
    state.SetLabel("table_batch_10");
}

// Benchmark: Batch FDM (10 queries)
static void BM_DirectFDM_Batch10(benchmark::State& state) {
    SetupPrecomputedTable(state);

    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 101,
        .dt = 0.001,
        .n_steps = 500
    };

    const double strike = 100.0;

    for (auto _ : state) {
        for (size_t i = 0; i < 10; i++) {
            double spot = g_query_moneyness[i] * strike;

            OptionData option = {
                .strike = strike,
                .volatility = g_query_volatility[i],
                .risk_free_rate = g_query_rate[i],
                .time_to_maturity = g_query_maturity[i],
                .option_type = OPTION_PUT,
                .n_dividends = 0,
                .dividend_times = nullptr,
                .dividend_amounts = nullptr
            };

            AmericanOptionResult result = american_option_price(&option, &grid);

            if (result.status == 0 && result.solver != nullptr) {
                double price = american_option_get_value_at_spot(result.solver, spot, strike);
                benchmark::DoNotOptimize(price);
                american_option_free_result(&result);
            }
        }
    }

    state.SetItemsProcessed(state.iterations() * 10);
    state.SetLabel("fdm_batch_10");
}

// Register benchmarks
BENCHMARK(BM_Interpolation_4D)
    ->Unit(benchmark::kNanosecond)
    ->Iterations(10000);

BENCHMARK(BM_DirectFDM_4D)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

BENCHMARK(BM_Interpolation_Batch10)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_DirectFDM_Batch10)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5);

// Custom main to add summary
int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }

    benchmark::RunSpecifiedBenchmarks();

    // Print summary
    std::cout << "\n========================================\n";
    std::cout << "PERFORMANCE SUMMARY\n";
    std::cout << "========================================\n";
    std::cout << "This benchmark directly compares:\n";
    std::cout << "  1. Precomputed table interpolation (sub-microsecond)\n";
    std::cout << "  2. Direct FDM solve (milliseconds)\n\n";
    std::cout << "Expected speedup: ~10,000x - 50,000x\n";
    std::cout << "Depending on grid resolution and query patterns.\n";
    std::cout << "========================================\n";

    // Cleanup
    if (g_precomputed_table) {
        price_table_destroy(g_precomputed_table);
    }

    benchmark::Shutdown();
    return 0;
}
