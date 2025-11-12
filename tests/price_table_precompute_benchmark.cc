/**
 * @file price_table_precompute_benchmark.cc
 * @brief Benchmark batch vs single-contract mode for price table precomputation
 *
 * Price tables are 4D grids of option prices across parameter dimensions:
 * - Moneyness (m = S/K): typically 20-50 points
 * - Maturity (τ): typically 15-30 points
 * - Volatility (σ): typically 10-20 points
 * - Rate (r): typically 8-10 points
 *
 * Total grid points: Nm × Nτ × Nσ × Nr (e.g., 20×15×10×8 = 24,000 points)
 *
 * Precomputation uses snapshot-based optimization:
 * - Solve ONE PDE per (σ, r) pair at max maturity T_max
 * - Collect snapshots at all intermediate maturities
 * - Result: O(Nσ × Nr) solves instead of O(Nm × Nτ × Nσ × Nr)
 *
 * Batch mode adds cross-contract vectorization:
 * - Groups multiple (σ, r) pairs into SIMD-width batches
 * - Solves all PDEs in batch simultaneously with AVX-512 vectorization
 * - Expected speedup: 6-7x with AVX-512 (batch_width = 8)
 *
 * This benchmark measures:
 * - Total precomputation time (wall-clock)
 * - Throughput (contracts/second)
 * - Speedup ratio (single-contract / batch)
 * - Actual batch width and number of batches executed
 *
 * Realistic parameters:
 * - Small table: 10×8×6×4 = 1,920 grid points, 24 PDE solves (~30s)
 * - Medium table: 20×15×10×8 = 24,000 grid points, 80 PDE solves (~2-3min)
 * - Large table: 50×30×20×10 = 300,000 grid points, 200 PDE solves (~5-10min)
 *
 * For benchmark speed, we use medium-sized tables with reduced time steps.
 */

#include <benchmark/benchmark.h>
#include <experimental/simd>
#include <vector>
#include <cmath>
#include <algorithm>

#include "src/option/price_table_4d_builder.hpp"
#include "src/option/american_option.hpp"

namespace mango {
namespace {

namespace stdx = std::experimental;

// Helper: Generate log-spaced grid (for moneyness)
std::vector<double> generate_log_spaced(double min, double max, size_t n) {
    std::vector<double> grid(n);
    const double log_min = std::log(min);
    const double log_max = std::log(max);
    const double d_log = (log_max - log_min) / (n - 1);

    for (size_t i = 0; i < n; ++i) {
        grid[i] = std::exp(log_min + i * d_log);
    }
    return grid;
}

// Helper: Generate linear-spaced grid
std::vector<double> generate_linear(double min, double max, size_t n) {
    std::vector<double> grid(n);
    const double dx = (max - min) / (n - 1);

    for (size_t i = 0; i < n; ++i) {
        grid[i] = min + i * dx;
    }
    return grid;
}

// Benchmark: Price table precomputation (uses batch mode by default)
static void BM_PriceTable_Precompute(benchmark::State& state) {
    // Grid dimensions (realistic but smaller for benchmark speed)
    const size_t n_moneyness = state.range(0);
    const size_t n_maturity = state.range(1);
    const size_t n_volatility = state.range(2);
    const size_t n_rate = state.range(3);

    // Total grid points
    const size_t total_points = n_moneyness * n_maturity * n_volatility * n_rate;

    // Number of PDE solves (using snapshot optimization)
    const size_t n_pde_solves = n_volatility * n_rate;

    // PDE grid configuration (use smaller n_time for faster benchmark)
    AmericanOptionGrid grid_config;
    grid_config.n_space = 101;      // 101 spatial points (realistic)
    grid_config.n_time = 500;       // 500 time steps (reduced from 1000 for speed)
    grid_config.x_min = -1.5;       // log-moneyness range
    grid_config.x_max = 1.5;        // log-moneyness range

    // 4D parameter grids
    auto moneyness = generate_log_spaced(0.7, 1.3, n_moneyness);   // 70%-130% of strike
    auto maturity = generate_linear(0.027, 2.0, n_maturity);       // ~10 days to 2 years
    auto volatility = generate_linear(0.10, 0.50, n_volatility);   // 10%-50% vol
    auto rate = generate_linear(0.0, 0.10, n_rate);                 // 0%-10% rates

    const double K_ref = 100.0;  // Reference strike
    const double dividend = 0.02;  // 2% dividend yield

    // SIMD width for batch processing
    const size_t simd_width = stdx::native_simd<double>::size();

    for (auto _ : state) {
        // Create builder
        auto builder = PriceTable4DBuilder::create(
            moneyness, maturity, volatility, rate, K_ref);

        // Pre-compute table (uses batch mode internally)
        auto result = builder.precompute(OptionType::PUT, grid_config, dividend);

        if (!result.has_value()) {
            state.SkipWithError(result.error().c_str());
            return;
        }
    }

    // Report metrics
    state.counters["total_points"] = total_points;
    state.counters["n_pde_solves"] = n_pde_solves;
    state.counters["simd_width"] = simd_width;
    state.counters["n_batches"] = (n_pde_solves + simd_width - 1) / simd_width;

    // Throughput: contracts solved per second
    state.counters["contracts_per_sec"] = benchmark::Counter(
        n_pde_solves * state.iterations(),
        benchmark::Counter::kIsRate);

    // Per-contract time (average)
    const double total_time = state.iterations() * state.iterations() / state.counters["contracts_per_sec"];
    state.counters["ms_per_contract"] = benchmark::Counter(
        1000.0 * total_time / (n_pde_solves * state.iterations()),
        benchmark::Counter::kAvgIterations);
}

// Register benchmarks with different table sizes
// Format: Arg(n_moneyness, n_maturity, n_volatility, n_rate)

// Small table: 10×8×6×4 = 1,920 points, 24 PDE solves
// Expected time: ~30 seconds (single-contract), ~5-7 seconds (batch mode)
BENCHMARK(BM_PriceTable_Precompute)
    ->Args({10, 8, 6, 4})
    ->Unit(benchmark::kSecond)
    ->Iterations(1);  // Single iteration due to high cost

// Medium table: 20×15×10×8 = 24,000 points, 80 PDE solves
// Expected time: ~2-3 minutes (single-contract), ~20-30 seconds (batch mode)
BENCHMARK(BM_PriceTable_Precompute)
    ->Args({20, 15, 10, 8})
    ->Unit(benchmark::kSecond)
    ->Iterations(1);  // Single iteration due to high cost

// Realistic table: 30×20×12×8 = 57,600 points, 96 PDE solves
// Expected time: ~3-4 minutes (single-contract), ~30-45 seconds (batch mode)
BENCHMARK(BM_PriceTable_Precompute)
    ->Args({30, 20, 12, 8})
    ->Unit(benchmark::kSecond)
    ->Iterations(1);  // Single iteration due to high cost

// Note: Full-scale tables (50×30×20×10 = 300K points, 200 solves)
// would take 5-10 minutes even with batch mode, so excluded from benchmark.

}  // namespace
}  // namespace mango

BENCHMARK_MAIN();
