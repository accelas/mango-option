#include <benchmark/benchmark.h>
#include <vector>
#include <cmath>
#include <random>

extern "C" {
#include "../src/iv_surface.h"
#include "../src/interp_multilinear.h"
#include "../src/interp_cubic.h"
}

// Test function for IV surface
static double test_iv_function(double m, double tau) {
    return 0.2 + 0.1 * (m - 1.0) + 0.05 * tau + 0.02 * (m - 1.0) * (m - 1.0);
}

// Helper to create and populate IV surface
static IVSurface* create_test_surface(size_t n_m, size_t n_tau,
                                       const InterpolationStrategy* strategy) {
    // Create grid
    std::vector<double> moneyness(n_m);
    std::vector<double> maturity(n_tau);

    for (size_t i = 0; i < n_m; i++) {
        moneyness[i] = 0.8 + i * (0.4 / (n_m - 1));
    }
    for (size_t j = 0; j < n_tau; j++) {
        maturity[j] = 0.1 + j * (1.9 / (n_tau - 1));
    }

    IVSurface* surface = iv_surface_create_with_strategy(
        moneyness.data(), n_m, maturity.data(), n_tau, strategy);

    if (!surface) return nullptr;

    // Populate with test data
    std::vector<double> iv_data(n_m * n_tau);
    for (size_t j = 0; j < n_tau; j++) {
        for (size_t i = 0; i < n_m; i++) {
            iv_data[j * n_m + i] = test_iv_function(moneyness[i], maturity[j]);
        }
    }

    iv_surface_set(surface, iv_data.data());

    return surface;
}

// Benchmark multilinear interpolation
static void BM_Multilinear(benchmark::State& state) {
    const size_t n_m = state.range(0);
    const size_t n_tau = state.range(1);

    IVSurface* surface = create_test_surface(n_m, n_tau, &INTERP_MULTILINEAR);
    if (!surface) {
        state.SkipWithError("Failed to create surface");
        return;
    }

    // Generate random query points
    std::mt19937 gen(42);
    std::uniform_real_distribution<> m_dist(0.8, 1.2);
    std::uniform_real_distribution<> tau_dist(0.1, 2.0);

    std::vector<double> query_m(1000);
    std::vector<double> query_tau(1000);
    for (size_t i = 0; i < 1000; i++) {
        query_m[i] = m_dist(gen);
        query_tau[i] = tau_dist(gen);
    }

    // Benchmark
    size_t idx = 0;
    for (auto _ : state) {
        double result = iv_surface_interpolate(surface, query_m[idx], query_tau[idx]);
        benchmark::DoNotOptimize(result);
        idx = (idx + 1) % 1000;
    }

    state.SetItemsProcessed(state.iterations());
    state.counters["grid_size"] = n_m * n_tau;

    iv_surface_destroy(surface);
}

// Benchmark cubic interpolation (with precompute)
static void BM_Cubic(benchmark::State& state) {
    const size_t n_m = state.range(0);
    const size_t n_tau = state.range(1);

    IVSurface* surface = create_test_surface(n_m, n_tau, &INTERP_CUBIC);
    if (!surface) {
        state.SkipWithError("Failed to create surface");
        return;
    }

    // Generate random query points
    std::mt19937 gen(42);
    std::uniform_real_distribution<> m_dist(0.8, 1.2);
    std::uniform_real_distribution<> tau_dist(0.1, 2.0);

    std::vector<double> query_m(1000);
    std::vector<double> query_tau(1000);
    for (size_t i = 0; i < 1000; i++) {
        query_m[i] = m_dist(gen);
        query_tau[i] = tau_dist(gen);
    }

    // Benchmark
    size_t idx = 0;
    for (auto _ : state) {
        double result = iv_surface_interpolate(surface, query_m[idx], query_tau[idx]);
        benchmark::DoNotOptimize(result);
        idx = (idx + 1) % 1000;
    }

    state.SetItemsProcessed(state.iterations());
    state.counters["grid_size"] = n_m * n_tau;

    iv_surface_destroy(surface);
}

// Register benchmarks with different grid sizes
BENCHMARK(BM_Multilinear)->Args({10, 10})->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Cubic)->Args({10, 10})->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Multilinear)->Args({20, 15})->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Cubic)->Args({20, 15})->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Multilinear)->Args({50, 30})->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Cubic)->Args({50, 30})->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Multilinear)->Args({100, 50})->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Cubic)->Args({100, 50})->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
