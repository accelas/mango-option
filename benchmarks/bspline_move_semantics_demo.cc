/**
 * @file bspline_move_semantics_demo.cc
 * @brief Demonstrate move semantics benefit for BSplineNDSeparable
 *
 * Shows that move semantics eliminates the array copy overhead
 * when the caller doesn't need the input values after fitting.
 */

#include "src/math/bspline_nd_separable.hpp"
#include <benchmark/benchmark.h>
#include <vector>
#include <random>

using namespace mango;

// Helper to create large test values
std::vector<double> create_large_values(size_t n) {
    std::vector<double> values(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (auto& v : values) {
        v = dist(rng);
    }
    return values;
}

//=============================================================================
// Benchmark: Copy semantics (lvalue) vs Move semantics (rvalue)
//=============================================================================

static void BM_Fit_CopySemantics_LargeGrid(benchmark::State& state) {
    // Large grid: 50×30×20×10 = 300,000 points
    std::array<std::vector<double>, 4> grids;
    std::array<size_t, 4> sizes = {50, 30, 20, 10};

    for (size_t i = 0; i < 4; ++i) {
        grids[i].resize(sizes[i]);
        for (size_t j = 0; j < sizes[i]; ++j) {
            grids[i][j] = static_cast<double>(j) / (sizes[i] - 1);
        }
    }

    auto fitter = BSplineNDSeparable<double, 4>::create(grids).value();

    for (auto _ : state) {
        // Create values each iteration (simulates user needing original values)
        auto values = create_large_values(300000);

        // Pass as lvalue - forces copy inside fit()
        auto result = fitter.fit(values, BSplineNDSeparableConfig<double>{.tolerance = 1e-6});

        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(values);  // User still has access to original
    }

    state.SetLabel("lvalue (copies array)");
}

static void BM_Fit_MoveSemantics_LargeGrid(benchmark::State& state) {
    // Large grid: 50×30×20×10 = 300,000 points
    std::array<std::vector<double>, 4> grids;
    std::array<size_t, 4> sizes = {50, 30, 20, 10};

    for (size_t i = 0; i < 4; ++i) {
        grids[i].resize(sizes[i]);
        for (size_t j = 0; j < sizes[i]; ++j) {
            grids[i][j] = static_cast<double>(j) / (sizes[i] - 1);
        }
    }

    auto fitter = BSplineNDSeparable<double, 4>::create(grids).value();

    for (auto _ : state) {
        // Create values each iteration
        auto values = create_large_values(300000);

        // Pass as rvalue - zero-copy move into fit()
        auto result = fitter.fit(std::move(values), BSplineNDSeparableConfig<double>{.tolerance = 1e-6});

        benchmark::DoNotOptimize(result);
        // Note: values is now empty (moved from)
    }

    state.SetLabel("rvalue (zero-copy move)");
}

//=============================================================================
// Register benchmarks
//=============================================================================

BENCHMARK(BM_Fit_CopySemantics_LargeGrid)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Fit_MoveSemantics_LargeGrid)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
