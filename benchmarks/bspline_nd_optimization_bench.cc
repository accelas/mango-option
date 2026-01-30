// SPDX-License-Identifier: MIT
/**
 * @file bspline_nd_optimization_bench.cc
 * @brief Benchmark for BSplineNDSeparable optimizations
 *
 * Measures performance impact of:
 * 1. Move semantics to avoid full array copy
 * 2. OpenMP SIMD for NaN/Inf validation
 * 3. OpenMP SIMD for slice extraction/write-back
 */

#include "src/math/bspline_nd_separable.hpp"
#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <cmath>

using namespace mango;

// Helper: Create test grids
template<size_t N>
std::array<std::vector<double>, N> create_grids(size_t points_per_axis) {
    std::array<std::vector<double>, N> grids;
    for (size_t i = 0; i < N; ++i) {
        grids[i].resize(points_per_axis);
        for (size_t j = 0; j < points_per_axis; ++j) {
            grids[i][j] = static_cast<double>(j) / (points_per_axis - 1);
        }
    }
    return grids;
}

// Helper: Create test values
std::vector<double> create_values(size_t n) {
    std::vector<double> values(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (auto& v : values) {
        v = dist(rng);
    }
    return values;
}

//=============================================================================
// Benchmark 1: Full fit() operation (measures array copy overhead)
//=============================================================================

static void BM_Fit4D_SmallGrid(benchmark::State& state) {
    // Small grid: 7×4×4×4 = 448 points
    auto grids = create_grids<4>(state.range(0));
    auto fitter = BSplineNDSeparable<double, 4>::create(grids).value();

    size_t total_points = 1;
    for (const auto& g : grids) total_points *= g.size();

    auto values = create_values(total_points);

    for (auto _ : state) {
        // Measures copy + validation + fitting
        auto result = fitter.fit(values, BSplineNDSeparableConfig<double>{.tolerance = 1e-6});
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * total_points);
    state.SetBytesProcessed(state.iterations() * total_points * sizeof(double));
}

static void BM_Fit4D_MediumGrid(benchmark::State& state) {
    // Medium grid: 20×15×10×8 = 24,000 points
    std::array<std::vector<double>, 4> grids;
    std::array<size_t, 4> sizes = {20, 15, 10, 8};

    for (size_t i = 0; i < 4; ++i) {
        grids[i].resize(sizes[i]);
        for (size_t j = 0; j < sizes[i]; ++j) {
            grids[i][j] = static_cast<double>(j) / (sizes[i] - 1);
        }
    }

    auto fitter = BSplineNDSeparable<double, 4>::create(grids).value();
    size_t total_points = 24000;
    auto values = create_values(total_points);

    for (auto _ : state) {
        auto result = fitter.fit(values, BSplineNDSeparableConfig<double>{.tolerance = 1e-6});
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * total_points);
    state.SetBytesProcessed(state.iterations() * total_points * sizeof(double));
}

static void BM_Fit4D_LargeGrid(benchmark::State& state) {
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
    size_t total_points = 300000;
    auto values = create_values(total_points);

    for (auto _ : state) {
        auto result = fitter.fit(values, BSplineNDSeparableConfig<double>{.tolerance = 1e-6});
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * total_points);
    state.SetBytesProcessed(state.iterations() * total_points * sizeof(double));
}

//=============================================================================
// Benchmark 2: Array copy overhead (isolated)
//=============================================================================

static void BM_ArrayCopy_SmallGrid(benchmark::State& state) {
    size_t n = 448;
    auto values = create_values(n);

    for (auto _ : state) {
        std::vector<double> copy = values;  // Current implementation
        benchmark::DoNotOptimize(copy);
    }

    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(double));
}

static void BM_ArrayCopy_MediumGrid(benchmark::State& state) {
    size_t n = 24000;
    auto values = create_values(n);

    for (auto _ : state) {
        std::vector<double> copy = values;
        benchmark::DoNotOptimize(copy);
    }

    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(double));
}

static void BM_ArrayCopy_LargeGrid(benchmark::State& state) {
    size_t n = 300000;
    auto values = create_values(n);

    for (auto _ : state) {
        std::vector<double> copy = values;
        benchmark::DoNotOptimize(copy);
    }

    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(double));
}

//=============================================================================
// Benchmark 3: NaN/Inf validation (isolated)
//=============================================================================

static void BM_Validation_Sequential(benchmark::State& state) {
    size_t n = state.range(0);
    auto values = create_values(n);

    for (auto _ : state) {
        bool valid = true;
        for (size_t i = 0; i < n; ++i) {
            if (std::isnan(values[i]) || std::isinf(values[i])) {
                valid = false;
                break;
            }
        }
        benchmark::DoNotOptimize(valid);
    }

    state.SetItemsProcessed(state.iterations() * n);
}

//=============================================================================
// Benchmark 4: Slice extraction (strided memory access)
//=============================================================================

static void BM_SliceExtraction_UnitStride(benchmark::State& state) {
    size_t n = state.range(0);
    std::vector<double> coeffs = create_values(n);
    std::vector<double> slice_buffer(100);

    for (auto _ : state) {
        // Extract unit-stride slice (best case)
        for (size_t i = 0; i < 100; ++i) {
            slice_buffer[i] = coeffs[i];
        }
        benchmark::DoNotOptimize(slice_buffer);
    }

    state.SetItemsProcessed(state.iterations() * 100);
}

static void BM_SliceExtraction_LargeStride(benchmark::State& state) {
    size_t n = state.range(0);
    size_t stride = 300;  // Realistic stride for 50×30×20×10 grid
    std::vector<double> coeffs = create_values(n);
    std::vector<double> slice_buffer(50);

    for (auto _ : state) {
        // Extract strided slice (worst case)
        for (size_t i = 0; i < 50; ++i) {
            slice_buffer[i] = coeffs[i * stride];
        }
        benchmark::DoNotOptimize(slice_buffer);
    }

    state.SetItemsProcessed(state.iterations() * 50);
}

//=============================================================================
// Register benchmarks
//=============================================================================

// Full fit operations
BENCHMARK(BM_Fit4D_SmallGrid)->Arg(7)->Arg(4)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Fit4D_MediumGrid)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Fit4D_LargeGrid)->Unit(benchmark::kMillisecond);

// Array copy overhead
BENCHMARK(BM_ArrayCopy_SmallGrid)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_ArrayCopy_MediumGrid)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ArrayCopy_LargeGrid)->Unit(benchmark::kMillisecond);

// Validation
BENCHMARK(BM_Validation_Sequential)
    ->Arg(448)
    ->Arg(24000)
    ->Arg(300000)
    ->Unit(benchmark::kNanosecond);

// Slice extraction
BENCHMARK(BM_SliceExtraction_UnitStride)
    ->Arg(300000)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_SliceExtraction_LargeStride)
    ->Arg(300000)
    ->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
