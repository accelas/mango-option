// SPDX-License-Identifier: MIT
/**
 * @file openmp_simd_benchmark.cc
 * @brief Benchmark OpenMP SIMD performance with target_clones
 *
 * Tests OpenMP SIMD auto-vectorization on:
 * - Uniform vs non-uniform grids
 * - First vs second derivatives
 * - Different grid sizes (101, 501, 1001)
 *
 * Compiler generates ISA-specific versions via [[gnu::target_clones]]:
 * - SSE2 baseline (2-wide)
 * - AVX2 (4-wide)
 * - AVX-512 (8-wide)
 */

#include <benchmark/benchmark.h>
#include "src/pde/operators/centered_difference_scalar.hpp"
#include "src/pde/core/grid.hpp"
#include <vector>
#include <cmath>

using namespace mango;
using namespace mango::operators;

// Grid sizes to test
constexpr size_t SMALL_GRID = 101;
constexpr size_t MEDIUM_GRID = 501;
constexpr size_t LARGE_GRID = 1001;

// Setup uniform grid
template<typename T>
auto setup_uniform_grid(size_t n) {
    std::vector<T> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<T>(i) / static_cast<T>(n - 1);
    }

    GridView<T> grid_view(x);
    GridSpacing<T> spacing(grid_view);

    std::vector<T> u(n);
    std::vector<T> result(n);
    for (size_t i = 0; i < n; ++i) {
        u[i] = std::sin(2.0 * M_PI * x[i]);
    }

    return std::make_tuple(spacing, u, result);
}

// Setup non-uniform grid (sinh-spaced)
template<typename T>
auto setup_nonuniform_grid(size_t n) {
    std::vector<T> x(n);
    const T stretch = 2.0;
    for (size_t i = 0; i < n; ++i) {
        T xi = static_cast<T>(i) / static_cast<T>(n - 1);
        x[i] = std::sinh(stretch * (2.0 * xi - 1.0)) / std::sinh(stretch);
    }

    GridView<T> grid_view(x);
    GridSpacing<T> spacing(grid_view);

    std::vector<T> u(n);
    std::vector<T> result(n);
    for (size_t i = 0; i < n; ++i) {
        u[i] = std::sin(2.0 * M_PI * x[i]);
    }

    return std::make_tuple(spacing, u, result);
}

// ============================================================================
// ScalarBackend Benchmarks (OpenMP SIMD)
// ============================================================================

static void BM_Scalar_UniformGrid_2ndDeriv(benchmark::State& state) {
    const size_t n = state.range(0);
    auto [spacing, u, result] = setup_uniform_grid<double>(n);
    ScalarBackend<double> backend(spacing);

    for (auto _ : state) {
        backend.compute_second_derivative_uniform(u, result, 1, n - 1);
        benchmark::DoNotOptimize(result.data());
    }

    state.SetItemsProcessed(state.iterations() * (n - 2));
    state.counters["grid_size"] = n;
}

static void BM_Scalar_NonUniformGrid_2ndDeriv(benchmark::State& state) {
    const size_t n = state.range(0);
    auto [spacing, u, result] = setup_nonuniform_grid<double>(n);
    ScalarBackend<double> backend(spacing);

    for (auto _ : state) {
        backend.compute_second_derivative_non_uniform(u, result, 1, n - 1);
        benchmark::DoNotOptimize(result.data());
    }

    state.SetItemsProcessed(state.iterations() * (n - 2));
    state.counters["grid_size"] = n;
}

static void BM_Scalar_UniformGrid_1stDeriv(benchmark::State& state) {
    const size_t n = state.range(0);
    auto [spacing, u, result] = setup_uniform_grid<double>(n);
    ScalarBackend<double> backend(spacing);

    for (auto _ : state) {
        backend.compute_first_derivative_uniform(u, result, 1, n - 1);
        benchmark::DoNotOptimize(result.data());
    }

    state.SetItemsProcessed(state.iterations() * (n - 2));
    state.counters["grid_size"] = n;
}

static void BM_Scalar_NonUniformGrid_1stDeriv(benchmark::State& state) {
    const size_t n = state.range(0);
    auto [spacing, u, result] = setup_nonuniform_grid<double>(n);
    ScalarBackend<double> backend(spacing);

    for (auto _ : state) {
        backend.compute_first_derivative_non_uniform(u, result, 1, n - 1);
        benchmark::DoNotOptimize(result.data());
    }

    state.SetItemsProcessed(state.iterations() * (n - 2));
    state.counters["grid_size"] = n;
}

// ============================================================================
// Register Benchmarks
// ============================================================================

BENCHMARK(BM_Scalar_UniformGrid_2ndDeriv)->Arg(SMALL_GRID)->Arg(MEDIUM_GRID)->Arg(LARGE_GRID);
BENCHMARK(BM_Scalar_NonUniformGrid_2ndDeriv)->Arg(SMALL_GRID)->Arg(MEDIUM_GRID)->Arg(LARGE_GRID);
BENCHMARK(BM_Scalar_UniformGrid_1stDeriv)->Arg(SMALL_GRID)->Arg(MEDIUM_GRID)->Arg(LARGE_GRID);
BENCHMARK(BM_Scalar_NonUniformGrid_1stDeriv)->Arg(SMALL_GRID)->Arg(MEDIUM_GRID)->Arg(LARGE_GRID);

BENCHMARK_MAIN();
