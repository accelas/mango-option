// SPDX-License-Identifier: MIT
#include <benchmark/benchmark.h>
#include "mango/math/cubic_spline_solver.hpp"
#include "mango/math/cubic_spline_nd.hpp"
#include <vector>
#include <random>
#include <cmath>

namespace {

// Test function for 2D interpolation: f(x,y) = sin(x) * cos(y)
double test_function_2d(double x, double y) {
    return std::sin(x) * std::cos(y);
}

// Setup common test data for 2D cubic splines
struct CubicSpline2DTestData {
    std::vector<double> x_grid;
    std::vector<double> y_grid;
    std::vector<double> z_values;  // Row-major: z[i*ny + j] = f(x[i], y[j])
    std::vector<std::array<double, 2>> query_points;

    CubicSpline2DTestData(size_t nx, size_t ny, size_t n_queries) {
        // Generate grids
        x_grid.resize(nx);
        y_grid.resize(ny);

        for (size_t i = 0; i < nx; ++i) {
            x_grid[i] = 0.0 + i * 3.14159 / (nx - 1);
        }
        for (size_t j = 0; j < ny; ++j) {
            y_grid[j] = 0.0 + j * 3.14159 / (ny - 1);
        }

        // Generate z-values from test function
        z_values.resize(nx * ny);
        for (size_t i = 0; i < nx; ++i) {
            for (size_t j = 0; j < ny; ++j) {
                z_values[i * ny + j] = test_function_2d(x_grid[i], y_grid[j]);
            }
        }

        // Generate random query points
        query_points.resize(n_queries);
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> x_dist(0.5, 2.5);
        std::uniform_real_distribution<double> y_dist(0.5, 2.5);

        for (auto& q : query_points) {
            q = {x_dist(rng), y_dist(rng)};
        }
    }
};

// Benchmark hardcoded CubicSpline2D
void BM_CubicSpline2D_Build(benchmark::State& state) {
    size_t nx = state.range(0);
    size_t ny = state.range(1);

    CubicSpline2DTestData data(nx, ny, 100);

    for (auto _ : state) {
        mango::CubicSpline2D<double> spline;
        auto error = spline.build(data.x_grid, data.y_grid, data.z_values);
        benchmark::DoNotOptimize(spline);
        benchmark::DoNotOptimize(error);
    }
}

void BM_CubicSpline2D_Eval(benchmark::State& state) {
    size_t nx = state.range(0);
    size_t ny = state.range(1);
    size_t n_queries = 1000;

    CubicSpline2DTestData data(nx, ny, n_queries);

    mango::CubicSpline2D<double> spline;
    auto error = spline.build(data.x_grid, data.y_grid, data.z_values);
    if (error.has_value()) {
        state.SkipWithError("Failed to build spline");
        return;
    }

    double sum = 0.0;
    for (auto _ : state) {
        for (const auto& q : data.query_points) {
            double val = spline.eval(q[0], q[1]);
            benchmark::DoNotOptimize(sum += val);
        }
    }

    state.SetItemsProcessed(state.iterations() * n_queries);
}

// Benchmark template CubicSplineND<double, 2>
void BM_CubicSplineND_Build(benchmark::State& state) {
    size_t nx = state.range(0);
    size_t ny = state.range(1);

    CubicSpline2DTestData data(nx, ny, 100);

    for (auto _ : state) {
        auto result = mango::CubicSplineND<double, 2>::create(
            {data.x_grid, data.y_grid},
            data.z_values);
        benchmark::DoNotOptimize(result);
    }
}

void BM_CubicSplineND_Eval(benchmark::State& state) {
    size_t nx = state.range(0);
    size_t ny = state.range(1);
    size_t n_queries = 1000;

    CubicSpline2DTestData data(nx, ny, n_queries);

    auto spline = mango::CubicSplineND<double, 2>::create(
        {data.x_grid, data.y_grid},
        data.z_values).value();

    double sum = 0.0;
    for (auto _ : state) {
        for (const auto& q : data.query_points) {
            double val = spline.eval(q);
            benchmark::DoNotOptimize(sum += val);
        }
    }

    state.SetItemsProcessed(state.iterations() * n_queries);
}

// Small grid (coarse interpolation)
BENCHMARK(BM_CubicSpline2D_Build)->Args({10, 10})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_CubicSplineND_Build)->Args({10, 10})->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_CubicSpline2D_Eval)->Args({10, 10})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_CubicSplineND_Eval)->Args({10, 10})->Unit(benchmark::kMicrosecond);

// Medium grid (typical use case)
BENCHMARK(BM_CubicSpline2D_Build)->Args({30, 30})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_CubicSplineND_Build)->Args({30, 30})->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_CubicSpline2D_Eval)->Args({30, 30})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_CubicSplineND_Eval)->Args({30, 30})->Unit(benchmark::kMicrosecond);

// Large grid (high accuracy)
BENCHMARK(BM_CubicSpline2D_Build)->Args({100, 100})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_CubicSplineND_Build)->Args({100, 100})->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_CubicSpline2D_Eval)->Args({100, 100})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_CubicSplineND_Eval)->Args({100, 100})->Unit(benchmark::kMicrosecond);

// Rectangular grids (common in finance: strikes × maturities)
BENCHMARK(BM_CubicSpline2D_Build)->Args({50, 20})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_CubicSplineND_Build)->Args({50, 20})->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_CubicSpline2D_Eval)->Args({50, 20})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_CubicSplineND_Eval)->Args({50, 20})->Unit(benchmark::kMicrosecond);

}  // namespace

BENCHMARK_MAIN();
