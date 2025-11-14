#include <benchmark/benchmark.h>
#include "src/interpolation/bspline_fitter_4d.hpp"
#include <vector>
#include <cmath>

namespace mango {

// Helper function to generate test data
static std::vector<double> generate_test_grid(size_t n) {
    std::vector<double> grid(n);
    for (size_t i = 0; i < n; ++i) {
        grid[i] = static_cast<double>(i) / (n - 1);
    }
    return grid;
}

static std::vector<double> generate_test_values(const std::vector<double>& grid) {
    std::vector<double> values(grid.size());
    for (size_t i = 0; i < grid.size(); ++i) {
        // Use smooth test function: f(x) = sin(2πx) + x²
        double x = grid[i];
        values[i] = std::sin(2.0 * M_PI * x) + x * x;
    }
    return values;
}

// Benchmark dense solver
static void BM_DenseSolver(benchmark::State& state) {
    const size_t n = state.range(0);

    // Setup test case
    auto grid = generate_test_grid(n);
    auto values = generate_test_values(grid);

    // Create collocation solver
    auto result = BSplineCollocation1D::create(grid);
    if (!result) {
        state.SkipWithError("Failed to create BSplineCollocation1D");
        return;
    }
    auto& solver = result.value();

    // Force dense solver mode
    solver.set_use_banded_solver(false);

    // Use relaxed tolerance for larger grids (numerical errors accumulate)
    double tolerance = 1e-6;

    for (auto _ : state) {
        auto fit_result = solver.fit(values, tolerance);
        benchmark::DoNotOptimize(fit_result);
        if (!fit_result.success) {
            state.SkipWithError("Fit failed");
            return;
        }
    }

    state.SetComplexityN(n);
}

// Benchmark banded solver
static void BM_BandedSolver(benchmark::State& state) {
    const size_t n = state.range(0);

    // Setup test case (same as dense)
    auto grid = generate_test_grid(n);
    auto values = generate_test_values(grid);

    // Create collocation solver
    auto result = BSplineCollocation1D::create(grid);
    if (!result) {
        state.SkipWithError("Failed to create BSplineCollocation1D");
        return;
    }
    auto& solver = result.value();

    // Force banded solver mode (default, but explicit for clarity)
    solver.set_use_banded_solver(true);

    // Use relaxed tolerance for larger grids (numerical errors accumulate)
    double tolerance = 1e-6;

    for (auto _ : state) {
        auto fit_result = solver.fit(values, tolerance);
        benchmark::DoNotOptimize(fit_result);
        if (!fit_result.success) {
            state.SkipWithError("Fit failed");
            return;
        }
    }

    state.SetComplexityN(n);
}

// Benchmark for n = 50, 100, 200 (typical axis sizes)
// These are realistic sizes from 4D price table construction
// Note: n=500 exhibits numerical stability issues in current banded LU implementation
// (needs pivoting or iterative refinement for very large grids)
BENCHMARK(BM_DenseSolver)
    ->Arg(50)
    ->Arg(100)
    ->Arg(200)
    ->Unit(benchmark::kMicrosecond)
    ->Complexity();

BENCHMARK(BM_BandedSolver)
    ->Arg(50)
    ->Arg(100)
    ->Arg(200)
    ->Unit(benchmark::kMicrosecond)
    ->Complexity();

} // namespace mango

BENCHMARK_MAIN();
