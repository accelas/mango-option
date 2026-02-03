// SPDX-License-Identifier: MIT
// OBSOLETE: This benchmark compared BSpline4D (removed) vs BSplineND template.
// BSpline4D was dead code - production now uses PriceTableSurface<4> directly.
// Kept for historical reference but no longer builds.

#include <benchmark/benchmark.h>
#include "mango/math/bspline_nd.hpp"
#include "mango/math/bspline_basis.hpp"
#include <vector>
#include <random>

namespace {

// Setup common test data
struct BSplineTestData {
    std::vector<double> m_grid;
    std::vector<double> tau_grid;
    std::vector<double> sigma_grid;
    std::vector<double> r_grid;

    std::vector<double> m_knots;
    std::vector<double> tau_knots;
    std::vector<double> sigma_knots;
    std::vector<double> r_knots;

    std::vector<double> coefficients;

    // Query points for evaluation
    std::vector<std::array<double, 4>> query_points;

    BSplineTestData(size_t n_m, size_t n_tau, size_t n_sigma, size_t n_r, size_t n_queries) {
        // Generate grids
        m_grid.resize(n_m);
        tau_grid.resize(n_tau);
        sigma_grid.resize(n_sigma);
        r_grid.resize(n_r);

        for (size_t i = 0; i < n_m; ++i) m_grid[i] = 0.7 + i * 0.6 / (n_m - 1);
        for (size_t i = 0; i < n_tau; ++i) tau_grid[i] = 0.027 + i * 1.973 / (n_tau - 1);
        for (size_t i = 0; i < n_sigma; ++i) sigma_grid[i] = 0.1 + i * 0.7 / (n_sigma - 1);
        for (size_t i = 0; i < n_r; ++i) r_grid[i] = 0.0 + i * 0.1 / (n_r - 1);

        // Generate knot vectors
        m_knots = mango::clamped_knots_cubic<double>(m_grid);
        tau_knots = mango::clamped_knots_cubic<double>(tau_grid);
        sigma_knots = mango::clamped_knots_cubic<double>(sigma_grid);
        r_knots = mango::clamped_knots_cubic<double>(r_grid);

        // Generate random coefficients
        size_t total_coeffs = n_m * n_tau * n_sigma * n_r;
        coefficients.resize(total_coeffs);
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(5.0, 50.0);
        for (auto& c : coefficients) c = dist(rng);

        // Generate random query points
        query_points.resize(n_queries);
        std::uniform_real_distribution<double> m_dist(0.75, 1.25);
        std::uniform_real_distribution<double> tau_dist(0.1, 1.5);
        std::uniform_real_distribution<double> sigma_dist(0.15, 0.6);
        std::uniform_real_distribution<double> r_dist(0.01, 0.08);

        for (auto& q : query_points) {
            q = {m_dist(rng), tau_dist(rng), sigma_dist(rng), r_dist(rng)};
        }
    }
};

// REMOVED: BSpline4D was dead code, removed in favor of PriceTableSurface<4>
// void BM_BSpline4D_Eval(benchmark::State& state) {
//     ...
// }

// Benchmark template BSplineND<double, 4>
void BM_BSplineND_Eval(benchmark::State& state) {
    size_t n_m = state.range(0);
    size_t n_tau = state.range(1);
    size_t n_sigma = state.range(2);
    size_t n_r = state.range(3);
    size_t n_queries = 1000;

    BSplineTestData data(n_m, n_tau, n_sigma, n_r, n_queries);

    auto spline = mango::BSplineND<double, 4>::create(
        {data.m_grid, data.tau_grid, data.sigma_grid, data.r_grid},
        {data.m_knots, data.tau_knots, data.sigma_knots, data.r_knots},
        data.coefficients);

    if (!spline.has_value()) {
        state.SkipWithError("Failed to create BSplineND");
        return;
    }

    double sum = 0.0;
    for (auto _ : state) {
        for (const auto& q : data.query_points) {
            double val = spline->eval(q);
            benchmark::DoNotOptimize(sum += val);
        }
    }

    state.SetItemsProcessed(state.iterations() * n_queries);
}

// Small grid (typical for coarse price tables)
BENCHMARK(BM_BSplineND_Eval)->Args({10, 8, 6, 4})->Unit(benchmark::kMicrosecond);

// Medium grid (typical for production)
BENCHMARK(BM_BSplineND_Eval)->Args({20, 15, 10, 8})->Unit(benchmark::kMicrosecond);

// Large grid (high accuracy requirements)
BENCHMARK(BM_BSplineND_Eval)->Args({50, 30, 20, 10})->Unit(benchmark::kMicrosecond);

}  // namespace

BENCHMARK_MAIN();
