/**
 * @file bspline_fitter_hardcoded_vs_template.cc
 * @brief Benchmark: Hardcoded 4-axis fitting vs Template-based axis fitting
 *
 * Compares:
 * 1. Current: 4 separate fit_axis0/1/2/3 methods (~400 lines)
 * 2. Template: Single fit_axis<Axis> method (~100 lines)
 *
 * Expected: Identical performance (compiler should generate same code)
 */

#include <benchmark/benchmark.h>
#include "src/math/bspline_nd_separable.hpp"
#include "src/math/bspline_nd_separable.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <array>

namespace {

// Test function: f(m, tau, sigma, r) = m² + tau·sigma + r
double test_function(double m, double tau, double sigma, double r) {
    return m * m + tau * sigma + r;
}

// Generate test data
struct TestData {
    std::vector<double> m_grid;
    std::vector<double> tau_grid;
    std::vector<double> sigma_grid;
    std::vector<double> r_grid;
    std::vector<double> values;

    TestData(size_t nm, size_t nt, size_t nv, size_t nr) {
        // Create grids
        m_grid.resize(nm);
        tau_grid.resize(nt);
        sigma_grid.resize(nv);
        r_grid.resize(nr);

        for (size_t i = 0; i < nm; ++i) m_grid[i] = 0.8 + i * 0.4 / (nm - 1);
        for (size_t i = 0; i < nt; ++i) tau_grid[i] = 0.1 + i * 1.9 / (nt - 1);
        for (size_t i = 0; i < nv; ++i) sigma_grid[i] = 0.1 + i * 0.4 / (nv - 1);
        for (size_t i = 0; i < nr; ++i) r_grid[i] = 0.0 + i * 0.1 / (nr - 1);

        // Generate function values
        values.resize(nm * nt * nv * nr);
        for (size_t i = 0; i < nm; ++i) {
            for (size_t j = 0; j < nt; ++j) {
                for (size_t k = 0; k < nv; ++k) {
                    for (size_t l = 0; l < nr; ++l) {
                        size_t idx = ((i * nt + j) * nv + k) * nr + l;
                        values[idx] = test_function(m_grid[i], tau_grid[j],
                                                    sigma_grid[k], r_grid[l]);
                    }
                }
            }
        }
    }
};

} // anonymous namespace

// Benchmark current generic template implementation
static void BM_Fitter4D_Hardcoded_Small(benchmark::State& state) {
    TestData data(7, 4, 4, 4);

    for (auto _ : state) {
        auto fitter_result = mango::BSplineNDSeparable<double, 4>::create(
            std::array<std::vector<double>, 4>{
                data.m_grid, data.tau_grid, data.sigma_grid, data.r_grid});

        if (!fitter_result.has_value()) {
            state.SkipWithError("Failed to create fitter");
            return;
        }

        auto result = fitter_result.value().fit(data.values, mango::BSplineNDSeparableConfig<double>{.tolerance = 1e-6});

        if (!result.success) {
            state.SkipWithError("Fitting failed");
            return;
        }

        benchmark::DoNotOptimize(result.coefficients);
        benchmark::ClobberMemory();
    }

    state.SetLabel("7x4x4x4 grid (hardcoded)");
}

static void BM_Fitter4D_Hardcoded_Medium(benchmark::State& state) {
    TestData data(20, 15, 10, 8);

    for (auto _ : state) {
        auto fitter_result = mango::BSplineNDSeparable<double, 4>::create(
            std::array<std::vector<double>, 4>{
                data.m_grid, data.tau_grid, data.sigma_grid, data.r_grid});

        if (!fitter_result.has_value()) {
            state.SkipWithError("Failed to create fitter");
            return;
        }

        auto result = fitter_result.value().fit(data.values, mango::BSplineNDSeparableConfig<double>{.tolerance = 1e-6});

        if (!result.success) {
            state.SkipWithError("Fitting failed");
            return;
        }

        benchmark::DoNotOptimize(result.coefficients);
        benchmark::ClobberMemory();
    }

    state.SetLabel("20x15x10x8 grid (hardcoded)");
}

static void BM_Fitter4D_Hardcoded_Large(benchmark::State& state) {
    TestData data(50, 30, 20, 10);

    for (auto _ : state) {
        auto fitter_result = mango::BSplineNDSeparable<double, 4>::create(
            std::array<std::vector<double>, 4>{
                data.m_grid, data.tau_grid, data.sigma_grid, data.r_grid});

        if (!fitter_result.has_value()) {
            state.SkipWithError("Failed to create fitter");
            return;
        }

        auto result = fitter_result.value().fit(data.values, mango::BSplineNDSeparableConfig<double>{.tolerance = 1e-6});

        if (!result.success) {
            state.SkipWithError("Fitting failed");
            return;
        }

        benchmark::DoNotOptimize(result.coefficients);
        benchmark::ClobberMemory();
    }

    state.SetLabel("50x30x20x10 grid (hardcoded)");
}

// Benchmark template version
static void BM_Fitter4D_Template_Small(benchmark::State& state) {
    TestData data(7, 4, 4, 4);

    for (auto _ : state) {
        std::array<std::vector<double>, 4> grids = {
            data.m_grid, data.tau_grid, data.sigma_grid, data.r_grid
        };

        auto fitter_result = mango::BSplineNDSeparable<double, 4>::create(std::move(grids));

        if (!fitter_result.has_value()) {
            state.SkipWithError("Failed to create fitter");
            return;
        }

        auto coeffs = fitter_result.value().fit(data.values);

        benchmark::DoNotOptimize(coeffs);
        benchmark::ClobberMemory();
    }

    state.SetLabel("7x4x4x4 grid (template)");
}

static void BM_Fitter4D_Template_Medium(benchmark::State& state) {
    TestData data(20, 15, 10, 8);

    for (auto _ : state) {
        std::array<std::vector<double>, 4> grids = {
            data.m_grid, data.tau_grid, data.sigma_grid, data.r_grid
        };

        auto fitter_result = mango::BSplineNDSeparable<double, 4>::create(std::move(grids));

        if (!fitter_result.has_value()) {
            state.SkipWithError("Failed to create fitter");
            return;
        }

        auto coeffs = fitter_result.value().fit(data.values);

        benchmark::DoNotOptimize(coeffs);
        benchmark::ClobberMemory();
    }

    state.SetLabel("20x15x10x8 grid (template)");
}

static void BM_Fitter4D_Template_Large(benchmark::State& state) {
    TestData data(50, 30, 20, 10);

    for (auto _ : state) {
        std::array<std::vector<double>, 4> grids = {
            data.m_grid, data.tau_grid, data.sigma_grid, data.r_grid
        };

        auto fitter_result = mango::BSplineNDSeparable<double, 4>::create(std::move(grids));

        if (!fitter_result.has_value()) {
            state.SkipWithError("Failed to create fitter");
            return;
        }

        auto coeffs = fitter_result.value().fit(data.values);

        benchmark::DoNotOptimize(coeffs);
        benchmark::ClobberMemory();
    }

    state.SetLabel("50x30x20x10 grid (template)");
}

BENCHMARK(BM_Fitter4D_Hardcoded_Small);
BENCHMARK(BM_Fitter4D_Template_Small);
BENCHMARK(BM_Fitter4D_Hardcoded_Medium);
BENCHMARK(BM_Fitter4D_Template_Medium);
BENCHMARK(BM_Fitter4D_Hardcoded_Large);
BENCHMARK(BM_Fitter4D_Template_Large);

BENCHMARK_MAIN();
