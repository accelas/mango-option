// SPDX-License-Identifier: MIT
/**
 * @file iv_solver_probe_benchmark.cc
 * @brief Benchmark comparing heuristic vs probe-based IVSolver
 *
 * Measures:
 * - Heuristic path (target_price_error = 0): Uses config_.grid for each Brent iteration
 * - Probe-based path (target_price_error = 0.01): Calibrates once at σ_high, reuses grid
 *
 * Expected behavior:
 * - Probe adds ~50-100ms calibration overhead (4-6 PDE solves)
 * - For multi-iteration IV solves, probe is ~20% faster due to grid reuse
 * - For single-iteration edge cases, heuristic may be faster (no calibration overhead)
 */

#include <benchmark/benchmark.h>
#include "src/option/iv_solver.hpp"

namespace {

// Create a standard test query
mango::IVQuery make_test_query() {
    mango::OptionSpec spec{
        .spot = 100.0,
        .strike = 100.0,
        .maturity = 1.0,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .option_type = mango::OptionType::PUT
    };
    return mango::IVQuery(spec, 5.50);  // ~17% IV
}

// Heuristic path: target_price_error = 0
static void BM_IVSolver_Heuristic(benchmark::State& state) {
    auto query = make_test_query();
    mango::IVSolverConfig config{.target_price_error = 0.0};  // Disable probe
    mango::IVSolver solver(config);

    for (auto _ : state) {
        auto result = solver.solve(query);
        benchmark::DoNotOptimize(result);
        if (!result.has_value()) {
            state.SkipWithError("Heuristic solve failed");
            break;
        }
    }
}
BENCHMARK(BM_IVSolver_Heuristic)->Unit(benchmark::kMillisecond);

// Probe-based path: target_price_error = 0.01 (default)
static void BM_IVSolver_ProbeBased(benchmark::State& state) {
    auto query = make_test_query();
    mango::IVSolverConfig config{.target_price_error = 0.01};  // Enable probe
    mango::IVSolver solver(config);

    for (auto _ : state) {
        auto result = solver.solve(query);
        benchmark::DoNotOptimize(result);
        if (!result.has_value()) {
            state.SkipWithError("Probe-based solve failed");
            break;
        }
    }
}
BENCHMARK(BM_IVSolver_ProbeBased)->Unit(benchmark::kMillisecond);

// Probe-based path with tighter tolerance
static void BM_IVSolver_ProbeBased_Tight(benchmark::State& state) {
    auto query = make_test_query();
    mango::IVSolverConfig config{.target_price_error = 0.001};  // Tighter tolerance
    mango::IVSolver solver(config);

    for (auto _ : state) {
        auto result = solver.solve(query);
        benchmark::DoNotOptimize(result);
        if (!result.has_value()) {
            state.SkipWithError("Probe-based tight solve failed");
            break;
        }
    }
}
BENCHMARK(BM_IVSolver_ProbeBased_Tight)->Unit(benchmark::kMillisecond);

// Test with different option types
static void BM_IVSolver_Probe_ITMPut(benchmark::State& state) {
    mango::OptionSpec spec{
        .spot = 80.0,  // ITM put
        .strike = 100.0,
        .maturity = 1.0,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .option_type = mango::OptionType::PUT
    };
    mango::IVQuery query(spec, 22.0);  // Higher price for ITM
    mango::IVSolverConfig config{.target_price_error = 0.01};
    mango::IVSolver solver(config);

    for (auto _ : state) {
        auto result = solver.solve(query);
        benchmark::DoNotOptimize(result);
        if (!result.has_value()) {
            state.SkipWithError("ITM put solve failed");
            break;
        }
    }
}
BENCHMARK(BM_IVSolver_Probe_ITMPut)->Unit(benchmark::kMillisecond);

// Test with short maturity (where Nt floor kicks in)
static void BM_IVSolver_Probe_ShortMaturity(benchmark::State& state) {
    mango::OptionSpec spec{
        .spot = 100.0,
        .strike = 100.0,
        .maturity = 0.05,  // ~18 days
        .rate = 0.05,
        .dividend_yield = 0.02,
        .option_type = mango::OptionType::PUT
    };
    mango::IVQuery query(spec, 1.50);  // Lower price for short maturity
    mango::IVSolverConfig config{.target_price_error = 0.01};
    mango::IVSolver solver(config);

    for (auto _ : state) {
        auto result = solver.solve(query);
        benchmark::DoNotOptimize(result);
        if (!result.has_value()) {
            state.SkipWithError("Short maturity solve failed");
            break;
        }
    }
}
BENCHMARK(BM_IVSolver_Probe_ShortMaturity)->Unit(benchmark::kMillisecond);

}  // namespace

BENCHMARK_MAIN();
