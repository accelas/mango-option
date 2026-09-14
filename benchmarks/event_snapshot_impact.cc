// SPDX-License-Identifier: MIT
// Build latency and fixed-grid capture cost for the event-sided table change.
// This harness also compiles against the pre-change API for paired comparisons.
#include <benchmark/benchmark.h>
#include "mango/option/american_option.hpp"
#include "mango/option/table/bspline/bspline_adaptive.hpp"
#include "mango/option/table/bspline/bspline_segmented_builder.hpp"
#include <cmath>
#include <span>
#include <vector>

using namespace mango;

namespace {

// Same ordinary-pricing cases as component_performance, without pulling its
// unrelated interpolation-factory/Arrow dependencies into this executable.
void BM_EventDirectSolve(benchmark::State& state) {
    const bool cash_call = state.range(0) == 1;
    PricingParams params(OptionSpec{.spot = 100, .strike = 100, .maturity = 1,
        .rate = 0.05, .dividend_yield = 0.02,
        .option_type = cash_call ? OptionType::CALL : OptionType::PUT}, 0.2);
    if (cash_call) params.discrete_dividends = {{0.25, 2}, {0.5, 2}, {0.75, 2}};
    double price = 0.0;
    for (auto _ : state) {
        auto solver = AmericanOptionSolver::create(params);
        if (!solver) { state.SkipWithError("solver creation failed"); break; }
        auto result = solver->solve();
        if (!result) { state.SkipWithError("PDE solve failed"); break; }
        price = result->value_at(params.spot);
        benchmark::DoNotOptimize(price);
    }
    state.counters["price"] = price;
}
BENCHMARK(BM_EventDirectSolve)->Arg(0)->Arg(1)->UseRealTime();

IVGrid domain() {
    return {.moneyness = {std::log(0.92), std::log(0.95), 0.0,
                         std::log(1.05), std::log(1.08)},
            .vol = {0.10, 0.15, 0.20, 0.30},
            .rate = {0.02, 0.03, 0.05, 0.07}};
}

std::vector<Dividend> dividends(int count) {
    std::vector<Dividend> result;
    for (int i = 1; i <= count; ++i) result.push_back({0.25 * i, 0.50});
    return result;
}

void BM_EventManualBuild(benchmark::State& state) {
    auto grid = domain();
    grid.moneyness.clear();
    for (size_t i = 0; i < 60; ++i) {
        grid.moneyness.push_back(std::lerp(std::log(0.92), std::log(1.08), i / 59.0));
    }
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0, .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.02,
                      .discrete_dividends = dividends(state.range(0))},
        .grid = grid, .maturity = 1.0,
    };
    size_t solves = 0, rows = 0, points = 0;
    double price = 0.0;
    for (auto _ : state) {
        auto result = SegmentedPriceTableBuilder::build_with_diagnostics(config);
        if (!result) { state.SkipWithError("manual build refused"); break; }
        solves = result->pde_solves;
        rows = result->sample_rows;
        points = result->sample_points;
        price = result->surface.price(100, 100, 0.6, 0.2, 0.05);
        benchmark::DoNotOptimize(price);
    }
    state.counters["pde_solves"] = solves;
    state.counters["sample_rows"] = rows;
    state.counters["sample_points"] = points;
    state.counters["price"] = price;
}
BENCHMARK(BM_EventManualBuild)->Arg(0)->Arg(1)->Arg(3)->UseRealTime();

void BM_EventAdaptiveBuild(benchmark::State& state) {
    const bool short_expiry = state.range(0) == 1;
    SegmentedAdaptiveConfig config{
        .spot = 100.0, .option_type = OptionType::PUT, .dividend_yield = 0.02,
        .discrete_dividends = short_expiry
            ? std::vector<Dividend>{{10.0 / 365.0, 0.5}} : dividends(3),
        .maturity = short_expiry ? 30.0 / 365.0 : 1.0,
        .kref_config = {.K_refs = {90, 92.5, 95, 97.5, 100, 102.5, 105, 107.5, 110}},
    };
    AdaptiveGridParams params{.target_iv_error = 1e-3};
    const auto grid = domain();
    size_t solves = 0, rows = 0, points = 0, iterations = 0;
    double error = 0.0;
    for (auto _ : state) {
        auto result = build_adaptive_bspline_segmented(params, config, grid);
        if (!result) { state.SkipWithError("adaptive build refused"); break; }
        solves = result->total_pde_solves;
        rows = result->diagnostics.sample_rows;
        points = result->diagnostics.sample_points;
        iterations = result->diagnostics.total_iterations;
        error = result->achieved_max_error;
        benchmark::DoNotOptimize(result->surface);
    }
    state.counters["pde_solves"] = solves;
    state.counters["sample_rows"] = rows;
    state.counters["sample_points"] = points;
    state.counters["refinement_iterations"] = iterations;
    state.counters["error_bps"] = error * 1e4;
}
BENCHMARK(BM_EventAdaptiveBuild)->Arg(0)->Arg(1)->UseRealTime();

template <typename Solver>
bool enable_capture(Solver& solver, std::span<const double> times) {
    if constexpr (requires { solver.set_before_event_snapshot_times(times); }) {
        solver.set_before_event_snapshot_times(times);
        return true;
    }
    return false;
}

void BM_EventFixedGridSolve(benchmark::State& state) {
    PricingParams params(OptionSpec{.spot = 100, .strike = 100, .maturity = 1,
        .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.2);
    params.discrete_dividends = dividends(3);
    std::vector<double> snapshots;
    for (int i = 0; i <= 20; ++i) snapshots.push_back(i / 20.0);
    const std::vector<double> events{0.25, 0.5, 0.75};
    PDEGridConfig grid{
        .grid_spec = GridSpec<double>::sinh_spaced(-2, 2, 201, 3.0).value(),
        .n_time = 1000, .mandatory_times = snapshots,
    };
    auto solver = AmericanOptionSolver::create(params, PDEGridSpec{grid}, snapshots);
    if (!solver) { state.SkipWithError("solver creation failed"); return; }
    if (state.range(0) && !enable_capture(*solver, events)) {
        state.SkipWithError("event capture API absent on baseline"); return;
    }
    double price = 0.0;
    size_t steps = 0;
    for (auto _ : state) {
        auto result = solver->solve();
        if (!result) { state.SkipWithError("PDE solve failed"); break; }
        price = result->value();
        steps = result->grid()->time().n_steps();
        benchmark::DoNotOptimize(price);
    }
    state.counters["price"] = price;
    state.counters["time_steps"] = steps;
    state.counters["ordinary_snapshot_bytes"] = snapshots.size() * 201 * sizeof(double);
    state.counters["extra_snapshot_bytes"] = state.range(0) ? events.size() * 201 * sizeof(double) : 0;
}
BENCHMARK(BM_EventFixedGridSolve)->Arg(0)->Arg(1)->UseRealTime();

}  // namespace

BENCHMARK_MAIN();
