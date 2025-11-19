/**
 * @file quantlib_performance.cc
 * @brief Performance comparison between mango-iv and QuantLib
 *
 * Compares performance of American option pricing between:
 * - mango-iv (C++20 FDM implementation)
 * - QuantLib (FDM implementation)
 *
 * Both use similar grid sizes for fair comparison.
 *
 * Run with: bazel run //benchmarks:quantlib_performance
 * Requires: libquantlib-dev (apt-get install libquantlib-dev)
 */

#include "src/option/american_option.hpp"
#include "src/option/american_solver_workspace.hpp"
#include <benchmark/benchmark.h>
#include <memory_resource>
#include <stdexcept>

// QuantLib includes
#include <ql/quantlib.hpp>

using namespace mango;
namespace ql = QuantLib;

// ============================================================================
// Helper: QuantLib American Option Pricer
// ============================================================================

double price_american_option_quantlib(
    double spot,
    double strike,
    double maturity,
    double volatility,
    double rate,
    double dividend_yield,
    bool is_call,
    size_t grid_steps = 100,
    size_t time_steps = 1000)
{
    // Setup QuantLib environment
    ql::Date today = ql::Date::todaysDate();
    ql::Settings::instance().evaluationDate() = today;

    // Option parameters
    ql::Option::Type option_type = is_call ? ql::Option::Call : ql::Option::Put;
    ql::Date maturity_date = today + ql::Period(static_cast<int>(maturity * 365), ql::Days);

    ql::ext::shared_ptr<ql::Exercise> exercise = ql::ext::make_shared<ql::AmericanExercise>(today, maturity_date);
    ql::ext::shared_ptr<ql::StrikedTypePayoff> payoff = ql::ext::make_shared<ql::PlainVanillaPayoff>(option_type, strike);

    ql::VanillaOption american_option(payoff, exercise);

    // Market data
    ql::Handle<ql::Quote> spot_handle(ql::ext::make_shared<ql::SimpleQuote>(spot));
    ql::Handle<ql::YieldTermStructure> rate_ts(
        ql::ext::make_shared<ql::FlatForward>(today, rate, ql::Actual365Fixed()));
    ql::Handle<ql::YieldTermStructure> div_ts(
        ql::ext::make_shared<ql::FlatForward>(today, dividend_yield, ql::Actual365Fixed()));
    ql::Handle<ql::BlackVolTermStructure> vol_ts(
        ql::ext::make_shared<ql::BlackConstantVol>(today, ql::NullCalendar(), volatility, ql::Actual365Fixed()));

    ql::ext::shared_ptr<ql::BlackScholesMertonProcess> bs_process =
        ql::ext::make_shared<ql::BlackScholesMertonProcess>(spot_handle, div_ts, rate_ts, vol_ts);

    // Finite difference pricing engine
    american_option.setPricingEngine(
        ql::ext::make_shared<ql::FdBlackScholesVanillaEngine>(
            bs_process,
            time_steps,  // timeSteps
            grid_steps   // gridPoints
        ));

    return american_option.NPV();
}

// ============================================================================
// Performance Comparison Benchmarks
// ============================================================================

static void BM_Mango_AmericanPut_ATM(benchmark::State& state) {
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::PUT,
        0.20    // volatility
    );

    auto [grid_spec, n_time] = estimate_grid_for_option(params);
    auto workspace_result = AmericanSolverWorkspace::create(grid_spec, n_time, std::pmr::get_default_resource());
    if (!workspace_result) {
        state.SkipWithError(workspace_result.error().c_str());
        return;
    }

    for (auto _ : state) {
        auto solver_result = AmericanOptionSolver::create(params, workspace_result.value());
        if (!solver_result) {
            state.SkipWithError(solver_result.error().c_str());
            return;
        }
        auto result = solver_result.value().solve();
        if (!result) {
            throw std::runtime_error(result.error().message);
        }
        benchmark::DoNotOptimize(*result);
    }

    state.SetLabel("mango-iv");
}
BENCHMARK(BM_Mango_AmericanPut_ATM);

static void BM_QuantLib_AmericanPut_ATM(benchmark::State& state) {
    for (auto _ : state) {
        double price = price_american_option_quantlib(
            100.0,  // spot
            100.0,  // strike
            1.0,    // maturity
            0.20,   // volatility
            0.05,   // rate
            0.02,   // dividend yield
            false,  // is_call (put)
            101,    // grid_steps
            1000    // time_steps
        );
        benchmark::DoNotOptimize(price);
    }

    state.SetLabel("QuantLib");
}
BENCHMARK(BM_QuantLib_AmericanPut_ATM);

// ----------------------------------------------------------------------------

static void BM_Mango_AmericanPut_OTM(benchmark::State& state) {
    AmericanOptionParams params(
        110.0,  // spot
        100.0,  // strike
        0.25,   // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::PUT,
        0.30    // volatility
    );

    auto [grid_spec, n_time] = estimate_grid_for_option(params);
    auto workspace_result = AmericanSolverWorkspace::create(grid_spec, n_time, std::pmr::get_default_resource());
    if (!workspace_result) {
        state.SkipWithError(workspace_result.error().c_str());
        return;
    }

    for (auto _ : state) {
        auto solver_result = AmericanOptionSolver::create(params, workspace_result.value());
        if (!solver_result) {
            state.SkipWithError(solver_result.error().c_str());
            return;
        }
        auto result = solver_result.value().solve();
        if (!result) {
            throw std::runtime_error(result.error().message);
        }
        benchmark::DoNotOptimize(*result);
    }

    state.SetLabel("mango-iv");
}
BENCHMARK(BM_Mango_AmericanPut_OTM);

static void BM_QuantLib_AmericanPut_OTM(benchmark::State& state) {
    for (auto _ : state) {
        double price = price_american_option_quantlib(
            110.0,  // spot
            100.0,  // strike
            0.25,   // maturity
            0.30,   // volatility
            0.05,   // rate
            0.02,   // dividend yield
            false,  // is_call (put)
            101,    // grid_steps
            1000    // time_steps
        );
        benchmark::DoNotOptimize(price);
    }

    state.SetLabel("QuantLib");
}
BENCHMARK(BM_QuantLib_AmericanPut_OTM);

// ----------------------------------------------------------------------------

static void BM_Mango_AmericanPut_ITM(benchmark::State& state) {
    AmericanOptionParams params(
        90.0,   // spot
        100.0,  // strike
        2.0,    // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::PUT,
        0.25    // volatility
    );

    auto [grid_spec, n_time] = estimate_grid_for_option(params);
    auto workspace_result = AmericanSolverWorkspace::create(grid_spec, n_time, std::pmr::get_default_resource());
    if (!workspace_result) {
        state.SkipWithError(workspace_result.error().c_str());
        return;
    }

    for (auto _ : state) {
        auto solver_result = AmericanOptionSolver::create(params, workspace_result.value());
        if (!solver_result) {
            state.SkipWithError(solver_result.error().c_str());
            return;
        }
        auto result = solver_result.value().solve();
        if (!result) {
            throw std::runtime_error(result.error().message);
        }
        benchmark::DoNotOptimize(*result);
    }

    state.SetLabel("mango-iv");
}
BENCHMARK(BM_Mango_AmericanPut_ITM);

static void BM_QuantLib_AmericanPut_ITM(benchmark::State& state) {
    for (auto _ : state) {
        double price = price_american_option_quantlib(
            90.0,   // spot
            100.0,  // strike
            2.0,    // maturity
            0.25,   // volatility
            0.05,   // rate
            0.02,   // dividend yield
            false,  // is_call (put)
            101,    // grid_steps
            1000    // time_steps
        );
        benchmark::DoNotOptimize(price);
    }

    state.SetLabel("QuantLib");
}
BENCHMARK(BM_QuantLib_AmericanPut_ITM);

// ============================================================================
// Grid Resolution Comparison
// ============================================================================

static void BM_Mango_GridResolution(benchmark::State& state) {
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::PUT,
        0.20    // volatility
    );

    auto [grid_spec, n_time] = estimate_grid_for_option(params);
    auto workspace_result = AmericanSolverWorkspace::create(grid_spec, n_time, std::pmr::get_default_resource());
    if (!workspace_result) {
        state.SkipWithError(workspace_result.error().c_str());
        return;
    }

    for (auto _ : state) {
        auto solver_result = AmericanOptionSolver::create(params, workspace_result.value());
        if (!solver_result) {
            state.SkipWithError(solver_result.error().c_str());
            return;
        }
        auto result = solver_result.value().solve();
        if (!result) {
            throw std::runtime_error(result.error().message);
        }
        benchmark::DoNotOptimize(*result);
    }

    state.SetLabel("mango " + std::to_string(grid_spec.n_points()) + "x" + std::to_string(n_time));
}
BENCHMARK(BM_Mango_GridResolution)
    ->Args({101, 1000})
    ->Args({201, 2000})
    ->Args({501, 5000});

static void BM_QuantLib_GridResolution(benchmark::State& state) {
    size_t n_space = state.range(0);
    size_t n_time = state.range(1);

    for (auto _ : state) {
        double price = price_american_option_quantlib(
            100.0,  // spot
            100.0,  // strike
            1.0,    // maturity
            0.20,   // volatility
            0.05,   // rate
            0.02,   // dividend yield
            false,  // is_call (put)
            n_space,
            n_time
        );
        benchmark::DoNotOptimize(price);
    }

    state.SetLabel("QuantLib " + std::to_string(n_space) + "x" + std::to_string(n_time));
}
BENCHMARK(BM_QuantLib_GridResolution)
    ->Args({101, 1000})
    ->Args({201, 2000})
    ->Args({501, 5000});

BENCHMARK_MAIN();
