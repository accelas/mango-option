/**
 * @file quantlib_accuracy.cc
 * @brief Accuracy comparison between mango-iv and QuantLib
 *
 * Compares numerical accuracy of American option pricing:
 * - Uses QuantLib as reference implementation
 * - Tests multiple scenarios (ATM, ITM, OTM, various maturities)
 * - Reports absolute and relative errors
 * - Tests convergence with increasing grid resolution
 * - Tests normalized chain solver accuracy
 *
 * Run with: bazel run //benchmarks:quantlib_accuracy --config=opt
 * Requires: libquantlib0-dev
 */

#include "src/option/american_option.hpp"
#include "src/option/normalized_chain_solver.hpp"
#include <benchmark/benchmark.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <algorithm>

// QuantLib includes
#include <ql/quantlib.hpp>

using namespace mango;
namespace ql = QuantLib;

// ============================================================================
// Helper: QuantLib American Option Pricer
// ============================================================================

struct PricingResult {
    double price;
    double delta;
    double gamma;
    double theta;
};

PricingResult price_american_option_quantlib(
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

    PricingResult result;
    result.price = american_option.NPV();
    result.delta = american_option.delta();
    result.gamma = american_option.gamma();
    result.theta = american_option.theta();

    return result;
}

// ============================================================================
// Accuracy Comparison: Single Scenario
// ============================================================================

static void compare_scenario(
    benchmark::State& state,
    const char* label,
    double spot,
    double strike,
    double maturity,
    double volatility,
    double rate,
    double dividend_yield,
    bool is_call)
{
    // Mango-IV pricing with workspace
    AmericanOptionParams mango_params{
        .strike = strike,
        .spot = spot,
        .maturity = maturity,
        .volatility = volatility,
        .rate = rate,
        .continuous_dividend_yield = dividend_yield,
        .option_type = is_call ? OptionType::CALL : OptionType::PUT,
        .discrete_dividends = {}
    };

    // Create workspace (high resolution for accuracy)
    auto workspace_result = AmericanSolverWorkspace::create(-3.0, 3.0, 201, 2000);
    if (!workspace_result) {
        throw std::runtime_error("Failed to create workspace: " + workspace_result.error());
    }
    auto workspace = std::move(workspace_result.value());

    AmericanOptionSolver solver(mango_params, workspace);
    auto mango_result_expected = solver.solve();
    if (!mango_result_expected) {
        throw std::runtime_error(mango_result_expected.error().message);
    }
    const AmericanOptionResult& mango_result = *mango_result_expected;

    // QuantLib pricing
    auto ql_result = price_american_option_quantlib(
        spot, strike, maturity, volatility, rate, dividend_yield, is_call,
        201, 2000);

    // Calculate errors
    double price_error = std::abs(mango_result.value - ql_result.price);
    double price_rel_error = price_error / ql_result.price * 100.0;

    double delta_error = std::abs(mango_result.delta - ql_result.delta);
    double gamma_error = std::abs(mango_result.gamma - ql_result.gamma);
    double theta_error = std::abs(mango_result.theta - ql_result.theta);

    // Report results (not part of benchmark timing)
    for (auto _ : state) {
        // Just iterate once to report
    }

    state.SetLabel(std::string(label) +
                  ": Price err=" + std::to_string(price_error) +
                  " (" + std::to_string(price_rel_error) + "%)");

    // Store custom counters
    state.counters["ql_price"] = ql_result.price;
    state.counters["mango_price"] = mango_result.value;
    state.counters["price_abs_err"] = price_error;
    state.counters["price_rel_err_%"] = price_rel_error;
    state.counters["delta_abs_err"] = delta_error;
    state.counters["gamma_abs_err"] = gamma_error;
    state.counters["theta_abs_err"] = theta_error;
}

// ============================================================================
// Accuracy Test Cases
// ============================================================================

static void BM_Accuracy_ATM_Put_1Y(benchmark::State& state) {
    compare_scenario(state, "ATM Put 1Y",
        100.0, 100.0, 1.0, 0.20, 0.05, 0.02, false);
}
BENCHMARK(BM_Accuracy_ATM_Put_1Y)->Iterations(1);

static void BM_Accuracy_OTM_Put_3M(benchmark::State& state) {
    compare_scenario(state, "OTM Put 3M",
        110.0, 100.0, 0.25, 0.30, 0.05, 0.02, false);
}
BENCHMARK(BM_Accuracy_OTM_Put_3M)->Iterations(1);

static void BM_Accuracy_ITM_Put_2Y(benchmark::State& state) {
    compare_scenario(state, "ITM Put 2Y",
        90.0, 100.0, 2.0, 0.25, 0.05, 0.02, false);
}
BENCHMARK(BM_Accuracy_ITM_Put_2Y)->Iterations(1);

static void BM_Accuracy_ATM_Call_1Y(benchmark::State& state) {
    compare_scenario(state, "ATM Call 1Y",
        100.0, 100.0, 1.0, 0.20, 0.05, 0.02, true);
}
BENCHMARK(BM_Accuracy_ATM_Call_1Y)->Iterations(1);

static void BM_Accuracy_DeepITM_Put_6M(benchmark::State& state) {
    compare_scenario(state, "Deep ITM Put 6M",
        80.0, 100.0, 0.5, 0.25, 0.05, 0.02, false);
}
BENCHMARK(BM_Accuracy_DeepITM_Put_6M)->Iterations(1);

static void BM_Accuracy_HighVol_Put_1Y(benchmark::State& state) {
    compare_scenario(state, "High Vol Put 1Y",
        100.0, 100.0, 1.0, 0.50, 0.05, 0.02, false);
}
BENCHMARK(BM_Accuracy_HighVol_Put_1Y)->Iterations(1);

static void BM_Accuracy_LowVol_Put_1Y(benchmark::State& state) {
    compare_scenario(state, "Low Vol Put 1Y",
        100.0, 100.0, 1.0, 0.10, 0.05, 0.02, false);
}
BENCHMARK(BM_Accuracy_LowVol_Put_1Y)->Iterations(1);

static void BM_Accuracy_LongMaturity_Put_5Y(benchmark::State& state) {
    compare_scenario(state, "Long Maturity Put 5Y",
        100.0, 100.0, 5.0, 0.20, 0.05, 0.02, false);
}
BENCHMARK(BM_Accuracy_LongMaturity_Put_5Y)->Iterations(1);

// ============================================================================
// Convergence Study: Grid Resolution
// ============================================================================

static void BM_Convergence_GridResolution(benchmark::State& state) {
    size_t n_space = state.range(0);
    size_t n_time = state.range(1);

    // Reference: Very high resolution QuantLib result
    static double ql_reference = 0.0;
    if (ql_reference == 0.0) {
        auto ref = price_american_option_quantlib(
            100.0, 100.0, 1.0, 0.20, 0.05, 0.02, false,
            1001, 10000);  // Very high resolution
        ql_reference = ref.price;
    }

    // Mango-IV pricing at given resolution
    AmericanOptionParams params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .discrete_dividends = {}
    };

    auto workspace_result = AmericanSolverWorkspace::create(-3.0, 3.0, n_space, n_time);
    if (!workspace_result) {
        throw std::runtime_error("Failed to create workspace");
    }
    auto workspace = std::move(workspace_result.value());

    AmericanOptionSolver solver(params, workspace);
    auto mango_result_expected = solver.solve();
    if (!mango_result_expected) {
        throw std::runtime_error(mango_result_expected.error().message);
    }
    const AmericanOptionResult& result = *mango_result_expected;

    // Error vs high-resolution reference
    double error = std::abs(result.value - ql_reference);
    double rel_error = error / ql_reference * 100.0;

    for (auto _ : state) {
        // Just iterate once to report
    }

    state.SetLabel("Grid " + std::to_string(n_space) + "x" + std::to_string(n_time));
    state.counters["abs_error"] = error;
    state.counters["rel_error_%"] = rel_error;
    state.counters["reference"] = ql_reference;
    state.counters["mango_price"] = result.value;
}

BENCHMARK(BM_Convergence_GridResolution)
    ->Args({51, 500})
    ->Args({101, 1000})
    ->Args({201, 2000})
    ->Args({501, 5000})
    ->Iterations(1);

// ============================================================================
// Greeks Accuracy Comparison
// ============================================================================

static void BM_Greeks_Accuracy_ATM(benchmark::State& state) {
    AmericanOptionParams params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .discrete_dividends = {}
    };

    auto workspace_result = AmericanSolverWorkspace::create(-3.0, 3.0, 201, 2000);
    if (!workspace_result) {
        throw std::runtime_error("Failed to create workspace");
    }
    auto workspace = std::move(workspace_result.value());

    AmericanOptionSolver solver(params, workspace);
    auto mango_result_expected = solver.solve();
    if (!mango_result_expected) {
        throw std::runtime_error(mango_result_expected.error().message);
    }
    const AmericanOptionResult& mango_result = *mango_result_expected;

    auto ql_result = price_american_option_quantlib(
        100.0, 100.0, 1.0, 0.20, 0.05, 0.02, false, 201, 2000);

    double delta_error = std::abs(mango_result.delta - ql_result.delta);
    double delta_rel = delta_error / std::abs(ql_result.delta) * 100.0;

    double gamma_error = std::abs(mango_result.gamma - ql_result.gamma);
    double gamma_rel = gamma_error / std::abs(ql_result.gamma) * 100.0;

    double theta_error = std::abs(mango_result.theta - ql_result.theta);
    double theta_rel = theta_error / std::abs(ql_result.theta) * 100.0;

    for (auto _ : state) {
        // Just iterate once to report
    }

    state.SetLabel("Greeks accuracy");
    state.counters["delta_ql"] = ql_result.delta;
    state.counters["delta_mango"] = mango_result.delta;
    state.counters["delta_rel_err_%"] = delta_rel;
    state.counters["gamma_ql"] = ql_result.gamma;
    state.counters["gamma_mango"] = mango_result.gamma;
    state.counters["gamma_rel_err_%"] = gamma_rel;
    state.counters["theta_ql"] = ql_result.theta;
    state.counters["theta_mango"] = mango_result.theta;
    state.counters["theta_rel_err_%"] = theta_rel;
}
BENCHMARK(BM_Greeks_Accuracy_ATM)->Iterations(1);

// ============================================================================
// Normalized Chain Solver Accuracy: Compare to QuantLib
// ============================================================================

static void compare_normalized_chain_accuracy(
    benchmark::State& state,
    const char* label,
    double spot,
    const std::vector<double>& strikes,
    const std::vector<double>& maturities,
    double volatility,
    double rate,
    double dividend_yield,
    bool is_call)
{
    // Build normalized solver request
    NormalizedSolveRequest request{
        .sigma = volatility,
        .rate = rate,
        .dividend = dividend_yield,
        .option_type = is_call ? OptionType::CALL : OptionType::PUT,
        .x_min = -3.0,
        .x_max = 3.0,
        .n_space = 201,  // High resolution for accuracy
        .n_time = 2000,
        .T_max = *std::max_element(maturities.begin(), maturities.end()),
        .tau_snapshots = maturities
    };

    // Solve with normalized chain solver
    auto workspace_result = NormalizedWorkspace::create(request);
    if (!workspace_result) {
        throw std::runtime_error("Failed to create workspace");
    }
    auto workspace = std::move(workspace_result.value());
    auto surface = workspace.surface_view();

    auto solve_result = NormalizedChainSolver::solve(request, workspace, surface);
    if (!solve_result) {
        throw std::runtime_error("Failed to solve normalized PDE");
    }

    // Compare prices for all (strike, maturity) combinations
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    double avg_abs_error = 0.0;
    double avg_rel_error = 0.0;
    size_t n_tests = 0;

    for (size_t i = 0; i < strikes.size(); ++i) {
        double K = strikes[i];
        for (size_t j = 0; j < maturities.size(); ++j) {
            double tau = maturities[j];

            // Mango normalized price
            double x = std::log(spot / K);
            double u = surface.interpolate(x, tau);
            double mango_price = K * u;

            // QuantLib reference price
            auto ql_result = price_american_option_quantlib(
                spot, K, tau, volatility, rate, dividend_yield, is_call,
                201, 2000);

            // Compute errors
            double abs_error = std::abs(mango_price - ql_result.price);
            double rel_error = abs_error / ql_result.price * 100.0;

            max_abs_error = std::max(max_abs_error, abs_error);
            max_rel_error = std::max(max_rel_error, rel_error);
            avg_abs_error += abs_error;
            avg_rel_error += rel_error;
            ++n_tests;
        }
    }

    avg_abs_error /= n_tests;
    avg_rel_error /= n_tests;

    // Report results
    for (auto _ : state) {
        // Just iterate once to report
    }

    state.SetLabel(std::string(label) +
                  ": n=" + std::to_string(n_tests) +
                  " max_rel=" + std::to_string(max_rel_error) + "%");

    state.counters["n_tests"] = n_tests;
    state.counters["max_abs_err"] = max_abs_error;
    state.counters["max_rel_err_%"] = max_rel_error;
    state.counters["avg_abs_err"] = avg_abs_error;
    state.counters["avg_rel_err_%"] = avg_rel_error;
}

// Test 1: ATM chain (5 strikes × 3 maturities = 15 options)
static void BM_NormalizedChain_ATM_5x3(benchmark::State& state) {
    compare_normalized_chain_accuracy(
        state, "Normalized Chain ATM 5x3",
        100.0,                                  // spot
        {90.0, 95.0, 100.0, 105.0, 110.0},     // strikes
        {0.25, 0.5, 1.0},                       // maturities
        0.20, 0.05, 0.02,                       // σ, r, q
        false);                                 // put
}
BENCHMARK(BM_NormalizedChain_ATM_5x3)->Iterations(1);

// Test 2: Wide strike range (7 strikes × 4 maturities = 28 options)
static void BM_NormalizedChain_Wide_7x4(benchmark::State& state) {
    compare_normalized_chain_accuracy(
        state, "Normalized Chain Wide 7x4",
        100.0,                                              // spot
        {80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0},    // strikes
        {0.25, 0.5, 1.0, 2.0},                             // maturities
        0.25, 0.05, 0.02,                                  // σ, r, q
        false);                                            // put
}
BENCHMARK(BM_NormalizedChain_Wide_7x4)->Iterations(1);

// Test 3: High volatility chain
static void BM_NormalizedChain_HighVol_5x3(benchmark::State& state) {
    compare_normalized_chain_accuracy(
        state, "Normalized Chain High Vol 5x3",
        100.0,                                  // spot
        {85.0, 90.0, 100.0, 110.0, 115.0},     // strikes
        {0.25, 0.5, 1.0},                       // maturities
        0.50, 0.05, 0.02,                       // σ=50%, r, q
        false);                                 // put
}
BENCHMARK(BM_NormalizedChain_HighVol_5x3)->Iterations(1);

// Test 4: Long maturity chain
static void BM_NormalizedChain_LongMat_5x4(benchmark::State& state) {
    compare_normalized_chain_accuracy(
        state, "Normalized Chain Long Mat 5x4",
        100.0,                                  // spot
        {85.0, 92.5, 100.0, 107.5, 115.0},     // strikes
        {0.5, 1.0, 2.0, 3.0},                   // maturities (up to 3 years)
        0.25, 0.05, 0.02,                       // σ, r, q
        false);                                 // put
}
BENCHMARK(BM_NormalizedChain_LongMat_5x4)->Iterations(1);

// Test 5: Call options
static void BM_NormalizedChain_Call_5x3(benchmark::State& state) {
    compare_normalized_chain_accuracy(
        state, "Normalized Chain Call 5x3",
        100.0,                                  // spot
        {90.0, 95.0, 100.0, 105.0, 110.0},     // strikes
        {0.25, 0.5, 1.0},                       // maturities
        0.20, 0.05, 0.02,                       // σ, r, q
        true);                                  // call
}
BENCHMARK(BM_NormalizedChain_Call_5x3)->Iterations(1);

BENCHMARK_MAIN();
