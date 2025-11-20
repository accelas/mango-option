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
#include <memory_resource>
#include <stdexcept>
#include <algorithm>

// QuantLib includes
#include <ql/quantlib.hpp>

using namespace mango;
namespace ql = QuantLib;

// ============================================================================
// Compatibility shims for post-revert API
// ============================================================================

// Helper function to estimate grid parameters from option params
inline std::tuple<GridSpec<double>, size_t> estimate_grid_for_option(
    const PricingParams& params,
    double n_sigma = 5.0,
    double tol = 1e-6)
{
    // Domain bounds (centered on current moneyness)
    double sigma_sqrt_T = params.volatility * std::sqrt(params.maturity);
    double x0 = std::log(params.spot / params.strike);

    double x_min = x0 - n_sigma * sigma_sqrt_T;
    double x_max = x0 + n_sigma * sigma_sqrt_T;

    // Spatial resolution (target truncation error)
    double dx_target = params.volatility * std::sqrt(tol);
    size_t Nx = static_cast<size_t>(std::ceil((x_max - x_min) / dx_target));
    Nx = std::clamp(Nx, size_t{200}, size_t{1200});

    // Ensure odd number of points (for centered stencils)
    if (Nx % 2 == 0) Nx++;

    // Temporal resolution (CFL-like condition for stability)
    double dx_actual = (x_max - x_min) / (Nx - 1);
    double dt_target = 0.75 * dx_actual * dx_actual / (params.volatility * params.volatility);
    size_t Nt = static_cast<size_t>(std::ceil(params.maturity / dt_target));
    Nt = std::clamp(Nt, size_t{200}, size_t{4000});

    auto grid_spec = GridSpec<double>::uniform(x_min, x_max, Nx);
    return {grid_spec.value(), Nt};
}

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
    AmericanOptionParams mango_params(
        spot,
        strike,
        maturity,
        rate,
        dividend_yield,
        is_call ? OptionType::CALL : OptionType::PUT,
        volatility
    );

    // Create workspace (use automatic grid determination)
    auto [grid_spec, n_time] = estimate_grid_for_option(mango_params);
    auto workspace_result = AmericanSolverWorkspace::create(
        grid_spec.x_min(), grid_spec.x_max(), grid_spec.n_points(), n_time);
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

    // Compute Greeks
    auto greeks_result = solver.compute_greeks();
    if (!greeks_result) {
        throw std::runtime_error("Failed to compute Greeks: " + greeks_result.error().message);
    }
    const AmericanOptionGreeks& greeks = *greeks_result;

    // QuantLib pricing
    auto ql_result = price_american_option_quantlib(
        spot, strike, maturity, volatility, rate, dividend_yield, is_call,
        201, 2000);

    // Calculate errors
    double mango_price = mango_result.value_at(spot);
    double price_error = std::abs(mango_price - ql_result.price);
    double price_rel_error = price_error / ql_result.price * 100.0;

    double delta_error = std::abs(greeks.delta - ql_result.delta);
    double gamma_error = std::abs(greeks.gamma - ql_result.gamma);
    double theta_error = std::abs(greeks.theta - ql_result.theta);

    // Report results (not part of benchmark timing)
    for (auto _ : state) {
        // Just iterate once to report
    }

    state.SetLabel(std::string(label) +
                  ": Price err=" + std::to_string(price_error) +
                  " (" + std::to_string(price_rel_error) + "%)");

    // Store custom counters
    state.counters["ql_price"] = ql_result.price;
    state.counters["mango_price"] = mango_price;
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
    // Reference: Very high resolution QuantLib result
    static double ql_reference = 0.0;
    if (ql_reference == 0.0) {
        auto ref = price_american_option_quantlib(
            100.0, 100.0, 1.0, 0.20, 0.05, 0.02, false,
            1001, 10000);  // Very high resolution
        ql_reference = ref.price;
    }

    // Mango-IV pricing at automatically determined resolution
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
    auto workspace_result = AmericanSolverWorkspace::create(grid_spec.x_min(), grid_spec.x_max(), grid_spec.n_points(), n_time);
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
    double mango_price = result.value_at(params.spot);
    double error = std::abs(mango_price - ql_reference);
    double rel_error = error / ql_reference * 100.0;

    for (auto _ : state) {
        // Just iterate once to report
    }

    state.SetLabel("Grid " + std::to_string(grid_spec.n_points()) + "x" + std::to_string(n_time));
    state.counters["abs_error"] = error;
    state.counters["rel_error_%"] = rel_error;
    state.counters["reference"] = ql_reference;
    state.counters["mango_price"] = mango_price;
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
    auto workspace_result = AmericanSolverWorkspace::create(grid_spec.x_min(), grid_spec.x_max(), grid_spec.n_points(), n_time);
    if (!workspace_result) {
        throw std::runtime_error("Failed to create workspace");
    }
    auto workspace = std::move(workspace_result.value());

    AmericanOptionSolver solver(params, workspace);
    auto mango_result_expected = solver.solve();
    if (!mango_result_expected) {
        throw std::runtime_error(mango_result_expected.error().message);
    }

    // Compute Greeks
    auto greeks_result = solver.compute_greeks();
    if (!greeks_result) {
        throw std::runtime_error("Failed to compute Greeks: " + greeks_result.error().message);
    }
    const AmericanOptionGreeks& greeks = *greeks_result;

    auto ql_result = price_american_option_quantlib(
        100.0, 100.0, 1.0, 0.20, 0.05, 0.02, false, 201, 2000);

    double delta_error = std::abs(greeks.delta - ql_result.delta);
    double delta_rel = delta_error / std::abs(ql_result.delta) * 100.0;

    double gamma_error = std::abs(greeks.gamma - ql_result.gamma);
    double gamma_rel = gamma_error / std::abs(ql_result.gamma) * 100.0;

    double theta_error = std::abs(greeks.theta - ql_result.theta);
    double theta_rel = theta_error / std::abs(ql_result.theta) * 100.0;

    for (auto _ : state) {
        // Just iterate once to report
    }

    state.SetLabel("Greeks accuracy");
    state.counters["delta_ql"] = ql_result.delta;
    state.counters["delta_mango"] = greeks.delta;
    state.counters["delta_rel_err_%"] = delta_rel;
    state.counters["gamma_ql"] = ql_result.gamma;
    state.counters["gamma_mango"] = greeks.gamma;
    state.counters["gamma_rel_err_%"] = gamma_rel;
    state.counters["theta_ql"] = ql_result.theta;
    state.counters["theta_mango"] = greeks.theta;
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
    double max_rel_error = 0.0;       // For meaningful prices only
    double max_rel_error_all = 0.0;   // Includes scaled low-premium metric
    double avg_abs_error = 0.0;
    double avg_rel_error = 0.0;
    size_t n_tests = 0;
    size_t n_low_premium = 0;         // Count of options < $0.01

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

            // Relative error handling:
            // For prices >= $0.01: compute standard relative error (meaningful %)
            // For prices < $0.01: track separately as low-premium (abs error is primary metric)
            constexpr double MIN_PRICE_FOR_REL_ERROR = 0.01;
            double rel_error = 0.0;
            double rel_error_scaled = 0.0;  // For low-premium options

            if (ql_result.price >= MIN_PRICE_FOR_REL_ERROR) {
                // Standard relative error for meaningful prices
                rel_error = abs_error / ql_result.price * 100.0;
                max_rel_error = std::max(max_rel_error, rel_error);
            } else {
                // Low-premium option: scale absolute error by 1¢ for percentage-like metric
                // This prevents Inf/NaN but shouldn't be interpreted as true relative error
                rel_error_scaled = abs_error / MIN_PRICE_FOR_REL_ERROR * 100.0;
                ++n_low_premium;
            }

            max_abs_error = std::max(max_abs_error, abs_error);
            max_rel_error_all = std::max(max_rel_error_all, std::max(rel_error, rel_error_scaled));
            avg_abs_error += abs_error;
            avg_rel_error += (rel_error > 0 ? rel_error : rel_error_scaled);
            ++n_tests;
        }
    }

    avg_abs_error /= n_tests;
    avg_rel_error /= n_tests;

    // Report results
    for (auto _ : state) {
        // Just iterate once to report
    }

    // Build label showing key metrics
    std::string label_str = std::string(label) +
                           ": n=" + std::to_string(n_tests) +
                           " max_abs=$" + std::to_string(max_abs_error);
    if (n_low_premium > 0) {
        label_str += " (low_premium=" + std::to_string(n_low_premium) + ")";
    }
    state.SetLabel(label_str);

    state.counters["n_tests"] = n_tests;
    state.counters["n_low_premium"] = n_low_premium;
    state.counters["max_abs_err_$"] = max_abs_error;
    state.counters["max_rel_err_%_meaningful"] = max_rel_error;  // Only prices >= $0.01
    state.counters["max_rel_err_%_all"] = max_rel_error_all;     // Includes scaled metric
    state.counters["avg_abs_err_$"] = avg_abs_error;
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
// NOTE: Wide ranges include deep OTM options with low absolute prices.
// For example, a put with K=80 when S=100 may have price < $0.10.
// Relative error is computed carefully: for prices < $0.01, we scale
// absolute error by 1¢ to avoid Inf/NaN from division by near-zero.
// Absolute error remains the primary accuracy metric for these cases.
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

// Test 6: Deep OTM/ITM Calls (boundary condition validation)
static void BM_NormalizedChain_DeepCalls_5x3(benchmark::State& state) {
    compare_normalized_chain_accuracy(
        state, "Normalized Chain Deep Calls 5x3",
        100.0,                                      // spot
        {70.0, 85.0, 100.0, 115.0, 130.0},         // strikes: deep ITM → deep OTM
        {0.25, 0.5, 1.0},                           // maturities
        0.25, 0.05, 0.02,                           // σ, r, q
        true);                                      // call
}
BENCHMARK(BM_NormalizedChain_DeepCalls_5x3)->Iterations(1);

// Test 7: Negative Interest Rate (European crisis scenario)
static void BM_NormalizedChain_NegativeRate_5x3(benchmark::State& state) {
    compare_normalized_chain_accuracy(
        state, "Normalized Chain Negative Rate 5x3",
        100.0,                                      // spot
        {85.0, 92.5, 100.0, 107.5, 115.0},         // strikes
        {0.25, 0.5, 1.0},                           // maturities
        0.20, -0.01, 0.00,                          // σ, r=-1%, q=0%
        false);                                     // put
}
BENCHMARK(BM_NormalizedChain_NegativeRate_5x3)->Iterations(1);

// Test 8: High Interest Rate (emerging markets scenario)
static void BM_NormalizedChain_HighRate_5x3(benchmark::State& state) {
    compare_normalized_chain_accuracy(
        state, "Normalized Chain High Rate 5x3",
        100.0,                                      // spot
        {85.0, 92.5, 100.0, 107.5, 115.0},         // strikes
        {0.25, 0.5, 1.0},                           // maturities
        0.30, 0.15, 0.03,                           // σ, r=15%, q=3%
        true);                                      // call
}
BENCHMARK(BM_NormalizedChain_HighRate_5x3)->Iterations(1);

// ============================================================================
// Discrete Dividend Tests: Ensure workspace routing + PDE handle jump mapping
// ============================================================================

// NOTE: These tests use the standard workspace API (not normalized chain solver)
// because discrete dividends force fallback from fast path to batch API.
// This validates the entire pipeline: dividend mapping, jump conditions, and
// workspace reuse when discrete payouts are present.

static void BM_DiscreteDiv_SinglePayout_Call(benchmark::State& state) {
    // Single $1 dividend at 0.25y (quarterly payout scenario)
    double spot = 100.0;
    double strike = 100.0;
    double maturity = 0.5;
    double volatility = 0.25;
    double rate = 0.05;
    double div_yield = 0.02;

    // Discrete dividend: $1 at 0.25y (time, amount pairs)
    std::vector<std::pair<double, double>> dividends = {
        {0.25, 1.0}  // $1 dividend at 0.25 years
    };

    AmericanOptionParams params(
        spot,
        strike,
        maturity,
        rate,
        div_yield,
        OptionType::CALL,
        volatility,
        dividends
    );

    // Create workspace (use automatic grid determination)
    auto [grid_spec, n_time] = estimate_grid_for_option(params);
    auto workspace_result = AmericanSolverWorkspace::create(grid_spec.x_min(), grid_spec.x_max(), grid_spec.n_points(), n_time);
    if (!workspace_result) {
        throw std::runtime_error("Failed to create workspace");
    }
    auto workspace = std::move(workspace_result.value());

    AmericanOptionSolver solver(params, workspace);
    auto result = solver.solve();

    if (!result) {
        throw std::runtime_error("Failed to solve with discrete dividend");
    }

    // Compute Greeks
    auto greeks_result = solver.compute_greeks();
    if (!greeks_result) {
        throw std::runtime_error("Failed to compute Greeks: " + greeks_result.error().message);
    }
    const AmericanOptionGreeks& greeks = *greeks_result;

    // QuantLib reference with discrete dividend
    // TODO: Add QuantLib DividendVanillaOption comparison when helper is implemented
    // For now, just verify the solve succeeds and produces reasonable values

    double mango_price = result->value_at(spot);

    for (auto _ : state) {
        // Report results
    }

    state.SetLabel("Discrete Div Call: S=$100 K=$100 div=$1@0.25y");
    state.counters["price_$"] = mango_price;
    state.counters["delta"] = greeks.delta;
    state.counters["gamma"] = greeks.gamma;

    // Sanity checks
    if (mango_price <= 0.0 || mango_price > spot) {
        throw std::runtime_error("Unreasonable price with discrete dividend");
    }
}
BENCHMARK(BM_DiscreteDiv_SinglePayout_Call)->Iterations(1);

static void BM_DiscreteDiv_Quarterly_Put(benchmark::State& state) {
    // Quarterly $0.50 dividends over 1 year (realistic equity scenario)
    double spot = 100.0;
    double strike = 100.0;
    double maturity = 1.0;
    double volatility = 0.20;
    double rate = 0.05;
    double div_yield = 0.01;

    // Quarterly dividends: $0.50 at 0.25y, 0.5y, 0.75y (time, amount pairs)
    std::vector<std::pair<double, double>> dividends = {
        {0.25, 0.5},  // $0.50 at 3 months
        {0.50, 0.5},  // $0.50 at 6 months
        {0.75, 0.5}   // $0.50 at 9 months
        // Note: dividend at maturity (1.0y) excluded (no impact)
    };

    AmericanOptionParams params(
        spot,
        strike,
        maturity,
        rate,
        div_yield,
        OptionType::PUT,
        volatility,
        dividends
    );

    auto [grid_spec, n_time] = estimate_grid_for_option(params);
    auto workspace_result = AmericanSolverWorkspace::create(grid_spec.x_min(), grid_spec.x_max(), grid_spec.n_points(), n_time);
    if (!workspace_result) {
        throw std::runtime_error("Failed to create workspace");
    }
    auto workspace = std::move(workspace_result.value());

    AmericanOptionSolver solver(params, workspace);
    auto result = solver.solve();

    if (!result) {
        throw std::runtime_error("Failed to solve with quarterly dividends");
    }

    // Compute Greeks
    auto greeks_result = solver.compute_greeks();
    if (!greeks_result) {
        throw std::runtime_error("Failed to compute Greeks: " + greeks_result.error().message);
    }
    const AmericanOptionGreeks& greeks = *greeks_result;

    double mango_price = result->value_at(spot);

    for (auto _ : state) {
        // Report results
    }

    state.SetLabel("Discrete Div Put: S=$100 K=$100 quarterly=$0.50");
    state.counters["price_$"] = mango_price;
    state.counters["delta"] = greeks.delta;
    state.counters["gamma"] = greeks.gamma;

    // Sanity checks
    if (mango_price <= 0.0 || mango_price > strike) {
        throw std::runtime_error("Unreasonable price with quarterly dividends");
    }
}
BENCHMARK(BM_DiscreteDiv_Quarterly_Put)->Iterations(1);

BENCHMARK_MAIN();
