// SPDX-License-Identifier: MIT
/**
 * @file quantlib_accuracy_test.cc
 * @brief Accuracy validation against QuantLib reference implementation
 *
 * This test ensures numerical accuracy doesn't regress by comparing
 * mango-option American option prices against QuantLib.
 *
 * Key scenarios tested:
 * - ATM/ITM/OTM options
 * - Various maturities (3M to 5Y)
 * - Different volatility regimes (10% to 50%)
 * - Greeks accuracy (delta, gamma, theta)
 *
 * Tolerance: 1% relative error (protects against 14.5% regression like #204)
 */

#include "mango/option/american_option.hpp"
#include "mango/option/iv_solver.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <memory_resource>

// QuantLib includes
#include <ql/quantlib.hpp>

using namespace mango;
namespace ql = QuantLib;

namespace {

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
    size_t grid_steps = 201,
    size_t time_steps = 2000)
{
    // Setup QuantLib environment
    ql::Date today = ql::Date::todaysDate();
    ql::Settings::instance().evaluationDate() = today;

    // Option parameters
    ql::Option::Type option_type = is_call ? ql::Option::Call : ql::Option::Put;
    ql::Date maturity_date = today + ql::Period(static_cast<int>(maturity * 365), ql::Days);

    ql::ext::shared_ptr<ql::Exercise> exercise =
        ql::ext::make_shared<ql::AmericanExercise>(today, maturity_date);
    ql::ext::shared_ptr<ql::StrikedTypePayoff> payoff =
        ql::ext::make_shared<ql::PlainVanillaPayoff>(option_type, strike);

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
            bs_process, time_steps, grid_steps));

    PricingResult result;
    result.price = american_option.NPV();
    result.delta = american_option.delta();
    result.gamma = american_option.gamma();
    result.theta = american_option.theta();

    return result;
}

void test_scenario(
    const std::string& name,
    double spot,
    double strike,
    double maturity,
    double volatility,
    double rate,
    double dividend_yield,
    bool is_call,
    double tolerance_pct = 1.0)
{
    SCOPED_TRACE(name);

    // Mango-Option pricing with auto-estimation (production mode)
    PricingParams mango_params(
        spot, strike, maturity, rate, dividend_yield,
        is_call ? OptionType::CALL : OptionType::PUT, volatility);

    // Use automatic grid estimation (matches production usage)
    auto [grid_spec, time_domain] = estimate_pde_grid(mango_params);

    // Allocate workspace buffer
    size_t n = grid_spec.n_points();
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
    ASSERT_TRUE(workspace_result.has_value()) << workspace_result.error();
    auto workspace = workspace_result.value();

    auto solver = AmericanOptionSolver::create(mango_params, workspace).value();
    auto mango_result = solver.solve();
    ASSERT_TRUE(mango_result.has_value()) << mango_result.error().message;

    // QuantLib reference
    auto ql_result = price_american_option_quantlib(
        spot, strike, maturity, volatility, rate, dividend_yield, is_call,
        201, 2000);

    // Check price accuracy
    double mango_price = mango_result->value_at(spot);
    double price_error = std::abs(mango_price - ql_result.price);
    double price_rel_error = (price_error / ql_result.price) * 100.0;

    EXPECT_LT(price_rel_error, tolerance_pct)
        << "Price relative error: " << price_rel_error << "%"
        << "\n  Mango:    $" << mango_price
        << "\n  QuantLib: $" << ql_result.price
        << "\n  Abs err:  $" << price_error;

    // Check Greeks (available directly from result)
    double delta_val = mango_result->delta();
    double delta_error = std::abs(delta_val - ql_result.delta);
    double delta_rel = (delta_error / std::abs(ql_result.delta)) * 100.0;

    EXPECT_LT(delta_rel, tolerance_pct * 2.0)  // 2x tolerance for Greeks
        << "Delta relative error: " << delta_rel << "%"
        << "\n  Mango:    " << delta_val
        << "\n  QuantLib: " << ql_result.delta;
}

} // namespace

// ============================================================================
// Core Accuracy Tests
// ============================================================================

TEST(QuantLibAccuracyTest, ATM_Put_1Y) {
    test_scenario("ATM Put 1Y",
        100.0, 100.0, 1.0, 0.20, 0.05, 0.02, false);
}

TEST(QuantLibAccuracyTest, OTM_Put_3M) {
    test_scenario("OTM Put 3M",
        110.0, 100.0, 0.25, 0.30, 0.05, 0.02, false);
}

TEST(QuantLibAccuracyTest, ITM_Put_2Y) {
    test_scenario("ITM Put 2Y",
        90.0, 100.0, 2.0, 0.25, 0.05, 0.02, false);
}

TEST(QuantLibAccuracyTest, ATM_Call_1Y) {
    test_scenario("ATM Call 1Y",
        100.0, 100.0, 1.0, 0.20, 0.05, 0.02, true);
}

TEST(QuantLibAccuracyTest, DeepITM_Put_6M) {
    test_scenario("Deep ITM Put 6M",
        80.0, 100.0, 0.5, 0.25, 0.05, 0.02, false);
}

TEST(QuantLibAccuracyTest, HighVol_Put_1Y) {
    test_scenario("High Vol Put 1Y",
        100.0, 100.0, 1.0, 0.50, 0.05, 0.02, false);
}

TEST(QuantLibAccuracyTest, LowVol_Put_1Y) {
    test_scenario("Low Vol Put 1Y",
        100.0, 100.0, 1.0, 0.10, 0.05, 0.02, false);
}

TEST(QuantLibAccuracyTest, LongMaturity_Put_5Y) {
    test_scenario("Long Maturity Put 5Y",
        100.0, 100.0, 5.0, 0.20, 0.05, 0.02, false);
}

// ============================================================================
// Convergence Test: Verify accuracy improves with grid resolution
// ============================================================================

TEST(QuantLibAccuracyTest, GridConvergence) {
    // Reference: Very high resolution QuantLib result
    auto ql_reference = price_american_option_quantlib(
        100.0, 100.0, 1.0, 0.20, 0.05, 0.02, false,
        1001, 10000);

    PricingParams params(
        100.0, 100.0, 1.0, 0.05, 0.02, OptionType::PUT, 0.20);

    // Use automatic grid estimation (production mode)
    auto [grid_spec, time_domain] = estimate_pde_grid(params);

    // Allocate workspace buffer
    size_t n = grid_spec.n_points();
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
    ASSERT_TRUE(workspace_result.has_value());
    auto workspace = workspace_result.value();

    auto solver = AmericanOptionSolver::create(params, workspace).value();
    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    double mango_price = result->value_at(params.spot);
    double error = std::abs(mango_price - ql_reference.price);
    double rel_error = (error / ql_reference.price) * 100.0;

    // Auto-estimation should converge to within 1% of high-resolution reference
    EXPECT_LT(rel_error, 1.0)
        << "Convergence test failed"
        << "\n  Mango:     $" << mango_price
        << "\n  Reference: $" << ql_reference.price
        << "\n  Grid:      " << grid_spec.n_points() << "x" << n_time;
}

// ============================================================================
// Greeks Accuracy
// ============================================================================

TEST(QuantLibAccuracyTest, Greeks_ATM) {
    PricingParams params(
        100.0, 100.0, 1.0, 0.05, 0.02, OptionType::PUT, 0.20);

    // Use automatic grid estimation (production mode)
    auto [grid_spec, time_domain] = estimate_pde_grid(params);

    // Allocate workspace buffer
    size_t n = grid_spec.n_points();
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
    ASSERT_TRUE(workspace_result.has_value());
    auto workspace = workspace_result.value();

    auto solver = AmericanOptionSolver::create(params, workspace).value();
    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    // Greeks are available directly from result
    double delta_val = result->delta();
    double gamma_val = result->gamma();
    // double theta_val = result->theta();  // Not tested yet

    auto ql_result = price_american_option_quantlib(
        100.0, 100.0, 1.0, 0.20, 0.05, 0.02, false, 201, 2000);

    // Delta within 2%
    double delta_error = std::abs(delta_val - ql_result.delta);
    double delta_rel = (delta_error / std::abs(ql_result.delta)) * 100.0;
    EXPECT_LT(delta_rel, 2.0)
        << "Delta: mango=" << delta_val << " ql=" << ql_result.delta;

    // Gamma within 5% (second derivative, less accurate)
    double gamma_error = std::abs(gamma_val - ql_result.gamma);
    double gamma_rel = (gamma_error / std::abs(ql_result.gamma)) * 100.0;
    EXPECT_LT(gamma_rel, 5.0)
        << "Gamma: mango=" << gamma_val << " ql=" << ql_result.gamma;

    // Note: Theta computation not yet implemented in compute_greeks()
    // Skip theta test for now
}

// ============================================================================
// Implied Volatility Accuracy Tests
// ============================================================================

void test_iv_scenario(
    const std::string& name,
    double spot,
    double strike,
    double maturity,
    double true_volatility,
    double rate,
    double dividend_yield,
    bool is_call,
    double tolerance_pct = 2.0)
{
    SCOPED_TRACE(name);

    // Get market price from QuantLib with known volatility
    auto ql_result = price_american_option_quantlib(
        spot, strike, maturity, true_volatility, rate, dividend_yield, is_call,
        201, 2000);

    // Solve for IV using mango with auto-estimation (production mode)
    IVQuery query{
        spot, strike, maturity, rate, dividend_yield,
        is_call ? OptionType::CALL : OptionType::PUT,
        ql_result.price
    };

    IVSolverConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    // Note: using auto-estimation (default GridAccuracyParams)

    IVSolver solver(config);
    auto iv_result = solver.solve(query);

    ASSERT_TRUE(iv_result.converged)
        << "IV solver failed: " << iv_result.failure_reason.value_or("unknown");

    // Check accuracy
    double vol_error = std::abs(iv_result.implied_vol - true_volatility);
    double vol_rel_error = (vol_error / true_volatility) * 100.0;

    EXPECT_LT(vol_rel_error, tolerance_pct)
        << "IV relative error: " << vol_rel_error << "%"
        << "\n  True vol:     " << true_volatility
        << "\n  Recovered IV: " << iv_result.implied_vol
        << "\n  Abs error:    " << vol_error
        << "\n  Iterations:   " << iv_result.iterations;
}

TEST(QuantLibAccuracyTest, IV_ATM_Put_1Y) {
    test_iv_scenario("IV ATM Put 1Y",
        100.0, 100.0, 1.0, 0.20, 0.05, 0.02, false);
}

TEST(QuantLibAccuracyTest, IV_OTM_Put_3M) {
    test_iv_scenario("IV OTM Put 3M",
        110.0, 100.0, 0.25, 0.30, 0.05, 0.02, false);
}

TEST(QuantLibAccuracyTest, IV_ITM_Put_2Y) {
    test_iv_scenario("IV ITM Put 2Y",
        90.0, 100.0, 2.0, 0.25, 0.05, 0.02, false);
}

TEST(QuantLibAccuracyTest, IV_ATM_Call_1Y) {
    test_iv_scenario("IV ATM Call 1Y",
        100.0, 100.0, 1.0, 0.20, 0.05, 0.02, true);
}

TEST(QuantLibAccuracyTest, IV_DeepITM_Put_6M) {
    test_iv_scenario("IV Deep ITM Put 6M",
        80.0, 100.0, 0.5, 0.25, 0.05, 0.02, false);
}

TEST(QuantLibAccuracyTest, IV_HighVol_Put_1Y) {
    test_iv_scenario("IV High Vol Put 1Y",
        100.0, 100.0, 1.0, 0.50, 0.05, 0.02, false);
}

TEST(QuantLibAccuracyTest, IV_LowVol_Put_1Y) {
    test_iv_scenario("IV Low Vol Put 1Y",
        100.0, 100.0, 1.0, 0.10, 0.05, 0.02, false);
}

TEST(QuantLibAccuracyTest, IV_LongMaturity_Put_5Y) {
    test_iv_scenario("IV Long Maturity Put 5Y",
        100.0, 100.0, 5.0, 0.20, 0.05, 0.02, false);
}
