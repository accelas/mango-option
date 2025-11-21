/**
 * @file quantlib_accuracy_test.cc
 * @brief Accuracy validation against QuantLib reference implementation
 *
 * This test ensures numerical accuracy doesn't regress by comparing
 * mango-iv American option prices against QuantLib.
 *
 * Key scenarios tested:
 * - ATM/ITM/OTM options
 * - Various maturities (3M to 5Y)
 * - Different volatility regimes (10% to 50%)
 * - Greeks accuracy (delta, gamma, theta)
 *
 * Tolerance: 1% relative error (protects against 14.5% regression like #204)
 */

#include "src/option/american_option.hpp"
#include <gtest/gtest.h>
#include <cmath>

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

    // Mango-IV pricing
    AmericanOptionParams mango_params(
        spot, strike, maturity, rate, dividend_yield,
        is_call ? OptionType::CALL : OptionType::PUT, volatility);

    // Create grid and workspace
    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0);
    ASSERT_TRUE(grid_spec.has_value());

    size_t n_time = 2000;
    auto workspace_result = AmericanSolverWorkspace::create(
        grid_spec.value(), n_time, std::pmr::get_default_resource());
    ASSERT_TRUE(workspace_result.has_value()) << workspace_result.error();
    auto workspace = workspace_result.value();

    AmericanOptionSolver solver(mango_params, workspace->workspace_spans());
    auto mango_result = solver.solve();
    ASSERT_TRUE(mango_result.has_value()) << mango_result.error().message;

    // Verify full surface was stored
    ASSERT_FALSE(mango_result->surface_2d.empty())
        << "Full surface should be automatically stored by solver";

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

    // Compute and check Greeks
    auto greeks_result = solver_result.value().compute_greeks();
    ASSERT_TRUE(greeks_result.has_value()) << greeks_result.error().message;
    const auto& greeks = greeks_result.value();

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

    AmericanOptionParams params(
        100.0, 100.0, 1.0, 0.05, 0.02, OptionType::PUT, 0.20);

    // Create grid and workspace
    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0);
    ASSERT_TRUE(grid_spec.has_value());

    size_t n_time = 2000;
    auto workspace_result = AmericanSolverWorkspace::create(
        grid_spec.value(), n_time, std::pmr::get_default_resource());
    ASSERT_TRUE(workspace_result.has_value());
    auto workspace = workspace_result.value();

    AmericanOptionSolver solver(params, workspace->workspace_spans());
    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    double mango_price = result->value_at(params.spot);
    double error = std::abs(mango_price - ql_reference.price);
    double rel_error = (error / ql_reference.price) * 100.0;

    // Should converge to within 0.1% of high-resolution reference
    EXPECT_LT(rel_error, 0.1)
        << "Convergence test failed"
        << "\n  Mango:     $" << mango_price
        << "\n  Reference: $" << ql_reference.price
        << "\n  Grid:      " << 201 << "x" << n_time;
}

// ============================================================================
// Greeks Accuracy
// ============================================================================

TEST(QuantLibAccuracyTest, Greeks_ATM) {
    AmericanOptionParams params(
        100.0, 100.0, 1.0, 0.05, 0.02, OptionType::PUT, 0.20);

    // Create grid and workspace
    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0);
    ASSERT_TRUE(grid_spec.has_value());

    size_t n_time = 2000;
    auto workspace_result = AmericanSolverWorkspace::create(
        grid_spec.value(), n_time, std::pmr::get_default_resource());
    ASSERT_TRUE(workspace_result.has_value());
    auto workspace = workspace_result.value();

    AmericanOptionSolver solver(params, workspace->workspace_spans());
    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    // Greeks are available directly from result
    double delta_val = result->delta();
    double gamma_val = result->gamma();
    double theta_val = result->theta();

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
