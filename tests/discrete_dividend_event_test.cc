// SPDX-License-Identifier: MIT
/**
 * @file discrete_dividend_event_test.cc
 * @brief Tests for discrete dividend handling in AmericanOptionSolver
 *
 * Verifies that discrete dividends are correctly applied during PDE solve
 * by testing observable effects on option prices.
 */
#include <gtest/gtest.h>
#include "mango/option/american_option.hpp"
#include <cmath>

using namespace mango;

namespace {

// Helper: solve an American option with the given params
std::expected<AmericanOptionResult, SolverError> solve(const PricingParams& params) {
    return solve_american_option(params);
}

PricingParams make_put(double spot, double strike, double maturity,
                       double vol, double rate, double div_yield,
                       std::vector<Dividend> dividends = {}) {
    PricingParams p(
        OptionSpec{.spot = spot, .strike = strike, .maturity = maturity,
                   .rate = rate, .dividend_yield = div_yield,
                   .option_type = OptionType::PUT},
        vol);
    p.discrete_dividends = std::move(dividends);
    return p;
}

PricingParams make_call(double spot, double strike, double maturity,
                        double vol, double rate, double div_yield,
                        std::vector<Dividend> dividends = {}) {
    PricingParams p(
        OptionSpec{.spot = spot, .strike = strike, .maturity = maturity,
                   .rate = rate, .dividend_yield = div_yield,
                   .option_type = OptionType::CALL},
        vol);
    p.discrete_dividends = std::move(dividends);
    return p;
}

}  // namespace

TEST(DiscreteDividendTest, PutValueIncreasesWithDividend) {
    // A discrete dividend lowers the effective spot, increasing put value
    auto no_div = solve(make_put(100, 100, 1.0, 0.20, 0.05, 0.0));
    auto with_div = solve(make_put(100, 100, 1.0, 0.20, 0.05, 0.0,
        {Dividend{.calendar_time = 0.25, .amount = 5.0}}));

    ASSERT_TRUE(no_div.has_value());
    ASSERT_TRUE(with_div.has_value());
    EXPECT_GT(with_div->value(), no_div->value())
        << "Put value should increase when a discrete dividend is present";
}

TEST(DiscreteDividendTest, CallValueDecreasesWithDividend) {
    // A discrete dividend lowers the effective spot, decreasing call value
    auto no_div = solve(make_call(100, 100, 1.0, 0.20, 0.05, 0.0));
    auto with_div = solve(make_call(100, 100, 1.0, 0.20, 0.05, 0.0,
        {Dividend{.calendar_time = 0.25, .amount = 5.0}}));

    ASSERT_TRUE(no_div.has_value());
    ASSERT_TRUE(with_div.has_value());
    EXPECT_LT(with_div->value(), no_div->value())
        << "Call value should decrease when a discrete dividend is present";
}

TEST(DiscreteDividendTest, ZeroDividendMatchesNoDividend) {
    auto no_div = solve(make_put(100, 100, 1.0, 0.20, 0.05, 0.0));
    auto zero_div = solve(make_put(100, 100, 1.0, 0.20, 0.05, 0.0,
        {Dividend{.calendar_time = 0.25, .amount = 0.0}}));

    ASSERT_TRUE(no_div.has_value());
    ASSERT_TRUE(zero_div.has_value());
    EXPECT_NEAR(zero_div->value(), no_div->value(), 1e-10)
        << "Zero dividend should produce same price as no dividend";
}

TEST(DiscreteDividendTest, LargerDividendIncreasesDeepITMPut) {
    // Deep ITM put with large dividend should push price closer to intrinsic
    auto small_div = solve(make_put(80, 100, 1.0, 0.20, 0.05, 0.0,
        {Dividend{.calendar_time = 0.5, .amount = 2.0}}));
    auto large_div = solve(make_put(80, 100, 1.0, 0.20, 0.05, 0.0,
        {Dividend{.calendar_time = 0.5, .amount = 10.0}}));

    ASSERT_TRUE(small_div.has_value());
    ASSERT_TRUE(large_div.has_value());
    EXPECT_GT(large_div->value(), small_div->value())
        << "Larger dividend should increase put value further";
}
