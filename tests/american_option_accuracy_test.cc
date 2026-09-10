// SPDX-License-Identifier: MIT
#include "mango/option/american_option.hpp"
#include <gtest/gtest.h>

using namespace mango;

// Regression: a stock-exhausting dividend retains discounted terminal strike.
// Bug: The dividend jump fallback replaced the continuation value with K.
TEST(AmericanOptionTest, NegativeRateDividendPutRetainsDiscountedStrike) {
    PricingParams params(
        OptionSpec{.spot = 1.0, .strike = 100.0, .maturity = 1.0,
                   .rate = -0.05, .dividend_yield = 0.0,
                   .option_type = OptionType::PUT},
        0.01, {{.calendar_time = 0.5, .amount = 5.0}});
    // Use the accuracy profile to separate the jump/boundary condition
    // from the default coarse grid's spatial truncation error.
    auto solver = AmericanOptionSolver::create(
        params, make_grid_accuracy(GridAccuracyProfile::High));
    ASSERT_TRUE(solver.has_value());
    auto result = solver->solve();
    ASSERT_TRUE(result.has_value());
    // The dividend exceeds spot by hundreds of standard deviations. The
    // absorbing-zero stock leaves a terminal payoff K, discounted at r<0.
    // A jump fallback of K at the ex-date loses half the discount growth.
    EXPECT_NEAR(result->value(), 105.1271096376024, 1e-3);
    const double left_spot = params.strike * std::exp(result->grid()->x().front());
    EXPECT_NEAR(result->value_at(left_spot), 105.1271096376024, 1e-3);
}

