// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/american_option.hpp"

namespace {

mango::PricingParams make_params() {
    return mango::PricingParams(
        mango::OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = mango::OptionType::PUT},
        0.20);
}

TEST(AmericanOptionAutoAPITest, AutoAPIProducesReasonableResult) {
    auto params = make_params();

    // Auto form uses thread-local PMR arena — no workspace parameter.
    auto solver = mango::AmericanOptionSolver::create(params).value();
    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    // Spot price at ATM put should be positive and below strike.
    double price = result->value_at(params.spot);
    EXPECT_GT(price, 0.0);
    EXPECT_LT(price, params.strike);

    // Delta of ATM put should be in (-1, 0).
    double delta = result->delta();
    EXPECT_LT(delta, 0.0);
    EXPECT_GT(delta, -1.0);

    // Off-spot evaluation exercises the spatial-operator pointer-aliasing
    // concern (variant init must not move solver objects).
    double price_otm = result->value_at(params.spot * 1.1);
    EXPECT_GE(price_otm, 0.0);
    EXPECT_LT(price_otm, price);  // OTM put cheaper than ATM
}

}  // namespace
