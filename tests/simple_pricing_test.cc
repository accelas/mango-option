// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/simple/pricing.hpp"
#include <cmath>

namespace {

using mango::OptionType;

// ===========================================================================
// price() tests
// ===========================================================================

TEST(SimplePricingTest, ATMPut) {
    auto result = mango::simple::price(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.20,   // volatility
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::PUT);

    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_GT(*result, 0.0);
    // ATM put with these params should be roughly 6-10
    EXPECT_GT(*result, 3.0);
    EXPECT_LT(*result, 15.0);
}

TEST(SimplePricingTest, ITMCall) {
    auto result = mango::simple::price(
        110.0,  // spot (ITM for call)
        100.0,  // strike
        1.0,    // maturity
        0.20,   // volatility
        0.05,   // rate
        0.0,    // no dividends
        OptionType::CALL);

    ASSERT_TRUE(result.has_value()) << result.error();
    // ITM call must be at least intrinsic value
    EXPECT_GT(*result, 10.0);
}

TEST(SimplePricingTest, DeepOTMPut) {
    auto result = mango::simple::price(
        200.0,  // spot (deep OTM for put)
        100.0,  // strike
        0.25,   // short maturity
        0.20,   // volatility
        0.05,   // rate
        0.0);

    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_NEAR(*result, 0.0, 0.1);
}

TEST(SimplePricingTest, InvalidSpotReturnsError) {
    auto result = mango::simple::price(
        -100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_FALSE(result.has_value());
}

TEST(SimplePricingTest, InvalidMaturityReturnsError) {
    auto result = mango::simple::price(
        100.0, 100.0, -1.0, 0.20, 0.05);
    EXPECT_FALSE(result.has_value());
}

TEST(SimplePricingTest, InvalidVolatilityReturnsError) {
    auto result = mango::simple::price(
        100.0, 100.0, 1.0, -0.20, 0.05);
    EXPECT_FALSE(result.has_value());
}

// ===========================================================================
// implied_vol() tests
// ===========================================================================

TEST(SimplePricingTest, ImpliedVolRoundtrip) {
    double spot = 100.0;
    double strike = 100.0;
    double maturity = 1.0;
    double vol = 0.25;
    double rate = 0.05;
    double div = 0.02;

    // First price with known vol
    auto price_result = mango::simple::price(
        spot, strike, maturity, vol, rate, div, OptionType::PUT);
    ASSERT_TRUE(price_result.has_value()) << price_result.error();

    // Then recover vol from price
    auto iv_result = mango::simple::implied_vol(
        spot, strike, maturity, *price_result, rate, div, OptionType::PUT);
    ASSERT_TRUE(iv_result.has_value()) << iv_result.error();

    // Should match original vol within solver tolerance
    EXPECT_NEAR(*iv_result, vol, 1e-3);
}

TEST(SimplePricingTest, ImpliedVolInvalidPriceReturnsError) {
    auto result = mango::simple::implied_vol(
        100.0, 100.0, 1.0, -5.0, 0.05);
    EXPECT_FALSE(result.has_value());
}

}  // anonymous namespace
