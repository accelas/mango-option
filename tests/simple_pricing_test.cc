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

// ===========================================================================
// price_batch() tests
// ===========================================================================

TEST(SimplePricingTest, PriceBatchMultipleStrikes) {
    // Chain-like batch: same maturity, varying strikes (eligible for normalized chain)
    std::vector<mango::PricingParams> batch;
    for (double K : {90.0, 95.0, 100.0, 105.0, 110.0}) {
        mango::PricingParams p;
        p.spot = 100.0;
        p.strike = K;
        p.maturity = 1.0;
        p.rate = 0.05;
        p.dividend_yield = 0.0;
        p.type = OptionType::PUT;
        p.volatility = 0.20;
        batch.push_back(p);
    }

    auto result = mango::simple::price_batch(batch);
    EXPECT_EQ(result.failed_count, 0u);
    ASSERT_EQ(result.prices.size(), 5u);

    for (auto& p : result.prices) {
        ASSERT_TRUE(p.has_value()) << p.error();
        EXPECT_GT(*p, 0.0);
    }

    // OTM puts (high strike) should be more expensive
    EXPECT_GT(*result.prices[4], *result.prices[0]);
}

TEST(SimplePricingTest, PriceBatchEmpty) {
    std::vector<mango::PricingParams> empty;
    auto result = mango::simple::price_batch(empty);
    EXPECT_EQ(result.prices.size(), 0u);
    EXPECT_EQ(result.failed_count, 0u);
}

// ===========================================================================
// implied_vol_batch() tests
// ===========================================================================

TEST(SimplePricingTest, ImpliedVolBatch) {
    // Price a few options first, then recover IV
    std::vector<mango::IVQuery> queries;
    for (double K : {95.0, 100.0, 105.0}) {
        auto price_result = mango::simple::price(
            100.0, K, 1.0, 0.25, 0.05, 0.0, OptionType::PUT);
        ASSERT_TRUE(price_result.has_value()) << price_result.error();

        mango::IVQuery q;
        q.spot = 100.0;
        q.strike = K;
        q.maturity = 1.0;
        q.rate = 0.05;
        q.dividend_yield = 0.0;
        q.type = OptionType::PUT;
        q.market_price = *price_result;
        queries.push_back(q);
    }

    auto result = mango::simple::implied_vol_batch(queries);
    EXPECT_EQ(result.failed_count, 0u);
    ASSERT_EQ(result.vols.size(), 3u);

    for (auto& v : result.vols) {
        ASSERT_TRUE(v.has_value()) << v.error();
        EXPECT_NEAR(*v, 0.25, 1e-3);
    }
}

}  // anonymous namespace
