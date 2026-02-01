// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/option_spec.hpp"
#include <cmath>

using namespace mango;

// ===========================================================================
// validate_option_spec tests
// ===========================================================================

TEST(OptionSpecValidationTest, ValidSpecPasses) {
    OptionSpec spec;
    spec.spot = 100.0;
    spec.strike = 100.0;
    spec.maturity = 1.0;
    spec.rate = 0.05;
    spec.dividend_yield = 0.0;
    spec.option_type = OptionType::PUT;
    auto result = validate_option_spec(spec);
    EXPECT_TRUE(result.has_value());
}

TEST(OptionSpecValidationTest, NegativeSpot) {
    OptionSpec spec;
    spec.spot = -100.0;
    spec.strike = 100.0;
    spec.maturity = 1.0;
    spec.rate = 0.05;
    spec.dividend_yield = 0.0;
    spec.option_type = OptionType::PUT;
    auto result = validate_option_spec(spec);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ValidationErrorCode::InvalidSpotPrice);
}

TEST(OptionSpecValidationTest, ZeroSpot) {
    OptionSpec spec;
    spec.spot = 0.0;
    spec.strike = 100.0;
    spec.maturity = 1.0;
    spec.rate = 0.05;
    spec.dividend_yield = 0.0;
    spec.option_type = OptionType::PUT;
    auto result = validate_option_spec(spec);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ValidationErrorCode::InvalidSpotPrice);
}

TEST(OptionSpecValidationTest, NegativeStrike) {
    OptionSpec spec;
    spec.spot = 100.0;
    spec.strike = -100.0;
    spec.maturity = 1.0;
    spec.rate = 0.05;
    spec.dividend_yield = 0.0;
    spec.option_type = OptionType::PUT;
    auto result = validate_option_spec(spec);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ValidationErrorCode::InvalidStrike);
}

TEST(OptionSpecValidationTest, NegativeMaturity) {
    OptionSpec spec;
    spec.spot = 100.0;
    spec.strike = 100.0;
    spec.maturity = -1.0;
    spec.rate = 0.05;
    spec.dividend_yield = 0.0;
    spec.option_type = OptionType::PUT;
    auto result = validate_option_spec(spec);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ValidationErrorCode::InvalidMaturity);
}

TEST(OptionSpecValidationTest, NegativeRateAllowed) {
    OptionSpec spec;
    spec.spot = 100.0;
    spec.strike = 100.0;
    spec.maturity = 1.0;
    spec.rate = -0.01;
    spec.dividend_yield = 0.0;
    spec.option_type = OptionType::PUT;
    auto result = validate_option_spec(spec);
    EXPECT_TRUE(result.has_value());
}

TEST(OptionSpecValidationTest, NegativeDividendYield) {
    OptionSpec spec;
    spec.spot = 100.0;
    spec.strike = 100.0;
    spec.maturity = 1.0;
    spec.rate = 0.05;
    spec.dividend_yield = -0.01;
    spec.option_type = OptionType::PUT;
    auto result = validate_option_spec(spec);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ValidationErrorCode::InvalidDividend);
}

// ===========================================================================
// validate_iv_query tests
// ===========================================================================

TEST(IVQueryValidationTest, ValidQueryPasses) {
    IVQuery query(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 10.0);
    auto result = validate_iv_query(query);
    EXPECT_TRUE(result.has_value());
}

TEST(IVQueryValidationTest, NegativeMarketPrice) {
    IVQuery query(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, -5.0);
    auto result = validate_iv_query(query);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ValidationErrorCode::InvalidMarketPrice);
}

TEST(IVQueryValidationTest, ArbitrageCallExceedsSpot) {
    // Call price > spot is arbitrage
    IVQuery query(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::CALL}, 150.0);
    auto result = validate_iv_query(query);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ValidationErrorCode::InvalidMarketPrice);
}

TEST(IVQueryValidationTest, ArbitragePutExceedsStrike) {
    // Put price > strike is arbitrage
    IVQuery query(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 150.0);
    auto result = validate_iv_query(query);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ValidationErrorCode::InvalidMarketPrice);
}

// ===========================================================================
// RateSpec helpers
// ===========================================================================

TEST(RateSpecTest, ConstantRateIsNotYieldCurve) {
    RateSpec spec = 0.05;
    EXPECT_FALSE(is_yield_curve(spec));
}

TEST(RateSpecTest, ConstantRateFn) {
    RateSpec spec = 0.05;
    auto fn = make_rate_fn(spec, 1.0);
    // For constant rate, function returns 0.05 regardless of tau
    EXPECT_DOUBLE_EQ(fn(0.5), 0.05);
    EXPECT_DOUBLE_EQ(fn(0.0), 0.05);
    EXPECT_DOUBLE_EQ(fn(1.0), 0.05);
}

TEST(RateSpecTest, GetZeroRateConstant) {
    RateSpec spec = 0.05;
    double rate = get_zero_rate(spec, 1.0);
    EXPECT_DOUBLE_EQ(rate, 0.05);
}

TEST(RateSpecTest, ForwardDiscountConstantRate) {
    RateSpec spec = 0.05;
    double T = 1.0;
    auto fn = make_forward_discount_fn(spec, T);
    // For constant rate: forward discount = exp(-r * tau)
    EXPECT_NEAR(fn(0.0), 1.0, 1e-10);
    EXPECT_NEAR(fn(1.0), std::exp(-0.05), 1e-10);
    EXPECT_NEAR(fn(0.5), std::exp(-0.025), 1e-10);
}
