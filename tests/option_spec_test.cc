// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/option_spec.hpp"
#include "mango/option/european_option.hpp"
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

// Regression: K is not an upper price bound for an American put when
// negative rates make the discounted strike exceed K.
// Bug: The put upper bound assumed the discount factor never exceeds one.
TEST(IVQueryValidationTest, MathReviewNegativeRatePutAboveStrikeIsValid) {
    for (const RateSpec& rate :
         {RateSpec{-0.05}, RateSpec{YieldCurve::flat(-0.05)}}) {
        SCOPED_TRACE(is_yield_curve(rate) ? "flat curve" : "scalar rate");
        const OptionSpec spec{
            .spot = 1.0, .strike = 100.0, .maturity = 1.0,
            .rate = rate, .dividend_yield = 0.0,
            .option_type = OptionType::PUT};
        // No early-exercise advantage at constant r<0 and q=0, so this
        // European price is also a legitimate American price (~$104.1271).
        const double price = EuropeanOptionResult(PricingParams(spec, 0.20)).value();
        ASSERT_GT(price, spec.strike);
        EXPECT_TRUE(validate_iv_query(IVQuery(spec, price)).has_value())
            << "A valid negative-rate price above strike must reach IV solving";

        // Adjusting the bound must still reject actual arbitrage.
        const double discounted_strike = spec.strike * std::exp(0.05 * spec.maturity);
        auto invalid = validate_iv_query(IVQuery(spec, discounted_strike + 1.0));
        ASSERT_FALSE(invalid.has_value());
        EXPECT_EQ(invalid.error().code, ValidationErrorCode::InvalidMarketPrice);
    }
}

// Regression: the put price bound includes all admitted exercise dates.
// Bug: An expiry-only discount can miss an interior discount maximum.
TEST(IVQueryValidationTest, PutBoundIncludesIntermediateExerciseDates) {
    // Discount grows to exp(0.1) at t=1, returns to 1 at expiry, and grows
    // again after expiry. Only stopping dates within this contract count.
    auto curve = YieldCurve::from_points({
        {0.0, 0.0}, {1.0, 0.1}, {2.0, 0.0}, {3.0, 0.5}});
    ASSERT_TRUE(curve.has_value());
    const OptionSpec spec{
        .spot = 1.0, .strike = 100.0, .maturity = 2.0,
        .rate = *curve, .option_type = OptionType::PUT};
    EXPECT_TRUE(validate_iv_query(IVQuery(spec, 109.0)).has_value());
    EXPECT_FALSE(validate_iv_query(IVQuery(spec, 111.0)).has_value());
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
