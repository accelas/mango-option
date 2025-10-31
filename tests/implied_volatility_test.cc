#include <gtest/gtest.h>
#include <cmath>

extern "C" {
#include "../src/implied_volatility.h"
#include "../src/american_option.h"
}

// Test fixture for American option implied volatility tests
class ImpliedVolatilityTest : public ::testing::Test {
protected:
    AmericanOptionGrid default_grid;

    void SetUp() override {
        default_grid.x_min = -0.7;
        default_grid.x_max = 0.7;
        default_grid.n_points = 141;
        default_grid.dt = 0.001;
        default_grid.n_steps = 1000;
    }
};

// Test ATM American put IV
TEST_F(ImpliedVolatilityTest, ATMPutIV) {
    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .market_price = 6.08,  // Typical American put price
        .is_call = false
    };

    IVResult result = calculate_iv(&params, &default_grid, 1e-6, 100);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.15);  // Reasonable range
    EXPECT_LT(result.implied_vol, 0.35);
    EXPECT_LT(result.iterations, 20);      // Should converge quickly
    EXPECT_EQ(result.error, nullptr);
}

// Test OTM American call IV
TEST_F(ImpliedVolatilityTest, OTMCallIV) {
    IVParams params = {
        .spot_price = 100.0,
        .strike = 110.0,
        .time_to_maturity = 0.5,
        .risk_free_rate = 0.03,
        .market_price = 3.0,
        .is_call = true
    };

    IVResult result = calculate_iv(&params, &default_grid, 1e-6, 100);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.10);
    EXPECT_LT(result.implied_vol, 0.50);
    EXPECT_EQ(result.error, nullptr);
}

// Test ITM American put IV
TEST_F(ImpliedVolatilityTest, ITMPutIV) {
    IVParams params = {
        .spot_price = 100.0,
        .strike = 110.0,
        .time_to_maturity = 0.25,
        .risk_free_rate = 0.05,
        .market_price = 11.0,  // ITM put
        .is_call = false
    };

    IVResult result = calculate_iv(&params, &default_grid, 1e-6, 100);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.05);
    EXPECT_LT(result.implied_vol, 0.60);
    EXPECT_EQ(result.error, nullptr);
}

// Test convenience function
TEST_F(ImpliedVolatilityTest, SimpleCalculation) {
    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .market_price = 6.0,
        .is_call = false
    };

    IVResult result = calculate_iv_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.0);
    EXPECT_LT(result.implied_vol, 1.0);
}

// Test invalid inputs
TEST_F(ImpliedVolatilityTest, InvalidSpot) {
    IVParams params = {
        .spot_price = -100.0,  // Invalid
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .market_price = 10.0,
        .is_call = true
    };

    IVResult result = calculate_iv(&params, &default_grid, 1e-6, 100);

    EXPECT_FALSE(result.converged);
    EXPECT_NE(result.error, nullptr);
}

// Test arbitrage violation (price below intrinsic)
TEST_F(ImpliedVolatilityTest, BelowIntrinsicValue) {
    IVParams params = {
        .spot_price = 100.0,
        .strike = 110.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .market_price = 5.0,  // Below intrinsic (10)
        .is_call = false
    };

    IVResult result = calculate_iv(&params, &default_grid, 1e-6, 100);

    EXPECT_FALSE(result.converged);
    EXPECT_NE(result.error, nullptr);
}

// Test short maturity
TEST_F(ImpliedVolatilityTest, ShortMaturity) {
    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 0.027,  // ~1 week
        .risk_free_rate = 0.05,
        .market_price = 2.0,
        .is_call = false
    };

    IVResult result = calculate_iv(&params, &default_grid, 1e-6, 100);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.0);
}

// Test consistency: reverse-engineer IV
TEST_F(ImpliedVolatilityTest, RoundTripConsistency) {
    // Start with known volatility, compute price, then recover IV
    double true_vol = 0.30;

    OptionData option = {
        .strike = 100.0,
        .volatility = true_vol,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = NULL,
        .dividend_amounts = NULL
    };

    // Compute American option price with known vol
    AmericanOptionResult price_result = american_option_price(&option, &default_grid);
    ASSERT_EQ(price_result.status, 0);

    double market_price = american_option_get_value_at_spot(
        price_result.solver, 100.0, 100.0);

    american_option_free_result(&price_result);

    // Now recover IV from that price
    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .market_price = market_price,
        .is_call = false
    };

    IVResult iv_result = calculate_iv(&params, &default_grid, 1e-6, 100);

    EXPECT_TRUE(iv_result.converged);
    // Should recover original volatility within tolerance
    EXPECT_NEAR(iv_result.implied_vol, true_vol, 0.01);  // Within 1%
}
