#include <gtest/gtest.h>
#include <cmath>

extern "C" {
#include "../src/lets_be_rational.h"
}

class LetsBeRationalTest : public ::testing::Test {};

// Test 1: ATM call option
TEST_F(LetsBeRationalTest, ATMCall) {
    double spot = 100.0;
    double strike = 100.0;
    double time_to_maturity = 1.0;
    double risk_free_rate = 0.05;
    double market_price = 10.45;  // Corresponds to σ≈0.20
    bool is_call = true;

    LBRResult result = lbr_implied_volatility(spot, strike, time_to_maturity,
                                               risk_free_rate, market_price, is_call);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, 0.20, 0.01);  // Within 1% of 20%
}

// Test 2: OTM put option
TEST_F(LetsBeRationalTest, OTMPut) {
    double spot = 100.0;
    double strike = 95.0;
    double time_to_maturity = 0.5;
    double risk_free_rate = 0.03;
    double market_price = 2.5;
    bool is_call = false;

    LBRResult result = lbr_implied_volatility(spot, strike, time_to_maturity,
                                               risk_free_rate, market_price, is_call);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.05);  // Reasonable vol > 5%
    EXPECT_LT(result.implied_vol, 1.0);   // Reasonable vol < 100%
}

// Test 3: Invalid inputs
TEST_F(LetsBeRationalTest, InvalidInputs) {
    LBRResult result = lbr_implied_volatility(-100.0, 100.0, 1.0, 0.05, 10.0, true);
    EXPECT_FALSE(result.converged);
    EXPECT_NE(result.error, nullptr);
}

// Test 4: Near expiry
TEST_F(LetsBeRationalTest, NearExpiry) {
    double spot = 100.0;
    double strike = 100.0;
    double time_to_maturity = 0.027;  // ~1 week
    double risk_free_rate = 0.05;
    double market_price = 2.0;
    bool is_call = true;

    LBRResult result = lbr_implied_volatility(spot, strike, time_to_maturity,
                                               risk_free_rate, market_price, is_call);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.0);
}
