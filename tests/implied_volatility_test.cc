#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" {
#include "../src/implied_volatility.h"
}

// Test fixture for implied volatility tests
class ImpliedVolatilityTest : public ::testing::Test {
protected:
    static constexpr double tolerance = 1e-6;
};

// Test Black-Scholes pricing directly
TEST_F(ImpliedVolatilityTest, BlackScholesPricing) {
    // Known Black-Scholes values
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    // ATM call option
    double call_price = black_scholes_price(spot, strike, T, r, sigma, true);
    EXPECT_NEAR(call_price, 10.4506, 0.001);  // Known value

    // ATM put option (put-call parity)
    double put_price = black_scholes_price(spot, strike, T, r, sigma, false);
    EXPECT_NEAR(put_price, 5.5735, 0.001);  // Known value

    // Verify put-call parity: C - P = S - K*e^(-rT)
    double parity_diff = call_price - put_price;
    double expected_diff = spot - strike * std::exp(-r * T);
    EXPECT_NEAR(parity_diff, expected_diff, 1e-6);
}

// Test vega calculation
TEST_F(ImpliedVolatilityTest, VegaCalculation) {
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double vega = black_scholes_vega(spot, strike, T, r, sigma);

    // ATM vega should be positive and reasonable
    EXPECT_GT(vega, 0.0);
    EXPECT_LT(vega, spot);  // Vega shouldn't exceed spot price

    // Known value for these parameters
    EXPECT_NEAR(vega, 37.524, 0.01);
}

// Test vega symmetry (same for calls and puts)
TEST_F(ImpliedVolatilityTest, VegaSymmetry) {
    double spot = 100.0;
    double strike = 105.0;
    double T = 0.5;
    double r = 0.03;
    double sigma = 0.25;

    double vega = black_scholes_vega(spot, strike, T, r, sigma);

    // Verify vega by finite difference
    double h = 1e-5;
    double price_up = black_scholes_price(spot, strike, T, r, sigma + h, true);
    double price_down = black_scholes_price(spot, strike, T, r, sigma - h, true);
    double vega_numeric = (price_up - price_down) / (2.0 * h);

    EXPECT_NEAR(vega, vega_numeric, 1e-3);
}

// Test ATM call option IV
TEST_F(ImpliedVolatilityTest, ATMCallOption) {
    double sigma_true = 0.2;  // True volatility
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;

    // Calculate theoretical price
    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, true);

    // Calculate IV from market price
    IVParams params = {spot, strike, T, r, market_price, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, tolerance);
    EXPECT_GT(result.vega, 0.0);
    EXPECT_LT(result.iterations, 20);
    EXPECT_EQ(result.error, nullptr);
}

// Test ATM put option IV
TEST_F(ImpliedVolatilityTest, ATMPutOption) {
    double sigma_true = 0.25;
    double spot = 100.0;
    double strike = 100.0;
    double T = 0.5;
    double r = 0.03;

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, false);

    IVParams params = {spot, strike, T, r, market_price, false};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, tolerance);
}

// Test OTM call option (strike > spot)
TEST_F(ImpliedVolatilityTest, OTMCallOption) {
    double sigma_true = 0.3;
    double spot = 100.0;
    double strike = 110.0;  // OTM call
    double T = 1.0;
    double r = 0.05;

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, true);

    IVParams params = {spot, strike, T, r, market_price, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, tolerance);
}

// Test ITM call option (strike < spot)
TEST_F(ImpliedVolatilityTest, ITMCallOption) {
    double sigma_true = 0.2;
    double spot = 100.0;
    double strike = 90.0;  // ITM call
    double T = 1.0;
    double r = 0.05;

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, true);

    IVParams params = {spot, strike, T, r, market_price, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, tolerance);
}

// Test OTM put option (strike < spot)
TEST_F(ImpliedVolatilityTest, OTMPutOption) {
    double sigma_true = 0.28;
    double spot = 100.0;
    double strike = 90.0;  // OTM put
    double T = 0.5;
    double r = 0.04;

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, false);

    IVParams params = {spot, strike, T, r, market_price, false};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, tolerance);
}

// Test ITM put option (strike > spot)
TEST_F(ImpliedVolatilityTest, ITMPutOption) {
    double sigma_true = 0.22;
    double spot = 100.0;
    double strike = 110.0;  // ITM put
    double T = 1.0;
    double r = 0.05;

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, false);

    IVParams params = {spot, strike, T, r, market_price, false};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, tolerance);
}

// Test short maturity
TEST_F(ImpliedVolatilityTest, ShortMaturity) {
    double sigma_true = 0.25;
    double spot = 100.0;
    double strike = 100.0;
    double T = 0.05;  // ~18 days
    double r = 0.05;

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, true);

    IVParams params = {spot, strike, T, r, market_price, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, tolerance);
}

// Test long maturity
TEST_F(ImpliedVolatilityTest, LongMaturity) {
    double sigma_true = 0.2;
    double spot = 100.0;
    double strike = 100.0;
    double T = 5.0;  // 5 years
    double r = 0.05;

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, true);

    IVParams params = {spot, strike, T, r, market_price, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, tolerance);
}

// Test low volatility
TEST_F(ImpliedVolatilityTest, LowVolatility) {
    double sigma_true = 0.05;  // 5% vol
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, true);

    IVParams params = {spot, strike, T, r, market_price, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, tolerance);
}

// Test high volatility
TEST_F(ImpliedVolatilityTest, HighVolatility) {
    double sigma_true = 1.5;  // 150% vol
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, true);

    IVParams params = {spot, strike, T, r, market_price, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, tolerance);
}

// Test very high volatility (stress test)
TEST_F(ImpliedVolatilityTest, VeryHighVolatility) {
    double sigma_true = 3.0;  // 300% vol
    double spot = 100.0;
    double strike = 100.0;
    double T = 0.5;
    double r = 0.05;

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, true);

    IVParams params = {spot, strike, T, r, market_price, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, 1e-4);  // Slightly relaxed tolerance
}

// Test zero interest rate
TEST_F(ImpliedVolatilityTest, ZeroInterestRate) {
    double sigma_true = 0.25;
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.0;

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, true);

    IVParams params = {spot, strike, T, r, market_price, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, tolerance);
}

// Test negative interest rate
TEST_F(ImpliedVolatilityTest, NegativeInterestRate) {
    double sigma_true = 0.2;
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = -0.01;  // Negative rate

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, true);

    IVParams params = {spot, strike, T, r, market_price, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, tolerance);
}

// Test custom tolerance
TEST_F(ImpliedVolatilityTest, CustomTolerance) {
    double sigma_true = 0.2;
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, true);

    IVParams params = {spot, strike, T, r, market_price, true};
    IVResult result = implied_volatility_calculate(&params, 0.01, 5.0, 1e-10, 100);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, 1e-10);
}

// Error case: negative spot price
TEST_F(ImpliedVolatilityTest, NegativeSpotPrice) {
    IVParams params = {-100.0, 100.0, 1.0, 0.05, 10.0, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_FALSE(result.converged);
    EXPECT_NE(result.error, nullptr);
}

// Error case: negative strike
TEST_F(ImpliedVolatilityTest, NegativeStrike) {
    IVParams params = {100.0, -100.0, 1.0, 0.05, 10.0, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_FALSE(result.converged);
    EXPECT_NE(result.error, nullptr);
}

// Error case: negative time to maturity
TEST_F(ImpliedVolatilityTest, NegativeTimeToMaturity) {
    IVParams params = {100.0, 100.0, -1.0, 0.05, 10.0, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_FALSE(result.converged);
    EXPECT_NE(result.error, nullptr);
}

// Error case: negative market price
TEST_F(ImpliedVolatilityTest, NegativeMarketPrice) {
    IVParams params = {100.0, 100.0, 1.0, 0.05, -10.0, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_FALSE(result.converged);
    EXPECT_NE(result.error, nullptr);
}

// Error case: call price exceeds spot (arbitrage)
TEST_F(ImpliedVolatilityTest, CallArbitrage) {
    IVParams params = {100.0, 90.0, 1.0, 0.05, 110.0, true};  // Price > spot
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_FALSE(result.converged);
    EXPECT_NE(result.error, nullptr);
}

// Error case: put price exceeds discounted strike (arbitrage)
TEST_F(ImpliedVolatilityTest, PutArbitrage) {
    IVParams params = {100.0, 100.0, 1.0, 0.05, 100.0, false};  // Price > K*e^(-rT)
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_FALSE(result.converged);
    EXPECT_NE(result.error, nullptr);
}

// Error case: price below intrinsic value
TEST_F(ImpliedVolatilityTest, PriceBelowIntrinsic) {
    IVParams params = {100.0, 90.0, 1.0, 0.05, 5.0, true};  // Deep ITM call with too low price
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_FALSE(result.converged);
    EXPECT_NE(result.error, nullptr);
}

// Stress test: Deep OTM call
TEST_F(ImpliedVolatilityTest, DeepOTMCall) {
    double sigma_true = 0.25;
    double spot = 100.0;
    double strike = 150.0;  // Very OTM
    double T = 0.25;
    double r = 0.05;

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, true);

    IVParams params = {spot, strike, T, r, market_price, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, tolerance);
}

// Stress test: Deep ITM put
TEST_F(ImpliedVolatilityTest, DeepITMPut) {
    double sigma_true = 0.3;
    double spot = 100.0;
    double strike = 150.0;  // Very ITM put
    double T = 1.0;
    double r = 0.05;

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, false);

    IVParams params = {spot, strike, T, r, market_price, false};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, tolerance);
}

// Stress test: Very short maturity with high volatility
TEST_F(ImpliedVolatilityTest, ShortMaturityHighVol) {
    double sigma_true = 1.0;
    double spot = 100.0;
    double strike = 100.0;
    double T = 0.01;  // ~4 days
    double r = 0.05;

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, true);

    IVParams params = {spot, strike, T, r, market_price, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, 1e-4);
}

// Stress test: Extreme moneyness
TEST_F(ImpliedVolatilityTest, ExtremeMoneyness) {
    double sigma_true = 0.4;
    double spot = 100.0;
    double strike = 200.0;  // Extreme OTM
    double T = 2.0;
    double r = 0.05;

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, true);

    IVParams params = {spot, strike, T, r, market_price, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, tolerance);
}

// Test convergence consistency across multiple runs
TEST_F(ImpliedVolatilityTest, ConvergenceConsistency) {
    double sigma_true = 0.25;
    double spot = 100.0;
    double strike = 105.0;
    double T = 0.5;
    double r = 0.04;

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, true);

    IVParams params = {spot, strike, T, r, market_price, true};

    // Run multiple times and verify consistent results
    std::vector<double> results;
    for (int i = 0; i < 10; i++) {
        IVResult result = implied_volatility_calculate_simple(&params);
        EXPECT_TRUE(result.converged);
        results.push_back(result.implied_vol);
    }

    // All results should be identical
    for (size_t i = 1; i < results.size(); i++) {
        EXPECT_DOUBLE_EQ(results[0], results[i]);
    }
}

// Test edge case: Zero time to maturity
TEST_F(ImpliedVolatilityTest, ZeroTimeToMaturity) {
    IVParams params = {100.0, 95.0, 0.0, 0.05, 5.0, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    // Should fail gracefully
    EXPECT_FALSE(result.converged);
    EXPECT_NE(result.error, nullptr);
}

// Numerical stability: Very small price
TEST_F(ImpliedVolatilityTest, VerySmallPrice) {
    double sigma_true = 0.15;
    double spot = 100.0;
    double strike = 120.0;
    double T = 0.1;
    double r = 0.05;

    double market_price = black_scholes_price(spot, strike, T, r, sigma_true, true);
    // This should be a very small price

    IVParams params = {spot, strike, T, r, market_price, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, sigma_true, tolerance);
}

// Numerical stability: Price near intrinsic value
TEST_F(ImpliedVolatilityTest, PriceNearIntrinsic) {
    double spot = 100.0;
    double strike = 90.0;
    double T = 1.0;
    double r = 0.05;

    // Price just above intrinsic value (implies very low vol)
    double intrinsic = spot - strike * std::exp(-r * T);
    double market_price = intrinsic + 0.01;

    IVParams params = {spot, strike, T, r, market_price, true};
    IVResult result = implied_volatility_calculate_simple(&params);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.0);
    EXPECT_LT(result.implied_vol, 0.1);  // Should be low volatility
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
