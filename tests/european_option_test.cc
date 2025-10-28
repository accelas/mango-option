#include <gtest/gtest.h>
#include <cmath>

extern "C" {
#include "../src/european_option.h"
}

// Test fixture for European option tests
class EuropeanOptionTest : public ::testing::Test {
protected:
    static constexpr double tolerance = 1e-6;
    static constexpr double loose_tolerance = 1e-4;
};

// ============================================================================
// Black-Scholes Pricing Tests
// ============================================================================

TEST_F(EuropeanOptionTest, ATMCallPricing) {
    // Test ATM call option with known Black-Scholes value
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double call_price = black_scholes_price(spot, strike, T, r, sigma, true);

    // Known value from Black-Scholes formula
    EXPECT_NEAR(call_price, 10.4506, 0.001);
}

TEST_F(EuropeanOptionTest, ATMPutPricing) {
    // Test ATM put option with known Black-Scholes value
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double put_price = black_scholes_price(spot, strike, T, r, sigma, false);

    // Known value from Black-Scholes formula
    EXPECT_NEAR(put_price, 5.5735, 0.001);
}

TEST_F(EuropeanOptionTest, ITMCallPricing) {
    // In-the-money call option
    double spot = 110.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double call_price = black_scholes_price(spot, strike, T, r, sigma, true);

    // ITM call should be worth at least intrinsic value
    double intrinsic_value = spot - strike;
    EXPECT_GT(call_price, intrinsic_value);

    // Known value
    EXPECT_NEAR(call_price, 16.7246, 0.001);
}

TEST_F(EuropeanOptionTest, OTMPutPricing) {
    // Out-of-the-money put option
    double spot = 110.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double put_price = black_scholes_price(spot, strike, T, r, sigma, false);

    // OTM put should be less than strike
    EXPECT_LT(put_price, strike);
    EXPECT_GT(put_price, 0.0);

    // Known value
    EXPECT_NEAR(put_price, 1.8478, 0.001);
}

TEST_F(EuropeanOptionTest, PutCallParity) {
    // Verify put-call parity: C - P = S - K*e^(-rT)
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double call_price = black_scholes_price(spot, strike, T, r, sigma, true);
    double put_price = black_scholes_price(spot, strike, T, r, sigma, false);

    double parity_lhs = call_price - put_price;
    double parity_rhs = spot - strike * std::exp(-r * T);

    EXPECT_NEAR(parity_lhs, parity_rhs, tolerance);
}

TEST_F(EuropeanOptionTest, PutCallParityOTM) {
    // Verify put-call parity for OTM options
    double spot = 90.0;
    double strike = 100.0;
    double T = 0.5;
    double r = 0.03;
    double sigma = 0.25;

    double call_price = black_scholes_price(spot, strike, T, r, sigma, true);
    double put_price = black_scholes_price(spot, strike, T, r, sigma, false);

    double parity_lhs = call_price - put_price;
    double parity_rhs = spot - strike * std::exp(-r * T);

    EXPECT_NEAR(parity_lhs, parity_rhs, tolerance);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(EuropeanOptionTest, ZeroTimeToMaturity) {
    double spot = 110.0;
    double strike = 100.0;
    double T = 0.0;
    double r = 0.05;
    double sigma = 0.2;

    // At expiration, call value is max(S - K, 0)
    double call_price = black_scholes_price(spot, strike, T, r, sigma, true);
    EXPECT_NEAR(call_price, spot - strike, tolerance);

    // At expiration, put value is max(K - S, 0)
    double put_price = black_scholes_price(spot, strike, T, r, sigma, false);
    EXPECT_NEAR(put_price, 0.0, tolerance);
}

TEST_F(EuropeanOptionTest, ZeroTimeOTMOptions) {
    double spot = 90.0;
    double strike = 100.0;
    double T = 0.0;
    double r = 0.05;
    double sigma = 0.2;

    // OTM call at expiration is worthless
    double call_price = black_scholes_price(spot, strike, T, r, sigma, true);
    EXPECT_NEAR(call_price, 0.0, tolerance);

    // ITM put at expiration is worth intrinsic value
    double put_price = black_scholes_price(spot, strike, T, r, sigma, false);
    EXPECT_NEAR(put_price, strike - spot, tolerance);
}

TEST_F(EuropeanOptionTest, ZeroVolatility) {
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.0;

    double call_price = black_scholes_price(spot, strike, T, r, sigma, true);
    double put_price = black_scholes_price(spot, strike, T, r, sigma, false);

    EXPECT_NEAR(call_price, 0.0, tolerance);
    EXPECT_NEAR(put_price, 0.0, tolerance);
}

TEST_F(EuropeanOptionTest, VeryHighVolatility) {
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 5.0;  // 500% volatility

    double call_price = black_scholes_price(spot, strike, T, r, sigma, true);

    // With very high vol, call should be worth close to spot
    EXPECT_GT(call_price, 0.0);
    EXPECT_LT(call_price, spot);
}

// ============================================================================
// Greeks: Delta Tests
// ============================================================================

TEST_F(EuropeanOptionTest, CallDeltaRange) {
    // Call delta should be between 0 and 1
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double delta = black_scholes_delta(spot, strike, T, r, sigma, true);

    EXPECT_GT(delta, 0.0);
    EXPECT_LT(delta, 1.0);
}

TEST_F(EuropeanOptionTest, PutDeltaRange) {
    // Put delta should be between -1 and 0
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double delta = black_scholes_delta(spot, strike, T, r, sigma, false);

    EXPECT_LT(delta, 0.0);
    EXPECT_GT(delta, -1.0);
}

TEST_F(EuropeanOptionTest, ATMDelta) {
    // ATM options should have delta around 0.5 for calls, -0.5 for puts
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double call_delta = black_scholes_delta(spot, strike, T, r, sigma, true);
    double put_delta = black_scholes_delta(spot, strike, T, r, sigma, false);

    EXPECT_NEAR(call_delta, 0.5, 0.1);  // Approximately 0.5
    EXPECT_NEAR(put_delta, -0.5, 0.1);  // Approximately -0.5
}

TEST_F(EuropeanOptionTest, DeepITMCallDelta) {
    // Deep ITM call should have delta close to 1
    double spot = 150.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double delta = black_scholes_delta(spot, strike, T, r, sigma, true);

    EXPECT_GT(delta, 0.9);
    EXPECT_LE(delta, 1.0);
}

TEST_F(EuropeanOptionTest, DeepOTMPutDelta) {
    // Deep OTM put should have delta close to 0
    double spot = 150.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double delta = black_scholes_delta(spot, strike, T, r, sigma, false);

    EXPECT_GT(delta, -0.1);
    EXPECT_LT(delta, 0.0);
}

TEST_F(EuropeanOptionTest, DeltaAtExpiration) {
    // At expiration, delta is 1 for ITM call, 0 for OTM
    double spot = 110.0;
    double strike = 100.0;
    double T = 0.0;
    double r = 0.05;
    double sigma = 0.2;

    double call_delta = black_scholes_delta(spot, strike, T, r, sigma, true);
    EXPECT_NEAR(call_delta, 1.0, tolerance);

    double put_delta = black_scholes_delta(spot, strike, T, r, sigma, false);
    EXPECT_NEAR(put_delta, -1.0, tolerance);
}

TEST_F(EuropeanOptionTest, DeltaNumericalVerification) {
    // Verify delta by numerical differentiation
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;
    double h = 0.01;  // Small bump

    double call_delta = black_scholes_delta(spot, strike, T, r, sigma, true);

    // Numerical delta: (C(S+h) - C(S-h)) / (2h)
    double price_up = black_scholes_price(spot + h, strike, T, r, sigma, true);
    double price_down = black_scholes_price(spot - h, strike, T, r, sigma, true);
    double numerical_delta = (price_up - price_down) / (2.0 * h);

    EXPECT_NEAR(call_delta, numerical_delta, 1e-4);
}

// ============================================================================
// Greeks: Gamma Tests
// ============================================================================

TEST_F(EuropeanOptionTest, GammaPositive) {
    // Gamma should always be positive
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double gamma = black_scholes_gamma(spot, strike, T, r, sigma);

    EXPECT_GT(gamma, 0.0);
}

TEST_F(EuropeanOptionTest, GammaMaximumATM) {
    // Gamma should be highest for ATM options
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double gamma_atm = black_scholes_gamma(spot, strike, T, r, sigma);

    // Compare with OTM option
    double gamma_otm = black_scholes_gamma(spot, strike * 1.2, T, r, sigma);
    EXPECT_GT(gamma_atm, gamma_otm);

    // Compare with ITM option
    double gamma_itm = black_scholes_gamma(spot, strike * 0.8, T, r, sigma);
    EXPECT_GT(gamma_atm, gamma_itm);
}

TEST_F(EuropeanOptionTest, GammaNumericalVerification) {
    // Verify gamma by numerical differentiation of delta
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;
    double h = 0.01;

    double gamma = black_scholes_gamma(spot, strike, T, r, sigma);

    // Numerical gamma: (Delta(S+h) - Delta(S-h)) / (2h)
    double delta_up = black_scholes_delta(spot + h, strike, T, r, sigma, true);
    double delta_down = black_scholes_delta(spot - h, strike, T, r, sigma, true);
    double numerical_gamma = (delta_up - delta_down) / (2.0 * h);

    EXPECT_NEAR(gamma, numerical_gamma, 1e-4);
}

TEST_F(EuropeanOptionTest, GammaZeroAtExpiration) {
    // Gamma should be 0 at expiration
    double spot = 100.0;
    double strike = 100.0;
    double T = 0.0;
    double r = 0.05;
    double sigma = 0.2;

    double gamma = black_scholes_gamma(spot, strike, T, r, sigma);

    EXPECT_NEAR(gamma, 0.0, tolerance);
}

// ============================================================================
// Greeks: Vega Tests
// ============================================================================

TEST_F(EuropeanOptionTest, VegaPositive) {
    // Vega should always be positive
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double vega = black_scholes_vega(spot, strike, T, r, sigma);

    EXPECT_GT(vega, 0.0);
}

TEST_F(EuropeanOptionTest, VegaMaximumATM) {
    // Vega should be highest for ATM options
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double vega_atm = black_scholes_vega(spot, strike, T, r, sigma);

    // Compare with OTM option
    double vega_otm = black_scholes_vega(spot, strike * 1.3, T, r, sigma);
    EXPECT_GT(vega_atm, vega_otm);
}

TEST_F(EuropeanOptionTest, VegaNumericalVerification) {
    // Verify vega by numerical differentiation
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;
    double h = 0.001;  // Small bump for volatility

    double vega = black_scholes_vega(spot, strike, T, r, sigma);

    // Numerical vega: (C(σ+h) - C(σ-h)) / (2h)
    double price_up = black_scholes_price(spot, strike, T, r, sigma + h, true);
    double price_down = black_scholes_price(spot, strike, T, r, sigma - h, true);
    double numerical_vega = (price_up - price_down) / (2.0 * h);

    EXPECT_NEAR(vega, numerical_vega, 1e-3);
}

TEST_F(EuropeanOptionTest, VegaIncreasesWithTime) {
    // Longer dated options have higher vega
    double spot = 100.0;
    double strike = 100.0;
    double r = 0.05;
    double sigma = 0.2;

    double vega_short = black_scholes_vega(spot, strike, 0.25, r, sigma);
    double vega_long = black_scholes_vega(spot, strike, 1.0, r, sigma);

    EXPECT_GT(vega_long, vega_short);
}

TEST_F(EuropeanOptionTest, VegaZeroAtExpiration) {
    // Vega should be 0 at expiration
    double spot = 100.0;
    double strike = 100.0;
    double T = 0.0;
    double r = 0.05;
    double sigma = 0.2;

    double vega = black_scholes_vega(spot, strike, T, r, sigma);

    EXPECT_NEAR(vega, 0.0, tolerance);
}

// ============================================================================
// Greeks: Theta Tests
// ============================================================================

TEST_F(EuropeanOptionTest, ThetaNegativeForLongOptions) {
    // Theta should be negative for long positions (time decay)
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double call_theta = black_scholes_theta(spot, strike, T, r, sigma, true);
    double put_theta = black_scholes_theta(spot, strike, T, r, sigma, false);

    // For most options, theta is negative
    EXPECT_LT(call_theta, 0.0);
    // Put theta can be positive for deep ITM European puts, but typically negative
}

TEST_F(EuropeanOptionTest, ThetaNumericalVerification) {
    // Verify theta by numerical differentiation with respect to time
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;
    double h = 1.0 / 365.0;  // One day

    double theta = black_scholes_theta(spot, strike, T, r, sigma, true);

    // Numerical theta: (C(T-h) - C(T)) / h (note: negative of derivative)
    // Theta = -∂C/∂t, so we compute -(C(T+h) - C(T))/h
    double price_now = black_scholes_price(spot, strike, T, r, sigma, true);
    double price_later = black_scholes_price(spot, strike, T + h, r, sigma, true);
    double numerical_theta = -(price_later - price_now) / h;

    EXPECT_NEAR(theta, numerical_theta, loose_tolerance);
}

TEST_F(EuropeanOptionTest, ThetaZeroAtExpiration) {
    // Theta should be 0 at expiration
    double spot = 100.0;
    double strike = 100.0;
    double T = 0.0;
    double r = 0.05;
    double sigma = 0.2;

    double theta = black_scholes_theta(spot, strike, T, r, sigma, true);

    EXPECT_NEAR(theta, 0.0, tolerance);
}

// ============================================================================
// Greeks: Rho Tests
// ============================================================================

TEST_F(EuropeanOptionTest, CallRhoPositive) {
    // Call rho should be positive (call value increases with rates)
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double rho = black_scholes_rho(spot, strike, T, r, sigma, true);

    EXPECT_GT(rho, 0.0);
}

TEST_F(EuropeanOptionTest, PutRhoNegative) {
    // Put rho should be negative (put value decreases with rates)
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double rho = black_scholes_rho(spot, strike, T, r, sigma, false);

    EXPECT_LT(rho, 0.0);
}

TEST_F(EuropeanOptionTest, RhoNumericalVerification) {
    // Verify rho by numerical differentiation
    double spot = 100.0;
    double strike = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;
    double h = 0.0001;  // 1 basis point

    double rho = black_scholes_rho(spot, strike, T, r, sigma, true);

    // Numerical rho: (C(r+h) - C(r-h)) / (2h)
    double price_up = black_scholes_price(spot, strike, T, r + h, sigma, true);
    double price_down = black_scholes_price(spot, strike, T, r - h, sigma, true);
    double numerical_rho = (price_up - price_down) / (2.0 * h);

    EXPECT_NEAR(rho, numerical_rho, 1e-3);
}

TEST_F(EuropeanOptionTest, RhoIncreasesWithMaturity) {
    // Longer dated options have higher rho (more sensitive to rates)
    double spot = 100.0;
    double strike = 100.0;
    double r = 0.05;
    double sigma = 0.2;

    double rho_short = black_scholes_rho(spot, strike, 0.25, r, sigma, true);
    double rho_long = black_scholes_rho(spot, strike, 2.0, r, sigma, true);

    EXPECT_GT(std::abs(rho_long), std::abs(rho_short));
}

TEST_F(EuropeanOptionTest, RhoZeroAtExpiration) {
    // Rho should be 0 at expiration
    double spot = 100.0;
    double strike = 100.0;
    double T = 0.0;
    double r = 0.05;
    double sigma = 0.2;

    double rho = black_scholes_rho(spot, strike, T, r, sigma, true);

    EXPECT_NEAR(rho, 0.0, tolerance);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(EuropeanOptionTest, ConsistencyAcrossParameters) {
    // Test consistency across different parameter combinations
    std::vector<double> spots = {80.0, 100.0, 120.0};
    std::vector<double> strikes = {90.0, 100.0, 110.0};
    std::vector<double> times = {0.25, 0.5, 1.0, 2.0};
    std::vector<double> rates = {0.01, 0.05, 0.10};
    std::vector<double> vols = {0.1, 0.2, 0.4};

    for (double spot : spots) {
        for (double strike : strikes) {
            for (double T : times) {
                for (double r : rates) {
                    for (double sigma : vols) {
                        // Prices should be non-negative
                        double call = black_scholes_price(spot, strike, T, r, sigma, true);
                        double put = black_scholes_price(spot, strike, T, r, sigma, false);
                        EXPECT_GE(call, 0.0);
                        EXPECT_GE(put, 0.0);

                        // Put-call parity
                        double parity_lhs = call - put;
                        double parity_rhs = spot - strike * std::exp(-r * T);
                        EXPECT_NEAR(parity_lhs, parity_rhs, tolerance);

                        // Greeks should be reasonable
                        double vega = black_scholes_vega(spot, strike, T, r, sigma);
                        double gamma = black_scholes_gamma(spot, strike, T, r, sigma);
                        EXPECT_GE(vega, 0.0);
                        EXPECT_GE(gamma, 0.0);
                    }
                }
            }
        }
    }
}
