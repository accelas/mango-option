// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/dimensionless/dimensionless_european.hpp"
#include "mango/option/european_option.hpp"
#include <cmath>

namespace mango {
namespace {

// Compare dimensionless European against standard Black-Scholes
TEST(DimensionlessEuropean, PutMatchesBlackScholes) {
    struct Case { double sigma; double r; double T; double moneyness; };
    std::vector<Case> cases = {
        {0.20, 0.05, 1.0, 1.0},   // ATM
        {0.20, 0.05, 1.0, 0.9},   // ITM put
        {0.20, 0.05, 1.0, 1.1},   // OTM put
        {0.30, 0.08, 0.5, 1.0},   // Higher vol, shorter maturity
        {0.10, 0.02, 2.0, 0.85},  // Low vol, long maturity
        {0.40, 0.10, 0.25, 1.15}, // High vol, short maturity
    };

    const double K = 100.0;
    for (const auto& c : cases) {
        double x = std::log(c.moneyness);
        double tau_prime = c.sigma * c.sigma * c.T / 2.0;
        double kappa = 2.0 * c.r / (c.sigma * c.sigma);

        // Dimensionless formula (returns V/K)
        double dim_price = dimensionless_european_put(x, tau_prime, kappa);

        // Standard Black-Scholes
        double S = K * c.moneyness;
        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = S, .strike = K, .maturity = c.T,
                       .rate = c.r, .dividend_yield = 0.0,
                       .option_type = OptionType::PUT},
            c.sigma).solve().value();
        double bs_price = eu.value() / K;  // Normalize by K

        EXPECT_NEAR(dim_price, bs_price, 1e-12)
            << "sigma=" << c.sigma << " r=" << c.r
            << " T=" << c.T << " m=" << c.moneyness;
    }
}

TEST(DimensionlessEuropean, CallMatchesBlackScholes) {
    const double K = 100.0;
    struct Case { double sigma; double r; double T; double moneyness; };
    std::vector<Case> cases = {
        {0.20, 0.05, 1.0, 1.0},
        {0.20, 0.05, 1.0, 1.1},
        {0.30, 0.08, 0.5, 0.9},
    };

    for (const auto& c : cases) {
        double x = std::log(c.moneyness);
        double tau_prime = c.sigma * c.sigma * c.T / 2.0;
        double kappa = 2.0 * c.r / (c.sigma * c.sigma);

        double dim_price = dimensionless_european_call(x, tau_prime, kappa);

        double S = K * c.moneyness;
        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = S, .strike = K, .maturity = c.T,
                       .rate = c.r, .dividend_yield = 0.0,
                       .option_type = OptionType::CALL},
            c.sigma).solve().value();
        double bs_price = eu.value() / K;

        EXPECT_NEAR(dim_price, bs_price, 1e-12)
            << "sigma=" << c.sigma << " r=" << c.r;
    }
}

TEST(DimensionlessEuropean, PutCallParity) {
    // Put-call parity: C - P = S/K - exp(-κτ') = exp(x) - exp(-κτ')
    double x = 0.05;
    double tau_prime = 0.03;
    double kappa = 1.5;

    double put = dimensionless_european_put(x, tau_prime, kappa);
    double call = dimensionless_european_call(x, tau_prime, kappa);
    double parity = std::exp(x) - std::exp(-kappa * tau_prime);

    EXPECT_NEAR(call - put, parity, 1e-14);
}

TEST(DimensionlessEuropean, ZeroTimeReturnsIntrinsic) {
    // At τ'=0, European = intrinsic value
    EXPECT_NEAR(dimensionless_european_put(-0.1, 0.0, 1.0),
                std::max(1.0 - std::exp(-0.1), 0.0), 1e-15);
    EXPECT_NEAR(dimensionless_european_put(0.1, 0.0, 1.0), 0.0, 1e-15);
    EXPECT_NEAR(dimensionless_european_call(0.1, 0.0, 1.0),
                std::max(std::exp(0.1) - 1.0, 0.0), 1e-15);
}

}  // namespace
}  // namespace mango
