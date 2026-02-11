// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/dimensionless/dimensionless_european.hpp"
#include "mango/option/european_option.hpp"
#include <cmath>

namespace mango {
namespace {

TEST(DimensionlessEuropeanTest, PutMatchesBlackScholes) {
    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, r = 0.05;
    double x = std::log(S / K);
    double tau_prime = sigma * sigma * tau / 2.0;
    double kappa = 2.0 * r / (sigma * sigma);

    double dim_price = dimensionless_european_put(x, tau_prime, kappa);

    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = S, .strike = K, .maturity = tau,
            .rate = r, .dividend_yield = 0.0,
            .option_type = OptionType::PUT}, sigma).solve().value();
    double bs_price = eu.value() / K;

    EXPECT_NEAR(dim_price, bs_price, 1e-12);
}

TEST(DimensionlessEuropeanTest, CallMatchesBlackScholes) {
    double S = 110.0, K = 100.0, tau = 0.5, sigma = 0.30, r = 0.03;
    double x = std::log(S / K);
    double tau_prime = sigma * sigma * tau / 2.0;
    double kappa = 2.0 * r / (sigma * sigma);

    double dim_price = dimensionless_european_call(x, tau_prime, kappa);

    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = S, .strike = K, .maturity = tau,
            .rate = r, .dividend_yield = 0.0,
            .option_type = OptionType::CALL}, sigma).solve().value();
    double bs_price = eu.value() / K;

    EXPECT_NEAR(dim_price, bs_price, 1e-12);
}

TEST(DimensionlessEuropeanTest, PutAtExpiry) {
    EXPECT_NEAR(dimensionless_european_put(-0.1, 0.0, 1.0),
                1.0 - std::exp(-0.1), 1e-14);
    EXPECT_NEAR(dimensionless_european_put(0.1, 0.0, 1.0),
                0.0, 1e-14);
}

TEST(DimensionlessEuropeanTest, Dispatch) {
    double x = 0.0, tp = 0.02, kappa = 2.5;
    EXPECT_EQ(dimensionless_european(x, tp, kappa, OptionType::PUT),
              dimensionless_european_put(x, tp, kappa));
    EXPECT_EQ(dimensionless_european(x, tp, kappa, OptionType::CALL),
              dimensionless_european_call(x, tp, kappa));
}

}  // namespace
}  // namespace mango
