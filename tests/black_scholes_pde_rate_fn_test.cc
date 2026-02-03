// SPDX-License-Identifier: MIT
#include "mango/pde/operators/black_scholes_pde.hpp"
#include "mango/math/yield_curve.hpp"
#include <gtest/gtest.h>

TEST(BlackScholesPDERateFnTest, ConstantRateFnViaLambda) {
    double sigma = 0.20;
    double d = 0.02;
    double r = 0.05;

    auto rate_fn = [r](double) { return r; };
    mango::operators::BlackScholesPDE pde(sigma, rate_fn, d);

    // L(V) = (sigma^2/2)*V_xx + (r-d-sigma^2/2)*V_x - r*V
    double half_sigma_sq = 0.5 * sigma * sigma;  // 0.02
    double drift = r - d - half_sigma_sq;         // 0.05 - 0.02 - 0.02 = 0.01

    double V_xx = 1.0;
    double V_x = 1.0;
    double V = 1.0;
    double t = 0.5;

    double expected = half_sigma_sq * V_xx + drift * V_x - r * V;
    // = 0.02 * 1 + 0.01 * 1 - 0.05 * 1 = -0.02

    EXPECT_NEAR(pde(t, V_xx, V_x, V), expected, 1e-10);
}

TEST(BlackScholesPDERateFnTest, TimeVaryingRate) {
    double sigma = 0.20;
    double d = 0.02;

    // Rate function: 5% for t < 1, 6% for t >= 1
    auto rate_fn = [](double t) { return t < 1.0 ? 0.05 : 0.06; };
    mango::operators::BlackScholesPDE pde(sigma, rate_fn, d);

    double V_xx = 1.0, V_x = 1.0, V = 1.0;

    // At t=0.5: r=0.05
    double r1 = 0.05;
    double drift1 = r1 - d - 0.5 * sigma * sigma;
    double expected1 = 0.02 * V_xx + drift1 * V_x - r1 * V;
    EXPECT_NEAR(pde(0.5, V_xx, V_x, V), expected1, 1e-10);

    // At t=1.5: r=0.06
    double r2 = 0.06;
    double drift2 = r2 - d - 0.5 * sigma * sigma;
    double expected2 = 0.02 * V_xx + drift2 * V_x - r2 * V;
    EXPECT_NEAR(pde(1.5, V_xx, V_x, V), expected2, 1e-10);
}

TEST(BlackScholesPDERateFnTest, WithYieldCurve) {
    double sigma = 0.20;
    double d = 0.02;

    // Create yield curve
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.0},
        {1.0, -0.05},
        {2.0, -0.11}
    };
    auto curve = mango::YieldCurve::from_points(points).value();
    auto rate_fn = [&curve](double t) { return curve.rate(t); };

    mango::operators::BlackScholesPDE pde(sigma, rate_fn, d);

    double V_xx = 1.0, V_x = 1.0, V = 1.0;

    // At t=0.5: r=0.05 (first segment)
    double expected = 0.02 * V_xx + (0.05 - d - 0.02) * V_x - 0.05 * V;
    EXPECT_NEAR(pde(0.5, V_xx, V_x, V), expected, 1e-10);
}
