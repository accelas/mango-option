#include "src/option/option_spec.hpp"
#include "src/math/yield_curve.hpp"
#include <gtest/gtest.h>

TEST(RateSpecTest, DefaultIsDouble) {
    mango::OptionSpec spec;
    EXPECT_TRUE(std::holds_alternative<double>(spec.rate));
    EXPECT_DOUBLE_EQ(std::get<double>(spec.rate), 0.0);
}

TEST(RateSpecTest, CanAssignDouble) {
    mango::OptionSpec spec;
    spec.rate = 0.05;

    EXPECT_TRUE(std::holds_alternative<double>(spec.rate));
    EXPECT_DOUBLE_EQ(std::get<double>(spec.rate), 0.05);
}

TEST(RateSpecTest, CanAssignYieldCurve) {
    mango::OptionSpec spec;
    spec.rate = mango::YieldCurve::flat(0.05);

    EXPECT_TRUE(std::holds_alternative<mango::YieldCurve>(spec.rate));
    auto& curve = std::get<mango::YieldCurve>(spec.rate);
    EXPECT_DOUBLE_EQ(curve.rate(0.5), 0.05);
}

TEST(RateSpecTest, MakeRateFnFromDouble) {
    mango::RateSpec spec = 0.05;
    double maturity = 1.0;
    auto fn = mango::make_rate_fn(spec, maturity);

    // For constant rate, returns same value regardless of time-to-expiry
    EXPECT_DOUBLE_EQ(fn(0.0), 0.05);
    EXPECT_DOUBLE_EQ(fn(0.5), 0.05);
    EXPECT_DOUBLE_EQ(fn(1.0), 0.05);
}

TEST(RateSpecTest, MakeRateFnFromCurve) {
    // Upward sloping curve: 5% for [0,1], 6% for [1,2]
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.0},
        {1.0, -0.05},
        {2.0, -0.11}
    };
    auto curve = mango::YieldCurve::from_points(points).value();
    mango::RateSpec spec = curve;

    double maturity = 2.0;  // 2-year option
    auto fn = mango::make_rate_fn(spec, maturity);

    // fn takes time-to-expiry τ, returns rate at calendar time s = T - τ
    // τ = 0 (at expiry): s = 2.0, rate = 6%
    EXPECT_NEAR(fn(0.0), 0.06, 1e-10);
    // τ = 0.5 (0.5 years to expiry): s = 1.5, rate = 6%
    EXPECT_NEAR(fn(0.5), 0.06, 1e-10);
    // τ = 1.0 (1 year to expiry): s = 1.0, rate = 6% (right edge of first segment)
    EXPECT_NEAR(fn(1.0), 0.06, 1e-10);
    // τ = 1.5 (1.5 years to expiry): s = 0.5, rate = 5%
    EXPECT_NEAR(fn(1.5), 0.05, 1e-10);
    // τ = 2.0 (at valuation): s = 0.0, rate = 5%
    EXPECT_NEAR(fn(2.0), 0.05, 1e-10);
}

// ===========================================================================
// Regression tests for time convention bug
// ===========================================================================

// Regression: make_rate_fn must convert time-to-expiry to calendar time
// Bug: Used curve.rate(τ) directly instead of curve.rate(T - τ)
TEST(RateSpecTest, TimeConversionForUpslopingCurve) {
    // Upward sloping curve: rates increase with time
    // Rate at s=0: 4%, rate at s=1: 5%, rate at s=2: 6%
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.0},
        {1.0, -0.04},   // 4% for [0,1]
        {2.0, -0.09}    // 5% for [1,2] (cumulative: 4% + 5% = 9%)
    };
    auto curve = mango::YieldCurve::from_points(points).value();
    mango::RateSpec spec = curve;

    double maturity = 2.0;
    auto fn = mango::make_rate_fn(spec, maturity);

    // Near expiry (τ small), calendar time s = T - τ is large
    // Should see HIGH rate (end of curve)
    EXPECT_NEAR(fn(0.1), 0.05, 1e-10);  // s = 1.9, second segment

    // Far from expiry (τ large), calendar time s = T - τ is small
    // Should see LOW rate (start of curve)
    EXPECT_NEAR(fn(1.9), 0.04, 1e-10);  // s = 0.1, first segment
}

// Regression: make_forward_discount_fn must return D(T)/D(s) not D(τ)
TEST(RateSpecTest, ForwardDiscountForUpslopingCurve) {
    // Same upward sloping curve
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.0},
        {1.0, -0.05},
        {2.0, -0.11}
    };
    auto curve = mango::YieldCurve::from_points(points).value();
    mango::RateSpec spec = curve;

    double maturity = 2.0;
    auto fn = mango::make_forward_discount_fn(spec, maturity);

    // At τ = 0 (at expiry): s = T, forward discount = D(T)/D(T) = 1
    EXPECT_NEAR(fn(0.0), 1.0, 1e-10);

    // At τ = T (at valuation): s = 0, forward discount = D(T)/D(0) = D(T)
    double D_T = std::exp(-0.11);  // curve.discount(2.0)
    EXPECT_NEAR(fn(2.0), D_T, 1e-10);

    // At τ = 1 (1 year to expiry): s = 1
    // Forward discount = D(2)/D(1) = exp(-0.11)/exp(-0.05) = exp(-0.06)
    double D_1 = std::exp(-0.05);
    EXPECT_NEAR(fn(1.0), D_T / D_1, 1e-10);
}

TEST(RateSpecTest, ForwardDiscountForConstantRate) {
    mango::RateSpec spec = 0.05;
    double maturity = 2.0;
    auto fn = mango::make_forward_discount_fn(spec, maturity);

    // For constant rate, forward discount is simply exp(-r*τ)
    EXPECT_NEAR(fn(0.0), 1.0, 1e-10);
    EXPECT_NEAR(fn(1.0), std::exp(-0.05), 1e-10);
    EXPECT_NEAR(fn(2.0), std::exp(-0.10), 1e-10);
}
