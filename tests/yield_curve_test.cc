// SPDX-License-Identifier: MIT
#include "mango/math/yield_curve.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(YieldCurveTest, FlatCurveReturnsConstantRate) {
    auto curve = mango::YieldCurve::flat(0.05);

    EXPECT_DOUBLE_EQ(curve.rate(0.0), 0.05);
    EXPECT_DOUBLE_EQ(curve.rate(0.5), 0.05);
    EXPECT_DOUBLE_EQ(curve.rate(1.0), 0.05);
    EXPECT_DOUBLE_EQ(curve.rate(10.0), 0.05);
}

TEST(YieldCurveTest, FlatCurveDiscountFactor) {
    auto curve = mango::YieldCurve::flat(0.05);

    // D(t) = exp(-r*t)
    EXPECT_NEAR(curve.discount(0.0), 1.0, 1e-10);
    EXPECT_NEAR(curve.discount(1.0), std::exp(-0.05), 1e-10);
    EXPECT_NEAR(curve.discount(2.0), std::exp(-0.10), 1e-10);
}

TEST(YieldCurveTest, FromPointsCreatesValidCurve) {
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.0},
        {1.0, -0.05},   // D(1) = exp(-0.05) ~ 0.9512
        {2.0, -0.10}    // D(2) = exp(-0.10) ~ 0.9048
    };

    auto result = mango::YieldCurve::from_points(points);
    ASSERT_TRUE(result.has_value());

    auto& curve = result.value();
    EXPECT_NEAR(curve.discount(0.0), 1.0, 1e-10);
    EXPECT_NEAR(curve.discount(1.0), std::exp(-0.05), 1e-10);
    EXPECT_NEAR(curve.discount(2.0), std::exp(-0.10), 1e-10);
}

TEST(YieldCurveTest, FromPointsFailsWithoutZeroTenor) {
    std::vector<mango::TenorPoint> points = {
        {0.5, -0.025},
        {1.0, -0.05}
    };

    auto result = mango::YieldCurve::from_points(points);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("t=0") != std::string::npos);
}

TEST(YieldCurveTest, FromDiscountsCreatesValidCurve) {
    std::vector<double> tenors = {0.0, 0.5, 1.0, 2.0};
    std::vector<double> discounts = {1.0, 0.9753, 0.9512, 0.9048};

    auto result = mango::YieldCurve::from_discounts(tenors, discounts);
    ASSERT_TRUE(result.has_value());

    auto& curve = result.value();
    EXPECT_NEAR(curve.discount(0.0), 1.0, 1e-10);
    EXPECT_NEAR(curve.discount(0.5), 0.9753, 1e-4);
    EXPECT_NEAR(curve.discount(1.0), 0.9512, 1e-4);
}

TEST(YieldCurveTest, FromDiscountsFailsOnSizeMismatch) {
    std::vector<double> tenors = {0.0, 0.5, 1.0};
    std::vector<double> discounts = {1.0, 0.9753};  // Wrong size

    auto result = mango::YieldCurve::from_discounts(tenors, discounts);
    ASSERT_FALSE(result.has_value());
}

TEST(YieldCurveTest, RateInterpolation) {
    // Upward sloping curve: 5% for first year, 6% for second year
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.0},
        {1.0, -0.05},   // 5% for [0,1]
        {2.0, -0.11}    // 6% for [1,2] (total: -0.05 - 0.06 = -0.11)
    };

    auto result = mango::YieldCurve::from_points(points);
    ASSERT_TRUE(result.has_value());
    auto& curve = result.value();

    // Rate in first segment [0,1]: 5%
    EXPECT_NEAR(curve.rate(0.0), 0.05, 1e-10);
    EXPECT_NEAR(curve.rate(0.5), 0.05, 1e-10);
    EXPECT_NEAR(curve.rate(0.99), 0.05, 1e-10);

    // Rate in second segment [1,2]: 6%
    EXPECT_NEAR(curve.rate(1.0), 0.06, 1e-10);
    EXPECT_NEAR(curve.rate(1.5), 0.06, 1e-10);
    EXPECT_NEAR(curve.rate(2.0), 0.06, 1e-10);
}

TEST(YieldCurveTest, RateExtrapolation) {
    auto curve = mango::YieldCurve::flat(0.05);

    // Extrapolation beyond curve should continue flat
    EXPECT_NEAR(curve.rate(50.0), 0.05, 1e-10);
    EXPECT_NEAR(curve.rate(100.0), 0.05, 1e-10);
}

TEST(YieldCurveTest, DiscountInterpolation) {
    // Curve with known discount factors
    std::vector<double> tenors = {0.0, 1.0, 2.0};
    std::vector<double> discounts = {1.0, 0.95, 0.90};

    auto result = mango::YieldCurve::from_discounts(tenors, discounts);
    ASSERT_TRUE(result.has_value());
    auto& curve = result.value();

    // Midpoint: log-linear interpolation
    // ln(D(0.5)) = 0.5 * ln(0.95) = -0.0256
    // D(0.5) = exp(-0.0256) ~ 0.9747
    double expected_d05 = std::exp(0.5 * std::log(0.95));
    EXPECT_NEAR(curve.discount(0.5), expected_d05, 1e-6);
}

// Edge case tests from code review feedback

TEST(YieldCurveTest, FromDiscountsFailsOnNegativeDiscount) {
    std::vector<double> tenors = {0.0, 1.0, 2.0};
    std::vector<double> discounts = {1.0, -0.95, 0.90};  // Negative discount

    auto result = mango::YieldCurve::from_discounts(tenors, discounts);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("positive") != std::string::npos);
}

TEST(YieldCurveTest, FromDiscountsFailsOnZeroDiscount) {
    std::vector<double> tenors = {0.0, 1.0, 2.0};
    std::vector<double> discounts = {1.0, 0.0, 0.90};  // Zero discount

    auto result = mango::YieldCurve::from_discounts(tenors, discounts);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("positive") != std::string::npos);
}

TEST(YieldCurveTest, FromPointsFailsOnNonZeroLogDiscountAtZero) {
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.05},   // Non-zero log_discount at t=0
        {1.0, -0.05}
    };

    auto result = mango::YieldCurve::from_points(points);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("log_discount at t=0") != std::string::npos);
}

// ===========================================================================
// Regression tests for bugs found during code review
// ===========================================================================

// Regression: Duplicate tenors cause division by zero in rate_between()
// Bug: from_points() didn't reject duplicate tenors, causing NaN in interpolation
TEST(YieldCurveTest, FromPointsRejectsDuplicateTenors) {
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.0},
        {1.0, -0.05},
        {1.0, -0.05},  // Duplicate tenor
        {2.0, -0.10}
    };

    auto result = mango::YieldCurve::from_points(points);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("strictly increasing") != std::string::npos);
}

// Regression: Nearly duplicate tenors also cause numerical issues
TEST(YieldCurveTest, FromPointsRejectsNearlyDuplicateTenors) {
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.0},
        {1.0, -0.05},
        {1.0 + 1e-15, -0.05},  // Nearly duplicate tenor
        {2.0, -0.10}
    };

    auto result = mango::YieldCurve::from_points(points);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("strictly increasing") != std::string::npos);
}

// Regression: Decreasing tenors after sorting should still work
TEST(YieldCurveTest, FromPointsSortsAndValidates) {
    std::vector<mango::TenorPoint> points = {
        {2.0, -0.10},
        {0.0, 0.0},
        {1.0, -0.05}
    };

    auto result = mango::YieldCurve::from_points(points);
    ASSERT_TRUE(result.has_value());

    auto& curve = result.value();
    EXPECT_NEAR(curve.discount(1.0), std::exp(-0.05), 1e-10);
}

// Test zero_rate calculation
TEST(YieldCurveTest, ZeroRateCalculation) {
    auto curve = mango::YieldCurve::flat(0.05);

    // For flat curve, zero rate should equal the constant rate
    EXPECT_NEAR(curve.zero_rate(1.0), 0.05, 1e-10);
    EXPECT_NEAR(curve.zero_rate(2.0), 0.05, 1e-10);
}

TEST(YieldCurveTest, ZeroRateForNonFlatCurve) {
    // Upward sloping curve: 5% for first year, 6% for second year
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.0},
        {1.0, -0.05},
        {2.0, -0.11}
    };

    auto result = mango::YieldCurve::from_points(points);
    ASSERT_TRUE(result.has_value());
    auto& curve = result.value();

    // Zero rate at t=1: -ln(D(1))/1 = 0.05/1 = 0.05
    EXPECT_NEAR(curve.zero_rate(1.0), 0.05, 1e-10);

    // Zero rate at t=2: -ln(D(2))/2 = 0.11/2 = 0.055
    EXPECT_NEAR(curve.zero_rate(2.0), 0.055, 1e-10);
}

TEST(YieldCurveTest, ZeroRateAtZeroReturnsFowardRate) {
    auto curve = mango::YieldCurve::flat(0.05);

    // At t=0, zero_rate falls back to forward rate at t=0
    EXPECT_NEAR(curve.zero_rate(0.0), 0.05, 1e-10);
}

// Test equality operator
TEST(YieldCurveTest, EqualitySameCurve) {
    auto curve1 = mango::YieldCurve::flat(0.05);
    auto curve2 = mango::YieldCurve::flat(0.05);

    EXPECT_TRUE(curve1 == curve2);
}

TEST(YieldCurveTest, EqualityDifferentRate) {
    auto curve1 = mango::YieldCurve::flat(0.05);
    auto curve2 = mango::YieldCurve::flat(0.06);

    EXPECT_FALSE(curve1 == curve2);
}

TEST(YieldCurveTest, EqualityFromPoints) {
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.0},
        {1.0, -0.05},
        {2.0, -0.10}
    };

    auto result1 = mango::YieldCurve::from_points(points);
    auto result2 = mango::YieldCurve::from_points(points);
    ASSERT_TRUE(result1.has_value() && result2.has_value());

    EXPECT_TRUE(result1.value() == result2.value());
}
