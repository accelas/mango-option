#include "src/math/yield_curve.hpp"
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
