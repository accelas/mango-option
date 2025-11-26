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
    auto fn = mango::make_rate_fn(spec);

    EXPECT_DOUBLE_EQ(fn(0.0), 0.05);
    EXPECT_DOUBLE_EQ(fn(1.0), 0.05);
    EXPECT_DOUBLE_EQ(fn(10.0), 0.05);
}

TEST(RateSpecTest, MakeRateFnFromCurve) {
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.0},
        {1.0, -0.05},
        {2.0, -0.11}
    };
    auto curve = mango::YieldCurve::from_points(points).value();
    mango::RateSpec spec = curve;

    auto fn = mango::make_rate_fn(spec);

    EXPECT_NEAR(fn(0.5), 0.05, 1e-10);  // First segment: 5%
    EXPECT_NEAR(fn(1.5), 0.06, 1e-10);  // Second segment: 6%
}
