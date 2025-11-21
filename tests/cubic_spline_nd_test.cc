/**
 * @file cubic_spline_nd_test.cc
 * @brief Basic tests for N-dimensional cubic spline template
 */

#include "src/math/cubic_spline_nd.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace mango;

// ============================================================================
// Basic N-D Template Tests
// ============================================================================

TEST(CubicSplineND, ConstantFunction3D) {
    // Test f(x, y, z) = 5.0 (constant) with N=3
    std::vector<double> x = {0.0, 1.0, 2.0};
    std::vector<double> y = {0.0, 1.0};
    std::vector<double> z = {0.0, 0.5, 1.0};

    // 3×2×3 = 18 values
    std::vector<double> values(18, 5.0);

    auto spline_result = CubicSplineND<double, 3>::create({x, y, z}, values);
    ASSERT_TRUE(spline_result.has_value());

    auto& spline = spline_result.value();

    EXPECT_NEAR(spline.eval({0.5, 0.5, 0.25}), 5.0, 1e-10);
    EXPECT_NEAR(spline.eval({1.5, 0.75, 0.75}), 5.0, 1e-10);
}

TEST(CubicSplineND, LinearFunction) {
    // Test f(x, y) = x + 2*y with N=2
    std::vector<double> x = {0.0, 1.0, 2.0};
    std::vector<double> y = {0.0, 1.0, 2.0};

    // 3×3 = 9 values
    std::vector<double> values;
    for (double xi : x) {
        for (double yi : y) {
            values.push_back(xi + 2.0 * yi);
        }
    }

    auto spline_result = CubicSplineND<double, 2>::create({x, y}, values);
    ASSERT_TRUE(spline_result.has_value());

    auto& spline = spline_result.value();

    // Linear functions should be reproduced exactly
    EXPECT_NEAR(spline.eval({0.5, 0.5}), 0.5 + 1.0, 1e-9);
    EXPECT_NEAR(spline.eval({1.5, 1.5}), 1.5 + 3.0, 1e-9);
}

TEST(CubicSplineND, InvalidGridSize) {
    std::vector<double> x = {0.0};  // Too small
    std::vector<double> y = {0.0, 1.0};
    std::vector<double> values(2, 1.0);

    auto result = CubicSplineND<double, 2>::create({x, y}, values);
    EXPECT_FALSE(result.has_value());
}

TEST(CubicSplineND, FloatPrecision) {
    // Test generic FloatingPoint support
    std::vector<float> x = {0.0f, 1.0f, 2.0f};
    std::vector<float> y = {0.0f, 1.0f};

    std::vector<float> values;
    for (float xi : x) {
        for (float yi : y) {
            values.push_back(xi + yi);
        }
    }

    auto spline_result = CubicSplineND<float, 2>::create({x, y}, values);
    ASSERT_TRUE(spline_result.has_value());

    auto& spline = spline_result.value();
    EXPECT_NEAR(spline.eval({0.5f, 0.5f}), 1.0f, 1e-5f);
}
