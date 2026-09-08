// SPDX-License-Identifier: MIT
#include "mango/math/chebyshev/chebyshev_polynomial.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
namespace {
using mango::detail::ChebyshevPolynomial;
TEST(ChebyshevPolynomialTest, ValueAndGreeksUseTheSameStoredPolynomial) {
    auto polynomial = ChebyshevPolynomial::from_coefficients({4}, {{-2, 2}}, {1, 2, 3, 4});
    ASSERT_TRUE(polynomial);
    for (double x : {-2., -1., 0., 1., 2.}) {
        const double s = x / 2;
        std::array<double, 1> query{x};
        EXPECT_DOUBLE_EQ(polynomial->eval(query), 16 * s * s * s + 6 * s * s - 10 * s - 2);
        EXPECT_DOUBLE_EQ(polynomial->partial(0, query), 24 * s * s + 6 * s - 5);
        EXPECT_DOUBLE_EQ(polynomial->eval_second_partial(0, query), 24 * s + 3);
    }
}
} // namespace

TEST(ChebyshevPolynomialTest, ConvertsNodalInputsToAnExplicitPolynomialAndPreservesFlatAxes) {
    using mango::detail::ChebyshevPolynomial;
    std::vector<double> values;
    for (int j = 0; j < 9; ++j) {
        const double x = -std::cos(std::numbers::pi * j / 8);
        values.push_back(16 * x * x * x + 6 * x * x - 10 * x - 2);
    }
    auto polynomial = ChebyshevPolynomial::from_cgl_values({9}, {{-2, 2}}, values);
    ASSERT_TRUE(polynomial);
    for (double q : {-2., -1.7, -.4, 0., .9, 1.8, 2.}) {
        const double x = q / 2;
        const std::array<double, 1> point{q};
        EXPECT_NEAR(polynomial->eval(point), 16 * x * x * x + 6 * x * x - 10 * x - 2, 1e-12);
        EXPECT_NEAR(polynomial->partial(0, point), 24 * x * x + 6 * x - 5, 1e-11);
        EXPECT_NEAR(polynomial->eval_second_partial(0, point), 24 * x + 3, 1e-10);
    }
    auto flat = ChebyshevPolynomial::from_cgl_values({257}, {{-1, 1}}, std::vector<double>(257, 3));
    ASSERT_TRUE(flat);
    EXPECT_EQ(flat->coefficients()[0], 3);
    for (std::size_t i = 1; i < 257; ++i)
        EXPECT_EQ(flat->coefficients()[i], 0);
}

TEST(ChebyshevPolynomialTest, Degree256ModalEndpointGreeksHaveExactPolynomialValues) {
    using mango::detail::ChebyshevPolynomial;
    std::vector<double> coefficients(257);
    coefficients.back() = 1;
    auto polynomial = ChebyshevPolynomial::from_coefficients({257}, {{-1, 1}}, coefficients);
    ASSERT_TRUE(polynomial);
    for (double x : {-1., 1.}) {
        const std::array<double, 1> point{x};
        EXPECT_DOUBLE_EQ(polynomial->eval(point), 1);
        EXPECT_DOUBLE_EQ(polynomial->partial(0, point), x * 65536);
        EXPECT_DOUBLE_EQ(polynomial->eval_second_partial(0, point), 1431633920);
    }
    const std::array<double, 1> center{0};
    EXPECT_DOUBLE_EQ(polynomial->eval(center), 1);
    EXPECT_DOUBLE_EQ(polynomial->partial(0, center), 0);
    EXPECT_DOUBLE_EQ(polynomial->eval_second_partial(0, center), -65536);
}

TEST(ChebyshevPolynomialTest, TensorConversionAndClampedGreeksPreserveAxisMeaning) {
    using mango::detail::ChebyshevPolynomial;
    std::vector<double> values;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j) {
            const double s = -std::cos(std::numbers::pi * i / 3),
                         t = -std::cos(std::numbers::pi * j / 2);
            values.push_back(s * t + 4 * s * s + 6 * t * t - 5);
        }
    auto polynomial = ChebyshevPolynomial::from_cgl_values({4, 3}, {{-2, 2}, {1, 3}}, values);
    ASSERT_TRUE(polynomial);
    const std::array<double, 2> point{.6, 2.2};
    const double s = .3, t = .2;
    EXPECT_NEAR(polynomial->eval(point), s * t + 4 * s * s + 6 * t * t - 5, 1e-12);
    EXPECT_NEAR(polynomial->partial(0, point), .5 * t + 4 * s, 1e-12);
    EXPECT_NEAR(polynomial->partial(1, point), s + 12 * t, 1e-12);
    EXPECT_NEAR(polynomial->eval_second_partial(0, point), 2, 1e-12);
    EXPECT_NEAR(polynomial->eval_second_partial(1, point), 12, 1e-12);
    const std::array<double, 2> outside{3, 2.2};
    EXPECT_EQ(polynomial->partial(0, outside), 0);
    EXPECT_NEAR(polynomial->partial(1, outside), 1 + 12 * t, 1e-12);
    const std::array<double, 2> invalid{3, std::numeric_limits<double>::quiet_NaN()};
    EXPECT_TRUE(std::isnan(polynomial->partial(0, invalid)));
}

TEST(ChebyshevPolynomialTest, InvalidDomainsAndCapacitiesAreRefusedWithoutReshaping) {
    using mango::detail::ChebyshevPolynomial;
    EXPECT_FALSE(
        ChebyshevPolynomial::from_coefficients({258}, {{-1, 1}}, std::vector<double>(258)));
    EXPECT_FALSE(ChebyshevPolynomial::from_coefficients({2}, {{1, 1}}, {0, 1}));
    EXPECT_FALSE(ChebyshevPolynomial::from_coefficients({2}, {{-1, 1}}, {0}));
    EXPECT_FALSE(ChebyshevPolynomial::from_coefficients({1}, {{-1, 1}},
                                                        {std::numeric_limits<double>::infinity()}));
    auto valid = ChebyshevPolynomial::from_coefficients({1}, {{-1, 1}}, {2});
    ASSERT_TRUE(valid);
    EXPECT_TRUE(std::isnan(valid->partial(1, std::array<double, 1>{0})));
    EXPECT_TRUE(std::isnan(valid->eval(std::array<double, 2>{0, 0})));
}
