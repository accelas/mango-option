// SPDX-License-Identifier: MIT
#include "mango/math/proof/bspline_cell.hpp"
#include <array>
#include <gtest/gtest.h>
#include <limits>
namespace {
using namespace mango::detail::proof;
TEST(ProofBSplineCellTest, PreservesStoredBezierCoefficientsAndPhysicalDerivativeScale) {
    const std::array<double, 8> knots{0, 0, 0, 0, 2, 2, 2, 2};
    const std::array<std::span<const double>, 1> axes{knots};
    const std::array<double, 4> coefficients{1, 2, 4, 8};
    const std::array<std::size_t, 1> spans{3};
    auto value = extract_cubic_bspline_cell(axes, coefficients, spans);
    ASSERT_TRUE(value);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(value->coefficients()[i].lower_bound(), coefficients[i]);
        EXPECT_EQ(value->coefficients()[i].upper_bound(), coefficients[i]);
    }
    auto derivative = extract_cubic_bspline_cell(axes, coefficients, spans, 0);
    ASSERT_TRUE(derivative);
    EXPECT_EQ(derivative->degrees(), std::vector<std::size_t>{2});
    EXPECT_EQ(derivative->bounds().lower_bound(), 1.5);
    EXPECT_EQ(derivative->bounds().upper_bound(), 6);
    EXPECT_EQ(prove_nonnegative(*derivative).status, ProofStatus::Certified);
}
} // namespace

TEST(ProofBSplineCellTest, NonuniformKnotCellsReproduceExactLinearPolynomial) {
    using namespace mango::detail::proof;
    const std::array<double, 11> knots{0, 0, 0, 0, .125, .5, .75, 1, 1, 1, 1};
    const std::array<std::span<const double>, 1> axes{knots};
    // Greville identity: these coefficients represent 3*x exactly.
    const std::array<double, 7> coefficients{0, .125, .625, 1.375, 2.25, 2.75, 3};
    for (std::size_t span = 3; span < 7; ++span) {
        const std::array<std::size_t, 1> spans{span};
        auto value = extract_cubic_bspline_cell(axes, coefficients, spans);
        ASSERT_TRUE(value);
        for (std::size_t i = 0; i < 4; ++i)
            EXPECT_TRUE(value->coefficients()[i].contains(3 * knots[span] +
                                                          i * (knots[span + 1] - knots[span])));
        auto derivative = extract_cubic_bspline_cell(axes, coefficients, spans, 0);
        ASSERT_TRUE(derivative);
        for (const auto &coefficient : derivative->coefficients())
            EXPECT_TRUE(coefficient.contains(3));
        EXPECT_EQ(prove_nonnegative(*derivative).status, ProofStatus::Certified);
    }
}

TEST(ProofBSplineCellTest, TensorDerivativeRetainsStructuralZeroBeforeConversion) {
    using namespace mango::detail::proof;
    const std::array<double, 11> knots{0, 0, 0, 0, .125, .5, .75, 1, 1, 1, 1};
    const std::array<std::span<const double>, 2> axes{knots, knots};
    std::vector<double> coefficients(49);
    for (std::size_t i = 0; i < 7; ++i)
        for (std::size_t j = 0; j < 7; ++j)
            coefficients[i * 7 + j] = static_cast<double>(i * i) - 3;
    const std::array<std::size_t, 2> spans{4, 5};
    auto derivative = extract_cubic_bspline_cell(axes, coefficients, spans, 1);
    ASSERT_TRUE(derivative);
    for (const auto &coefficient : derivative->coefficients())
        EXPECT_TRUE(coefficient.exact_zero());
    auto proof = prove_nonnegative(*derivative);
    EXPECT_EQ(proof.status, ProofStatus::Certified);
    EXPECT_EQ(proof.nodes, 1);
}

TEST(ProofBSplineCellTest, PositiveStoredPriceCoefficientsDoNotHideNegativeDerivativePocket) {
    using namespace mango::detail::proof;
    constexpr double a = 17.0 / 32, c = a * a - 1.0 / (128 * 128);
    const std::array<double, 8> knots{0, 0, 0, 0, 1, 1, 1, 1};
    const std::array<std::span<const double>, 1> axes{knots};
    // This cubic's derivative is 3*((x-a)^2-(1/128)^2).
    const std::array<double, 4> coefficients{1, 1 + c, 1 + 2 * c - a, 2 + 3 * c - 3 * a};
    for (auto coefficient : coefficients)
        ASSERT_GT(coefficient, 0);
    const std::array<std::size_t, 1> spans{3};
    auto derivative = extract_cubic_bspline_cell(axes, coefficients, spans, 0);
    ASSERT_TRUE(derivative);
    EXPECT_EQ(prove_nonnegative(*derivative).status, ProofStatus::NegativeWitness);
}

TEST(ProofBSplineCellTest, RejectsMalformedAndDiscontinuousKnotRepresentations) {
    using namespace mango::detail::proof;
    std::array<double, 12> knots{0, 0, 0, 0, .5, .5, .5, .5, 1, 1, 1, 1};
    const std::array<std::span<const double>, 1> axes{knots};
    const std::array<double, 8> coefficients{};
    const std::array<std::size_t, 1> spans{3};
    auto result = extract_cubic_bspline_cell(axes, coefficients, spans);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), InputError::Knots);
    knots[7] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(extract_cubic_bspline_cell(axes, coefficients, spans));
}

TEST(ProofBSplineCellTest, FourDimensionalRowMajorPartialUsesTheSelectedAxis) {
    using namespace mango::detail::proof;
    const std::array<double, 11> knots{0, 0, 0, 0, .125, .5, .75, 1, 1, 1, 1};
    const std::array<std::span<const double>, 4> axes{knots, knots, knots, knots};
    const std::array<double, 7> greville_times_three{0, .125, .625, 1.375, 2.25, 2.75, 3};
    std::vector<double> coefficients(7 * 7 * 7 * 7);
    // The exact represented polynomial is 3*x + 6*y + 9*z + 12*w.
    for (std::size_t i = 0; i < coefficients.size(); ++i) {
        auto index = i;
        for (std::size_t d = 4; d > 0; --d) {
            coefficients[i] += d * greville_times_three[index % 7];
            index /= 7;
        }
    }
    const std::array<std::size_t, 4> spans{3, 4, 5, 6};
    for (std::size_t axis = 0; axis < 4; ++axis) {
        auto derivative = extract_cubic_bspline_cell(axes, coefficients, spans, axis);
        ASSERT_TRUE(derivative);
        for (const auto &coefficient : derivative->coefficients())
            EXPECT_TRUE(coefficient.contains(3.0 * (axis + 1)));
        EXPECT_EQ(prove_nonnegative(*derivative).status, ProofStatus::Certified);
    }
}

TEST(ProofBSplineCellTest, NonuniformCubicAndItsDerivativeMatchExactPolynomialControls) {
    using namespace mango::detail::proof;
    const std::array<double, 11> knots{0, 0, 0, 0, .125, .5, .75, 1, 1, 1, 1};
    const std::array<std::span<const double>, 1> axes{knots};
    std::array<double, 7> coefficients{};
    // Marsden's polynomial identity gives the coefficients of x^3.
    for (std::size_t i = 0; i < 7; ++i)
        coefficients[i] = knots[i + 1] * knots[i + 2] * knots[i + 3];
    for (std::size_t span = 3; span < 7; ++span) {
        const std::array<std::size_t, 1> spans{span};
        const double a = knots[span], b = knots[span + 1];
        const std::array<double, 4> values{a * a * a, a * a * b, a * b * b, b * b * b};
        const std::array<double, 3> derivatives{3 * a * a, 3 * a * b, 3 * b * b};
        auto value = extract_cubic_bspline_cell(axes, coefficients, spans);
        auto derivative = extract_cubic_bspline_cell(axes, coefficients, spans, 0);
        ASSERT_TRUE(value);
        ASSERT_TRUE(derivative);
        for (std::size_t i = 0; i < 4; ++i)
            EXPECT_TRUE(value->coefficients()[i].contains(values[i]));
        for (std::size_t i = 0; i < 3; ++i)
            EXPECT_TRUE(derivative->coefficients()[i].contains(derivatives[i]));
        EXPECT_EQ(prove_nonnegative(*derivative).status, ProofStatus::Certified);
    }
}

TEST(ProofBSplineCellTest, ContinuousPiecewiseSplineAllowsFlatThenRisingIntervals) {
    using namespace mango::detail::proof;
    const std::array<double, 11> knots{0, 0, 0, 0, .5, .5, .5, 1, 1, 1, 1};
    const std::array<std::span<const double>, 1> axes{knots};
    const std::array<double, 7> coefficients{1, 1, 1, 1, 2, 3, 4};
    const std::array<std::size_t, 1> left{3}, right{6};
    auto flat = extract_cubic_bspline_cell(axes, coefficients, left, 0);
    auto rising = extract_cubic_bspline_cell(axes, coefficients, right, 0);
    ASSERT_TRUE(flat);
    ASSERT_TRUE(rising);
    EXPECT_TRUE(flat->bounds().exact_zero());
    EXPECT_TRUE(rising->bounds().strictly_positive());
    EXPECT_EQ(prove_nonnegative(*flat).status, ProofStatus::Certified);
    EXPECT_EQ(prove_nonnegative(*rising).status, ProofStatus::Certified);
}
