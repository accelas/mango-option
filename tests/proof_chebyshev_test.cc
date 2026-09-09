// SPDX-License-Identifier: MIT
#include "mango/math/proof/chebyshev.hpp"
#include <gtest/gtest.h>
namespace {
using namespace mango::detail;
using namespace mango::detail::proof;
TEST(ProofChebyshevTest, CertifiesPositiveAndFlatPhysicalPartialsAtDegree256) {
    std::vector<double> coefficients(257 * 2);
    coefficients[1] = 2;
    coefficients[256 * 2 + 1] = .1;
    auto positive =
        ChebyshevPolynomial::from_coefficients({257, 2}, {{-1, 1}, {.1, .5}}, coefficients);
    ASSERT_TRUE(positive);
    auto bound = enclose_chebyshev(*positive, UnitBox{{0, 1}, {0, 1}}, 1);
    ASSERT_TRUE(bound);
    EXPECT_GT(bound->lower_bound(), 9.4);
    auto proof = prove_chebyshev_partial(*positive, 1, {.max_nodes = 1});
    ASSERT_TRUE(proof);
    EXPECT_EQ(proof->status, ProofStatus::Certified);
    coefficients[1] = 0;
    coefficients[256 * 2 + 1] = 0;
    coefficients[256 * 2] = 1;
    auto flat = ChebyshevPolynomial::from_coefficients({257, 2}, {{-1, 1}, {.1, .5}}, coefficients);
    ASSERT_TRUE(flat);
    auto flat_proof = prove_chebyshev_partial(*flat, 1, {.max_nodes = 1});
    ASSERT_TRUE(flat_proof);
    EXPECT_EQ(flat_proof->status, ProofStatus::Certified);
}
} // namespace

TEST(ProofChebyshevTest, BernsteinConversionProvesAZeroDerivativeMinimum) {
    using namespace mango::detail;
    using namespace mango::detail::proof;
    // P(x)=x^3-(9/4)x^2+(27/16)x on [0,1]; P'=3*(x-3/4)^2.
    auto polynomial =
        ChebyshevPolynomial::from_coefficients({4}, {{0, 1}}, {.3125, .1875, -.09375, .03125});
    ASSERT_TRUE(polynomial);
    auto derivative = chebyshev_to_bernstein(*polynomial, 0);
    ASSERT_TRUE(derivative);
    auto proof = prove_chebyshev_partial(*polynomial, 0, {.max_nodes = 31});
    ASSERT_TRUE(proof);
    EXPECT_EQ(proof->status, ProofStatus::Certified);
}

TEST(ProofChebyshevTest, RejectsHiddenNegativePocketOfTheEvaluatedPolynomial) {
    using namespace mango::detail;
    using namespace mango::detail::proof;
    constexpr double a = 17.0 / 32, c = a * a - 1.0 / (128 * 128);
    auto polynomial = ChebyshevPolynomial::from_coefficients({4}, {{0, 1}},
                                                             {5.0 / 16 - 9 * a / 8 + 3 * c / 2,
                                                              15.0 / 32 - 3 * a / 2 + 3 * c / 2,
                                                              3.0 / 16 - 3 * a / 8, 1.0 / 32});
    ASSERT_TRUE(polynomial);
    for (int i = 0; i <= 16; ++i) {
        std::array<double, 1> point{i / 16.0};
        ASSERT_GT(polynomial->partial(0, point), 0);
        if (i > 0) {
            const std::array<double, 1> previous{(i - 1) / 16.0};
            ASSERT_GT(polynomial->eval(point), polynomial->eval(previous));
        }
    }
    auto proof = prove_chebyshev_partial(*polynomial, 0);
    ASSERT_TRUE(proof);
    EXPECT_EQ(proof->status, ProofStatus::NegativeWitness);
    EXPECT_TRUE(proof->witness_bound.strictly_negative());
    ASSERT_EQ(proof->witness_box.size(), 1);
    const std::array<double, 1> point{(proof->witness_box[0].first + proof->witness_box[0].second) /
                                      2};
    EXPECT_LT(polynomial->partial(0, point), 0);
    EXPECT_EQ(prove_chebyshev_partial(*polynomial, 0, {.max_nodes = 0})->status,
              ProofStatus::Indeterminate);
}

TEST(ProofChebyshevTest, RestrictedIntervalsFindDegree256OscillationWithoutBernsteinConversion) {
    using namespace mango::detail;
    using namespace mango::detail::proof;
    std::vector<double> coefficients(257);
    coefficients.back() = 1;
    auto polynomial = ChebyshevPolynomial::from_coefficients({257}, {{-1, 1}}, coefficients);
    ASSERT_TRUE(polynomial);
    EXPECT_FALSE(chebyshev_to_bernstein(*polynomial, 0));
    auto proof = prove_chebyshev_partial(*polynomial, 0, {.max_nodes = 64, .max_depth = 24});
    ASSERT_TRUE(proof);
    EXPECT_EQ(proof->status, ProofStatus::NegativeWitness);
    ASSERT_EQ(proof->witness_box.size(), 1);
    const std::array<double, 1> point{proof->witness_box[0].first + proof->witness_box[0].second -
                                      1};
    EXPECT_LT(polynomial->partial(0, point), 0);
    EXPECT_TRUE(proof->witness_bound.strictly_negative());
    EXPECT_LE(proof->nodes, 64);
    auto center = enclose_chebyshev(*polynomial, UnitBox{{.5, .5}});
    ASSERT_TRUE(center);
    EXPECT_EQ(center->lower_bound(), 1);
    EXPECT_EQ(center->upper_bound(), 1);
}

TEST(ProofChebyshevTest, PhysicalModalBoundsIncludeClampedCoordinatesAndRetainedTinySigns) {
    using namespace mango::detail;
    using namespace mango::detail::proof;
    auto polynomial = ChebyshevPolynomial::from_coefficients({3}, {{-1, 1}}, {0, 0, 1});
    ASSERT_TRUE(polynomial);
    const std::array<Interval, 1> edge{Interval::hull(-2, -1)};
    auto value = enclose_chebyshev_physical(*polynomial, edge);
    ASSERT_TRUE(value);
    EXPECT_EQ(value->lower_bound(), 1);
    EXPECT_EQ(value->upper_bound(), 1);
    auto derivative = enclose_chebyshev_physical(*polynomial, edge, 0);
    ASSERT_TRUE(derivative);
    EXPECT_TRUE(derivative->contains(-4));
    EXPECT_TRUE(derivative->contains(0));
    auto outside =
        enclose_chebyshev_physical(*polynomial, std::array<Interval, 1>{Interval(-2)}, 0);
    ASSERT_TRUE(outside);
    EXPECT_TRUE(outside->exact_zero());
    auto tiny = ChebyshevPolynomial::from_coefficients(
        {2}, {{0, 4}}, {0, -std::numeric_limits<double>::denorm_min()});
    ASSERT_TRUE(tiny);
    auto small = enclose_chebyshev_physical(*tiny, std::array<Interval, 1>{Interval(1)}, 0);
    ASSERT_TRUE(small);
    EXPECT_TRUE(small->strictly_negative());
    EXPECT_EQ(small->upper_bound(), 0);
}

TEST(ProofChebyshevTest, OutsidePartialDoesNotMaskInvalidOtherCoordinate) {
    using namespace mango::detail;
    using namespace mango::detail::proof;
    auto polynomial =
        ChebyshevPolynomial::from_coefficients({2, 2}, {{-1, 1}, {-1, 1}}, {1, 0, 0, 0});
    ASSERT_TRUE(polynomial);
    const std::array<Interval, 2> box{Interval(2),
                                      Interval(std::numeric_limits<double>::quiet_NaN())};
    EXPECT_FALSE(enclose_chebyshev_physical(*polynomial, box, 0));
}
