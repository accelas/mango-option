// SPDX-License-Identifier: MIT
#include "mango/math/proof/bernstein.hpp"
#include <array>
#include <gtest/gtest.h>
#include <limits>
namespace {
using namespace mango::detail::proof;
TEST(ProofBernsteinTest, CertifiesNonnegativeCoefficientHullIncludingZeroPlateau) {
    auto polynomial = BernsteinTensor::create({2}, {Interval(0), Interval(1), Interval(2)});
    ASSERT_TRUE(polynomial);
    auto proof = prove_nonnegative(*polynomial);
    EXPECT_EQ(proof.status, ProofStatus::Certified);
    EXPECT_EQ(proof.nodes, 1);
    auto flat = BernsteinTensor::create({3, 3}, std::vector<Interval>(16));
    ASSERT_TRUE(flat);
    EXPECT_EQ(prove_nonnegative(*flat).status, ProofStatus::Certified);
}
} // namespace

TEST(ProofBernsteinTest, SubdivisionProvesSquareDespiteNegativeCoarseCoefficient) {
    using namespace mango::detail::proof;
    // (x - 1/2)^2 has Bernstein coefficients [1/4, -1/4, 1/4].
    auto polynomial = BernsteinTensor::create({2}, {Interval(.25), Interval(-.25), Interval(.25)});
    ASSERT_TRUE(polynomial);
    EXPECT_EQ(prove_nonnegative(*polynomial, {.max_nodes = 1, .max_depth = 0}).status,
              ProofStatus::Indeterminate);
    auto proof = prove_nonnegative(*polynomial);
    EXPECT_EQ(proof.status, ProofStatus::Certified);
    EXPECT_EQ(proof.nodes, 3);
}

TEST(ProofBernsteinTest, FindsNegativePocketBetweenSeventeenPositiveScanNodes) {
    using namespace mango::detail::proof;
    constexpr double center = 17.0 / 32;
    constexpr double radius = 1.0 / 128;
    constexpr double c = center * center - radius * radius;
    for (int i = 0; i <= 16; ++i) {
        const double x = i / 16.0;
        ASSERT_GT((x - center) * (x - center) - radius * radius, 0);
    }
    auto polynomial = BernsteinTensor::create(
        {2}, {Interval(c), Interval(c - center), Interval(c - 2 * center + 1)});
    ASSERT_TRUE(polynomial);
    auto proof = prove_nonnegative(*polynomial);
    EXPECT_EQ(proof.status, ProofStatus::NegativeWitness);
    ASSERT_EQ(proof.witness_box.size(), 1);
    EXPECT_TRUE(proof.witness_bound.strictly_negative());
    double x = (proof.witness_box[0].first + proof.witness_box[0].second) / 2;
    EXPECT_LT((x - center) * (x - center) - radius * radius, 0);
    EXPECT_EQ(prove_nonnegative(*polynomial, {.max_nodes = 0}).status, ProofStatus::Indeterminate);
    auto repeat = prove_nonnegative(*polynomial);
    EXPECT_EQ(proof.nodes, repeat.nodes);
    EXPECT_EQ(proof.witness_box, repeat.witness_box);
}

TEST(ProofBernsteinTest, InvalidEnclosuresAndOversizedTensorsCannotProduceProofs) {
    using namespace mango::detail::proof;
    EXPECT_FALSE(BernsteinTensor::create({65}, {}));
    EXPECT_FALSE(BernsteinTensor::create({64, 64}, {}));
    EXPECT_FALSE(BernsteinTensor::create({1}, {Interval(0)}));
    EXPECT_FALSE(
        BernsteinTensor::create({0}, {Interval(std::numeric_limits<double>::quiet_NaN())}));
    auto uncertain = BernsteinTensor::create({0}, {Interval::hull(-1, 1)});
    ASSERT_TRUE(uncertain);
    auto proof = prove_nonnegative(*uncertain);
    EXPECT_EQ(proof.status, ProofStatus::Indeterminate);
    EXPECT_EQ(proof.reason, StopReason::Arithmetic);
}

TEST(ProofBernsteinTest, NegativeWitnessSurvivesBinary64DiagnosticUnderflow) {
    using namespace mango::detail::proof;
    const auto negative = Interval(-std::numeric_limits<double>::denorm_min()) / Interval(2);
    auto polynomial = BernsteinTensor::create({0}, {negative});
    ASSERT_TRUE(polynomial);
    auto proof = prove_nonnegative(*polynomial);
    EXPECT_EQ(proof.status, ProofStatus::NegativeWitness);
    EXPECT_EQ(proof.witness_bound.upper_bound(), 0);
    EXPECT_TRUE(proof.witness_bound.strictly_negative());
}

TEST(ProofBernsteinTest, TensorSubdivisionProvesSumOfSquaresWithinBudget) {
    using namespace mango::detail::proof;
    const std::array<double, 3> square{.25, -.25, .25};
    std::vector<Interval> coefficients;
    for (std::size_t i = 0; i < 81; ++i) {
        auto index = i;
        double value = 0;
        for (std::size_t axis = 0; axis < 4; ++axis) {
            value += square[index % 3];
            index /= 3;
        }
        coefficients.emplace_back(value);
    }
    auto polynomial = BernsteinTensor::create({2, 2, 2, 2}, std::move(coefficients));
    ASSERT_TRUE(polynomial);
    EXPECT_EQ(prove_nonnegative(*polynomial, {.max_nodes = 1}).status, ProofStatus::Indeterminate);
    auto proof = prove_nonnegative(*polynomial, {.max_nodes = 31, .max_depth = 4});
    EXPECT_EQ(proof.status, ProofStatus::Certified);
    EXPECT_LE(proof.nodes, 31);
}

TEST(ProofBernsteinTest, PartialCellRestrictionAndClampedFacesPreserveThePolynomial) {
    using namespace mango::detail::proof;
    // f(u)=(u-.5)^2 has a negative coarse Bernstein coefficient. On [.75,1]
    // its exact restricted controls are {1/16,1/8,1/4}.
    auto source = BernsteinTensor::create({2}, {Interval(.25), Interval(-.25), Interval(.25)});
    ASSERT_TRUE(source);
    auto restricted = source->restrict_axis(0, Interval(.75), Interval(1));
    ASSERT_TRUE(restricted);
    const std::array<double, 3> expected{.0625, .125, .25};
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(restricted->coefficients()[i].lower_bound(), expected[i]);
        EXPECT_EQ(restricted->coefficients()[i].upper_bound(), expected[i]);
    }
    EXPECT_EQ(prove_nonnegative(*restricted, {1, 0}).status, ProofStatus::Certified);
    auto face = source->restrict_axis(0, Interval(.5), Interval(.5));
    ASSERT_TRUE(face);
    EXPECT_TRUE(face->bounds().exact_zero());
}
