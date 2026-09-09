// SPDX-License-Identifier: MIT
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/splits/multi_kref.hpp"
#include "mango/option/table/transform_leaf.hpp"
#include "mango/option/table/transforms/standard_4d.hpp"
#include <gtest/gtest.h>
#include <limits>

namespace mango {
namespace {
struct ConstantInterp {
    double value = 10;
    double eval(const std::array<double, 4> &) const { return value; }
    double partial(size_t, const std::array<double, 4> &) const { return 0; }
    double eval_second_partial(size_t, const std::array<double, 4> &) const { return 0; }
};

struct PolynomialInterp {
    double eval(const std::array<double, 4> &p) const {
        return 10 + p[0] + p[0] * p[0] + 4 * (p[1] - 2) + 3 * (p[2] - .2) + 5 * (p[3] + .05);
    }
    double partial(size_t axis, const std::array<double, 4> &p) const {
        return axis == 0 ? 1 + 2 * p[0] : axis == 1 ? 4 : axis == 2 ? 3 : 5;
    }
    double eval_second_partial(size_t axis, const std::array<double, 4> &) const {
        return axis == 0 ? 2 : 0;
    }
};

TEST(FiniteScaleCompositionTest, FinitePremiumDoesNotOverflowBeforeReferenceNormalization) {
    const double scale = std::numeric_limits<double>::max();
    TransformLeaf leaf(ConstantInterp{10}, StandardTransform4D{}, 100);
    const double price = leaf.price(scale, scale, 2, .2, -.05);
    ASSERT_TRUE(std::isfinite(price));
    EXPECT_NEAR(price / scale, 0.1, 1e-15);
}

TEST(FiniteScaleCompositionTest, FinitePremiumSurvivesExtremeNormalizationRatios) {
    TransformLeaf tiny(ConstantInterp{1e-300}, StandardTransform4D{}, 1e100);
    const double small = tiny.price(1e300, 1e300, 1, .2, .05);
    ASSERT_GT(small, 0.0);
    EXPECT_NEAR(small / 1e-100, 1.0, 1e-15);

    TransformLeaf large(ConstantInterp{1e100}, StandardTransform4D{}, 1e-300);
    const double big = large.price(1e-300, 1e-300, 1, .2, .05);
    ASSERT_TRUE(std::isfinite(big));
    EXPECT_NEAR(big / 1e100, 1.0, 1e-15);
}

TEST(FiniteScaleCompositionTest, EuropeanPutDoesNotOverflowItsDiscountedStrike) {
    const double scale = std::numeric_limits<double>::max();
    AnalyticalEEP eep(OptionType::PUT, 0);
    // Independent ordinary-scale solver plus exact strike homogeneity.
    const auto reference = EuropeanOptionSolver(OptionSpec{.spot = 70,
                                                           .strike = 100,
                                                           .maturity = 2,
                                                           .rate = -.05,
                                                           .option_type = OptionType::PUT},
                                                .2)
                               .solve()
                               .value();
    const double price = eep.european_price(.7 * scale, scale, 2, .2, -.05);
    ASSERT_TRUE(std::isfinite(price));
    EXPECT_NEAR(price / scale, reference.value() / 100, 1e-14);
}

TEST(FiniteScaleCompositionTest, LeafGreeksKeepTheirChainFactorsAtLargeScale) {
    const double scale = std::numeric_limits<double>::max();
    TransformLeaf leaf(PolynomialInterp{}, StandardTransform4D{}, 100);
    PricingParams p(OptionSpec{.spot = scale, .strike = scale, .maturity = 2, .rate = -.05}, .2);
    const auto delta = leaf.greek(Greek::Delta, p), theta = leaf.greek(Greek::Theta, p),
               rho = leaf.greek(Greek::Rho, p), gamma = leaf.gamma(p);
    ASSERT_TRUE(delta && theta && rho && gamma);
    EXPECT_NEAR(*delta, .01, 1e-14);
    EXPECT_NEAR(leaf.vega(scale, scale, 2, .2, -.05) / scale, .03, 1e-14);
    EXPECT_NEAR(*theta / scale, -.04, 1e-14);
    EXPECT_NEAR(*rho / scale, .05, 1e-14);
    ASSERT_GT(*gamma, 0.0);
    EXPECT_NEAR(*gamma * scale, .01, 1e-14);
}

TEST(FiniteScaleCompositionTest, EuropeanGreeksRespectTheirHomogeneityDegrees) {
    const double scale = std::numeric_limits<double>::max();
    for (auto type : {OptionType::CALL, OptionType::PUT}) {
        AnalyticalEEP eep(type, 0);
        const auto reference =
            EuropeanOptionSolver(
                OptionSpec{
                    .spot = 90, .strike = 100, .maturity = 2, .rate = -.05, .option_type = type},
                2)
                .solve()
                .value();
        EXPECT_NEAR(eep.european_delta(.9 * scale, scale, 2, 2, -.05), reference.delta(), 1e-14);
        EXPECT_NEAR(eep.european_vega(.9 * scale, scale, 2, 2, -.05) / scale,
                    reference.vega() / 100, 1e-14);
        EXPECT_NEAR(eep.european_theta(.9 * scale, scale, 2, 2, -.05) / scale,
                    reference.theta() / 100, 1e-14);
        const double gamma = eep.european_gamma(.9 * scale, scale, 2, 2, -.05);
        EXPECT_GT(gamma, 0.0);
        EXPECT_NEAR(gamma * scale, reference.gamma() * 100, 1e-14);
        if (type == OptionType::CALL) {
            EXPECT_NEAR(eep.european_rho(.9 * scale, scale, 2, 2, -.05) / scale,
                        reference.rho() / 100, 1e-14);
        } else {
            // Put rho's normalized magnitude exceeds one: the final quote
            // sensitivity itself cannot be represented at this scale.
            ASSERT_LT(reference.rho() / 100, -1.0);
            EXPECT_EQ(eep.european_rho(.9 * scale, scale, 2, 2, -.05),
                      -std::numeric_limits<double>::infinity());
        }
    }
}

TEST(FiniteScaleCompositionTest, ActiveReferenceSpotMapAvoidsOverflowingItsJacobian) {
    MultiKRefSplit split({.1, 1e308});
    const auto bracket = split.bracket(.5, .5, 1, .2, .05);
    ASSERT_EQ(bracket.count, 2u);
    ASSERT_GT(bracket.entries[1].weight, 0.0);
    const auto [spot, strike, tau, sigma, rate] = split.to_local(1, .5, .5, 1, .2, .05);
    EXPECT_EQ(spot, 1e308);
    EXPECT_EQ(strike, 1e308);
    EXPECT_EQ(tau, 1);
    EXPECT_EQ(sigma, .2);
    EXPECT_EQ(rate, .05);

    struct Piece {
        int *calls;
        double price(double s, double k, double, double, double) const {
            ++*calls;
            return std::isfinite(s) ? .1 * k : std::numeric_limits<double>::quiet_NaN();
        }
    };
    int low_calls = 0, high_calls = 0;
    SplitSurface surface(std::vector<Piece>{{&low_calls}, {&high_calls}}, split);
    EXPECT_NEAR(surface.price(.5, .5, 1, .2, .05), .05, 1e-15);
    EXPECT_EQ(low_calls, 1);
    EXPECT_EQ(high_calls, 1);
    EXPECT_NEAR(surface.price(.1, .1, 1, .2, .05), .01, 1e-15);
    EXPECT_EQ(low_calls, 2);
    EXPECT_EQ(high_calls, 1); // Exact inactive neighbor remains unevaluated.
}

TEST(FiniteScaleCompositionTest, UnrepresentableAndInvalidRawOutputsAreNotClampedFinite) {
    const double scale = std::numeric_limits<double>::max();
    TransformLeaf too_large(ConstantInterp{1000}, StandardTransform4D{}, 100);
    EXPECT_EQ(too_large.price(scale, scale, 1, .2, .05), std::numeric_limits<double>::infinity());
    TransformLeaf invalid(ConstantInterp{std::numeric_limits<double>::quiet_NaN()},
                          StandardTransform4D{}, 100);
    EXPECT_TRUE(std::isnan(invalid.price(100, 100, 1, .2, .05)));
    TransformLeaf inactive(ConstantInterp{-1}, StandardTransform4D{}, 100);
    EXPECT_EQ(inactive.price(scale, scale, 1, .2, .05), 0.0);
}

TEST(FiniteScaleCompositionTest, NormalizationPreservesExactPayoffClosure) {
    const double strike = 1e100;
    const double spot = std::nextafter(strike, 0.0);
    AnalyticalEEP eep(OptionType::PUT, 0);
    // Subtraction is exact for these neighboring doubles. Dividing to unit
    // strike first would throw away meaningful digits of the payoff.
    EXPECT_EQ(eep.european_price(spot, strike, 0, .2, -.05), strike - spot);
}
} // namespace
} // namespace mango
