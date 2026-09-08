// SPDX-License-Identifier: MIT
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/certification/continuous_cell.hpp"
#include <gtest/gtest.h>
using namespace mango;
using namespace mango::detail::certification;
namespace {
BSplineND<double, 4> bezier(const std::array<double, 4> &sigma_controls, const SurfaceBounds &b) {
    const std::array<std::pair<double, double>, 4> ranges{{{b.m_min, b.m_max},
                                                           {b.tau_min, b.tau_max},
                                                           {b.sigma_min, b.sigma_max},
                                                           {b.rate_min, b.rate_max}}};
    std::array<std::vector<double>, 4> grids, knots;
    for (std::size_t d = 0; d < 4; ++d) {
        auto [a, z] = ranges[d];
        grids[d] = {a, a + (z - a) / 3, a + 2 * (z - a) / 3, z};
        knots[d] = {a, a, a, a, z, z, z, z};
    }
    std::vector<double> coefficients(256);
    for (std::size_t i = 0; i < 256; ++i)
        coefficients[i] = sigma_controls[(i / 4) % 4];
    return BSplineND<double, 4>::create(std::move(grids), std::move(knots), std::move(coefficients))
        .value();
}
} // namespace
TEST(PhysicalCellProofTest, EuropeanAddBackCanCertifyDecreasingPremium) {
    const SurfaceBounds b{-.001, .001, .99, 1.01, .19, .21, -.001, .001};
    // f(sigma) = 4 - 10*sigma. European vega exceeds ten dollars here.
    auto spline = bezier({2.1, 2.1 - .2 / 3, 2.1 - 2 * .2 / 3, 1.9}, b);
    auto result = prove_continuous_bspline_cell(spline, {3, 3, 3, 3}, 100, OptionType::PUT, 0, b);
    EXPECT_EQ(result.status, PriceProofStatus::Certified);
    EXPECT_FALSE(result.witness);
}

TEST(PhysicalCellProofTest, FindsPhysicalNegativePocketBetweenAllSeventeenScanPoints) {
    const SurfaceBounds b{-.001, .001, .1, .11, .2, .20001, .04, .06};
    const double a = 17. / 32, c = a * a - 1. / (128 * 128);
    auto spline = bezier({.1, .1 + c, .1 + 2 * c - a, 1.1 + 3 * c - 3 * a}, b);
    auto source = std::make_shared<const BSplineND<double, 4>>(spline);
    BSplineLeaf leaf(BSplineTransformLeaf(SharedBSplineInterp<4>(source), {}, 100),
                     AnalyticalEEP(OptionType::PUT, 0));
    double previous = leaf.price(100, 100, .105, .2, .05);
    for (int i = 1; i <= 16; ++i) {
        const double next = leaf.price(100, 100, .105, .2 + (.20001 - .2) * i / 16, .05);
        ASSERT_GT(next, previous);
        previous = next;
    }
    auto result = prove_continuous_bspline_cell(spline, {3, 3, 3, 3}, 100, OptionType::PUT, 0, b);
    ASSERT_EQ(result.status, PriceProofStatus::NegativeWitness)
        << "nodes=" << result.nodes << " reason=" << static_cast<int>(result.reason);
    ASSERT_TRUE(result.witness);
    EXPECT_TRUE(result.witness_vega_per_strike.strictly_negative());
    auto p = *result.witness;
    EXPECT_LT(
        leaf.vega(p.spot, p.strike, p.maturity, p.volatility, get_zero_rate(p.rate, p.maturity)),
        0);
    EXPECT_TRUE(
        MoneynessDomain({std::exp(b.m_min), std::exp(b.m_max)}).contains_quote(p.spot, p.strike));
}

TEST(PhysicalCellProofTest, StructuralFlatAndFloorMaskedIntervalsRemainValidAtExpiryLimit) {
    const SurfaceBounds b{-.01, .01, 0., .01, .1, .3, -.01, .01};
    for (const std::array<double, 4> controls :
         {std::array<double, 4>{0, 0, 0, 0}, std::array<double, 4>{-1, -2, -3, -4}}) {
        auto spline = bezier(controls, b);
        auto result =
            prove_continuous_bspline_cell(spline, {3, 3, 3, 3}, 100, OptionType::PUT, 0, b);
        EXPECT_EQ(result.status, PriceProofStatus::Certified);
        EXPECT_EQ(result.nodes, 1u);
        EXPECT_FALSE(result.witness);
    }
}

TEST(PhysicalCellProofTest, ExhaustedBudgetsAndUnreachableNegativityAreIndeterminate) {
    const SurfaceBounds b{-.001, .001, .99, 1.01, .19, .21, -.001, .001};
    auto spline = bezier({20, 20 - 2. / 3, 20 - 4. / 3, 18}, b);
    auto zero_budget =
        prove_continuous_bspline_cell(spline, {3, 3, 3, 3}, 100, OptionType::PUT, 0, b, {0, 24});
    EXPECT_EQ(zero_budget.status, PriceProofStatus::Indeterminate);
    EXPECT_EQ(zero_budget.reason, mango::detail::proof::StopReason::NodeBudget);
    EXPECT_EQ(zero_budget.nodes, 0u);
    auto outside = b;
    outside.m_min = std::log(10.);
    outside.m_max = std::log(11.);
    outside.ratio_bounds = MoneynessBounds{10, 11};
    auto unreachable =
        prove_continuous_bspline_cell(spline, {3, 3, 3, 3}, 100, OptionType::PUT, 0, outside);
    EXPECT_EQ(unreachable.status, PriceProofStatus::Indeterminate);
    EXPECT_FALSE(unreachable.witness);
    auto reachable =
        prove_continuous_bspline_cell(spline, {3, 3, 3, 3}, 100, OptionType::PUT, 0, b);
    EXPECT_EQ(reachable.status, PriceProofStatus::NegativeWitness);
}
