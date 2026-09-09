// SPDX-License-Identifier: MIT
#include "mango/option/table/bspline/bspline_3d_surface.hpp"
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

TEST(PhysicalCellProofTest, InvalidClampMetadataCannotReceiveCellEvidence) {
    const SurfaceBounds b{-.001, .001, .99, 1.01, .19, .21, -.001, .001};
    auto source = bezier({0, 0, 0, 0}, b);
    std::array<std::vector<double>, 4> grids, knots;
    for (std::size_t d = 0; d < 4; ++d) {
        grids[d] = source.grid(d);
        knots[d] = source.knots(d);
    }
    grids[0][0] = std::numeric_limits<double>::quiet_NaN();
    auto invalid =
        BSplineND<double, 4>::create(std::move(grids), std::move(knots), source.coefficients());
    ASSERT_TRUE(invalid); // Raw math construction does not validate this metadata.
    auto result = prove_continuous_bspline_cell(*invalid, {3, 3, 3, 3}, 100, OptionType::PUT, 0, b);
    EXPECT_EQ(result.status, PriceProofStatus::Indeterminate);
}

TEST(PhysicalCellProofTest, WholeDomainUsesZeroSlopeOnSigmaClampedExtension) {
    const SurfaceBounds support{-.001, .001, .99, 1.01, .19, .21, -.001, .001};
    auto spline = bezier({20, 20 - 2. / 3, 20 - 4. / 3, 18}, support);
    auto requested = support;
    requested.sigma_min = .3;
    requested.sigma_max = .4;
    auto result = prove_continuous_bspline(spline, 100, OptionType::PUT, 0, requested);
    EXPECT_EQ(result.status, PriceProofStatus::Certified);
    EXPECT_FALSE(result.witness);
}

TEST(PhysicalCellProofTest, RestrictedDomainExcludesPocketButSingletonRateCanStillWitnessIt) {
    const SurfaceBounds support{-.001, .001, .1, .11, .2, .20001, .04, .06};
    const double a = 17. / 32, c = a * a - 1. / (128 * 128);
    auto spline = bezier({.1, .1 + c, .1 + 2 * c - a, 1.1 + 3 * c - 3 * a}, support);
    auto requested = support;
    requested.sigma_max = .200001;
    auto restricted = prove_continuous_bspline(spline, 100, OptionType::PUT, 0, requested);
    EXPECT_EQ(restricted.status, PriceProofStatus::Certified);
    requested = support;
    requested.rate_min = .05;
    requested.rate_max = .05;
    requested.tau_min = .105;
    requested.tau_max = .105;
    auto negative = prove_continuous_bspline(spline, 100, OptionType::PUT, 0, requested);
    ASSERT_EQ(negative.status, PriceProofStatus::NegativeWitness) << "nodes=" << negative.nodes;
    ASSERT_TRUE(negative.witness);
    EXPECT_EQ(negative.witness->maturity, .105);
    EXPECT_EQ(std::get<double>(negative.witness->rate), .05);
}

TEST(PhysicalCellProofTest, WholeProofCoversEveryStoredCellWithinOneSharedBudget) {
    const SurfaceBounds b{-.001, .001, .1, .11, .2, .20004, .04, .06};
    auto base = bezier({1, 2, 3, 4}, b);
    std::array<std::vector<double>, 4> grids, knots;
    for (std::size_t d = 0; d < 4; ++d) {
        grids[d] = base.grid(d);
        knots[d] = base.knots(d);
    }
    grids[2] = {.2, .20001, .20002, .20003, .20004};
    knots[2] = {.2, .2, .2, .2, .20002, .20004, .20004, .20004, .20004};
    std::vector<double> controls(320);
    const std::array<double, 5> sigma{1, 2, 3, 4, 0};
    for (std::size_t i = 0; i < controls.size(); ++i)
        controls[i] = sigma[(i / 4) % 5];
    auto spline =
        BSplineND<double, 4>::create(std::move(grids), std::move(knots), std::move(controls))
            .value();
    auto first = prove_continuous_bspline_cell(spline, {3, 3, 3, 3}, 100, OptionType::PUT, 0, b);
    ASSERT_EQ(first.status, PriceProofStatus::Certified);
    auto limited = prove_continuous_bspline(spline, 100, OptionType::PUT, 0, b, {1, 24});
    EXPECT_EQ(limited.status, PriceProofStatus::Indeterminate);
    EXPECT_EQ(limited.reason, mango::detail::proof::StopReason::NodeBudget);
    EXPECT_EQ(limited.nodes, 1u);
    auto whole = prove_continuous_bspline(spline, 100, OptionType::PUT, 0, b);
    ASSERT_EQ(whole.status, PriceProofStatus::NegativeWitness);
    ASSERT_TRUE(whole.witness);
    EXPECT_GT(whole.witness->volatility, .20002);
}

namespace {
BSplineND<double, 3> dimensionless_linear(double u_partial, double z_partial, double offset) {
    std::array<std::vector<double>, 3> grids{
        {{-.01, -.003, .003, .01}, {.4, .45, .55, .6}, {-.1, -.03, .03, .1}}};
    std::array<std::vector<double>, 3> knots;
    for (std::size_t d = 0; d < 3; ++d) {
        const double a = grids[d].front(), b = grids[d].back();
        knots[d] = {a, a, a, a, b, b, b, b};
    }
    std::vector<double> coefficients(64);
    for (std::size_t i = 0; i < 64; ++i)
        coefficients[i] = offset + u_partial * (.4 + .2 * ((i / 4) % 4) / 3) +
                          z_partial * (-.1 + .2 * (i % 4) / 3);
    return BSplineND<double, 3>::create(std::move(grids), std::move(knots), std::move(coefficients))
        .value();
}
} // namespace
TEST(PhysicalCellProofTest, DimensionlessProofUsesBothActualCoordinateDerivatives) {
    const SurfaceBounds b{-.001, .001, .99, 1.01, .99, 1.01, .49, .51};
    auto positive = dimensionless_linear(1, -1, .5);
    auto result = prove_dimensionless_bspline(positive, 100, OptionType::PUT, b);
    EXPECT_EQ(result.status, PriceProofStatus::Certified);
    EXPECT_FALSE(result.witness);
    auto negative = dimensionless_linear(1, 20, 10);
    auto violation = prove_dimensionless_bspline(negative, 100, OptionType::PUT, b);
    EXPECT_EQ(violation.status, PriceProofStatus::NegativeWitness);
    ASSERT_TRUE(violation.witness);
    EXPECT_TRUE(violation.witness_vega_per_strike.strictly_negative());
    BSpline3DLeaf leaf(
        BSpline3DTransformLeaf(
            SharedBSplineInterp<3>(std::make_shared<const BSplineND<double, 3>>(negative)), {},
            100),
        AnalyticalEEP(OptionType::PUT, 0));
    const auto &query = *violation.witness;
    EXPECT_LT(leaf.vega(query.spot, query.strike, query.maturity, query.volatility,
                        get_zero_rate(query.rate, query.maturity)),
              0);
}

TEST(PhysicalCellProofTest, DimensionlessSubdivisionCertifiesPositiveQuadraticSensitivity) {
    auto base = dimensionless_linear(1, 0, 100);
    std::array<std::vector<double>, 3> grids, knots;
    for (std::size_t d = 0; d < 3; ++d) {
        grids[d] = base.grid(d);
        knots[d] = base.knots(d);
    }
    // f_u=10000*((u-.5)^2+.0001)>0, whose coarse Bernstein middle
    // derivative coefficient is -99. f stays positive everywhere.
    const std::array<double, 4> u{100, 100 + 101 * .2 / 3, 100 + 2 * .2 / 3, 100 + 103 * .2 / 3};
    std::vector<double> coefficients(64);
    for (std::size_t i = 0; i < 64; ++i)
        coefficients[i] = u[(i / 4) % 4];
    auto spline =
        BSplineND<double, 3>::create(std::move(grids), std::move(knots), std::move(coefficients))
            .value();
    const SurfaceBounds b{-.001, .001, 1, 1, std::sqrt(.8), std::sqrt(1.2), .5, .5};
    auto result = prove_dimensionless_bspline(spline, 100, OptionType::PUT, b, {4096, 24});
    EXPECT_EQ(result.status, PriceProofStatus::Certified) << "nodes=" << result.nodes;
    EXPECT_GT(result.nodes, 2u);
}

TEST(PhysicalCellProofTest, DimensionlessClampAndBudgetKeepTheirPhysicalMeaning) {
    auto spline = dimensionless_linear(1, -1, .5);
    SurfaceBounds b{-.001, .001, 3, 4, .99, 1.01, .49, .51};
    // Transformed time is entirely above its grid; its partial is zero.
    EXPECT_EQ(prove_dimensionless_bspline(spline, 100, OptionType::PUT, b).status,
              PriceProofStatus::Certified);
    auto limited = prove_dimensionless_bspline(spline, 100, OptionType::PUT, b, {1, 24});
    EXPECT_EQ(limited.status, PriceProofStatus::Indeterminate);
    EXPECT_EQ(limited.reason, mango::detail::proof::StopReason::NodeBudget);
    EXPECT_EQ(limited.nodes, 1u);
    b.rate_min = 0;
    auto unsupported = prove_dimensionless_bspline(spline, 100, OptionType::PUT, b);
    EXPECT_EQ(unsupported.status, PriceProofStatus::Indeterminate);
    EXPECT_FALSE(unsupported.witness);
}

namespace {
BSplineMultiKRefInner two_references(double left_slope, double right_slope) {
    const SurfaceBounds b{-.1, .1, 0, 1, .1, .4, .01, .1};
    std::vector<BSplineSegmentedSurface> members;
    const std::array<double, 2> refs{80, 120}, slopes{left_slope, right_slope};
    for (std::size_t i = 0; i < 2; ++i) {
        const double slope = slopes[i];
        auto spline = std::make_shared<const BSplineND<double, 4>>(
            bezier({.5 - .1 * slope, .5, .5 + .1 * slope, .5 + .2 * slope}, b));
        std::vector<BSplineSegmentedLeaf> leaves;
        leaves.emplace_back(SharedBSplineInterp<4>(spline), StandardTransform4D{}, refs[i]);
        members.emplace_back(std::move(leaves), TauSegmentSplit({0}, {1}, {0}, {1}, refs[i]));
    }
    return BSplineMultiKRefInner(std::move(members), MultiKRefSplit({80, 120}));
}
} // namespace
TEST(PhysicalCellProofTest, PositiveReferenceMixtureCanMaskADecreasingRawMember) {
    auto inner = two_references(-.1, .4);
    SurfaceBounds b{-.01, .01, .1, .9, .15, .35, .04, .06, StrikeBounds{95, 105}};
    auto result = prove_segmented_bspline(inner, OptionType::CALL, 0, b);
    EXPECT_EQ(result.status, PriceProofStatus::Certified);
    EXPECT_FALSE(result.witness);
    b.strike_bounds = StrikeBounds{80, 105};
    auto negative = prove_segmented_bspline(inner, OptionType::CALL, 0, b);
    ASSERT_EQ(negative.status, PriceProofStatus::NegativeWitness);
    ASSERT_TRUE(negative.witness);
    const auto &p = *negative.witness;
    EXPECT_LT(
        inner.vega(p.spot, p.strike, p.maturity, p.volatility, get_zero_rate(p.rate, p.maturity)),
        0);
    EXPECT_TRUE(inner.contains_maturity(p.maturity));
    EXPECT_TRUE(b.strike_bounds->contains(p.strike));
}

TEST(PhysicalCellProofTest, EmptyRawInterpolantSnapshotRemainsInspectableForAdmission) {
    SharedBSplineInterp<4> empty(nullptr);
    EXPECT_FALSE(empty.has_value());
    auto snapshot = empty.immutable_snapshot();
    EXPECT_FALSE(snapshot.has_value());
}

TEST(PhysicalCellProofTest, SegmentedWitnessCannotLandInAnOmittedTimeGap) {
    auto source = two_references(-.1, -.1);
    std::vector<BSplineSegmentedSurface> members;
    for (std::size_t i = 0; i < 2; ++i) {
        const auto leaf = source.pieces()[i].pieces().front();
        members.emplace_back(
            std::vector<BSplineSegmentedLeaf>{leaf, leaf},
            TauSegmentSplit({0, .6}, {.4, 1}, {0, 0}, {.4, .4}, source.split().k_refs()[i]));
    }
    BSplineMultiKRefInner inner(std::move(members), MultiKRefSplit({80, 120}));
    const SurfaceBounds b{-.01, .01, .1, .9, .15, .35, .04, .06, StrikeBounds{95, 105}};
    ASSERT_FALSE(inner.contains_maturity(.5));
    auto result = prove_segmented_bspline(inner, OptionType::CALL, 0, b);
    ASSERT_EQ(result.status, PriceProofStatus::NegativeWitness);
    ASSERT_TRUE(result.witness);
    EXPECT_TRUE(inner.contains_maturity(result.witness->maturity));
    EXPECT_NE(result.witness->maturity, .5);
}

TEST(PhysicalCellProofTest, SegmentedFloorsMaskInactiveRawNegativeSlopes) {
    const SurfaceBounds support{-.1, .1, 0, 1, .1, .4, .01, .1};
    auto spline = std::make_shared<const BSplineND<double, 4>>(bezier({-1, -2, -3, -4}, support));
    auto source = two_references(0, 0);
    std::vector<BSplineSegmentedSurface> members;
    members.emplace_back(std::vector<BSplineSegmentedLeaf>{BSplineSegmentedLeaf(
                             SharedBSplineInterp<4>(spline), {}, 80)},
                         TauSegmentSplit({0}, {1}, {0}, {1}, 80));
    members.push_back(source.pieces()[1]);
    BSplineMultiKRefInner inner(std::move(members), MultiKRefSplit({80, 120}));
    const SurfaceBounds b{-.01, .01, .1, .9, .15, .35, .04, .06, StrikeBounds{100, 100}};
    auto result = prove_segmented_bspline(inner, OptionType::CALL, 0, b);
    EXPECT_EQ(result.status, PriceProofStatus::Certified);
    EXPECT_FALSE(result.witness);
}

TEST(PhysicalCellProofTest, SegmentedNullPayloadAndInconsistentReferencesAreIndeterminate) {
    auto source = two_references(0, 0);
    std::vector<BSplineSegmentedSurface> members = source.pieces();
    members[0] = BSplineSegmentedSurface(std::vector<BSplineSegmentedLeaf>{BSplineSegmentedLeaf(
                                             SharedBSplineInterp<4>(nullptr), {}, 80)},
                                         TauSegmentSplit({0}, {1}, {0}, {1}, 80));
    const SurfaceBounds b{-.01, .01, .1, .9, .15, .35, .04, .06, StrikeBounds{95, 105}};
    BSplineMultiKRefInner empty(std::move(members), MultiKRefSplit({80, 120}));
    EXPECT_EQ(prove_segmented_bspline(empty, OptionType::CALL, 0, b).status,
              PriceProofStatus::Indeterminate);
    BSplineMultiKRefInner mismatched(source.pieces(), MultiKRefSplit({70, 120}));
    EXPECT_EQ(prove_segmented_bspline(mismatched, OptionType::CALL, 0, b).status,
              PriceProofStatus::Indeterminate);
}

TEST(PhysicalCellProofTest, ModalContinuousProofAndEvaluatorUseIdenticalStoredCoefficients) {
    Domain<4> domain{{-.001, .99, .19, -.001}, {.001, 1.01, .21, .001}};
    const std::array<std::size_t, 4> shape{2, 2, 4, 2};
    const SurfaceBounds b{-.001, .001, .99, 1.01, .19, .21, -.001, .001};
    std::vector<double> coefficients(32);
    coefficients[0] = 2;
    coefficients[2] = -.1; // f_sigma=-10; European dominates.
    auto interpolant =
        ChebyshevModalInterpolant<4>::build_from_coefficients(coefficients, domain, shape);
    ASSERT_TRUE(interpolant);
    auto positive =
        prove_continuous_chebyshev(interpolant->polynomial(), 100, OptionType::PUT, 0, b);
    EXPECT_EQ(positive.status, PriceProofStatus::Certified);
    coefficients[2] = -1; // f_sigma=-100; full physical price decreases.
    auto negative =
        ChebyshevModalInterpolant<4>::build_from_coefficients(coefficients, domain, shape);
    ASSERT_TRUE(negative);
    auto violation = prove_continuous_chebyshev(negative->polynomial(), 100, OptionType::PUT, 0, b);
    ASSERT_EQ(violation.status, PriceProofStatus::NegativeWitness);
    ASSERT_TRUE(violation.witness);
    using Leaf =
        EEPLayer<TransformLeaf<ChebyshevModalInterpolant<4>, StandardTransform4D>, AnalyticalEEP>;
    Leaf leaf(TransformLeaf<ChebyshevModalInterpolant<4>, StandardTransform4D>(*negative, {}, 100),
              AnalyticalEEP(OptionType::PUT, 0));
    const auto &p = *violation.witness;
    EXPECT_LT(
        leaf.vega(p.spot, p.strike, p.maturity, p.volatility, get_zero_rate(p.rate, p.maturity)),
        0);
}

TEST(PhysicalCellProofTest, ModalPhysicalProofFindsTheNarrowOffScanPocket) {
    Domain<4> domain{{-.001, .1, .2, .04}, {.001, .11, .20001, .06}};
    const SurfaceBounds b{-.001, .001, .1, .11, .2, .20001, .04, .06};
    const double a = 17. / 32, c = a * a - 1. / (128 * 128);
    const double p0 = .1 + 1.5 * c - .75 * a + .125, p1 = 1.5 * c - 1.5 * a + .375,
                 p2 = -.75 * a + .375, p3 = .125;
    std::vector<double> coefficients(32);
    coefficients[0] = p0 + .5 * p2;
    coefficients[2] = p1 + .75 * p3;
    coefficients[4] = .5 * p2;
    coefficients[6] = .25 * p3;
    auto interp =
        ChebyshevModalInterpolant<4>::build_from_coefficients(coefficients, domain, {2, 2, 4, 2});
    ASSERT_TRUE(interp);
    auto result = prove_continuous_chebyshev(interp->polynomial(), 100, OptionType::PUT, 0, b);
    EXPECT_EQ(result.status, PriceProofStatus::NegativeWitness) << "nodes=" << result.nodes;
    EXPECT_TRUE(result.witness);
}

TEST(PhysicalCellProofTest, ModalDimensionlessProofUsesTheQueryPolynomialAndCoupledDerivative) {
    Domain<3> domain{{-.01, .4, -.1}, {.01, .6, .1}};
    const SurfaceBounds b{-.001, .001, .99, 1.01, .99, 1.01, .49, .51};
    std::vector<double> coefficients(8);
    coefficients[0] = 1;
    coefficients[2] = .1;
    coefficients[1] = -.1;
    auto positive =
        ChebyshevModalInterpolant<3>::build_from_coefficients(coefficients, domain, {2, 2, 2});
    ASSERT_TRUE(positive);
    EXPECT_EQ(prove_dimensionless_chebyshev(positive->polynomial(), 100, OptionType::PUT, b).status,
              PriceProofStatus::Certified);
    coefficients[0] = 10.5;
    coefficients[1] = 2;
    auto negative =
        ChebyshevModalInterpolant<3>::build_from_coefficients(coefficients, domain, {2, 2, 2});
    ASSERT_TRUE(negative);
    auto proof = prove_dimensionless_chebyshev(negative->polynomial(), 100, OptionType::PUT, b);
    ASSERT_EQ(proof.status, PriceProofStatus::NegativeWitness);
    ASSERT_TRUE(proof.witness);
    using Transform = TransformLeaf<ChebyshevModalInterpolant<3>, DimensionlessTransform3D>;
    EEPLayer<Transform, AnalyticalEEP> leaf(Transform(*negative, {}, 100),
                                            AnalyticalEEP(OptionType::PUT, 0));
    const auto &p = *proof.witness;
    EXPECT_LT(
        leaf.vega(p.spot, p.strike, p.maturity, p.volatility, get_zero_rate(p.rate, p.maturity)),
        0);
}

TEST(PhysicalCellProofTest, SingletonSigmaIsPriceableShapeNotANegativeExtensionWitness) {
    const SurfaceBounds support{-.001, .001, .99, 1.01, .19, .21, -.001, .001};
    auto spline = bezier({20, 20 - 2. / 3, 20 - 4. / 3, 18}, support);
    auto requested = support;
    requested.sigma_min = .2;
    requested.sigma_max = .2;
    auto point = prove_continuous_bspline(spline, 100, OptionType::PUT, 0, requested);
    EXPECT_EQ(point.status, PriceProofStatus::Certified);
    EXPECT_FALSE(point.witness);
    auto full_cell =
        prove_continuous_bspline_cell(spline, {3, 3, 3, 3}, 100, OptionType::PUT, 0, requested);
    EXPECT_EQ(full_cell.status, PriceProofStatus::Indeterminate);
    EXPECT_FALSE(full_cell.witness);
    const SurfaceBounds d3domain{-.001, .001, .99, 1.01, 1, 1, .49, .51};
    auto d3 = dimensionless_linear(1, 20, 10);
    EXPECT_EQ(prove_dimensionless_bspline(d3, 100, OptionType::PUT, d3domain).status,
              PriceProofStatus::Certified);
    auto segmented = two_references(-.1, -.1);
    SurfaceBounds split_domain{-.01, .01, .1, .9, .2, .2, .04, .06, StrikeBounds{95, 105}};
    auto split = prove_segmented_bspline(segmented, OptionType::CALL, 0, split_domain);
    EXPECT_EQ(split.status, PriceProofStatus::Certified);
    EXPECT_FALSE(split.witness);
    Domain<4> modal_domain{{-.001, .99, .19, -.001}, {.001, 1.01, .21, .001}};
    std::vector<double> coefficients(16);
    coefficients[0] = 2;
    coefficients[2] = -1;
    auto modal = ChebyshevModalInterpolant<4>::build_from_coefficients(coefficients, modal_domain,
                                                                       {2, 2, 2, 2});
    ASSERT_TRUE(modal);
    EXPECT_EQ(
        prove_continuous_chebyshev(modal->polynomial(), 100, OptionType::PUT, 0, requested).status,
        PriceProofStatus::Certified);
}

TEST(PhysicalCellProofTest, ModalSegmentedProofCertifiesTheFinalPositiveReferenceBlend) {
    Domain<4> domain{{-.1, 0, .1, .01}, {.1, 1, .4, .1}};
    const std::array<double, 2> refs{80, 120}, slopes{-.1, .4};
    std::vector<ModalTauInner> members;
    for (std::size_t i = 0; i < 2; ++i) {
        std::vector<double> coefficients(16);
        coefficients[0] = .5 + .05 * slopes[i];
        coefficients[2] = .15 * slopes[i];
        auto interpolant = ChebyshevModalInterpolant<4>::build_from_coefficients(
            coefficients, domain, {2, 2, 2, 2});
        ASSERT_TRUE(interpolant);
        members.emplace_back(
            std::vector<ModalSegmentedLeaf>{ModalSegmentedLeaf(*interpolant, {}, refs[i])},
            TauSegmentSplit({0}, {1}, {0}, {1}, refs[i]));
    }
    ModalMultiKRefInner inner(std::move(members), MultiKRefSplit({80, 120}));
    SurfaceBounds b{-.01, .01, .1, .9, .15, .35, .04, .06, StrikeBounds{95, 105}};
    EXPECT_EQ(prove_segmented_chebyshev(inner, OptionType::CALL, 0, b).status,
              PriceProofStatus::Certified);
    b.strike_bounds = StrikeBounds{80, 105};
    auto negative = prove_segmented_chebyshev(inner, OptionType::CALL, 0, b);
    ASSERT_EQ(negative.status, PriceProofStatus::NegativeWitness);
    ASSERT_TRUE(negative.witness);
    const auto &p = *negative.witness;
    EXPECT_LT(
        inner.vega(p.spot, p.strike, p.maturity, p.volatility, get_zero_rate(p.rate, p.maturity)),
        0);
    b.sigma_min = .2;
    b.sigma_max = .2;
    EXPECT_EQ(prove_segmented_chebyshev(inner, OptionType::CALL, 0, b).status,
              PriceProofStatus::Certified);
}

TEST(PhysicalCellProofTest, ModalProofHandlesTheActual257By9By9By5ShapeWithinDeclaredWork) {
    const std::array<std::size_t, 4> shape{257, 9, 9, 5};
    Domain<4> domain{{-.1, .1, .1, -.05}, {.1, 1, .5, .1}};
    std::vector<double> coefficients(257 * 9 * 9 * 5);
    coefficients[0] = 2;
    coefficients[5] = .1;
    coefficients[((256 * 9) * 9 + 1) * 5] = .01;
    // f_sigma=(.1+.01*T256(x))*2/.4 >= .45 everywhere.
    auto polynomial =
        ChebyshevModalInterpolant<4>::build_from_coefficients(coefficients, domain, shape);
    ASSERT_TRUE(polynomial);
    const SurfaceBounds b{-.01, .01, .1, 1, .1, .5, -.05, .1};
    auto result =
        prove_continuous_chebyshev(polynomial->polynomial(), 100, OptionType::PUT, 0, b, {3, 24});
    EXPECT_EQ(result.status, PriceProofStatus::Certified);
    EXPECT_LE(result.nodes, 3u);
    EXPECT_EQ(polynomial->num_pts(), shape);
    EXPECT_EQ(polynomial->polynomial().coefficients().size(), coefficients.size());
}
