// SPDX-License-Identifier: MIT
#include "mango/option/table/accuracy/final_candidate.hpp"
#include "mango/option/european_option.hpp"
#include <gtest/gtest.h>

namespace mango::detail::accuracy {
namespace {
BSplinePriceTable certified_call(const SurfaceBounds& b, double premium) {
    const std::array<std::pair<double,double>,4> ranges{{
        {b.m_min,b.m_max},{b.tau_min,b.tau_max},
        {b.sigma_min,b.sigma_max},{b.rate_min,b.rate_max}}};
    BSplineND<double,4>::GridArray grid;
    BSplineND<double,4>::KnotArray knots;
    for (size_t d=0; d<4; ++d) {
        const auto [lo,hi]=ranges[d];
        grid[d]={lo,std::lerp(lo,hi,1./3),std::lerp(lo,hi,2./3),hi};
        knots[d]={lo,lo,lo,lo,hi,hi,hi,hi};
    }
    auto spline=BSplineND<double,4>::create(
        std::move(grid),std::move(knots),std::vector<double>(256,premium));
    auto source=std::make_shared<const BSplineND<double,4>>(std::move(spline.value()));
    BSplineLeaf leaf(BSplineTransformLeaf(SharedBSplineInterp<4>(source),{},100),
                     AnalyticalEEP(OptionType::CALL,0));
    return BSplinePriceTable::create(leaf,b,OptionType::CALL,0).value();
}

PhysicalReferenceRow call_reference(double spot=100, double tau=1, double sigma=.2) {
    PhysicalReferenceRow row;
    row.query=PricingParams(OptionSpec{.spot=spot,.strike=100,.maturity=tau,
        .rate=0.,.dividend_yield=0.,.option_type=OptionType::CALL},sigma);
    // Cash-free q=0,r=0 American call equals this independent closed form.
    row.reference_price=bs_price(spot,100,tau,sigma,0,0,OptionType::CALL);
    row.price_qualification=PriceQualification::Qualified;
    row.iv_qualification=IvQualification::Measurable;
    row.price_uncertainty=0.;
    row.iv_uncertainty=0.;
    row.reference_vega_lower_bound=bs_vega(spot,100,tau,sigma,0);
    row.provenance.source=ReferenceSource::Analytic;
    row.provenance.qualification_digest[0]=1; // Declared BSM identity in this test.
    return row;
}

TEST(AccuracyFinalCandidateTest, UsesRealCertifiedPayloadAndActualConditionedInversion) {
    const SurfaceBounds b{-.01,.01,.99,1.01,.18,.22,-.001,.001};
    const auto table=certified_call(b,.03);
    const std::vector rows{call_reference()};
    const AccuracyRequest strict{.max_price_error=.05,.max_iv_error=.0005};
    const auto result=collect_final_candidate(table,rows,strict);
    EXPECT_EQ(result.assessment.prerequisites().certificate,PriceProofStatus::Certified);
    EXPECT_EQ(result.assessment.decision(),Decision::TargetsMissed);
    EXPECT_EQ(result.assessment.price_target_met(),true);
    EXPECT_EQ(result.assessment.iv_target_met(),false);
    ASSERT_TRUE(result.assessment.evidence().iv->max_error);
    EXPECT_GT(*result.assessment.evidence().iv->max_error,.0007);
    EXPECT_LT(*result.assessment.evidence().iv->max_error,.0008);
    EXPECT_NEAR(*result.assessment.evidence().price->max_error,.03,1e-10);
    auto best=strict;
    best.policy=AccuracyPolicy::BestEffort;
    EXPECT_EQ(collect_final_candidate(table,rows,best).assessment.decision(),
              Decision::AcceptedBestEffort);
    EXPECT_EQ(&table.inner(),&BSplinePriceTable(table).inner());
}

TEST(AccuracyFinalCandidateTest, AllFilteredPriceOnlyPassesButUnmeasurableIvDoesNot) {
    const SurfaceBounds b{std::log(1.99),std::log(2.01),.00009,.00011,.009,.011,-.001,.001};
    const auto table=certified_call(b,0.);
    auto row=call_reference(200,.0001,.01);
    row.iv_qualification=IvQualification::FilteredTimeValue;
    row.reference_vega_lower_bound.reset();
    const std::vector rows{row};
    auto request=AccuracyRequest{};
    request.max_iv_error.reset();
    const auto price_only=collect_final_candidate(table,rows,request);
    EXPECT_EQ(price_only.assessment.decision(),Decision::Accepted);
    EXPECT_EQ(price_only.assessment.evidence().price->measured,1u);
    EXPECT_EQ(price_only.assessment.evidence().iv->filtered,1u);
    EXPECT_FALSE(price_only.assessment.evidence().iv->max_error);
    EXPECT_EQ(collect_final_candidate(table,rows,AccuracyRequest{}).assessment.decision(),
              Decision::IvUnmeasured);

    const auto miss=certified_call(b,.03);
    request.policy=AccuracyPolicy::BestEffort;
    const auto refused=collect_final_candidate(miss,rows,request);
    EXPECT_EQ(refused.assessment.decision(),Decision::ViabilityUnassessed);
    EXPECT_EQ(refused.assessment.price_target_met(),false);
}

TEST(AccuracyFinalCandidateTest, MissingSensitivityAndModelMismatchCannotInventViability) {
    const SurfaceBounds b{-.01,.01,.99,1.01,.18,.22,-.001,.001};
    const auto table=certified_call(b,.03);
    auto row=call_reference();
    row.reference_vega_lower_bound.reset();
    AccuracyRequest request{.max_price_error=.01,.max_iv_error=std::nullopt,
        .policy=AccuracyPolicy::BestEffort};
    EXPECT_EQ(collect_final_candidate(table,std::vector{row},request).assessment.decision(),
              Decision::ViabilityUnassessed);
    row.query.option_type=OptionType::PUT;
    const auto wrong_model=collect_final_candidate(table,std::vector{row},request);
    EXPECT_EQ(wrong_model.assessment.evidence().price->refused,1u);
    EXPECT_NE(wrong_model.assessment.decision(),Decision::AcceptedBestEffort);
}

TEST(AccuracyFinalCandidateTest, BestEffortRetainsTheIndependentHardGuard) {
    const SurfaceBounds b{-.01,.01,.99,1.01,.18,.22,-.001,.001};
    const auto table=certified_call(b,10.);
    auto row=call_reference();
    row.iv_applicable=false;
    const AccuracyRequest request{.max_price_error=.01,.max_iv_error=std::nullopt,
        .policy=AccuracyPolicy::BestEffort};
    const auto result=collect_final_candidate(table,std::vector{row},request);
    EXPECT_EQ(result.assessment.decision(),Decision::ViabilityFailed);
    EXPECT_EQ(result.assessment.prerequisites().certificate,PriceProofStatus::Certified);
    EXPECT_EQ(result.assessment.evidence().price->measured,1u);
    EXPECT_EQ(result.assessment.price_target_met(),false);
    EXPECT_FALSE(result.assessment.evidence().iv->max_error);
}

struct ForgedSurface {
    PriceProofStatus proof_status() const { return PriceProofStatus::Certified; }
};
static_assert(!FinalAccuracyTable<ForgedSurface>);
static_assert(FinalAccuracyTable<BSplinePriceTable>);
} // namespace
} // namespace mango::detail::accuracy
