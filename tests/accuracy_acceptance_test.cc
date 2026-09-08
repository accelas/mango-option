// SPDX-License-Identifier: MIT
#include "mango/option/table/accuracy/acceptance.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace mango::detail::accuracy {
namespace {

ReferenceAccuracySummary measured(double price_max, double iv_max) {
    return {
        .price=ReferenceErrorSummary{.requested=10, .measured=10,
            .max_error=price_max, .rms_error=price_max/10},
        .iv=ReferenceErrorSummary{.requested=10, .measured=10,
            .max_error=iv_max, .rms_error=iv_max/10},
    };
}

Assessment assess_actual_iv(const ReferenceAccuracySummary& evidence,
    Prerequisites prerequisites, Targets targets={}, Policy policy=Policy::Strict) {
    return assess(evidence, prerequisites, targets, policy, IvMetricKind::AbsoluteIvError);
}

const Prerequisites viable_certified{
    .viability=Viability::Passed, .certificate=PriceProofStatus::Certified};

TEST(AccuracyAcceptanceTest, DefaultsGateMaximumQuoteAndDecimalIvErrors) {
    auto passing=assess_actual_iv(measured(0.01, 2e-5), viable_certified);
    EXPECT_EQ(passing.decision(), Decision::Accepted);
    EXPECT_EQ(passing.price_target_met(), true);
    EXPECT_EQ(passing.iv_target_met(), true);
    EXPECT_DOUBLE_EQ(passing.targets().price, 0.01);
    EXPECT_EQ(passing.targets().iv, 2e-5);  // 0.2 absolute-IV bp, not 2 bp.

    auto price_miss=assess_actual_iv(measured(0.0101, 1e-5), viable_certified);
    EXPECT_EQ(price_miss.decision(), Decision::TargetsMissed);
    EXPECT_EQ(price_miss.price_target_met(), false);
    EXPECT_LT(*price_miss.evidence().price->rms_error, 0.01);

    auto iv_miss=assess_actual_iv(measured(0.005, 2.01e-5), viable_certified);
    EXPECT_EQ(iv_miss.decision(), Decision::TargetsMissed);
    EXPECT_EQ(iv_miss.iv_target_met(), false);
    EXPECT_LT(*iv_miss.evidence().iv->rms_error, 2e-5);
}

TEST(AccuracyAcceptanceTest, BestEffortNeverBypassesViabilityOrCertification) {
    const auto evidence=measured(0.02, 3e-5);
    auto best=assess_actual_iv(evidence, viable_certified, {}, Policy::BestEffort);
    EXPECT_EQ(best.decision(), Decision::AcceptedBestEffort);
    EXPECT_EQ(best.price_target_met(), false);
    EXPECT_EQ(best.iv_target_met(), false);
    EXPECT_EQ(best.policy(), Policy::BestEffort);

    for (auto policy : {Policy::Strict, Policy::BestEffort}) {
        for (auto viability : {Viability::Unassessed, Viability::Failed}) {
            auto assessment=assess_actual_iv(evidence, {viability, PriceProofStatus::Certified}, {}, policy);
            EXPECT_EQ(assessment.decision(), viability==Viability::Failed
                ? Decision::ViabilityFailed : Decision::ViabilityUnassessed);
            EXPECT_EQ(assessment.prerequisites().viability, viability);
            EXPECT_EQ(assessment.price_target_met(), false);
        }
        for (auto certificate : {PriceProofStatus::NotRun, PriceProofStatus::Indeterminate, PriceProofStatus::NegativeWitness}) {
            auto assessment=assess_actual_iv(evidence, {Viability::Passed, certificate}, {}, policy);
            EXPECT_EQ(assessment.decision(), certificate==PriceProofStatus::NegativeWitness
                ? Decision::CertificateViolated : certificate==PriceProofStatus::NotRun
                    ? Decision::CertificateNotRun : Decision::CertificateIndeterminate);
            EXPECT_EQ(assessment.prerequisites().certificate, certificate);
            EXPECT_EQ(assessment.evidence().price->measured, 10u);
        }
    }
}

TEST(AccuracyAcceptanceTest, AllFilteredIvSupportsOnlyPriceOnlyAcceptance) {
    auto evidence=measured(0.009, 1e-5);
    evidence.iv=ReferenceErrorSummary{.requested=10, .filtered=10};
    for (auto policy : {Policy::Strict, Policy::BestEffort}) {
        auto price_only=assess_actual_iv(evidence, viable_certified, {.iv=std::nullopt}, policy);
        EXPECT_EQ(price_only.decision(), Decision::Accepted);
        EXPECT_EQ(price_only.price_target_met(), true);
        EXPECT_FALSE(price_only.iv_target_met().has_value());
        EXPECT_EQ(price_only.evidence().iv->measured, 0u);
        EXPECT_EQ(price_only.evidence().iv->filtered, 10u);
        EXPECT_FALSE(price_only.evidence().iv->max_error.has_value());
        EXPECT_FALSE(price_only.evidence().iv->rms_error.has_value());

        auto iv_targeted=assess_actual_iv(evidence, viable_certified, {}, policy);
        EXPECT_EQ(iv_targeted.decision(), Decision::IvUnmeasured);
        EXPECT_FALSE(iv_targeted.iv_target_met().has_value());
    }
    evidence.iv=ReferenceErrorSummary{.requested=10, .measured=3, .filtered=7,
        .max_error=1e-5, .rms_error=5e-6};
    auto mixed=assess_actual_iv(evidence, viable_certified);
    EXPECT_EQ(mixed.decision(), Decision::Accepted);
    EXPECT_EQ(mixed.iv_target_met(), true);
    EXPECT_EQ(mixed.evidence().iv->requested, 10u);
}

TEST(AccuracyAcceptanceTest, MissingRequiredRowsCannotImproveAccuracyByDisappearing) {
    for (auto counter : {&ReferenceErrorSummary::unresolved, &ReferenceErrorSummary::refused,
                         &ReferenceErrorSummary::untested}) {
        for (bool price_channel : {false, true}) for (auto policy : {Policy::Strict, Policy::BestEffort}) {
            auto evidence=measured(0.009, 1e-5);
            auto& channel=price_channel ? evidence.price : evidence.iv;
            channel->measured=9;
            (*channel).*counter=1;
            auto result=assess_actual_iv(evidence, viable_certified, {}, policy);
            EXPECT_EQ(result.decision(), price_channel
                ? Decision::IncompletePriceEvidence : Decision::IncompleteIvEvidence);
            EXPECT_FALSE((price_channel ? result.price_target_met() : result.iv_target_met()).has_value());
            EXPECT_EQ((price_channel ? result.evidence().price : result.evidence().iv)->requested, 10u);

            // A known violation remains visible even though the rest of the
            // population was not completely assessed.
            channel->max_error=0.1;
            auto witnessed=assess_actual_iv(evidence, viable_certified, {}, policy);
            EXPECT_EQ(price_channel ? witnessed.price_target_met() : witnessed.iv_target_met(), false);
        }
    }
    auto missing_price=assess_actual_iv({}, viable_certified, {}, Policy::BestEffort);
    EXPECT_EQ(missing_price.decision(), Decision::PriceUnmeasured);
    EXPECT_FALSE(missing_price.price_target_met().has_value());
}

TEST(AccuracyAcceptanceTest, MalformedOrStructuralEvidenceCannotBecomeMeasuredAccuracy) {
    const double nan=std::numeric_limits<double>::quiet_NaN();
    const ReferenceErrorSummary invalid[] = {
        {.requested=10, .measured=9, .max_error=0.0},
        {.requested=10, .filtered=10, .max_error=0.0},
        {.requested=10, .filtered=10, .rms_error=0.0},
        {.requested=10, .measured=10},
        {.requested=10, .measured=10, .max_error=nan},
        {.requested=10, .measured=10, .max_error=0.0, .rms_error=-1.0},
        {.requested=10, .measured=10, .max_error=0.0, .max_uncertainty=nan},
        {.requested=10, .structurally_exact=10},
        {.requested=10, .measured=9, .max_error=0.0, .structurally_exact=1},
        {.requested=std::numeric_limits<size_t>::max(),
            .measured=std::numeric_limits<size_t>::max(), .filtered=1, .max_error=0.0},
    };
    for (const auto& channel : invalid) for (bool price_channel : {false, true}) {
        auto evidence=measured(0.009, 1e-5);
        (price_channel ? evidence.price : evidence.iv)=channel;
        auto result=assess_actual_iv(evidence, viable_certified, {}, Policy::BestEffort);
        EXPECT_EQ(result.decision(), Decision::InvalidEvidence);
        EXPECT_FALSE(result.price_target_met().has_value());
        EXPECT_FALSE(result.iv_target_met().has_value());
        EXPECT_EQ((price_channel ? result.evidence().price : result.evidence().iv)->requested,
                  channel.requested);
    }
    auto filtered_price=measured(0.0, 0.0);
    filtered_price.price=ReferenceErrorSummary{.requested=10, .filtered=10};
    EXPECT_EQ(assess_actual_iv(filtered_price, viable_certified).decision(), Decision::InvalidEvidence);
}

TEST(AccuracyAcceptanceTest, ConfiguredTargetsMustBeFiniteAndPositive) {
    for (double invalid : {0.0, -0.01, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        for (bool price_target : {false, true}) {
            Targets targets;
            if (price_target) targets.price=invalid;
            else targets.iv=invalid;
            auto result=assess_actual_iv(measured(0.001, 1e-6), viable_certified, targets, Policy::BestEffort);
            EXPECT_EQ(result.decision(), Decision::InvalidTargets);
            EXPECT_FALSE(result.price_target_met().has_value());
            EXPECT_FALSE(result.iv_target_met().has_value());
        }
    }
    auto configured=assess_actual_iv(measured(0.02, 3e-5), viable_certified, {.price=0.03, .iv=4e-5});
    EXPECT_EQ(configured.decision(), Decision::Accepted);
}

TEST(AccuracyAcceptanceTest, ProxyOrUnknownIvEvidenceCannotClaimActualIvAccuracy) {
    for (auto kind : {IvMetricKind::Unknown, IvMetricKind::PriceVegaProxy}) {
        for (auto policy : {Policy::Strict, Policy::BestEffort}) {
            auto iv_targeted=assess(measured(0.001, 1e-6), viable_certified, {}, policy, kind);
            EXPECT_EQ(iv_targeted.decision(), Decision::IvMetricNotActual);
            EXPECT_FALSE(iv_targeted.iv_target_met().has_value());
            EXPECT_EQ(iv_targeted.iv_metric_kind(), kind);
            EXPECT_EQ(iv_targeted.evidence().iv->measured, 10u);

            auto price_only=assess(measured(0.001, 1e-6), viable_certified,
                {.iv=std::nullopt}, policy, kind);
            EXPECT_EQ(price_only.decision(), Decision::Accepted);
            EXPECT_FALSE(price_only.iv_target_met().has_value());
            EXPECT_EQ(price_only.iv_metric_kind(), kind);
        }
    }
    auto absent_kind=assess(measured(0.001, 1e-6), viable_certified);
    EXPECT_EQ(absent_kind.decision(), Decision::IvMetricNotActual);
}

TEST(AccuracyAcceptanceTest, AssessmentOwnsImmutableEvidenceEvenOnRefusal) {
    static_assert(!std::is_copy_assignable_v<Assessment>);
    static_assert(std::is_trivially_copyable_v<Assessment>);
    auto source=measured(0.02, 3e-5);
    const auto result=assess_actual_iv(source, viable_certified);
    source.price->max_error=0.0;
    source.iv.reset();
    EXPECT_EQ(result.decision(), Decision::TargetsMissed);
    EXPECT_EQ(result.evidence().price->max_error, 0.02);
    ASSERT_TRUE(result.evidence().iv.has_value());
    EXPECT_EQ(result.evidence().iv->requested, 10u);
    EXPECT_EQ(result.evidence().iv->max_error, 3e-5);
}

TEST(AccuracyAcceptanceTest, OnlyAffirmativePrerequisiteStatusesPermitAdmission) {
    auto unknown_viability=assess_actual_iv(measured(0.001, 1e-6),
        {static_cast<Viability>(255), PriceProofStatus::Certified});
    EXPECT_EQ(unknown_viability.decision(), Decision::ViabilityUnassessed);
    auto unknown_proof=assess_actual_iv(measured(0.001, 1e-6),
        {Viability::Passed, static_cast<PriceProofStatus>(255)});
    EXPECT_EQ(unknown_proof.decision(), Decision::CertificateIndeterminate);
}

} // namespace
} // namespace mango::detail::accuracy
