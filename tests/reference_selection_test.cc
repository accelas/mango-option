// SPDX-License-Identifier: MIT
#include "mango/option/table/reference_selection.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <limits>

namespace mango {
namespace {

// Analytic normalized reference value1/K represents a constant quote price.
// Positive linear-in-K blending of these samples has a known approximation
// error; adequacy depends on that error, not a mocked reference-count cutoff.
ReferenceCandidateMetrics reciprocal_metrics(std::span<const double> refs,
                                              double target) {
    constexpr std::array<double, 6> queries{90, 91.3, 97.1, 103.7, 108.9, 110};
    double max_error = 0.0, squared = 0.0;
    for (double K : queries) {
        const size_t upper = std::clamp<size_t>(
            std::upper_bound(refs.begin(), refs.end(), K) - refs.begin(), 1, refs.size()-1);
        const double lo = refs[upper-1], hi = refs[upper];
        const double w = (K-lo)/(hi-lo);
        const double error = std::abs(K*((1-w)/lo+w/hi)-1.0);
        max_error = std::max(max_error, error);
        squared += error*error;
    }
    ReferenceCandidateMetrics result;
    result.decision = max_error <= target ? ReferenceCandidateDecision::Adequate
                                         : ReferenceCandidateDecision::RefineReferences;
    result.ideal_blend.price = ReferenceErrorSummary{
        .requested = queries.size(), .measured = queries.size(),
        .max_error = max_error, .rms_error = std::sqrt(squared/queries.size()),
        .max_uncertainty = 0.0};
    return result;
}

ReferenceCandidateMetrics exact_composed_metrics(std::span<const double> refs) {
    auto result = reciprocal_metrics(refs, 1e-12);
    // The composed analytic price is exactly1. Its correction cancels the
    // ideal blend error; a private component budget is not the user target.
    result.fit = result.ideal_blend;  // absolute size of the cancelling correction
    result.total.price = ReferenceErrorSummary{
        .requested=6, .measured=6, .max_error=0.0, .rms_error=0.0};
    result.total_target_met = true;
    result.decision = ReferenceCandidateDecision::Adequate;
    result.prefer_refinement = *result.ideal_blend.price->max_error > 1e-12;
    return result;
}

TEST(ReferenceSelectionTest, RefinesAnalyticErrorWithinCeilings) {
    MultiKRefConfig config;
    constexpr double target = 1e-5;
    auto result = select_k_refs(config, {90, 110}, [](std::span<const double> refs)
        -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
        return reciprocal_metrics(refs, target);
    });
    ASSERT_TRUE(result.has_value()) << static_cast<int>(result.error().stop_reason);
    ASSERT_EQ(result->candidates.size(), 5u);
    constexpr std::array<size_t, 5> counts{3,5,9,17,33};
    for (size_t i = 0; i < counts.size(); ++i) {
        const auto& candidate = result->candidates[i];
        EXPECT_EQ(candidate.refs.size(), counts[i]);
        EXPECT_DOUBLE_EQ(candidate.refs.front(), 90.0);
        EXPECT_DOUBLE_EQ(candidate.refs.back(), 110.0);
        EXPECT_LE(candidate.refs.size(), config.max_references);
        ASSERT_TRUE(candidate.metrics.has_value());
        EXPECT_EQ(candidate.metrics->ideal_blend.price->measured, 6u);
        if (i) {
            EXPECT_TRUE(std::includes(candidate.refs.begin(), candidate.refs.end(),
                result->candidates[i-1].refs.begin(), result->candidates[i-1].refs.end()));
        }
    }
    EXPECT_LE(result->candidates.size(), config.max_selection_rounds);
    EXPECT_EQ(result->picked_candidate, 4u);
    EXPECT_EQ(result->refs, result->candidates.back().refs);
    EXPECT_LE(*result->candidates.back().metrics->ideal_blend.price->max_error, target);
}

TEST(ReferenceSelectionTest, DoesNotGrowToResolveUnqualifiedOracleOrThreshold) {
    for (bool unqualified : {true, false}) {
        SCOPED_TRACE(unqualified);
        auto result = select_k_refs({}, {90,110}, [=](std::span<const double> refs)
            -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
            auto metrics = reciprocal_metrics(refs, 0.002);
            if (unqualified) {
                metrics.decision = ReferenceCandidateDecision::ReferenceUnqualified;
                metrics.ideal_blend.price = ReferenceErrorSummary{
                    .requested=6, .unresolved=6};
            } else {
                auto& price = *metrics.ideal_blend.price;
                price.max_uncertainty = 0.0005;
                EXPECT_LT(*price.max_error-*price.max_uncertainty, 0.002);
                EXPECT_GT(*price.max_error+*price.max_uncertainty, 0.002);
                metrics.decision = ReferenceCandidateDecision::ThresholdAmbiguous;
            }
            return metrics;
        });
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().stop_reason, unqualified
            ? ReferenceSelectionStopReason::ReferenceUnqualified
            : ReferenceSelectionStopReason::ThresholdAmbiguous);
        ASSERT_EQ(result.error().candidates.size(), 1u);
        const auto& price = *result.error().candidates[0].metrics->ideal_blend.price;
        EXPECT_EQ(price.requested, 6u);
        if (unqualified) {
            EXPECT_EQ(price.unresolved, 6u);
            EXPECT_FALSE(price.max_error.has_value());
        }
    }
}

TEST(ReferenceSelectionTest, FitLimitedSelectsAdequateReferencesWithoutWholeTableRejection) {
    auto result = select_k_refs({}, {90,110}, [](std::span<const double> refs)
        -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
        auto metrics = reciprocal_metrics(refs, 0.1);
        // A constant additive fitting bias cannot be cured by more K refs.
        metrics.fit.price = ReferenceErrorSummary{
            .requested=6, .measured=6, .max_error=0.25, .rms_error=0.25};
        metrics.total.price = ReferenceErrorSummary{
            .requested=6, .measured=6,
            .max_error=0.25+*metrics.ideal_blend.price->max_error};
        metrics.total_target_met = false;
        metrics.decision = ReferenceCandidateDecision::FitLimited;
        return metrics;
    });
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->stop_reason, ReferenceSelectionStopReason::FitLimited);
    ASSERT_EQ(result->candidates.size(), 1u);
    EXPECT_EQ(result->candidates[0].metrics->total_target_met, false);
    EXPECT_EQ(result->refs, (std::vector<double>{90,100,110}));
}

TEST(ReferenceSelectionTest, UnmeasuredIvSelectsPriceAdequateReferencesWithoutInventedZeroError) {
    auto result = select_k_refs({}, {90,110}, [](std::span<const double> refs)
        -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
        auto metrics = reciprocal_metrics(refs, 0.1);
        // The analytic quote price is constant in sigma: IV is unmeasurable.
        metrics.ideal_blend.iv = ReferenceErrorSummary{.requested=6, .filtered=6};
        metrics.decision = ReferenceCandidateDecision::IvUnmeasured;
        return metrics;
    });
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->stop_reason, ReferenceSelectionStopReason::IvUnmeasured);
    ASSERT_EQ(result->candidates.size(), 1u);
    const auto& metrics = *result->candidates[0].metrics;
    EXPECT_FALSE(metrics.total_target_met.has_value());
    EXPECT_EQ(metrics.ideal_blend.iv->filtered, 6u);
    EXPECT_FALSE(metrics.ideal_blend.iv->max_error.has_value());
    EXPECT_FALSE(metrics.ideal_blend.iv->rms_error.has_value());
}

TEST(ReferenceSelectionTest, FullTargetSuccessSurvivesPrivateHeadroomCeilings) {
    for (bool round_limit : {false, true}) {
        SCOPED_TRACE(round_limit);
        MultiKRefConfig config;
        config.max_references = round_limit ? 65 : 9;
        config.max_selection_rounds = round_limit ? 2 : 7;
        auto result = select_k_refs(config, {90,110}, [](std::span<const double> refs)
            -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
            return exact_composed_metrics(refs);
        });
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->stop_reason, round_limit ? ReferenceSelectionStopReason::RoundLimit
                                                   : ReferenceSelectionStopReason::ReferenceLimit);
        EXPECT_EQ(result->candidates.size(), round_limit ? 2u : 3u);
        EXPECT_EQ(result->picked_candidate, result->candidates.size()-1);
        EXPECT_EQ(result->refs.size(), round_limit ? 5u : 9u);
        const auto& picked = *result->candidates[result->picked_candidate].metrics;
        EXPECT_EQ(picked.total_target_met, true);
        EXPECT_GT(*picked.ideal_blend.price->max_error, 1e-12);
    }
}

TEST(ReferenceSelectionTest, OptionalRefinementRetainsPassingCandidateAfterEvidenceFailure) {
    for (bool evaluator_failed : {false, true}) {
        auto result = select_k_refs({}, {90,110}, [=](std::span<const double> refs)
            -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
            // This oracle has qualified data at90/100/110 only. Extra
            // reference locations require evidence it cannot supply.
            if (std::ranges::all_of(refs, [](double k) { return std::fmod(k, 10.0)==0; })) {
                return exact_composed_metrics(refs);
            }
            if (evaluator_failed) return std::unexpected(
                PriceTableError{PriceTableErrorCode::ExtractionFailed, 2, 6});
            ReferenceCandidateMetrics metrics;
            metrics.decision = ReferenceCandidateDecision::ReferenceUnqualified;
            metrics.ideal_blend.price = ReferenceErrorSummary{.requested=6, .unresolved=6};
            return metrics;
        });
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->stop_reason, evaluator_failed ? ReferenceSelectionStopReason::EvaluatorFailed
                                                       : ReferenceSelectionStopReason::ReferenceUnqualified);
        ASSERT_EQ(result->candidates.size(), 2u);
        EXPECT_EQ(result->picked_candidate, 0u);
        EXPECT_EQ(result->refs, (std::vector<double>{90,100,110}));
        if (evaluator_failed) {
            ASSERT_TRUE(result->candidates[1].evaluator_error.has_value());
            EXPECT_EQ(result->candidates[1].evaluator_error->code, PriceTableErrorCode::ExtractionFailed);
            EXPECT_FALSE(result->candidates[1].metrics.has_value());
        } else {
            EXPECT_EQ(result->candidates[1].metrics->ideal_blend.price->unresolved, 6u);
        }
    }
}

TEST(ReferenceSelectionTest, ExplicitReferencesUseFullComposedSuccessWithoutResizing) {
    MultiKRefConfig config{.K_refs={110,90,100}};
    auto result = select_k_refs(config, {90,110}, [](std::span<const double> refs)
        -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
        return exact_composed_metrics(refs);
    });
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->candidates.size(), 1u);
    EXPECT_EQ(result->refs, (std::vector<double>{90,100,110}));
    EXPECT_TRUE(result->candidates[0].metrics->prefer_refinement);
}

TEST(ReferenceSelectionTest, InadequateExplicitSetIsMeasuredOnceAndPreserved) {
    MultiKRefConfig config{.K_refs={110,90,100}};
    auto result = select_k_refs(config, {90,110}, [](std::span<const double> refs)
        -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
        return reciprocal_metrics(refs, 1e-5);
    });
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().stop_reason, ReferenceSelectionStopReason::ExplicitReferencesInadequate);
    ASSERT_EQ(result.error().candidates.size(), 1u);
    EXPECT_EQ(result.error().candidates[0].refs, (std::vector<double>{90,100,110}));
}

TEST(ReferenceSelectionTest, PartialPointCeilingKeepsEveryEarlierReferenceDeterministically) {
    MultiKRefConfig config{.max_references=20};
    const auto evaluator = [](std::span<const double> refs)
        -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
        return reciprocal_metrics(refs, 1e-12);
    };
    auto first = select_k_refs(config, {90,110}, evaluator);
    auto second = select_k_refs(config, {90,110}, evaluator);
    ASSERT_FALSE(first.has_value());
    ASSERT_FALSE(second.has_value());
    EXPECT_EQ(first.error().stop_reason, ReferenceSelectionStopReason::ReferenceLimit);
    const auto& history = first.error().candidates;
    ASSERT_EQ(history.size(), 5u);
    EXPECT_EQ(history.back().refs.size(), 20u);
    EXPECT_EQ(history.back().refs, second.error().candidates.back().refs);
    for (size_t i=1; i<history.size(); ++i) {
        EXPECT_TRUE(std::includes(history[i].refs.begin(), history[i].refs.end(),
            history[i-1].refs.begin(), history[i-1].refs.end()));
        EXPECT_DOUBLE_EQ(history[i].refs.front(), 90.0);
        EXPECT_DOUBLE_EQ(history[i].refs.back(), 110.0);
        EXPECT_LE(history[i].refs.size(), 20u);
    }
}

TEST(ReferenceSelectionTest, SeedCountsAsFirstRoundAndCapsAreFrozenBeforeCallback) {
    MultiKRefConfig config{.max_references=65, .max_selection_rounds=1};
    auto result = select_k_refs(config, {90,110}, [&](std::span<const double> refs)
        -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
        config.max_selection_rounds = 7;  // callback cannot enlarge this call's budget
        return reciprocal_metrics(refs, 1e-12);
    });
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().stop_reason, ReferenceSelectionStopReason::RoundLimit);
    EXPECT_EQ(result.error().candidates.size(), 1u);
}

TEST(ReferenceSelectionTest, RejectsInvalidRequestsBeforeCallingEvaluator) {
    std::array<MultiKRefConfig, 6> invalid{
        MultiKRefConfig{.max_references=0},
        MultiKRefConfig{.max_selection_rounds=0},
        MultiKRefConfig{.K_refs={90,90,110}},
        MultiKRefConfig{.K_refs={90,std::numeric_limits<double>::infinity()}},
        MultiKRefConfig{.K_refs={95,105}},
        MultiKRefConfig{.K_refs={90,100,110}, .max_references=2},
    };
    for (const auto& config : invalid) {
        bool called = false;
        auto result = select_k_refs(config, {90,110}, [&](std::span<const double>)
            -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
            called = true;
            return ReferenceCandidateMetrics{};
        });
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().stop_reason, ReferenceSelectionStopReason::InvalidConfig);
        EXPECT_TRUE(result.error().error.has_value());
        EXPECT_TRUE(result.error().candidates.empty());
        EXPECT_FALSE(called);
    }
    auto missing = select_k_refs({}, {90,110}, {});
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().stop_reason, ReferenceSelectionStopReason::InvalidConfig);
}

TEST(ReferenceSelectionTest, SingletonDomainAndExhaustedFloatingPointIntervalsAreBounded) {
    auto single = select_k_refs({}, {100,100}, [](std::span<const double> refs)
        -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
        EXPECT_EQ(refs.size(), 1u);
        ReferenceCandidateMetrics metrics;
        metrics.decision = ReferenceCandidateDecision::Adequate;
        metrics.ideal_blend.price = ReferenceErrorSummary{
            .requested=1, .measured=1, .max_error=0.0};
        return metrics;
    });
    ASSERT_TRUE(single.has_value());
    EXPECT_EQ(single->refs, std::vector<double>{100});
    const double hi = std::nextafter(1.0, 2.0);
    auto exhausted = select_k_refs({}, {1.0,hi}, [](std::span<const double> refs)
        -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
        const double width = refs.back()-refs.front();
        // Exact maximum linear-interpolation error for the quadratic K^2.
        ReferenceCandidateMetrics metrics;
        metrics.ideal_blend.price = ReferenceErrorSummary{
            .requested=1, .measured=1, .max_error=width*width/4};
        metrics.decision = ReferenceCandidateDecision::RefineReferences;
        return metrics;
    });
    ASSERT_FALSE(exhausted.has_value());
    EXPECT_EQ(exhausted.error().stop_reason, ReferenceSelectionStopReason::NoRepresentableRefinement);
    EXPECT_EQ(exhausted.error().candidates.size(), 1u);
}

TEST(ReferenceSelectionTest, RejectsIncoherentMetricsWithoutInventingMeasurements) {
    const std::array<ReferenceErrorSummary, 5> invalid{
        ReferenceErrorSummary{.requested=6, .filtered=6, .max_error=0.0},
        ReferenceErrorSummary{.requested=5, .measured=3, .filtered=3, .max_error=0.1},
        ReferenceErrorSummary{.requested=2, .measured=2},
        ReferenceErrorSummary{.requested=2, .measured=2,
            .max_error=std::numeric_limits<double>::quiet_NaN()},
        ReferenceErrorSummary{.requested=std::numeric_limits<size_t>::max(),
            .measured=std::numeric_limits<size_t>::max(), .filtered=1, .max_error=0.1},
    };
    for (const auto& bad : invalid) {
        auto result = select_k_refs({}, {90,110}, [&](std::span<const double> refs)
            -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
            auto metrics = reciprocal_metrics(refs, 0.1);
            metrics.fit.iv = bad;
            return metrics;
        });
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().stop_reason, ReferenceSelectionStopReason::InvalidMetrics);
        ASSERT_EQ(result.error().candidates.size(), 1u);
        EXPECT_EQ(result.error().candidates[0].metrics->fit.iv->requested, bad.requested);
    }
}

TEST(ReferenceSelectionTest, StructuralHomogeneityStopsAtCheapSeedWithoutNumericIvClaims) {
    auto result = select_k_refs({}, {90,110}, [](std::span<const double>)
        -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
        ReferenceCandidateMetrics metrics;
        // A cash-free price is K*f(S/K); reference blending is identically
        // exact. This evidence is not a measured zero IV error.
        metrics.decision = ReferenceCandidateDecision::Adequate;
        metrics.ideal_blend.price = ReferenceErrorSummary{.requested=6, .structurally_exact=6};
        metrics.ideal_blend.iv = ReferenceErrorSummary{.requested=6, .structurally_exact=6};
        return metrics;
    });
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->candidates.size(), 1u);
    EXPECT_EQ(result->refs, (std::vector<double>{90,100,110}));
    const auto& iv = *result->candidates[0].metrics->ideal_blend.iv;
    EXPECT_EQ(iv.measured, 0u);
    EXPECT_EQ(iv.structurally_exact, 6u);
    EXPECT_FALSE(iv.max_error.has_value());
}

TEST(ReferenceSelectionTest, EarlyQualifiedWitnessPreservesUntestedPopulation) {
    MultiKRefConfig config{.max_selection_rounds=1};
    auto result = select_k_refs(config, {90,110}, [](std::span<const double> refs)
        -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
        // At fixed K97.1, the reciprocal-reference interpolation has a
        // positive analytic error, enough to reject the current density.
        const double K=97.1;
        const double w=(K-refs[0])/(refs[1]-refs[0]);
        const double error=K*((1-w)/refs[0]+w/refs[1])-1;
        ReferenceCandidateMetrics metrics;
        metrics.decision=ReferenceCandidateDecision::RefineReferences;
        metrics.ideal_blend.price=ReferenceErrorSummary{
            .requested=6, .measured=1, .max_error=error, .rms_error=error, .untested=5};
        return metrics;
    });
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().stop_reason, ReferenceSelectionStopReason::RoundLimit);
    ASSERT_EQ(result.error().candidates.size(), 1u);
    EXPECT_EQ(result.error().candidates[0].metrics->ideal_blend.price->untested, 5u);
}

TEST(ReferenceSelectionTest, AcceptingIncompleteEvidenceRequiresFullComposedRescue) {
    for (bool rescue : {false,true}) {
        auto result = select_k_refs({}, {90,110}, [=](std::span<const double> refs)
            -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
            auto metrics=exact_composed_metrics(refs);
            metrics.prefer_refinement=false;
            metrics.ideal_blend.price=ReferenceErrorSummary{
                .requested=6, .measured=1, .max_error=0.1, .untested=5};
            if (!rescue) {
                metrics.total={};
                metrics.total_target_met.reset();
            }
            return metrics;
        });
        if (rescue) {
            ASSERT_TRUE(result.has_value());
            EXPECT_EQ(result->candidates.size(), 1u);
        } else {
            ASSERT_FALSE(result.has_value());
            EXPECT_EQ(result.error().stop_reason, ReferenceSelectionStopReason::InvalidMetrics);
        }
    }
}

TEST(ReferenceSelectionTest, IvFilteringCannotRemovePriceObligations) {
    auto result=select_k_refs({}, {90,110}, [](std::span<const double>)
        -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
        ReferenceCandidateMetrics metrics;
        metrics.decision=ReferenceCandidateDecision::Adequate;
        metrics.ideal_blend.price=ReferenceErrorSummary{.requested=6, .filtered=6};
        metrics.ideal_blend.iv=ReferenceErrorSummary{.requested=6, .measured=6, .max_error=0.0};
        return metrics;
    });
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().stop_reason, ReferenceSelectionStopReason::InvalidMetrics);
}

TEST(ReferenceSelectionTest, ComposedPriceCanRescueReferencesWithoutAnIvClaim) {
    for (bool complete_price : {false, true}) {
        auto result=select_k_refs(MultiKRefConfig{.K_refs={90,100,110}}, {90,110},
            [=](std::span<const double>)
                -> std::expected<ReferenceCandidateMetrics, PriceTableError> {
                ReferenceCandidateMetrics metrics;
                metrics.decision=ReferenceCandidateDecision::IvUnmeasured;
                metrics.ideal_blend.price=ReferenceErrorSummary{.requested=6, .unresolved=6};
                metrics.ideal_blend.iv=ReferenceErrorSummary{.requested=6, .unresolved=6};
                metrics.total.price=complete_price
                    ? ReferenceErrorSummary{.requested=6, .measured=6, .max_error=0.001}
                    : ReferenceErrorSummary{.requested=6, .measured=5,
                        .unresolved=1, .max_error=0.001};
                // Ideal homogeneity does not measure composed IV accuracy:
                // retain unassessed rows rather than inventing filtered IVs.
                metrics.total.iv=ReferenceErrorSummary{.requested=6, .untested=6};
                return metrics;
            });
        if (!complete_price) {
            ASSERT_FALSE(result.has_value());
            EXPECT_EQ(result.error().stop_reason, ReferenceSelectionStopReason::InvalidMetrics);
            continue;
        }
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->refs, (std::vector<double>{90,100,110}));
        EXPECT_EQ(result->stop_reason, ReferenceSelectionStopReason::IvUnmeasured);
        ASSERT_EQ(result->candidates.size(), 1u);
        const auto& metrics=*result->candidates[0].metrics;
        EXPECT_FALSE(metrics.total_target_met.has_value());
        EXPECT_EQ(metrics.ideal_blend.price->unresolved, 6u);
        EXPECT_EQ(metrics.total.iv->untested, 6u);
        EXPECT_FALSE(metrics.total.iv->max_error.has_value());
    }
}

}  // namespace
}  // namespace mango
