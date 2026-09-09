// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/reference_strike_evaluator.hpp"
#include <cmath>

namespace mango {
namespace {

SurfaceBounds requested_bounds() {
    return {.m_min = std::log(0.92), .m_max = std::log(1.08),
        .tau_min = 0.0, .tau_max = 1.0, .sigma_min = 0.05, .sigma_max = 0.1,
        .rate_min = 0.05, .rate_max = 0.05,
        .strike_bounds = StrikeBounds{90.0, 110.0},
        .ratio_bounds = MoneynessBounds{0.92, 1.08}};
}

TEST(ReferenceStrikeEvaluatorTest, RolledCashFreeDomainUsesExactHomogeneity) {
    SegmentedAdaptiveConfig config{
        .spot = 100.0, .option_type = OptionType::PUT, .dividend_yield = 0.02,
        .discrete_dividends = {{0.25, 3.0}}, .maturity = 1.0,
    };
    auto bounds = requested_bounds();
    bounds.tau_max = 0.5;  // Every query is after the anchored payment.
    auto evaluator = ReferenceStrikeEvaluator::create(config, bounds, {{0.0, 0.5}});
    ASSERT_TRUE(evaluator.has_value());
    const std::vector<double> refs{90.0, 100.0, 110.0};
    auto result = evaluator->evaluate(refs);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->decision, ReferenceCandidateDecision::Adequate);
    EXPECT_EQ(result->pde_solves, 0u);
    ASSERT_TRUE(result->provider_work.has_value());
    EXPECT_EQ(result->provider_work->attempted, 0u);
    EXPECT_EQ(result->provider_work->completed, 0u);
    EXPECT_EQ(result->provider_work->failed, 0u);
    ASSERT_TRUE(result->ideal_blend.price.has_value());
    EXPECT_GT(result->ideal_blend.price->requested, 0u);
    EXPECT_EQ(result->ideal_blend.price->structurally_exact, result->ideal_blend.price->requested);
    ASSERT_TRUE(result->ideal_blend.iv.has_value());
    EXPECT_EQ(result->ideal_blend.iv->measured, 0u);
    EXPECT_FALSE(result->ideal_blend.iv->max_error.has_value());
    EXPECT_FALSE(result->total_target_met.has_value());
}

TEST(ReferenceStrikeEvaluatorTest, ExactReferenceIdentityNeedsNoDensityOracle) {
    SegmentedAdaptiveConfig config{
        .spot = 100.0, .option_type = OptionType::CALL, .dividend_yield = 0.02,
        .discrete_dividends = {{0.5, 3.0}}, .maturity = 1.0,
        .kref_config = {.K_refs = {100.0}}, .strike_bounds = StrikeBounds{100.0, 100.0},
    };
    auto bounds = requested_bounds();
    bounds.strike_bounds = config.strike_bounds;
    auto evaluator = ReferenceStrikeEvaluator::create(config, bounds, {{0.0, 0.4995}, {0.5005, 1.0}});
    ASSERT_TRUE(evaluator.has_value());
    auto result = evaluator->evaluate(config.kref_config.K_refs);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->decision, ReferenceCandidateDecision::Adequate);
    EXPECT_EQ(result->pde_solves, 0u);
    EXPECT_FALSE(result->ideal_blend.iv->max_error.has_value());
}

// With zero carry and a deeply ITM put, a one-cent future cash payment
// supplies TV exactly at the 1e-4*K threshold while vega is negligible.
// TV ambiguity must not prevent the independent vega branch of the OR filter.
TEST(ReferenceStrikeEvaluatorTest, SmallVegaFiltersDespiteTimeValueThresholdStraddle) {
    SegmentedAdaptiveConfig config{
        .spot = 100.0, .option_type = OptionType::PUT, .dividend_yield = 0.0,
        .discrete_dividends = {{0.005, 0.01}}, .maturity = 0.01,
        .kref_config = {.K_refs = {90.0, 110.0}},
        .strike_bounds = StrikeBounds{100.0, 100.0},
    };
    SurfaceBounds bounds{
        .m_min = std::log(0.899), .m_max = std::log(0.9),
        .tau_min = 0.0, .tau_max = 0.01, .sigma_min = 0.05, .sigma_max = 0.05,
        .rate_min = 0.0, .rate_max = 0.0,
        .strike_bounds = StrikeBounds{100.0, 100.0},
        .ratio_bounds = MoneynessBounds{0.899, 0.9},
    };
    auto evaluator = ReferenceStrikeEvaluator::create(config, bounds,
        {{0.0, 0.0045}, {0.0055, 0.01}});
    ASSERT_TRUE(evaluator.has_value());
    auto result = evaluator->evaluate(config.kref_config.K_refs);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->decision, ReferenceCandidateDecision::IvUnmeasured);
    ASSERT_TRUE(result->ideal_blend.iv.has_value());
    EXPECT_GT(result->ideal_blend.iv->filtered, 0u);
    EXPECT_EQ(result->ideal_blend.iv->unresolved, 0u);
    EXPECT_FALSE(result->ideal_blend.iv->max_error.has_value());
}

TEST(ReferenceStrikeEvaluatorTest, RejectedSolverConstructionIsNotPdeWork) {
    SegmentedAdaptiveConfig config{
        .spot = 105.0, .option_type = OptionType::PUT,
        .discrete_dividends = {{0.5, 1.0}}, .maturity = 1.0,
        .kref_config = {.K_refs = {100.0, 110.0}},
        .strike_bounds = StrikeBounds{105.0, 105.0},
    };
    // The requested rate violates the existing projected-LCP domain for
    // every remaining life here, so construction rejects before solve().
    SurfaceBounds bounds{
        .m_min = 0.0, .m_max = 0.0, .tau_min = 0.9, .tau_max = 1.0,
        .sigma_min = 0.2, .sigma_max = 0.2, .rate_min = -4.0, .rate_max = -4.0,
        .strike_bounds = config.strike_bounds, .ratio_bounds = MoneynessBounds{1.0, 1.0},
    };
    auto evaluator = ReferenceStrikeEvaluator::create(config, bounds, {{0.9, 1.0}});
    ASSERT_TRUE(evaluator.has_value());
    auto result = evaluator->evaluate(config.kref_config.K_refs);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->decision, ReferenceCandidateDecision::ReferenceUnqualified);
    EXPECT_GT(result->pde_solves, 0u);  // Retained attempted-request budget.
    ASSERT_TRUE(result->provider_work.has_value());
    EXPECT_EQ(result->provider_work->attempted, 0u);
    EXPECT_EQ(result->provider_work->completed, 0u);
    EXPECT_EQ(result->provider_work->failed, 0u);

    auto invalid = evaluator->evaluate(std::vector<double>{});
    ASSERT_FALSE(invalid.has_value());
    ASSERT_TRUE(invalid.error().work);
    const auto work = invalid.error().work->total_pde();
    ASSERT_TRUE(work.has_value());
    EXPECT_EQ(work->attempted, 0u);  // This failed request did no new work.
}

// Independent controlled-FDE audit at K=S97.1 finds a roughly .030552
// quote-unit residual for K90/K110. This exceeds the .01 price criterion
// without relying on an unqualified or filtered IV observation.
TEST(ReferenceStrikeEvaluatorTest, SparseCashReferencesHaveQualifiedPriceWitness) {
    SegmentedAdaptiveConfig config{
        .spot = 100.0, .option_type = OptionType::PUT, .dividend_yield = 0.02,
        .discrete_dividends = {{0.5, 3.0}}, .maturity = 1.0,
        .kref_config = {.K_refs = {90.0, 110.0}},
    };
    auto evaluator = ReferenceStrikeEvaluator::create(
        config, requested_bounds(), {{0.0, 0.4995}, {0.5005, 1.0}});
    ASSERT_TRUE(evaluator.has_value());
    auto result = evaluator->evaluate(config.kref_config.K_refs);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->decision, ReferenceCandidateDecision::RefineReferences);
    ASSERT_TRUE(result->ideal_blend.price->max_error.has_value());
    EXPECT_GT(*result->ideal_blend.price->max_error, 0.02);
    EXPECT_LT(*result->ideal_blend.price->max_error, 0.04);
    ASSERT_TRUE(result->ideal_blend.price->max_uncertainty.has_value());
    EXPECT_LT(*result->ideal_blend.price->max_uncertainty, 0.001);
    EXPECT_GT(result->ideal_blend.price->untested, 0u);
    EXPECT_EQ(result->ideal_blend.price->requested,
        result->ideal_blend.price->measured + result->ideal_blend.price->filtered +
        result->ideal_blend.price->unresolved + result->ideal_blend.price->refused +
        result->ideal_blend.price->structurally_exact + result->ideal_blend.price->untested);
    EXPECT_FALSE(result->total_target_met.has_value());
    ASSERT_TRUE(result->provider_work.has_value());
    EXPECT_GT(result->provider_work->attempted, 0u);
    EXPECT_EQ(result->provider_work->completed, result->provider_work->attempted);
    EXPECT_EQ(result->provider_work->failed, 0u);

    auto cached = evaluator->evaluate(config.kref_config.K_refs);
    ASSERT_TRUE(cached.has_value());
    ASSERT_TRUE(cached->provider_work.has_value());
    EXPECT_EQ(cached->provider_work->attempted, 0u);
    EXPECT_EQ(cached->provider_work->completed, 0u);
    EXPECT_EQ(cached->provider_work->failed, 0u);
    EXPECT_EQ(cached->decision, result->decision);

}

}  // namespace
}  // namespace mango
