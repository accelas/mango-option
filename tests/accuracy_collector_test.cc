// SPDX-License-Identifier: MIT
#include "mango/option/table/accuracy/collector.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <limits>

namespace mango::detail::accuracy {
namespace {

PhysicalReferenceRow analytic_row() {
    PhysicalReferenceRow row;
    row.query=PricingParams{OptionSpec{.spot=100, .strike=100, .maturity=1}, 0.2};
    row.reference_price=0.04;
    row.price_qualification=PriceQualification::Qualified;
    row.iv_qualification=IvQualification::Measurable;
    row.price_uncertainty=0.0;
    row.iv_uncertainty=0.0;
    row.provenance.source=ReferenceSource::Analytic;
    row.provenance.qualification_digest[0]=1; // Declared exact fixture identity.
    return row;
}

const Prerequisites viable_certified{
    .viability=Viability::Passed, .certificate=PriceProofStatus::Certified};

TEST(AccuracyCollectorTest, MeasuresActualNonlinearInversionInsteadOfPriceVegaProxy) {
    const std::vector rows{analytic_row()};
    // Independent oracle P(sigma)=sigma^2; fitted Q(sigma)=(sigma+0.1)^2.
    // At sigma=.2: price error=.05, actual IV error=.1, proxy=.05/.4=.125.
    auto result=collect(rows,
        [](const PricingParams& query) -> std::optional<double> {
            return std::pow(query.volatility+0.1, 2);
        },
        [](const IVQuery& query) -> std::optional<double> {
            EXPECT_DOUBLE_EQ(query.market_price, 0.04);
            return std::sqrt(query.market_price)-0.1;
        }, viable_certified, {.price=0.1, .iv=0.11});
    EXPECT_EQ(result.assessment.decision(), Decision::Accepted);
    EXPECT_EQ(result.assessment.iv_metric_kind(), IvMetricKind::AbsoluteIvError);
    const auto& evidence=result.assessment.evidence();
    ASSERT_TRUE(evidence.price->max_error.has_value());
    ASSERT_TRUE(evidence.iv->max_error.has_value());
    EXPECT_NEAR(*evidence.price->max_error, 0.05, 1e-15);
    EXPECT_NEAR(*evidence.iv->max_error, 0.1, 1e-15);
    EXPECT_NEAR(*evidence.iv->rms_error, 0.1, 1e-15);
    EXPECT_GT(0.05/0.4, *result.assessment.request().max_iv_error);
    ASSERT_EQ(result.rows.size(), 1u);
    EXPECT_EQ(result.rows[0].iv, ObservationOutcome::Measured);
    ASSERT_TRUE(result.rows[0].iv_error.has_value());
    EXPECT_NEAR(*result.rows[0].iv_error, 0.1, 1e-15);
    EXPECT_EQ(result.rows[0].provenance.qualification_digest, rows[0].provenance.qualification_digest);
}

TEST(AccuracyCollectorTest, ProxyUnderestimationCannotFalselyPassAnActualIvTarget) {
    // P(sigma)=sigma^2 and fitted Q(sigma)=sigma^2+.03.
    // The reference quote .04 inverts to .1: actual IV error .1 exceeds .09,
    // although the price/reference-vega proxy .03/.4=.075 appears adequate.
    auto result=collect(std::vector{analytic_row()},
        [](const PricingParams& query) -> std::optional<double> {
            return query.volatility*query.volatility+0.03;
        },
        [](const IVQuery& query) -> std::optional<double> {
            return std::sqrt(query.market_price-0.03);
        }, viable_certified, {.price=0.04, .iv=0.09});
    EXPECT_LT(0.03/0.4, *result.assessment.request().max_iv_error);
    EXPECT_EQ(result.assessment.decision(), Decision::TargetsMissed);
    EXPECT_EQ(result.assessment.price_target_met(), true);
    EXPECT_EQ(result.assessment.iv_target_met(), false);
    EXPECT_NEAR(*result.assessment.evidence().iv->max_error, 0.1, 1e-15);
}

TEST(AccuracyCollectorTest, UnqualifiedReferencesNeverReachBackendAsValidQuotes) {
    std::vector rows(7, analytic_row());
    rows[1].price_qualification=PriceQualification::Unresolved;
    rows[2].price_qualification=PriceQualification::Refused;
    rows[3].provenance.source=ReferenceSource::Unspecified;
    rows[4].provenance.qualification_digest={};
    rows[5].price_uncertainty.reset();
    rows[6].query.volatility=std::numeric_limits<double>::quiet_NaN();
    size_t price_calls=0, iv_calls=0;
    auto result=collect(rows,
        [&](const PricingParams&) -> std::optional<double> { ++price_calls; return 0.04; },
        [&](const IVQuery&) -> std::optional<double> { ++iv_calls; return 0.2; },
        viable_certified);
    EXPECT_EQ(price_calls, 1u);
    EXPECT_EQ(iv_calls, 1u);
    EXPECT_EQ(result.assessment.decision(), Decision::IncompletePriceEvidence);
    for (const auto* channel : {&result.assessment.evidence().price, &result.assessment.evidence().iv}) {
        ASSERT_TRUE(channel->has_value());
        EXPECT_EQ((*channel)->requested, 7u);
        EXPECT_EQ((*channel)->measured, 1u);
        EXPECT_EQ((*channel)->unresolved, 5u);
        EXPECT_EQ((*channel)->refused, 1u);
        EXPECT_EQ((*channel)->max_error, 0.0); // One genuinely zero observation.
    }
    ASSERT_EQ(result.rows.size(), rows.size());
    EXPECT_EQ(result.rows[1].iv, ObservationOutcome::ReferenceUnresolved);
    EXPECT_EQ(result.rows[2].iv, ObservationOutcome::ReferenceRefused);
    EXPECT_EQ(result.rows[3].price, ObservationOutcome::InvalidReference);
    EXPECT_EQ(result.rows[6].iv, ObservationOutcome::InvalidReference);
}

TEST(AccuracyCollectorTest, IndependentIvFiltersNeverRemovePriceChecks) {
    std::vector rows(6, analytic_row());
    rows[1].iv_qualification=IvQualification::FilteredTimeValue;
    rows[2].iv_qualification=IvQualification::FilteredVega;
    rows[3].iv_qualification=IvQualification::Unresolved;
    rows[4].iv_qualification=IvQualification::Refused;
    rows[5].iv_uncertainty.reset();
    size_t price_calls=0, iv_calls=0;
    auto result=collect(rows,
        [&](const PricingParams&) -> std::optional<double> { ++price_calls; return 0.04; },
        [&](const IVQuery&) -> std::optional<double> { ++iv_calls; return 0.2; },
        viable_certified);
    EXPECT_EQ(price_calls, 6u);
    EXPECT_EQ(iv_calls, 1u);
    EXPECT_EQ(result.assessment.evidence().price->measured, 6u);
    const auto& iv=*result.assessment.evidence().iv;
    EXPECT_EQ(iv.requested, 6u);
    EXPECT_EQ(iv.measured, 1u);
    EXPECT_EQ(iv.filtered, 2u);
    EXPECT_EQ(iv.unresolved, 2u);
    EXPECT_EQ(iv.refused, 1u);
    EXPECT_EQ(result.rows[1].iv, ObservationOutcome::FilteredTimeValue);
    EXPECT_EQ(result.rows[2].iv, ObservationOutcome::FilteredVega);

    const std::vector filtered_rows{rows[1], rows[2]};
    auto filtered=collect(filtered_rows,
        [](const PricingParams&) -> std::optional<double> { return 0.04; },
        [](const IVQuery&) -> std::optional<double> { ADD_FAILURE(); return 0.2; },
        viable_certified, {.iv=std::nullopt});
    EXPECT_EQ(filtered.assessment.decision(), Decision::Accepted);
    EXPECT_FALSE(filtered.assessment.evidence().iv->max_error.has_value());
    EXPECT_FALSE(filtered.assessment.evidence().iv->rms_error.has_value());
    EXPECT_EQ(filtered.assessment.evidence().iv->filtered, 2u);
    auto iv_targeted=collect(filtered_rows,
        [](const PricingParams&) -> std::optional<double> { return 0.04; }, {},
        viable_certified, {}, Policy::BestEffort);
    EXPECT_EQ(iv_targeted.assessment.decision(), Decision::IvUnmeasured);
    auto price_refused=collect(filtered_rows, {}, {}, viable_certified, {.iv=std::nullopt});
    EXPECT_EQ(price_refused.assessment.decision(), Decision::IncompletePriceEvidence);
    EXPECT_EQ(price_refused.assessment.evidence().price->refused, 2u);
    EXPECT_EQ(price_refused.assessment.evidence().iv->filtered, 2u);
}

TEST(AccuracyCollectorTest, BackendFailuresRetainBothChannelsAndNeverMeasureNanAsZero) {
    const std::optional<double> bad_values[]={std::nullopt,
        std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -1.0};
    for (bool price_channel : {false, true}) {
        std::vector rows(5, analytic_row());
        size_t price_calls=0, iv_calls=0;
        auto result=collect(rows,
            [&](const PricingParams&) -> std::optional<double> {
                const size_t index=price_calls++;
                return price_channel && index<4 ? bad_values[index] : std::optional{0.04};
            },
            [&](const IVQuery&) -> std::optional<double> {
                const size_t index=iv_calls++;
                return !price_channel && index<4 ? bad_values[index] : std::optional{0.2};
            }, viable_certified);
        EXPECT_EQ(price_calls, 5u);
        EXPECT_EQ(iv_calls, 5u);
        const auto& failed=price_channel ? result.assessment.evidence().price : result.assessment.evidence().iv;
        const auto& other=price_channel ? result.assessment.evidence().iv : result.assessment.evidence().price;
        EXPECT_EQ(failed->requested, 5u);
        EXPECT_EQ(failed->measured, 1u);
        EXPECT_EQ(failed->refused, 4u);
        EXPECT_EQ(failed->rms_error, 0.0);
        EXPECT_EQ(other->measured, 5u);
        EXPECT_EQ(price_channel ? result.rows[0].price : result.rows[0].iv, ObservationOutcome::BackendRefused);
        EXPECT_EQ(price_channel ? result.rows[1].price : result.rows[1].iv, ObservationOutcome::BackendNonfinite);
        EXPECT_EQ(price_channel ? result.rows[3].price : result.rows[3].iv, ObservationOutcome::BackendInvalidValue);
    }
    auto absent=collect(std::vector{analytic_row()}, {}, {}, viable_certified);
    EXPECT_EQ(absent.assessment.evidence().price->refused, 1u);
    EXPECT_EQ(absent.assessment.evidence().iv->refused, 1u);
    EXPECT_FALSE(absent.assessment.evidence().price->rms_error.has_value());
}

TEST(AccuracyCollectorTest, DeclaredApplicabilityStaysSeparateFromIvFiltering) {
    std::vector rows(4, analytic_row());
    rows[1].iv_applicable=false;
    rows[2].price_applicable=false;
    rows[3].price_applicable=rows[3].iv_applicable=false;
    size_t price_calls=0, iv_calls=0;
    auto result=collect(rows,
        [&](const PricingParams&) -> std::optional<double> { ++price_calls; return 0.04; },
        [&](const IVQuery&) -> std::optional<double> { ++iv_calls; return 0.2; }, viable_certified);
    EXPECT_EQ(price_calls, 2u);
    EXPECT_EQ(iv_calls, 2u);
    EXPECT_EQ(result.assessment.evidence().price->requested, 2u);
    EXPECT_EQ(result.assessment.evidence().iv->requested, 2u);
    EXPECT_EQ(result.assessment.evidence().iv->filtered, 0u);
    ASSERT_EQ(result.rows.size(), 4u);
    EXPECT_EQ(result.rows[1].iv, ObservationOutcome::NotApplicable);
    EXPECT_EQ(result.rows[2].price, ObservationOutcome::NotApplicable);
    EXPECT_EQ(result.rows[3].price, ObservationOutcome::NotApplicable);
    EXPECT_EQ(result.rows[3].iv, ObservationOutcome::NotApplicable);
}

TEST(AccuracyCollectorTest, RmsDoesNotTurnSmallErrorsIntoZeroOrLargeErrorsIntoInfinity) {
    auto row=analytic_row();
    row.reference_price=0.0;
    row.iv_applicable=false;
    for (double error : {1e-200, 1e200}) {
        auto result=collect(std::vector{row},
            [=](const PricingParams&) -> std::optional<double> { return error; },
            {}, viable_certified, {.iv=std::nullopt});
        EXPECT_EQ(result.assessment.evidence().price->max_error, error);
        EXPECT_EQ(result.assessment.evidence().price->rms_error, error);
        EXPECT_NE(result.assessment.decision(), Decision::InvalidEvidence);
    }
    size_t calls=0;
    auto result=collect(std::vector{row, row},
        [&](const PricingParams&) -> std::optional<double> { return ++calls==1 ? 3.0 : 4.0; },
        {}, viable_certified, {.price=5, .iv=std::nullopt});
    EXPECT_EQ(result.assessment.evidence().price->max_error, 4.0);
    EXPECT_NEAR(*result.assessment.evidence().price->rms_error, 3.5355339059327376, 1e-15);
    calls=0;
    auto subnormal=collect(std::vector{row, row, row, row},
        [&](const PricingParams&) -> std::optional<double> {
            return calls++==0 ? std::numeric_limits<double>::denorm_min() : 0.0;
        }, {}, viable_certified, {.iv=std::nullopt});
    EXPECT_GT(*subnormal.assessment.evidence().price->rms_error, 0.0);
}

TEST(AccuracyCollectorTest, InversionUsesTheSameRolledPhysicalContractAndReferenceQuote) {
    std::vector rows(2, analytic_row());
    rows[0].query=PricingParams{OptionSpec{
        .spot=97, .strike=103, .maturity=0.5+1.0/365, .rate=-0.01,
        .dividend_yield=0.02, .option_type=OptionType::PUT}, 0.2,
        {{.calendar_time=1.0/365, .amount=3.0}}};
    rows[1].query=PricingParams{OptionSpec{
        .spot=109, .strike=103, .maturity=0.5-1.0/365, .rate=0.03,
        .dividend_yield=0.01, .option_type=OptionType::CALL}, 0.3, {}};
    for (auto& row : rows) row.reference_price=10.0;
    size_t price_calls=0, iv_calls=0;
    auto check_contract=[](const auto& observed, const PricingParams& expected) {
        EXPECT_EQ(observed.spot, expected.spot);
        EXPECT_EQ(observed.strike, expected.strike);
        EXPECT_EQ(observed.maturity, expected.maturity);
        EXPECT_EQ(std::get<double>(observed.rate), std::get<double>(expected.rate));
        EXPECT_EQ(observed.dividend_yield, expected.dividend_yield);
        EXPECT_EQ(observed.option_type, expected.option_type);
        ASSERT_EQ(observed.discrete_dividends.size(), expected.discrete_dividends.size());
        for (size_t i=0; i<expected.discrete_dividends.size(); ++i) {
            EXPECT_EQ(observed.discrete_dividends[i].calendar_time, expected.discrete_dividends[i].calendar_time);
            EXPECT_EQ(observed.discrete_dividends[i].amount, expected.discrete_dividends[i].amount);
        }
    };
    auto result=collect(rows,
        [&](const PricingParams& query) -> std::optional<double> {
            check_contract(query, rows[price_calls++].query);
            return 10.001; // Must not be substituted for the independent quote.
        },
        [&](const IVQuery& query) -> std::optional<double> {
            const auto& reference=rows[iv_calls++];
            check_contract(query, reference.query);
            EXPECT_EQ(query.market_price, 10.0);
            return reference.query.volatility;
        }, viable_certified);
    EXPECT_EQ(result.assessment.decision(), Decision::Accepted);
    EXPECT_EQ(price_calls, 2u);
    EXPECT_EQ(iv_calls, 2u);
}

} // namespace
} // namespace mango::detail::accuracy
