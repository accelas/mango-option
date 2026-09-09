// SPDX-License-Identifier: MIT
#include "mango/option/table/accuracy/population.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>

namespace mango::detail::accuracy {
namespace {
PopulationRequest family(size_t events, bool zero_rate=false) {
    PopulationRequest input;
    input.domain={std::log(.8),std::log(1.2),events ? 0. : .05,1.,.1,.5,
        zero_rate ? -.05 : .01,zero_rate ? .1 : .08,StrikeBounds{95.,105.},
        MoneynessBounds{.8,1.2}};
    input.option_type=OptionType::CALL;
    input.spot=100.;
    input.max_moneyness_nodes=160;
    if (events) {
        FixedExpiryMetadata fixed{1.,{}};
        for (size_t i=0;i<events;++i)
            fixed.discrete_dividends.push_back({double(i+1)/double(events+1),1.});
        input.fixed_expiry=std::move(fixed);
    }
    return input;
}

TEST(AccuracyPopulationTest, ApprovedDefaultFamiliesHaveFrozenCompleteCounts) {
    const std::array<size_t,4> events{0,1,4,8};
    const std::array<size_t,4> ordinary{198,214,259,319};
    const std::array<size_t,4> with_zero{201,220,274,346};
    for (size_t i=0;i<events.size();++i) {
        for (bool zero : {false,true}) {
            const auto result=make_accuracy_population(family(events[i],zero));
            ASSERT_TRUE(result);
            EXPECT_EQ(result->profile_version(),1u);
            EXPECT_EQ(result->rows().size(),zero ? with_zero[i] : ordinary[i]);
            EXPECT_EQ(result->limits().max_population_rows,512u);
            EXPECT_EQ(result->limits().max_reference_requests,8192u);
            EXPECT_EQ(result->limits().max_qualification_rounds,2u);
            EXPECT_EQ(result->counts().declared_rows,result->rows().size());
        }
    }
}

TEST(AccuracyPopulationTest, InsufficientCapRefusesWithRequiredCountBeforeNumericalWork) {
    auto input=family(8,true);
    input.limits.max_population_rows=345;
    const auto result=make_accuracy_population(input);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().reason,PopulationFailureReason::RowLimit);
    EXPECT_EQ(result.error().required_rows,346u);
    EXPECT_EQ(result.error().limits.max_population_rows,345u);
    EXPECT_EQ(result.error().work.attempted,0u);
}

TEST(AccuracyPopulationTest, CashRowsOwnCorrectRolledSchedulesAndRefusalEntries) {
    auto input=family(1);
    const auto population=make_accuracy_population(input);
    ASSERT_TRUE(population);
    size_t events=0,expired=0;
    for (const auto& row : population->rows()) {
        const auto expected=rolled_dividends(input.fixed_expiry->discrete_dividends,1.,row.query.maturity);
        ASSERT_EQ(row.query.discrete_dividends.size(),expected.size());
        for (size_t i=0;i<expected.size();++i) {
            EXPECT_EQ(row.query.discrete_dividends[i].calendar_time,expected[i].calendar_time);
            EXPECT_EQ(row.query.discrete_dividends[i].amount,expected[i].amount);
        }
        if (has_stratum(row.strata,AccuracyStratum::EventAdmission)) {
            ++events;
            EXPECT_EQ(row.expectation,PopulationExpectation::Admission);
        }
        if (has_stratum(row.strata,AccuracyStratum::ExpiryAdmission)) {
            ++expired;
            EXPECT_EQ(row.query.maturity,0.);
            EXPECT_EQ(row.expectation,PopulationExpectation::Refusal);
        }
    }
    EXPECT_EQ(events,3u);
    EXPECT_EQ(expired,1u);
    input.fixed_expiry->discrete_dividends[0].amount=99.;
    const auto found=std::ranges::find_if(population->rows(),[](const auto& row) {
        return !row.query.discrete_dividends.empty();
    });
    ASSERT_NE(found,population->rows().end());
    EXPECT_EQ(found->query.discrete_dividends[0].amount,1.);
}

TEST(AccuracyPopulationTest, DeclaredManualNodesAbove160AreNeverSilentlyTruncated) {
    auto input=family(0);
    input.max_moneyness_nodes=600;
    const auto limited=make_accuracy_population(input);
    ASSERT_FALSE(limited);
    EXPECT_EQ(limited.error().reason,PopulationFailureReason::RowLimit);
    input.limits.max_population_rows=1024;
    const auto population=make_accuracy_population(input);
    ASSERT_TRUE(population);
    size_t strip=0;
    for (const auto& row : population->rows())
        strip+=has_stratum(row.strata,AccuracyStratum::MoneynessStrip);
    EXPECT_EQ(strip,601u);
    EXPECT_GT(population->rows().size(),600u);
}

TEST(AccuracyPopulationTest, PointDomainsDeduplicateWithoutWidening) {
    auto input=family(0);
    input.domain={0.,0.,.1,.1,.2,.2,.03,.03,StrikeBounds{100.,100.},MoneynessBounds{1.,1.}};
    const auto population=make_accuracy_population(input);
    ASSERT_TRUE(population);
    ASSERT_EQ(population->rows().size(),1u);
    const auto& query=population->rows().front().query;
    EXPECT_EQ(query.spot,100.);
    EXPECT_EQ(query.strike,100.);
    EXPECT_EQ(query.maturity,.1);
    EXPECT_EQ(query.volatility,.2);
    EXPECT_EQ(get_zero_rate(query.rate,query.maturity),.03);
    EXPECT_FALSE(population->counts().off_node_moneyness_available);
}

TEST(AccuracyPopulationTest, HugeNodeConstraintAndTinyRepresentableDomainStayBounded) {
    auto input=family(0);
    input.max_moneyness_nodes=std::numeric_limits<size_t>::max()-1;
    const auto limited=make_accuracy_population(input);
    ASSERT_FALSE(limited);
    EXPECT_EQ(limited.error().reason,PopulationFailureReason::RowLimit);
    EXPECT_EQ(limited.error().required_rows,513u);
    EXPECT_FALSE(limited.error().required_rows_exact);
    input.domain.ratio_bounds=MoneynessBounds{1.,std::nextafter(1.,2.)};
    input.domain.m_min=0.; input.domain.m_max=std::log(input.domain.ratio_bounds->max);
    const auto narrow=make_accuracy_population(input);
    ASSERT_TRUE(narrow);
    EXPECT_LE(narrow->rows().size(),512u);
    EXPECT_FALSE(narrow->counts().off_node_moneyness_available);
}

TEST(AccuracyPopulationTest, ExplicitRoundsAndStorageOverflowAreValidated) {
    auto input=family(0);
    for (size_t rounds : {1u,2u,3u}) {
        input.limits.max_qualification_rounds=rounds;
        ASSERT_TRUE(make_accuracy_population(input));
    }
    input.limits.max_qualification_rounds=4;
    EXPECT_FALSE(make_accuracy_population(input));
    input.limits={};
    EXPECT_EQ(input.limits.max_scalar_cache_bytes(),32u*1024u*1024u);
    input.limits.max_reference_requests=std::numeric_limits<size_t>::max();
    EXPECT_FALSE(input.limits.max_scalar_cache_bytes());
    EXPECT_FALSE(make_accuracy_population(input));
}

TEST(AccuracyPopulationTest, InteriorStrikesAvoidEveryPredeclaredReferenceCandidate) {
    auto input=family(0);
    const double a=(3.-std::sqrt(3.))/6.;
    input.possible_reference_strikes={95.,std::lerp(95.,105.,a),100.,std::lerp(95.,105.,1.-a),105.};
    const auto population=make_accuracy_population(input);
    ASSERT_TRUE(population);
    for (const auto& row:population->rows()) {
        if (has_stratum(row.strata,AccuracyStratum::RegimeInterior)) {
            EXPECT_FALSE(std::ranges::contains(input.possible_reference_strikes,row.query.strike));
            EXPECT_GT(row.query.strike,95.); EXPECT_LT(row.query.strike,105.);
        }
    }
}

} // namespace
} // namespace mango::detail::accuracy
