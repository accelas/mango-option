// SPDX-License-Identifier: MIT
#include "mango/option/table/accuracy/report.hpp"
#include "mango/option/table/accuracy/acceptance.hpp"
#include "mango/support/error_types.hpp"
#include "mango/option/detail/price_table_error_mapping.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <type_traits>

namespace mango {
namespace {
TEST(AccuracyReportTest, SnapshotOwnsEvidenceAndPreservesUnknownOutcomes) {
    ReferenceAccuracySummary evidence;
    evidence.price = ReferenceErrorSummary{.requested=2, .measured=1, .unresolved=1,
        .max_error=.02, .rms_error=.02, .max_uncertainty=.001};
    const AccuracyRequest request{.max_price_error=.01, .max_iv_error=std::nullopt};
    const AccuracyReport report=detail::accuracy::assess_request(evidence, {}, request);
    evidence.price->max_error=0.;
    EXPECT_EQ(report.evidence().price->max_error, .02);
    EXPECT_FALSE(report.evidence().iv);
    EXPECT_FALSE(report.iv_target_met());
    EXPECT_EQ(report.request().max_price_error, .01);
    EXPECT_FALSE(report.request().max_iv_error);
    EXPECT_EQ(report.decision(), AccuracyDecision::ViabilityUnassessed);
    EXPECT_EQ(report.prerequisites().certificate, PriceProofStatus::NotRun);
    static_assert(!std::is_default_constructible_v<AccuracyReport>);
    static_assert(!std::is_copy_assignable_v<AccuracyReport>);
}

TEST(AccuracyReportTest, PublicFailureMappingRetainsSameOwnedReport) {
    auto report=std::make_shared<const AccuracyReport>(
        detail::accuracy::assess_request({}, {}, AccuracyRequest{}));
    PriceTableError source{PriceTableErrorCode::NoViableSurface};
    source.accuracy_report=report;
    const auto mapped=detail::to_validation_error(source);
    EXPECT_EQ(mapped.accuracy_report, report);
    const auto roundtrip=convert_to_price_table_error(mapped);
    ASSERT_EQ(roundtrip.accuracy_report, report);
    source.accuracy_report.reset();
    report.reset();
    EXPECT_EQ(roundtrip.accuracy_report->decision(), AccuracyDecision::ViabilityUnassessed);
}
} // namespace
} // namespace mango
