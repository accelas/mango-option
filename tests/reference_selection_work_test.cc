// SPDX-License-Identifier: MIT
#include "mango/option/table/reference_selection_builder.hpp"
#include <gtest/gtest.h>

namespace mango {
namespace {
struct UnbuiltSurface {};

TEST(ReferenceSelectionWork, FitFailureKeepsSuccessfulSelectionHistory) {
    SegmentedAdaptiveConfig config{.spot = 100.0, .option_type = OptionType::PUT,
        .dividend_yield = 0.0, .discrete_dividends = {}, .maturity = .2,
        .kref_config = {.K_refs = {100.0}}, .strike_bounds = StrikeBounds{100.0, 100.0}};
    SurfaceBounds bounds{0.0, 0.0, .1, .2, .2, .2, .05, .05};
    bounds.strike_bounds = config.strike_bounds;
    size_t builds = 0;
    auto result = detail::build_with_reference_selection(config, bounds, {{.1, .2}},
        [&](std::span<const double>) -> std::expected<UnbuiltSurface, PriceTableError> {
            ++builds;
            return std::unexpected(PriceTableError{PriceTableErrorCode::FittingFailed});
        }, [](const UnbuiltSurface&, double, double, double, double, double) { return 0.0; });
    ASSERT_FALSE(result);
    EXPECT_EQ(builds, 1u);
    ASSERT_TRUE(result.error().reference_selection);
    const auto* selected = std::get_if<ReferenceSelectionResult>(
        &result.error().reference_selection->outcome());
    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->refs, (std::vector<double>{100.0}));
    EXPECT_EQ(selected->stop_reason, ReferenceSelectionStopReason::Adequate);
    ASSERT_EQ(selected->candidates.size(), 1u);
    ASSERT_TRUE(result.error().work);
    EXPECT_FALSE(result.error().work->tables.pde);  // Legacy failed callback reports no work.
    ASSERT_TRUE(result.error().work->selection.pde);
    EXPECT_EQ(result.error().work->selection.pde->attempted, 0u);
}

TEST(ReferenceSelectionWork, SelectionFailureKeepsAvailableMetricsAndTypedCause) {
    SegmentedAdaptiveConfig config{.spot = 105.0, .option_type = OptionType::PUT,
        .dividend_yield = 0.0, .discrete_dividends = {{.5, 1.0}}, .maturity = 1.0,
        .kref_config = {.K_refs = {100.0, 110.0}}, .strike_bounds = StrikeBounds{105.0, 105.0}};
    // The existing projected-LCP domain rejects this rate before any solve.
    // This cheaply exercises a real unqualified evaluator and failed rescue.
    SurfaceBounds bounds{0.0, 0.0, .9, 1.0, .2, .2, -4.0, -4.0};
    bounds.strike_bounds = config.strike_bounds;
    size_t builds = 0;
    auto result = detail::build_with_reference_selection(config, bounds, {{.9, 1.0}},
        [&](std::span<const double>) -> std::expected<UnbuiltSurface, PriceTableError> {
            ++builds;
            return std::unexpected(PriceTableError{PriceTableErrorCode::FittingFailed});
        }, [](const UnbuiltSurface&, double, double, double, double, double) { return 0.0; });
    ASSERT_FALSE(result);
    EXPECT_EQ(builds, 1u);
    ASSERT_TRUE(result.error().reference_selection);
    const auto* failed = std::get_if<ReferenceSelectionFailure>(
        &result.error().reference_selection->outcome());
    ASSERT_NE(failed, nullptr);
    EXPECT_EQ(failed->stop_reason, ReferenceSelectionStopReason::EvaluatorFailed);
    ASSERT_EQ(failed->candidates.size(), 1u);
    const auto& candidate = failed->candidates.front();
    EXPECT_EQ(candidate.refs, (std::vector<double>{100.0, 110.0}));
    ASSERT_TRUE(candidate.evaluator_error);
    EXPECT_EQ(candidate.evaluator_error->code, PriceTableErrorCode::FittingFailed);
    ASSERT_TRUE(candidate.metrics);  // The ideal assessment must survive failed fitting.
    EXPECT_EQ(candidate.metrics->decision, ReferenceCandidateDecision::ReferenceUnqualified);
    ASSERT_TRUE(candidate.metrics->provider_work);
    EXPECT_EQ(candidate.metrics->provider_work->attempted, 0u);
}
} // namespace
} // namespace mango
