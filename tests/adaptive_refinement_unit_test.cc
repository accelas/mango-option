// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/support/error_types.hpp"

TEST(AdaptiveGridParamsTest, DefaultMaxIterIsEight) {
    EXPECT_EQ(mango::AdaptiveGridParams{}.max_iter, 8u);
}

TEST(BuildDiagnosticsTest, DefaultsAreEmpty) {
    mango::BuildDiagnostics d;
    EXPECT_FALSE(d.target_met);
    EXPECT_EQ(d.holdout_points, 0u);
}

// ===========================================================================
// Task 3: PrepareRefsFn / ScoreErrorFn split (spec D4)
// ===========================================================================

// Score equivalence with the old arithmetic:
TEST(ScoreFnTest, MatchesComputeIvError) {
    mango::AdaptiveGridParams p;  // target 2e-5, floor 1e-4
    auto score = mango::make_iv_score_fn(p, mango::OptionType::PUT);
    mango::ErrorRefs refs{.ref_price = 5.0, .vega = 20.0};
    // price_error 0.01 / vega 20 = 5e-4
    EXPECT_NEAR(score(5.01, refs, 100.0, 100.0, 1.0, 0.2, 0.05), 5e-4, 1e-12);
}

TEST(ScoreFnTest, TvkFilterZeroesDeepItm) {
    mango::AdaptiveGridParams p;
    auto score = mango::make_iv_score_fn(p, mango::OptionType::PUT);
    // K=100, S=100 put ref 0.005 -> TV/K = 5e-5 < 1e-4 -> filtered
    mango::ErrorRefs refs{.ref_price = 0.005, .vega = 1.0};
    EXPECT_EQ(score(1.0, refs, 100.0, 100.0, 0.01, 0.2, 0.05), 0.0);
}

TEST(PrepareRefsTest, PropagatesSolveFailure) {
    mango::ValidateFn failing = [](double, double, double, double, double)
        -> std::expected<double, mango::SolverError> {
        return std::unexpected(mango::SolverError{});
    };
    auto prep = mango::make_fd_vega_refs_fn(mango::AdaptiveGridParams{}, failing);
    EXPECT_FALSE(prep(100, 100, 1.0, 0.2, 0.05).has_value());
}
