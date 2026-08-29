// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/option/table/bspline/bspline_adaptive.hpp"
#include "mango/support/error_types.hpp"
#include <algorithm>
#include <utility>
#include <vector>

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

// ===========================================================================
// Task 4: RefineFn/RefineOutcome + B-spline refiner rewrite (spec D6, D2)
// ===========================================================================

TEST(BSplineRefineFnTest, NoOpAtCapReturnsUnchanged) {
    mango::AdaptiveGridParams p;
    p.max_points_per_dim = 4;
    auto fn = mango::make_bspline_refine_fn(p);
    std::vector<double> m{0.0, 0.1, 0.2, 0.3}, t{0.1, 0.5, 1.0, 2.0},
                        v{0.1, 0.2, 0.3, 0.4}, r{0.01, 0.03, 0.05, 0.08};
    auto out = fn(0, {}, m, t, v, r);
    EXPECT_FALSE(out.changed);
    EXPECT_EQ(out.changed_dim, -1);
    EXPECT_EQ(m.size(), 4u);
}

TEST(BSplineRefineFnTest, UniformWhenNoFocus) {
    mango::AdaptiveGridParams p;
    p.refinement_factor = 2.0;  // enough budget to fill every gap
    auto fn = mango::make_bspline_refine_fn(p);
    std::vector<double> m{0.0, 0.3, 0.6, 1.0};
    std::vector<double> t{0.1, 0.5, 1.0, 2.0};
    std::vector<double> v{0.1, 0.2, 0.3, 0.4};
    std::vector<double> r{0.01, 0.03, 0.05, 0.08};

    auto out = fn(0, {}, m, t, v, r);

    EXPECT_TRUE(out.changed);
    EXPECT_EQ(out.changed_dim, 0);
    // Grew toward size * refinement_factor (= 8), capped by the number of
    // gaps actually available to insert into (3) in a single pass.
    size_t target = std::min<size_t>(
        static_cast<size_t>(4 * p.refinement_factor), p.max_points_per_dim);
    EXPECT_GT(target, 4u);
    EXPECT_EQ(m.size(), 7u);

    // Uniform refinement (no focus) spreads new points across the whole
    // axis, not just one localized region.
    EXPECT_TRUE(std::any_of(m.begin(), m.end(),
        [](double x) { return x > 0.0 && x < 0.3; }));
    EXPECT_TRUE(std::any_of(m.begin(), m.end(),
        [](double x) { return x > 0.3 && x < 0.6; }));
    EXPECT_TRUE(std::any_of(m.begin(), m.end(),
        [](double x) { return x > 0.6 && x < 1.0; }));
}

TEST(BSplineRefineFnTest, FocusIntervalTargetsBin) {
    mango::AdaptiveGridParams p;
    p.refinement_factor = 3.0;  // plenty of insertion budget
    auto fn = mango::make_bspline_refine_fn(p);
    std::vector<double> m{0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
    std::vector<double> t{0.1, 0.5, 1.0, 2.0};
    std::vector<double> v{0.1, 0.2, 0.3, 0.4};
    std::vector<double> r{0.01, 0.03, 0.05, 0.08};
    const std::vector<double> original = m;

    std::vector<std::pair<double, double>> focus = {{0.55, 0.85}};
    auto out = fn(0, focus, m, t, v, r);

    EXPECT_TRUE(out.changed);
    EXPECT_EQ(out.changed_dim, 0);
    EXPECT_GT(m.size(), original.size());

    // Every newly inserted point lies inside the provided focus interval.
    for (double x : m) {
        bool is_original =
            std::find(original.begin(), original.end(), x) != original.end();
        if (!is_original) {
            EXPECT_GE(x, 0.55);
            EXPECT_LE(x, 0.85);
        }
    }
}
