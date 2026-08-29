// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/option/table/bspline/bspline_adaptive.hpp"
#include "mango/support/error_types.hpp"
#include <algorithm>
#include <cmath>
#include <expected>
#include <span>
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

// ===========================================================================
// Task 5: fit domain vs. sample domain separation (spec D2, D3)
// ===========================================================================

// Regression: headroom used to be 3 * width / (n_strikes - 1), which for a
// 7-strike chain gave 3 * w / 6 -- an order of magnitude too wide.  Spec D3
// requires the *expected seeded moneyness density* instead.
TEST(ExtractChainDomainTest, HeadroomUsesExpectedKnots) {
    mango::OptionGrid chain;
    chain.spot = 100.0;
    chain.strikes = {80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0};
    chain.maturities = {0.1, 1.0};
    chain.implied_vols = {0.1, 0.4};
    chain.rates = {0.02, 0.08};

    auto ctx = mango::extract_chain_domain(chain, 60);
    ASSERT_TRUE(ctx.has_value());

    const double w = ctx->sample_bounds.m_max - ctx->sample_bounds.m_min;
    const double expected_h = 3.0 * w / 59.0;
    EXPECT_NEAR(ctx->bounds.m_max - ctx->sample_bounds.m_max, expected_h, 1e-12);
    EXPECT_NEAR(ctx->sample_bounds.m_min - ctx->bounds.m_min, expected_h, 1e-12);

    // The old (n_strikes - 1) rule would have been 3 * w / 6 -- far wider.
    EXPECT_LT(expected_h, 3.0 * w / 6.0);

    // tau / vol / rate: fit == sample (headroom on moneyness only).
    EXPECT_EQ(ctx->bounds.tau_min, ctx->sample_bounds.tau_min);
    EXPECT_EQ(ctx->bounds.tau_max, ctx->sample_bounds.tau_max);
    EXPECT_EQ(ctx->bounds.sigma_min, ctx->sample_bounds.sigma_min);
    EXPECT_EQ(ctx->bounds.sigma_max, ctx->sample_bounds.sigma_max);
    EXPECT_EQ(ctx->bounds.rate_min, ctx->sample_bounds.rate_min);
    EXPECT_EQ(ctx->bounds.rate_max, ctx->sample_bounds.rate_max);
}

// sample_bounds is the user's own range (after minimum-spread widening,
// which is a usability floor rather than headroom).
TEST(ExtractChainDomainTest, SampleBoundsAreTheUserRange) {
    mango::OptionGrid chain;
    chain.spot = 100.0;
    chain.strikes = {80.0, 100.0, 120.0};
    chain.maturities = {0.25, 1.0};
    chain.implied_vols = {0.15, 0.35};
    chain.rates = {0.02, 0.08};

    auto ctx = mango::extract_chain_domain(chain, 60);
    ASSERT_TRUE(ctx.has_value());

    EXPECT_NEAR(ctx->sample_bounds.m_min, std::log(100.0 / 120.0), 1e-12);
    EXPECT_NEAR(ctx->sample_bounds.m_max, std::log(100.0 / 80.0), 1e-12);
    EXPECT_NEAR(ctx->sample_bounds.sigma_min, 0.15, 1e-12);
    EXPECT_NEAR(ctx->sample_bounds.sigma_max, 0.35, 1e-12);
    EXPECT_NEAR(ctx->sample_bounds.rate_min, 0.02, 1e-12);
    EXPECT_NEAR(ctx->sample_bounds.rate_max, 0.08, 1e-12);
    // tau minimum spread is 0.5: [0.25, 1.0] is 0.75 wide, so untouched.
    EXPECT_NEAR(ctx->sample_bounds.tau_min, 0.25, 1e-12);
    EXPECT_NEAR(ctx->sample_bounds.tau_max, 1.0, 1e-12);
}

// Spec D2: run_refinement draws every validation sample from sample_bounds
// while the grids it builds still span the (wider) fit domain.
TEST(RunRefinementDomainTest, ValidationSamplesStayInSampleBounds) {
    mango::AdaptiveGridParams p;
    p.validation_samples = 32;
    p.max_iter = 1;
    p.min_moneyness_points = 6;

    mango::RefinementContext ctx{
        .spot = 100.0,
        .dividend_yield = 0.0,
        .option_type = mango::OptionType::PUT,
        .bounds = {.m_min = -1.0, .m_max = 1.0,
                   .tau_min = 0.1, .tau_max = 2.0,
                   .sigma_min = 0.1, .sigma_max = 0.5,
                   .rate_min = 0.01, .rate_max = 0.09},
        .sample_bounds = {.m_min = -0.2, .m_max = 0.2,
                          .tau_min = 0.1, .tau_max = 2.0,
                          .sigma_min = 0.1, .sigma_max = 0.5,
                          .rate_min = 0.01, .rate_max = 0.09},
    };

    std::vector<double> queried_m;
    std::vector<double> built_m_grid;

    mango::BuildFn build_fn =
        [&](std::span<const double> m, std::span<const double>,
            std::span<const double>, std::span<const double>)
        -> std::expected<mango::SurfaceHandle, mango::PriceTableError> {
        built_m_grid.assign(m.begin(), m.end());
        return mango::SurfaceHandle{
            .price = [&queried_m](double spot, double strike, double,
                                  double, double) {
                queried_m.push_back(std::log(spot / strike));
                return 1.0;
            },
            .pde_solves = 0,
        };
    };

    mango::RefineFn refine_fn =
        [](size_t, std::span<const std::pair<double, double>>,
           std::vector<double>&, std::vector<double>&,
           std::vector<double>&, std::vector<double>&) {
            return mango::RefineOutcome{.changed = false, .changed_dim = -1};
        };

    mango::PrepareRefsFn prepare_refs =
        [](double, double, double, double, double)
        -> std::expected<mango::ErrorRefs, mango::SolverError> {
        return mango::ErrorRefs{.ref_price = 1.0, .vega = 20.0};
    };
    mango::ScoreErrorFn score =
        [](double, const mango::ErrorRefs&, double, double, double,
           double, double) { return 0.0; };

    auto result = mango::run_refinement(p, build_fn, refine_fn, ctx,
                                        prepare_refs, score);
    ASSERT_TRUE(result.has_value());

    // Grids span the fit domain ...
    ASSERT_FALSE(built_m_grid.empty());
    EXPECT_NEAR(built_m_grid.front(), ctx.bounds.m_min, 1e-12);
    EXPECT_NEAR(built_m_grid.back(), ctx.bounds.m_max, 1e-12);

    // ... but every validation sample lies inside the user domain.
    ASSERT_FALSE(queried_m.empty());
    for (double m : queried_m) {
        EXPECT_GE(m, ctx.sample_bounds.m_min - 1e-12);
        EXPECT_LE(m, ctx.sample_bounds.m_max + 1e-12);
    }
}
