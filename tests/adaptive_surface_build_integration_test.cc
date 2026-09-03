// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/american_option.hpp"
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/bspline/bspline_adaptive.hpp"
#include "mango/option/table/bspline/bspline_pde_cache.hpp"
#include "mango/option/table/bspline/bspline_segmented_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"
#include "mango/option/table/adaptive_metrics.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/math/chebyshev/chebyshev_nodes.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/interpolated_iv_solver.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace mango {
namespace {

/// Convert S/K moneyness to log-moneyness for internal builder APIs.
std::vector<double> to_log_m(std::initializer_list<double> sk) {
    std::vector<double> v;
    v.reserve(sk.size());
    for (double m : sk) v.push_back(std::log(m));
    return v;
}

TEST(AdaptiveGridBuilderTest, BuildsWithSyntheticChain) {
    // Create a minimal synthetic chain
    OptionGrid chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;

    // Add strikes and maturities
    chain.strikes = {90.0, 95.0, 100.0, 105.0, 110.0};
    chain.maturities = {0.25, 0.5, 1.0};
    chain.implied_vols = {0.18, 0.20, 0.22};  // Some variation
    chain.rates = {0.04, 0.05, 0.06};

    AdaptiveGridParams params;
    params.target_iv_error = 0.002;  // 20 bps - relaxed for test speed
    params.max_iter = 2;
    params.validation_samples = 8;  // Fewer for test speed

    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0).value();
    auto result = build_adaptive_bspline(params, chain,
        PDEGridConfig{grid_spec, 200, {}}, OptionType::PUT);

    if (!result.has_value()) {
        std::cerr << "Build failed with error code: "
                  << static_cast<int>(result.error().code) << "\n";
    }
    ASSERT_TRUE(result.has_value());

    // Should have at least one iteration
    EXPECT_GE(result->iterations.size(), 1);

    // Spline should be populated
    EXPECT_NE(result->spline, nullptr);

    // Should have done some PDE solves
    EXPECT_GT(result->total_pde_solves, 0);
}


// ===========================================================================
// BSplinePDECache unit tests
// ===========================================================================


// ===========================================================================
// ErrorBins unit tests
// ===========================================================================


// ===========================================================================
// Regression tests for bugs found during code review
// ===========================================================================

// Regression: Single-value axes must be expanded to provide distinct grid points
// Bug: linspace(x, x, 5) produces {x, x, x, x, x} which dedupes to 1 point,
// causing B-spline fitting failure (requires >= 4 points)
TEST(AdaptiveGridBuilderTest, RegressionSingleValueAxes) {
    OptionGrid chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;

    // Single strike = single moneyness value (needs expansion)
    chain.strikes = {100.0};
    // Multiple maturities (don't need expansion for this test to be valid)
    chain.maturities = {0.25, 0.5, 1.0};
    // Single vol (needs expansion)
    chain.implied_vols = {0.20};
    // Single rate (needs expansion)
    chain.rates = {0.05};

    AdaptiveGridParams params;
    params.target_iv_error = 0.01;  // Very relaxed
    params.max_iter = 1;
    params.validation_samples = 8;  // spec D3 minimum

    auto grid_spec = GridSpec<double>::uniform(-3.0, 3.0, 31).value();
    auto result = build_adaptive_bspline(params, chain,
        PDEGridConfig{grid_spec, 100, {}}, OptionType::PUT);

    // Should succeed (bounds expanded) rather than fail with InsufficientGridPoints
    ASSERT_TRUE(result.has_value())
        << "Single-value axes should be expanded to valid ranges. "
        << "Error code: " << (result.has_value() ? 0 : static_cast<int>(result.error().code));

    // Spline should be usable
    EXPECT_NE(result->spline, nullptr);
}

// Regression: Cache should clear on new build
// Bug: reuse of AdaptiveGridBuilder re-used previous slices because cache wasn't cleared
TEST(AdaptiveGridBuilderTest, RegressionCacheClearedBetweenBuilds) {
    OptionGrid chain1;
    chain1.spot = 100.0;
    chain1.dividend_yield = 0.0;
    chain1.strikes = {90.0, 100.0, 110.0};
    chain1.maturities = {0.25, 0.5, 1.0};
    chain1.implied_vols = {0.18, 0.22};
    chain1.rates = {0.04, 0.05};

    OptionGrid chain2 = chain1;
    // Different spot => cache must not reuse chain1 slices.  90, not 50: at a
    // spot of 50 against strikes of 90-110 every holdout point is a deep-ITM
    // put whose time value is below the TV/K filter, so the build is measured
    // nowhere and now refuses (spec D4/D5) -- a real contract, but not the
    // one this test is about.
    chain2.spot = 90.0;

    AdaptiveGridParams params;
    params.max_iter = 1;
    params.validation_samples = 8;  // spec D3 minimum

    auto grid_spec = GridSpec<double>::uniform(-3.0, 3.0, 31).value();

    // Free functions create fresh caches each call, so no cross-contamination
    auto result1 = build_adaptive_bspline(params, chain1,
        PDEGridConfig{grid_spec, 100, {}}, OptionType::PUT);
    ASSERT_TRUE(result1.has_value());
    size_t solves1 = result1->iterations[0].pde_solves_table;

    auto result2 = build_adaptive_bspline(params, chain2,
        PDEGridConfig{grid_spec, 100, {}}, OptionType::PUT);
    ASSERT_TRUE(result2.has_value());
    size_t solves2 = result2->iterations[0].pde_solves_table;

    EXPECT_EQ(solves1, solves2) << "Second build should recompute all slices for new chain";
}

// A two-K_ref list cannot blend accurately across the strike range it is
// asked to serve, and the build refuses rather than shipping the blend.
//
// K_refs {90, 110} against S/K in [0.91, 1.1] means strikes in [90.9, 109.9]
// served by exactly two surfaces: every query but the two endpoints is a
// linear-in-strike blend across a 20-point gap.  The assembled surface
// measures **0.4756 (4,756 bps) max IV error** on the D9 validation set
// (avg 0.1191, 15 of 16 points measured), and the bumped-grid retry measures
// 0.4766 -- both far outside the 0.20 viability bound, so the build returns
// `NoViableSurface`.
//
// This shipped silently before #434.  The test previously asserted success:
// with the full dividend schedule handed to every reference solve, each
// sample below the dividend date lost its reference and only the long-tau
// tail was measured, which was not enough to expose the blend.  Once
// `make_validate_fn` filters the schedule by the sampled maturity all 16
// samples measure, and the sparse-K_ref error is unavoidable.
//
// Tracked as the sparse-K_ref accuracy follow-up (MultiKRefSplit blend
// resolution); the refusal is the correct behavior until it lands.
TEST(AdaptiveGridBuilderTest, BuildSegmentedSmallKRefList) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 1;
    params.validation_samples = 16;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {Dividend{.calendar_time = 0.25, .amount = 1.50}},
        .maturity = 0.5,
        .kref_config = {.K_refs = {90.0, 110.0}},  // < 3 K_refs — probe all
    };

    auto m_domain = to_log_m({0.91, 0.95, 1.0, 1.05, 1.1});
    std::vector<double> v_domain = {0.10, 0.15, 0.20, 0.30};
    std::vector<double> r_domain = {0.02, 0.03, 0.05, 0.07};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m_domain, v_domain, r_domain});
    ASSERT_FALSE(result.has_value())
        << "a two-K_ref blend measuring 4,756 bps must not be returned";
    EXPECT_EQ(result.error().code, PriceTableErrorCode::NoViableSurface);
}

// Large discrete dividend (total_div/K_ref > 0.2, stresses moneyness expansion)
//
// $20 of *absolute* dividends against a $100 spot does not produce a usable
// surface: the assembled multi-K_ref surface measures 58.4 IV error (583,897
// bps) on plain user-domain validation, and the worst probe measures 0.97
// (9,740 bps) against the 0.20 viability bound.  Returning that surface
// silently was the pre-#434 behavior and is the defect this branch exists to
// fix -- refusal is the contract (spec D5).
TEST(AdaptiveGridBuilderTest, BuildSegmentedLargeDividend) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 2;
    params.validation_samples = 16;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {Dividend{.calendar_time = 0.25, .amount = 10.0},
                               Dividend{.calendar_time = 0.75, .amount = 10.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {70.0, 100.0, 130.0}},
    };

    auto m_domain = to_log_m({0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5});
    std::vector<double> v_domain = {0.05, 0.10, 0.20, 0.30, 0.50};
    std::vector<double> r_domain = {0.01, 0.03, 0.05, 0.10};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m_domain, v_domain, r_domain});
    ASSERT_FALSE(result.has_value())
        << "an unusable surface must not be returned";
    EXPECT_EQ(result.error().code, PriceTableErrorCode::NoViableSurface);
}

// No dividends (single segment, degenerates to simple case)
TEST(AdaptiveGridBuilderTest, BuildSegmentedNoDividends) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 1;
    params.validation_samples = 8;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {},  // No discrete dividends
        .maturity = 1.0,
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    };

    auto m_domain = to_log_m({0.85, 0.9, 1.0, 1.1, 1.2});
    std::vector<double> v_domain = {0.10, 0.15, 0.20, 0.30};
    std::vector<double> r_domain = {0.02, 0.03, 0.05, 0.07};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m_domain, v_domain, r_domain});
    ASSERT_TRUE(result.has_value()) << "code " << static_cast<int>(result.error().code);

    double price = result->surface.price(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
    EXPECT_TRUE(std::isfinite(price));
}

// ===========================================================================
// Probe measurement bands (spec D2/D9)
//
// The assembled multi-K_ref surface routes each query to the K_ref nearest
// its strike, so each probe is measured only over the strike band it serves:
// the geometric midpoints to its neighbours, clipped to the user's own
// strike range.  These two tests pin the band's degenerate cases.
// ===========================================================================

// A band thinner than the loop's non-degeneracy tolerance is widened about
// its midpoint rather than being handed to run_refinement as m_max == m_min
// (which would fail the build with InvalidConfig).  K_refs one basis point
// apart give the middle probe a band ~1e-4 wide in log-moneyness.
//
// Such a config cannot produce a usable surface: three K_refs within one
// basis point of 100 cannot resolve strikes spanning [90.9, 111.1], and the
// assembled surface measures 0.278 (2,776 bps) on the final validation
// against the 0.20 viability bound, so the build refuses (spec D9).  What
// this test pins is *which* refusal: `NoViableSurface` from the final gate
// means the degenerate band was widened and every probe loop ran;
// `InvalidConfig` would mean the band was handed over degenerate.
TEST(AdaptiveGridBuilderTest, BuildSegmentedDegenerateProbeBandWidened) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.01;
    params.max_iter = 1;
    params.validation_samples = 8;
    params.min_moneyness_points = 10;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {},
        .maturity = 1.0,
        .kref_config = {.K_refs = {99.99, 100.0, 100.01}},
    };

    auto m = to_log_m({0.9, 0.95, 1.0, 1.05, 1.1});
    std::vector<double> v = {0.15, 0.20, 0.30, 0.40};
    std::vector<double> r = {0.02, 0.03, 0.05, 0.07};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m, v, r});
    ASSERT_FALSE(result.has_value())
        << "three K_refs a basis point apart cannot serve [90.9, 111.1]";
    EXPECT_EQ(result.error().code, PriceTableErrorCode::NoViableSurface)
        << "a degenerate band must be widened and measured, not rejected up "
           "front (InvalidConfig would mean it reached run_refinement "
           "degenerate)";
}

// A probe whose served band lies entirely outside the user's strike range is
// skipped: no refinement loop, its seed sizes still feed the aggregate, and
// the skip is recorded with the refined_dim = -3 sentinel.  K_ref = 50 with
// user strikes in [91.7, 108.7] serves nothing: its band ends at the
// geometric midpoint to its neighbour, sqrt(50 * 90) = 67.1.
TEST(AdaptiveGridBuilderTest, BuildSegmentedEmptyProbeBandSkipped) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.01;
    params.max_iter = 1;
    params.validation_samples = 8;
    params.min_moneyness_points = 10;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {},
        .maturity = 1.0,
        .kref_config = {.K_refs = {50.0, 90.0, 100.0, 110.0}},
    };

    auto m = to_log_m({0.92, 0.95, 1.0, 1.05, 1.09});
    std::vector<double> v = {0.15, 0.20, 0.30, 0.40};
    std::vector<double> r = {0.02, 0.03, 0.05, 0.07};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m, v, r});
    ASSERT_TRUE(result.has_value())
        << "a probe serving no queryable strike must be skipped, not fatal: "
        << "code " << static_cast<int>(result.error().code);

    size_t skipped = 0;
    for (const auto& it : result->iterations) {
        if (it.refined_dim == -3) ++skipped;
    }
    EXPECT_EQ(skipped, 1u) << "the K_ref = 50 probe should be recorded skipped";
}

// Coverage: Single auto-generated K_ref (count=1)
//
// One K_ref cannot serve a +/-30 % strike range.  The assembled surface
// prices every query as (K / K_ref) * P(S, K_ref) -- the multi-K_ref split
// substitutes K_ref for the query strike while holding the spot fixed -- so
// it measures 4.69 (46,924 bps) on the final validation and the build
// refuses (spec D9).  The coverage here is that `K_ref_count = 1` resolves
// to a single K_ref and the build runs all the way to the final gate rather
// than failing configuration validation.
//
// Revisit when MultiKRefSplit spot-scaling is fixed (follow-up): a split
// that mapped the query onto the K_ref problem instead of substituting the
// strike would make a single K_ref usable, and this test would go back to
// asserting a successful build.
TEST(AdaptiveGridBuilderTest, BuildSegmentedSingleAutoKRef) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 1;
    params.validation_samples = 8;
    params.min_moneyness_points = 10;  // Use smaller grid for test speed

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {}, .K_ref_count = 1, .K_ref_span = 0.3},
    };

    auto m = to_log_m({0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3});
    std::vector<double> v = {0.10, 0.15, 0.20, 0.30};
    std::vector<double> r = {0.02, 0.03, 0.05, 0.07};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m, v, r});
    ASSERT_FALSE(result.has_value())
        << "a lone K_ref cannot serve a +/-30 % strike range";
    EXPECT_EQ(result.error().code, PriceTableErrorCode::NoViableSurface)
        << "K_ref_count = 1 must resolve and build, then fail the final "
           "viability gate -- not fail configuration validation";
}

// Coverage: Very short maturity — tau domain compressed, max_tau clamped
//
// A 0.05-year maturity with a discrete dividend has vega near zero, so any
// price error divides into an enormous IV error: the assembled surface
// measures 229 (2.29 million bps) on user-domain validation and the worst
// probe 3,847.  Returning it silently was the pre-#434 behavior; the build
// now refuses (spec D5).
TEST(AdaptiveGridBuilderTest, BuildSegmentedVeryShortMaturity) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 1;
    params.validation_samples = 8;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {Dividend{.calendar_time = 0.02, .amount = 1.0}},
        .maturity = 0.05,  // Very short
        .kref_config = {.K_refs = {90.0, 100.0, 110.0}},
    };

    auto m = to_log_m({0.8, 0.9, 1.0, 1.1, 1.2});
    std::vector<double> v = {0.10, 0.20, 0.30, 0.40};
    std::vector<double> r = {0.02, 0.03, 0.05, 0.07};

    // The tau domain is still built and clamped to the maturity -- the
    // refusal below comes from the accuracy gate, not from a domain error.
    auto bounds = expand_segmented_domain(
        {m, v, r}, seg_config.maturity, seg_config.dividend_yield,
        seg_config.discrete_dividends, 90.0);
    ASSERT_TRUE(bounds.has_value());
    EXPECT_LE(bounds->tau_max, seg_config.maturity);
    EXPECT_GT(bounds->tau_min, 0.0);

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m, v, r});
    ASSERT_FALSE(result.has_value())
        << "an unusable surface must not be returned";
    EXPECT_EQ(result.error().code, PriceTableErrorCode::NoViableSurface);
}

// ===========================================================================
// Coverage gap tests — Priority 3 (Medium)
// ===========================================================================

// Coverage: Large expansion clamps moneyness to 0.01
//
// The clamp itself is asserted directly (no build needed).  The adaptive
// build over the same config is then refused: $50 of absolute dividends
// against a $100 spot leaves no usable surface, and the D5 viability gate in
// the probe loop says so -- `NoViableSurface`.
//
// This expected `ValidationFailed` before #434, and the reason it no longer
// does is the point: with the full schedule handed to every reference solve,
// every sampled tau below the last dividend date (0.75 of a 1y surface) lost
// its reference and the holdout fell under the `max(4, n/4)` floor, so the
// build died at reference validation without ever scoring a surface.  Now
// that `make_validate_fn` filters the schedule by the sampled maturity those
// references solve, the holdout clears the floor, and the build proceeds far
// enough for the viability gate to do the refusing.  Both codes mean "this
// must not be returned"; the build simply gets further before saying it.
//
// The D4 `ValidationFailed` path keeps its own coverage elsewhere, at both
// levels: `SegmentedFinalContract.SparseReferencesFailValidation` drives
// `prepare_final_validation` past the floor (and back under it) directly, and
// `RunRefinementTest.HoldoutValidityThresholds` does the same for the
// refinement loop's holdout.
TEST(AdaptiveGridBuilderTest, BuildSegmentedMoneynessClampedToFloor) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 1;
    params.validation_samples = 8;
    params.min_moneyness_points = 10;  // Use smaller grid for test speed

    // total_div = 50, K_ref_min = 50 → expansion = 1.0
    // min_m = 0.5, expanded = max(0.5 - 1.0, 0.01) = 0.01
    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {Dividend{.calendar_time = 0.25, .amount = 25.0},
                               Dividend{.calendar_time = 0.75, .amount = 25.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {50.0, 100.0, 150.0}},
    };

    auto m = to_log_m({0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5});
    std::vector<double> v = {0.10, 0.20, 0.30, 0.50};
    std::vector<double> r = {0.02, 0.05, 0.07, 0.10};

    // The moneyness floor prevents a negative/zero domain: expansion = 1.0
    // against a lowest moneyness of 0.5 clamps to 0.01 rather than -0.5.
    auto bounds = expand_segmented_domain(
        {m, v, r}, seg_config.maturity, seg_config.dividend_yield,
        seg_config.discrete_dividends, 50.0);
    ASSERT_TRUE(bounds.has_value());
    EXPECT_NEAR(bounds->m_min, std::log(0.01), 1e-12);
    EXPECT_GT(bounds->m_max, bounds->m_min);

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m, v, r});
    ASSERT_FALSE(result.has_value())
        << "a surface this config cannot support must not be certified";
    EXPECT_EQ(result.error().code, PriceTableErrorCode::NoViableSurface);
}

// Coverage: Negative K_ref in explicit list (K_ref_min <= 0 guard)
TEST(AdaptiveGridBuilderTest, BuildSegmentedNegativeKRefExpansionGuard) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 1;
    params.validation_samples = 8;

    // K_ref_min=0.01 is very small, making expansion = total_div / 0.01 = 200
    // This exercises the K_ref_min > 0 guard and the moneyness clamp.
    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {0.01, 100.0, 200.0}},
    };

    auto m = to_log_m({0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0});
    std::vector<double> v = {0.10, 0.20, 0.30, 0.50};
    std::vector<double> r = {0.02, 0.05, 0.07, 0.10};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m, v, r});
    // With K_ref=0.01, the per-K_ref PDE build will likely fail.
    // The important thing is it doesn't crash or divide by zero.
    // It should either succeed or return a clean error.
    if (!result.has_value()) {
        // Acceptable: clean error propagation, no crash
        SUCCEED();
    } else {
        // Also acceptable: managed to build despite extreme K_ref
        SUCCEED();
    }
}

// Regression: empty tau grid must return error, not crash
// Bug: Very short maturity with mid-tau dividend made all segments narrower
// than kMinSegmentWidth. The tau grid was empty, causing UB when
// build callback dereferenced tau_nodes.back().
// Regression: narrow real segments must not be treated as gaps.
// Bug: width-based gap detection (hi - lo < kMinSegmentWidth) misclassified
// narrow real segments as gaps, producing zero prices or errors.
TEST(AdaptiveGridBuilderTest, SegmentedChebyshevNarrowSegmentsStillWork) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.01;
    params.max_iter = 1;
    params.validation_samples = 8;  // spec D3 minimum

    // Maturity=0.02 (~7 days) with dividend at mid-point.
    // Gap ε=5e-4 on each side of tau_split=0.01 creates segments
    // [0.005, 0.0095] and [0.0105, 0.015] — narrow but real.
    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {Dividend{.calendar_time = 0.01, .amount = 0.50}},
        .maturity = 0.02,
        .kref_config = {.K_refs = {100.0}},
    };

    auto m_domain = to_log_m({0.9, 1.0, 1.1});
    std::vector<double> v_domain = {0.15, 0.25};
    std::vector<double> r_domain = {0.05};

    // A 7-day option has near-zero vega, so any price error divides into an
    // enormous IV error: the assembled surface measures 970 (9.7 million
    // bps) on the final validation, and the adaptive path now refuses it
    // (spec D9) exactly as BuildSegmentedVeryShortMaturity does.  The
    // regression under test is in segment classification, not in
    // refinement, so it is pinned on the fixed-level build of the same
    // configuration.  Revisit when MultiKRefSplit spot-scaling is fixed
    // (follow-up): part of this error is the lone K_ref, not the maturity.
    auto adaptive = build_adaptive_chebyshev_segmented(
        params, seg_config, {m_domain, v_domain, r_domain});
    ASSERT_FALSE(adaptive.has_value());
    EXPECT_EQ(adaptive.error().code, PriceTableErrorCode::NoViableSurface)
        << "the segments must build and be measured; a gap misclassification "
           "would surface as a build error instead";

    auto surface = build_chebyshev_segmented_manual(
        seg_config, {m_domain, v_domain, r_domain});
    // Narrow real segments should build successfully, not be rejected as gaps
    ASSERT_TRUE(surface.has_value())
        << "Narrow real segments should produce valid prices, not errors";

    // Price at ATM should be positive
    double p = surface->price(100.0, 100.0, 0.01, 0.20, 0.05);
    EXPECT_GT(p, 0.0) << "ATM put price should be positive";
}

// Regression (#437): the adaptive cached path bypasses build()'s upfront
// explicit-grid coverage validation (bspline_builder.cpp:73-84), so an
// explicit PDE grid narrower than the moneyness fit axis was silently
// accepted and its tails cubic-spline-extrapolated by extract_tensor.
TEST(AdaptiveGridBuilderTest, RejectsExplicitGridNotCoveringMoneyness) {
    OptionGrid chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;
    chain.strikes = {60.0, 80.0, 100.0, 120.0, 140.0};
    chain.maturities = {0.05, 0.1};
    chain.implied_vols = {0.10, 0.15, 0.20};
    chain.rates = {0.03, 0.05};

    AdaptiveGridParams params;
    params.target_iv_error = 0.002;
    params.max_iter = 1;
    params.validation_samples = 4;

    // Half-width 0.25 vs required |ln(100/60)| ~= 0.51 (+ headroom).
    auto grid_spec = GridSpec<double>::sinh_spaced(-0.25, 0.25, 101, 2.0).value();
    auto result = build_adaptive_bspline(params, chain,
        PDEGridConfig{grid_spec, 200, {}}, OptionType::PUT);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}

// Regression: a log-moneyness lower bound that does not survive an exp/log
// round trip used to inject three near-duplicate knots.
// Bug: expand_log_moneyness_grid compared log(exp(x_min) - total_div/K_ref)
// against x_min with no tolerance.  With no dividends the subtraction is a
// no-op, but the round trip can land one ULP below x_min, so the "expansion"
// branch fired and inserted three knots spaced ~1e-17 apart.  The cubic
// collocation solver then rejected the grid as unsorted and the whole build
// failed with FittingFailed.
TEST(SegmentedPriceTableBuilderTest, RegressionUlpRoundTripDoesNotExpand) {
    // log(exp(x)) < x for this value.
    constexpr double kHostileXMin = -0.38815151385769298;
    ASSERT_LT(std::log(std::exp(kHostileXMin)), kHostileXMin);

    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.02, .discrete_dividends = {}},
        .grid = {.moneyness = {kHostileXMin, -0.2, 0.0, 0.15, 0.29},
                 .vol = {0.10, 0.15, 0.20, 0.30},
                 .rate = {0.02, 0.03, 0.05, 0.07}},
        .maturity = 1.0,
        .tau_points_per_segment = 5,
    };

    auto surface = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(surface.has_value())
        << "build failed with code "
        << static_cast<int>(surface.error().code);
}


// ===========================================================================
// Segmented final-surface contracts (spec D9)
//
// The segmented builders assemble their final surface outside the refinement
// loop, so it gets its own references, its own score, and its own viability
// gate.  These tests pin the selection arithmetic directly (no PDE solves)
// and then check the assembled B-spline path reports its *returned* surface.
// ===========================================================================


// The numbers a segmented build reports must describe the surface it
// returned.  Pre-#434 the bumped-grid retry was returned carrying the
// *pre-retry* error numbers; here the returned surface is re-scored on an
// independently reproduced reference set and must match what it reported.
TEST(SegmentedFinalContract, ReportedErrorsDescribeReturnedSurface) {
    AdaptiveGridParams params;
    params.target_iv_error = 1e-6;  // unreachable => the retry path is taken
    params.max_iter = 1;
    // 16, not 8: `solve_american_option` refuses a schedule whose dividend
    // date is at or beyond the requested maturity, so every sample with
    // tau <= 0.25 loses its reference -- half the tau range here.  Eight
    // samples would leave the validation set sitting exactly on the
    // `max(4, n/4)` floor.
    params.validation_samples = 16;
    params.min_moneyness_points = 8;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {Dividend{.calendar_time = 0.25, .amount = 1.50}},
        .maturity = 0.5,
        .kref_config = {.K_refs = {90.0, 95.0, 100.0, 105.0, 110.0}},
    };

    auto m_domain = to_log_m({0.95, 1.0, 1.05});
    std::vector<double> v_domain = {0.10, 0.15, 0.20, 0.30};
    std::vector<double> r_domain = {0.02, 0.03, 0.05, 0.07};
    IVGrid domain{m_domain, v_domain, r_domain};

    auto result = build_adaptive_bspline_segmented(params, seg_config, domain);
    ASSERT_TRUE(result.has_value())
        << "code " << static_cast<int>(result.error().code);

    // The target is unreachable, so the builder must have tried the retry and
    // reported the miss honestly.
    EXPECT_FALSE(result->target_met);
    EXPECT_EQ(result->diagnostics.target_met, result->target_met);
    EXPECT_DOUBLE_EQ(result->diagnostics.achieved_max_error,
                     result->achieved_max_error);
    EXPECT_DOUBLE_EQ(result->diagnostics.achieved_avg_error,
                     result->achieved_avg_error);
    EXPECT_GT(result->diagnostics.holdout_points, 0u);
    EXPECT_LE(result->achieved_max_error, kViabilityBound);

    // Reproduce the builder's final validation set exactly (same sample
    // domain, same seed, same references) and re-score the surface we were
    // handed.  A retry returned with the original's numbers fails here.
    auto K_refs = resolve_k_refs(seg_config.kref_config, seg_config.spot);
    ASSERT_TRUE(K_refs.has_value());
    auto sample = expand_segmented_domain(domain, seg_config.maturity,
                                          seg_config.dividend_yield, {},
                                          K_refs->front());
    ASSERT_TRUE(sample.has_value());

    RefinementContext ctx{
        .spot = seg_config.spot,
        .dividend_yield = seg_config.dividend_yield,
        .option_type = seg_config.option_type,
        .bounds = *sample,
        .sample_bounds = *sample,
    };
    auto validate_fn = make_validate_fn(seg_config.dividend_yield,
                                        seg_config.option_type,
                                        seg_config.discrete_dividends);
    auto refs_fn = make_fd_vega_refs_fn(params, validate_fn);
    auto points = detail::prepare_final_validation(params, ctx, refs_fn,
                                                   params.lhs_seed + 999);
    ASSERT_TRUE(points.has_value());

    const SurfaceHandle returned{
        .price = [&](double spot, double strike, double tau, double sigma,
                     double rate) {
            return result->surface.price(spot, strike, tau, sigma, rate);
        }};
    auto measured = detail::score_final_surface(
        points->points, returned, make_iv_score_fn(params, seg_config.option_type),
        ctx);

    EXPECT_EQ(measured.measured,
              result->diagnostics.holdout_points_measured);
    EXPECT_NEAR(measured.max_error, result->achieved_max_error, 1e-12)
        << "reported max error does not describe the returned surface"
        << " (used_retry = " << result->used_retry << ")";
    EXPECT_NEAR(measured.avg_error, result->achieved_avg_error, 1e-12);

    // Which of the two surfaces wins is deliberately NOT pinned.
    //
    // This config sits at its own accuracy floor, so the bumped grids buy
    // nothing and the two scores land on top of each other: measured here,
    // original 0.020230 vs retry 0.020352 -- 0.6 % apart, with the original
    // winning by 1.2e-4.  Sweeping `min_moneyness_points` over 5..12 shows
    // the retry winning at 6 and losing at 5, 7, 8, 9, 10 and 12, with every
    // score in 0.0196-0.0278 and no trend in the grid size: the outcome is
    // numerical noise, not a property of the design.  An earlier revision
    // asserted `used_retry` here and duly broke when an unrelated fix to the
    // reference solves shifted the validation set.
    //
    // The contract this test exists for is the identity above -- the
    // *reported* numbers describe the surface actually returned -- and it is
    // checked unconditionally.  The grid check below extends that identity to
    // the reported grid sizes, for whichever surface won.
    //
    // With max_iter = 1 no probe refines, so every probe returns its seed and
    // the aggregate is exactly the seed sizes; the retry adds (+2, +2, +1, +1)
    // on (moneyness, tau, vol, rate).
    auto support = expand_segmented_domain(
        domain, seg_config.maturity, seg_config.dividend_yield,
        seg_config.discrete_dividends, K_refs->front());
    ASSERT_TRUE(support.has_value());
    SurfaceBounds fit = *support;
    const double headroom = spline_support_headroom(
        sample->m_max - sample->m_min,
        std::max(domain.moneyness.size(), params.min_moneyness_points));
    fit.m_min -= headroom;
    fit.m_max += headroom;

    RefinementContext seed_ctx{
        .spot = seg_config.spot,
        .dividend_yield = seg_config.dividend_yield,
        .option_type = seg_config.option_type,
        .bounds = fit,
        .sample_bounds = *sample,
    };
    auto seeded = seed_refinement_grids(
        params, seed_ctx,
        InitialGrids{.moneyness = domain.moneyness,
                     .vol = domain.vol,
                     .rate = domain.rate});

    const size_t m_bump = result->used_retry ? 2 : 0;
    const size_t v_bump = result->used_retry ? 1 : 0;
    const size_t r_bump = result->used_retry ? 1 : 0;
    const int tau_bump = result->used_retry ? 2 : 0;

    EXPECT_EQ(result->grid.moneyness.size(),
              std::min(seeded.moneyness.size() + m_bump,
                       params.max_points_per_dim))
        << "reported moneyness grid does not describe the returned surface"
        << " (used_retry = " << result->used_retry << ")";
    EXPECT_EQ(result->grid.vol.size(),
              std::min(seeded.vol.size() + v_bump, params.max_points_per_dim));
    EXPECT_EQ(result->grid.rate.size(),
              std::min(seeded.rate.size() + r_bump, params.max_points_per_dim));
    EXPECT_EQ(result->tau_points_per_segment,
              std::min(static_cast<int>(seeded.tau.size()) + tau_bump,
                       static_cast<int>(params.max_points_per_dim)));
}

// Reference FDM price with a PINNED explicit configuration (spec: the
// tolerance floor must not drift if solve_american_option defaults
// change).
double fdm_reference_price(double spot, double strike, double tau,
                           double sigma, double rate) {
    PricingParams ref_params(
        OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
                   .rate = rate, .dividend_yield = 0.0,
                   .option_type = OptionType::PUT},
        sigma);
    auto solver = AmericanOptionSolver::create(
        ref_params, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::High)});
    EXPECT_TRUE(solver.has_value());
    auto ref = solver->solve();
    EXPECT_TRUE(ref.has_value());
    return ref->value_at(spot);
}

// Regression (#437): the adaptive cached path (GridAccuracyParams branch)
// solved gridless, so per-normalized-group estimation gave each sigma
// slice half-width n_sigma * sigma * sqrt(tau).  make_batch() solves at
// spot=strike=K_ref (x0=0) with maturity fixed to the FIT tau axis's
// upper bound -- which extract_chain_domain widens to a 0.5y floor
// regardless of the chain's own maturities (measured: 0.500001, not the
// chain's raw max of 0.1) -- so with the default n_sigma=5.0 and
// sigma=0.10 the half-width is 5.0*0.10*sqrt(0.500001) ~= 0.3536,
// against the fit axis's lower endpoint |ln(100/140)| ~= 0.3365, widened by
// ~0.043 of B-spline support headroom to ~0.3796 -- the endpoint where the
// pre-fix failure below was actually measured.
// extract_tensor extrapolates that tail. Min-sigma assertions guard the
// routing defect specifically: a widening-only fix covers the max-sigma
// slice while every lower-sigma slice still extrapolates.
// Pre-fix max abs error on this branch's parent: 0.4263 (m=-0.379555,
// sigma=0.10; constant across all queried tau -- the underlying PDE
// domain is identical regardless of snapshot tau, confirming genuine
// extrapolation rather than ordinary interpolation error).
TEST(AdaptiveGridBuilderTest, TensorTailsMatchFdmAtExtremeMoneyness) {
    OptionGrid chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;
    chain.strikes = {60.0, 80.0, 100.0, 120.0, 140.0};
    chain.maturities = {0.05, 0.1};
    chain.implied_vols = {0.10, 0.15, 0.20};
    chain.rates = {0.03, 0.05};

    AdaptiveGridParams params;
    params.target_iv_error = 0.002;  // relaxed: accuracy is asserted below
    params.max_iter = 2;
    params.validation_samples = 8;

    auto result = build_adaptive_bspline(
        params, chain, make_grid_accuracy(GridAccuracyProfile::High),
        OptionType::PUT);
    ASSERT_TRUE(result.has_value());

    auto wrapper = make_bspline_surface(
        result->spline, result->K_ref, result->dividend_yield,
        OptionType::PUT);
    ASSERT_TRUE(wrapper.has_value());

    const auto& m_axis = result->axes.grids[0];
    const auto& tau_axis = result->axes.grids[1];
    const auto& vol_axis = result->axes.grids[2];
    const auto& rate_axis = result->axes.grids[3];
    const double K = result->K_ref;
    const double tau = tau_axis.back();
    const double r = rate_axis.front();

    // Tolerance in $ per K_ref=100 strike: post-fix max observed deviation
    // is 6.9e-09 (m_axis.back(), sigma=vol_axis.back()).  TOL is
    // deliberately loosened well above the plan's "~10x post-fix" guideline
    // (which would pin ~7e-08) to stay robust against cross-toolchain
    // numerical noise -- this compares two independently-run pipelines
    // (batch PDE solve + B-spline fit vs. a separate High-profile FDM
    // solve) and CI should not depend on bit-level agreement between them.
    // 1e-5 is still ~43,000x below the recorded 0.4263 pre-fix error, so it
    // keeps full discriminating power between domain coverage and ordinary
    // interpolation error.
    constexpr double TOL = 1e-5;

    for (double m : {m_axis.front(), m_axis.back()}) {
        for (double sigma : {vol_axis.front(), vol_axis.back()}) {
            const double S = K * std::exp(m);
            const double ref = fdm_reference_price(S, K, tau, sigma, r);
            const double got = wrapper->price(S, K, tau, sigma, r);
            EXPECT_NEAR(got, ref, TOL)
                << "m=" << m << " sigma=" << sigma;
        }
    }
}

// Regression (#437, fallback branch): an explicit grid that covers the
// fit axis but violates MAX_DX falls back to accuracy estimation, which
// also solved gridless.  As above, make_batch() fixes maturity to the fit
// tau axis's widened upper bound (measured 0.500001) for every entry, so
// the fallback's required_n_sigma is derived from
// max_sigma_sqrt_tau = 0.20*sqrt(0.500001) ~= 0.1414, not the chain's raw
// max maturity of 0.1 -- an order-of-magnitude difference from a naive
// reading of the formula.  The explicit bounds [-0.6, 0.6] are chosen so
// that required_n_sigma = (0.6/0.1414)*1.1 ~= 4.67 sits BELOW the
// n_sigma=5.0 floor: the fallback clamps to that floor, reproducing
// exactly the default-profile branch's undershoot for the min-sigma
// (0.10) group (half-width 5.0*0.10*sqrt(0.500001) ~= 0.3536 < the
// ~0.3796 fit-axis reach at m_axis.front()) while max-sigma (0.20,
// half-width ~0.7071) stays covered.  17 points over width 1.2 gives
// max_dx ~= 0.094 > 0.05, forcing the fallback branch; width 1.2 still
// exceeds min_required_width (6*0.1414 ~= 0.849) so only MAX_DX trips.
// Pre-fix max abs error on this branch: 0.0569 (m=-0.379555, sigma=0.10).
// Smaller than the GridAccuracyParams branch's 0.4263 despite the same
// floored n_sigma=5.0 -- the multi-sinh point placement here differs from
// the default-profile grid's, so the cubic-spline extrapolation just past
// the domain edge is milder -- but it is still a genuine, real
// out-of-domain extrapolation, not ordinary interpolation error.
TEST(AdaptiveGridBuilderTest, FallbackExplicitGridCoversMoneynessTails) {
    OptionGrid chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;
    chain.strikes = {60.0, 80.0, 100.0, 120.0, 140.0};
    chain.maturities = {0.05, 0.1};
    chain.implied_vols = {0.10, 0.15, 0.20};
    chain.rates = {0.03, 0.05};

    AdaptiveGridParams params;
    params.target_iv_error = 0.002;
    params.max_iter = 2;
    params.validation_samples = 8;

    // Covers the fit axis (upfront check passes) but 17 points over
    // width 1.2 makes max_dx > 0.05, forcing the fallback branch.
    auto grid_spec = GridSpec<double>::sinh_spaced(-0.6, 0.6, 17, 2.0).value();
    auto result = build_adaptive_bspline(params, chain,
        PDEGridConfig{grid_spec, 200, {}}, OptionType::PUT);
    ASSERT_TRUE(result.has_value());

    auto wrapper = make_bspline_surface(
        result->spline, result->K_ref, result->dividend_yield,
        OptionType::PUT);
    ASSERT_TRUE(wrapper.has_value());

    const auto& m_axis = result->axes.grids[0];
    const auto& vol_axis = result->axes.grids[2];
    const double K = result->K_ref;
    const double tau = result->axes.grids[1].back();
    const double r = result->axes.grids[3].front();

    // Tolerance in $ per K_ref=100 strike: post-fix max observed deviation
    // is 1.19e-05 (m_axis.back(), sigma=vol_axis.back()).  TOL is
    // deliberately loosened well above that -- this is the same
    // coarse-fallback pipeline (multi-sinh explicit grid + accuracy
    // re-estimation) already flagged as noisier across toolchains, so 1e-3
    // stays robust to that noise while still ~57x below the recorded 0.0569
    // pre-fix error, keeping full discriminating power between domain
    // coverage and ordinary interpolation error.
    constexpr double TOL = 1e-3;

    for (double m : {m_axis.front(), m_axis.back()}) {
        for (double sigma : {vol_axis.front(), vol_axis.back()}) {
            const double S = K * std::exp(m);
            const double ref = fdm_reference_price(S, K, tau, sigma, r);
            const double got = wrapper->price(S, K, tau, sigma, r);
            EXPECT_NEAR(got, ref, TOL)
                << "m=" << m << " sigma=" << sigma;
        }
    }
}

// Regression (#480, S1): the continuous Chebyshev build solved its
// (sigma, rate) batch gridless.  extract_chain_domain floors the tau axis
// to a 0.5y spread and build_adaptive_chebyshev adds CC headroom, so for
// this chain the PDE maturity is 1.01 * 0.6875 and the old batch-union
// half-width is 5 * sigma_hi * sqrt(0.694) ~= 5 * 0.225 * 0.833 ~= 0.94
// (the batch is normalized-ineligible: its first param is the sigma_lo =
// 0.01 node, whose margin is far below 0.35).  The moneyness nodes reach
// +-ln(2.5) * (1 + 6/32) ~= +-1.09, so both endpoint nodes were
// cubic-spline extrapolations -- and a Chebyshev interpolant is a global
// polynomial, so the garbage reaches the user's own strikes.
// Pre-fix max abs error on this branch's parent: 27.85, at the node
// m_lo = -1.088095 with sigma=0.15 -- got 94.16 for a put whose
// reference price is 66.31, i.e. above the K=100 intrinsic ceiling.  The
// same ~27.85 shows at every queried sigma (27.850638466 / 27.850638459 /
// 27.850638458), and that sigma-independence to 8 significant figures is
// the signature of extrapolating one slice past the PDE domain edge
// rather than of interpolation error.  Post-fix the node queries agree
// with FDM to <= 6.7e-09 and the user-strike queries to <= 0.0176; the
// two classes are therefore asserted at different tolerances below.
TEST(AdaptiveGridBuilderTest, ChebyshevNodesMatchFdmAtExtremeMoneyness) {
    OptionGrid chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;
    chain.strikes = {40.0, 60.0, 100.0, 160.0, 250.0};
    chain.maturities = {0.05, 0.1};
    chain.implied_vols = {0.10};
    chain.rates = {0.03, 0.05};

    AdaptiveGridParams params;
    params.target_iv_error = 0.002;  // relaxed: accuracy is asserted below
    params.max_iter = 2;
    params.validation_samples = 8;

    auto result = build_adaptive_chebyshev(params, chain, OptionType::PUT);
    ASSERT_TRUE(result.has_value())
        << "build failed: " << static_cast<int>(result.error().code);
    ASSERT_NE(result->surface, nullptr);
    const auto& surface = *result->surface;

    // Node span of the fit domain (CC-extended), read back from the
    // interpolant so the test cannot drift from the builder's headroom.
    const auto& dom = surface.inner().interpolant().domain();
    const double K = chain.spot;
    const double tau = dom.hi[1];   // a tau node: isolates m extraction
    const double r = dom.hi[3];     // a rate node
    const auto& sb = result->sample_bounds;

    // Tolerances in $ per K=100, split by query class because the two
    // classes measure different things.
    //
    // TOL_NODE guards the two on-node m endpoints -- the queries that
    // actually discriminate this defect.  Post-fix max deviation over that
    // class is 6.7e-09 (m_lo, sigma=0.15), so the plan's ">= 10x post-fix"
    // rule would pin ~7e-08; it is loosened to 1e-5 for exactly the reason
    // the #437 test above uses 1e-5 -- this compares two independently-run
    // pipelines (batch PDE solve + Chebyshev fit vs. a separate
    // High-profile FDM solve) and CI must not depend on bit-level agreement
    // between them across toolchains.  1e-5 is still ~2.8e6x below the
    // 27.85 pre-fix error, far inside the "<= 1/50 of pre-fix" bound, so
    // the class keeps its full discriminating power.
    constexpr double TOL_NODE = 1e-5;
    //
    // TOL_USER guards the two user strikes.  These sit off-node in m, so
    // the class carries ordinary Chebyshev interpolation error across the
    // intrinsic-value kink: post-fix max deviation is 0.0176, at strike 250
    // with the sigma_lo = 0.01 node, where the exact American put value is
    // its intrinsic 60 and a global polynomial cannot follow the kink.
    // TOL_USER = 0.2 is ~11x that per the ">= 10x post-fix" rule, and
    // ~139x below the 27.85 pre-fix error.  This class is a user-visible
    // sanity assertion, not the discriminator: pre-fix its worst deviation
    // was only 0.0139 (strike 250, sigma=0.01), so it would have passed at
    // this tolerance on the unfixed code.  The bug is caught by TOL_NODE.
    constexpr double TOL_USER = 0.2;

    struct Query { double m; const char* what; double tol; };
    const Query queries[] = {
        {dom.lo[0], "node m_lo", TOL_NODE},
        {dom.hi[0], "node m_hi", TOL_NODE},
        {std::log(100.0 / 250.0), "user strike 250", TOL_USER},
        {std::log(100.0 / 40.0), "user strike 40", TOL_USER},
    };
    // sigma at the node endpoints (on-axis) and the user-facing sample
    // bounds (interpolated in sigma).
    const double sigmas[] = {dom.lo[2], dom.hi[2], sb.sigma_min, sb.sigma_max};

    for (const auto& q : queries) {
        for (double sigma : sigmas) {
            const double S = K * std::exp(q.m);
            const double ref = fdm_reference_price(S, K, tau, sigma, r);
            const double got = surface.price(S, K, tau, sigma, r);
            EXPECT_NEAR(got, ref, q.tol)
                << q.what << " m=" << q.m << " sigma=" << sigma;
        }
    }
}

}  // namespace
}  // namespace mango

