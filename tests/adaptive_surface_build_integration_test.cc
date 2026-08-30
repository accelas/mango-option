// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
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


TEST(AdaptiveGridBuilderTest, BuildSegmentedBasic) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;  // 50 bps — relaxed for test speed
    params.max_iter = 2;
    params.validation_samples = 16;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {90.0, 95.0, 100.0, 105.0, 110.0}},
    };

    auto m_domain = to_log_m({0.92, 0.95, 1.0, 1.05, 1.08});
    std::vector<double> v_domain = {0.10, 0.15, 0.20, 0.30};
    std::vector<double> r_domain = {0.02, 0.03, 0.05, 0.07};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m_domain, v_domain, r_domain});
    ASSERT_TRUE(result.has_value())
        << "build_adaptive_bspline_segmented failed: code "
        << static_cast<int>(result.error().code);

    // On a K_ref, where the multi-K_ref bracket resolves to a single entry
    double price = result->surface.price(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
    EXPECT_TRUE(std::isfinite(price));

    // And off every K_ref: 97.5 sits midway between 95 and 100, so the query
    // exercises the two-entry blend rather than resolving to one surface.
    double price2 = result->surface.price(100.0, 97.5, 0.5, 0.20, 0.05);
    EXPECT_GT(price2, 0.0);
    EXPECT_TRUE(std::isfinite(price2));
    EXPECT_LT(price2, price) << "a lower-struck put must be worth less";
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

// ===========================================================================
// Coverage gap tests — Priority 1 (Critical)
// ===========================================================================


// ===========================================================================
// Coverage gap tests — Priority 2 (High)
// ===========================================================================

// Coverage: ATM K_ref coincides with lowest K_ref — dedup prevents 3rd probe
TEST(AdaptiveGridBuilderTest, BuildSegmentedATMEqualsLowest) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 1;
    // 16, not 8: a schedule entry at or beyond the queried tau makes
    // `solve_american_option` refuse, so every sample below the first
    // dividend date loses its reference -- roughly half the tau range
    // here.  Eight samples would leave the validation set sitting
    // exactly on the `max(4, n/4)` floor.
    params.validation_samples = 16;
    params.min_moneyness_points = 10;  // Use smaller grid for test speed

    // spot=100, K_refs sorted: {100, 110, 120, 130}
    // Lowest=100, highest=130, ATM=100 (closest to spot)
    // ATM == lowest → only 2 probes (100, 130)
    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 1.50}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {100.0, 110.0, 120.0, 130.0}},
    };

    // Strikes must stay inside the K_ref span: S/K in [0.77, 1.0] maps to
    // K in [100, 130].  Outside it the multi-K_ref blend clamps to the
    // nearest K_ref and the assembled surface is not viable.
    auto m = to_log_m({0.77, 0.85, 0.9, 0.95, 1.0});
    std::vector<double> v = {0.10, 0.15, 0.20, 0.30};
    std::vector<double> r = {0.02, 0.03, 0.05, 0.07};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m, v, r});
    ASSERT_TRUE(result.has_value());
    double price = result->surface.price(100.0, 110.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
}

// Coverage: ATM K_ref coincides with highest K_ref
TEST(AdaptiveGridBuilderTest, BuildSegmentedATMEqualsHighest) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 1;
    // 16, not 8: a schedule entry at or beyond the queried tau makes
    // `solve_american_option` refuse, so every sample below the first
    // dividend date loses its reference -- roughly half the tau range
    // here.  Eight samples would leave the validation set sitting
    // exactly on the `max(4, n/4)` floor.
    params.validation_samples = 16;
    params.min_moneyness_points = 10;  // Use smaller grid for test speed

    // spot=100, K_refs sorted: {70, 80, 90, 100}
    // Lowest=70, highest=100, ATM=100 (closest to spot)
    // ATM == highest → only 2 probes (70, 100)
    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 1.50}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {70.0, 80.0, 90.0, 100.0}},
    };

    // Strikes inside the K_ref span: S/K in [1.0, 1.42] maps to K in [70, 100].
    auto m = to_log_m({1.0, 1.1, 1.2, 1.3, 1.42});
    std::vector<double> v = {0.10, 0.15, 0.20, 0.30};
    std::vector<double> r = {0.02, 0.03, 0.05, 0.07};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m, v, r});
    ASSERT_TRUE(result.has_value());
    double price = result->surface.price(100.0, 90.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
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


// Regression: Standard path deep OTM IV accuracy requires domain headroom
// Bug: AdaptiveGridBuilder::build() used expand_bounds(min, max, 0.10) which
// is a no-op when the domain is already >0.10 wide.  Queries near the
// log-moneyness boundary (e.g. K=80 with S=100, x=0.223 vs domain max=0.262)
// hit clamped B-spline endpoint effects, producing 1000+ bps IV errors.
// Fix: add 3*dx spline-support headroom to domain bounds after expand_bounds.
TEST(AdaptiveGridBuilderTest, RegressionDeepOTMPutIVAccuracy) {
    // Build a vanilla adaptive surface covering K=80..120
    OptionGrid chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.02;
    chain.strikes = {76.9, 83.3, 90.9, 100.0, 111.1, 125.0, 142.9};
    chain.maturities = {0.01, 0.06, 0.20, 0.60, 1.0, 2.0, 2.5};
    chain.implied_vols = {0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50};
    chain.rates = {0.01, 0.03, 0.05, 0.10};

    AdaptiveGridParams params;
    params.target_iv_error = 2e-5;  // 2 bps
    // Spec D3: headroom is now 3 * w / (min_moneyness_points - 1) instead of
    // 3 * w / (n_strikes - 1), so this chain's support band shrinks from
    // +/-0.31 to +/-0.03 log-moneyness.  A single build on the seeded grid is
    // exactly what this regression is about -- whether that band is wide
    // enough for a K=80 query to clear B-spline endpoint effects.  The
    // refinement loop beyond it is a separate (pre-existing) pathology:
    // focused refinement piles knots into one bin until the collocation fit
    // fails.  Spec D5 retention keeps the viable seed candidate through such
    // a failure, so the build runs at the default budget.

    GridAccuracyParams accuracy;
    accuracy.min_spatial_points = 200;
    accuracy.max_spatial_points = 200;

    auto result = build_adaptive_bspline(params, chain, accuracy, OptionType::PUT);
    ASSERT_TRUE(result.has_value()) << "Adaptive build failed";

    // Wrap spline for price queries
    auto wrapper = make_bspline_surface(result->spline, result->K_ref, result->dividend_yield, OptionType::PUT);
    ASSERT_TRUE(wrapper.has_value()) << wrapper.error();

    // Query at K=80, T=1y, σ=15% — this was 1574 bps error before the fix
    double spot = 100.0, strike = 80.0, tau = 1.0, sigma = 0.15, rate = 0.05;
    double price = wrapper->price(spot, strike, tau, sigma, rate);
    EXPECT_TRUE(std::isfinite(price));
    EXPECT_GT(price, 0.0);

    // Verify the recovered price allows reasonable IV recovery.
    // Reference: FDM solve at the same parameters.
    PricingParams ref_params;
    ref_params.spot = spot;
    ref_params.strike = strike;
    ref_params.maturity = tau;
    ref_params.rate = rate;
    ref_params.dividend_yield = 0.02;
    ref_params.option_type = OptionType::PUT;
    ref_params.volatility = sigma;

    auto ref = solve_american_option(ref_params);
    ASSERT_TRUE(ref.has_value());
    double ref_price = ref->value();

    // Price error should be small enough that IV round-trip works.
    // Before fix: |price - ref| was ~$1.3 on a ~$0.30 option.
    // After fix: should be within $0.05 (< 50 bps IV error).
    double price_error = std::abs(price - ref_price);
    EXPECT_LT(price_error, 0.10)
        << "Surface price " << price << " vs FDM " << ref_price
        << " (error $" << price_error << ")";
}

// ===========================================================================
// Regression tests for segmented Chebyshev dividend edge cases
// ===========================================================================

// Regression: gap queries must route to nearest real segment by distance
// Bug: Always routed to seg_idx+1 (right), so queries in left half of gap
// mapped to post-dividend segment instead of pre-dividend segment.
TEST(AdaptiveGridBuilderTest, SegmentedChebyshevGapRoutesNearest) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.01;  // 100 bps — relaxed for test speed
    params.max_iter = 1;
    // 16, not 8: a schedule entry at or beyond the queried tau makes
    // `solve_american_option` refuse, so every sample below the first
    // dividend date loses its reference -- roughly half the tau range
    // here.  Eight samples would leave the validation set sitting
    // exactly on the `max(4, n/4)` floor.
    params.validation_samples = 16;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        // The assembled surface blends K_ref-struck prices linearly in
        // strike, so the K_refs must span (and resolve) the queryable strike
        // range [90.9, 111.1]; a lone K_ref = 100 measures 0.59 on the final
        // validation and is refused by the viability gate (spec D9).
        .kref_config = {.K_refs = {91.0, 100.0, 111.0}},
    };

    auto m_domain = to_log_m({0.9, 0.95, 1.0, 1.05, 1.1});
    std::vector<double> v_domain = {0.10, 0.20, 0.30};
    std::vector<double> r_domain = {0.03, 0.05};

    auto result = build_adaptive_chebyshev_segmented(
        params, seg_config, {m_domain, v_domain, r_domain});
    ASSERT_TRUE(result.has_value())
        << "build_adaptive_chebyshev_segmented failed";

    // Dividend at cal_time=0.5 → tau_split=0.5.
    // Gap is [0.5-ε, 0.5+ε] with ε=5e-4.
    //
    // With nearest-side routing:
    //   tau=0.4999 (left of gap mid) → clamps to RIGHT edge of left segment
    //   tau=0.5001 (right of gap mid) → clamps to LEFT edge of right segment
    //   These are different segment edges with different values.
    //
    // If routing were always-right (the old bug):
    //   Both would clamp to LEFT edge of right segment → identical prices.
    double tau_left  = 0.4999;   // left of gap mid
    double tau_right = 0.5001;   // right of gap mid

    auto pf = [&](double tau) {
        return result->surface.price(100.0, 100.0, tau, 0.20, 0.05);
    };

    double p_left  = pf(tau_left);
    double p_right = pf(tau_right);

    EXPECT_TRUE(std::isfinite(p_left));
    EXPECT_TRUE(std::isfinite(p_right));
    EXPECT_GT(p_left, 0.0);
    EXPECT_GT(p_right, 0.0);

    // If nearest-side routing works, these route to different segments
    // and thus produce different prices. If both route to the same
    // segment (the old bug), they clamp to the same local_tau=0 and
    // produce identical prices.
    EXPECT_NE(p_left, p_right)
        << "Gap queries on both sides of midpoint gave identical prices ("
        << p_left << ") — both likely routed to same segment";

    // Additionally verify the prices differ by a meaningful amount
    // (not just floating-point noise), since there's a $2 dividend
    // discontinuity between segments.
    double diff = std::abs(p_left - p_right);
    EXPECT_GT(diff, 0.001)
        << "Gap queries differ by only " << diff
        << " — routing may not be splitting correctly";
}

// Regression: duplicate dividend dates must be merged to avoid non-monotonic
// segment boundaries
// Bug: compute_segment_boundaries pushed split-ε/split+ε for every dividend
// without merging same-date entries, causing overlapping gaps.
TEST(AdaptiveGridBuilderTest, SegmentedChebyshevDuplicateDividends) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.01;
    params.max_iter = 1;
    // 16, not 8: a schedule entry at or beyond the queried tau makes
    // `solve_american_option` refuse, so every sample below the first
    // dividend date loses its reference -- roughly half the tau range
    // here.  Eight samples would leave the validation set sitting
    // exactly on the `max(4, n/4)` floor.
    params.validation_samples = 16;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        // Two dividends at the exact same date
        .discrete_dividends = {
            Dividend{.calendar_time = 0.5, .amount = 1.0},
            Dividend{.calendar_time = 0.5, .amount = 1.5},
        },
        .maturity = 1.0,
        // The assembled surface blends K_ref-struck prices linearly in
        // strike, so the K_refs must span (and resolve) the queryable strike
        // range [90.9, 111.1]; a lone K_ref = 100 measures 0.59 on the final
        // validation and is refused by the viability gate (spec D9).
        .kref_config = {.K_refs = {91.0, 100.0, 111.0}},
    };

    auto m_domain = to_log_m({0.9, 0.95, 1.0, 1.05, 1.1});
    std::vector<double> v_domain = {0.10, 0.20, 0.30};
    std::vector<double> r_domain = {0.03, 0.05};

    auto result = build_adaptive_chebyshev_segmented(
        params, seg_config, {m_domain, v_domain, r_domain});
    ASSERT_TRUE(result.has_value())
        << "build_adaptive_chebyshev_segmented failed with duplicate dividends";

    // Should be able to query across the entire tau range
    for (double tau : {0.1, 0.3, 0.5, 0.7, 0.9}) {
        double p = result->surface.price(100.0, 100.0, tau, 0.20, 0.05);
        EXPECT_TRUE(std::isfinite(p))
            << "Price not finite at tau=" << tau;
        EXPECT_GT(p, 0.0) << "Price not positive at tau=" << tau;
    }
}

// Regression: nearly-coincident dividend dates must not create overlapping gaps
// Bug: Two dividends 1 day apart produce gaps that overlap, making boundaries
// non-monotonic.
TEST(AdaptiveGridBuilderTest, SegmentedChebyshevNearlyCoincidentDividends) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.01;
    params.max_iter = 1;
    // 16, not 8: a schedule entry at or beyond the queried tau makes
    // `solve_american_option` refuse, so every sample below the first
    // dividend date loses its reference -- roughly half the tau range
    // here.  Eight samples would leave the validation set sitting
    // exactly on the `max(4, n/4)` floor.
    params.validation_samples = 16;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        // Two dividends ~1 day apart
        .discrete_dividends = {
            Dividend{.calendar_time = 0.500, .amount = 1.0},
            Dividend{.calendar_time = 0.503, .amount = 1.0},  // ~1 day later
        },
        .maturity = 1.0,
        // The assembled surface blends K_ref-struck prices linearly in
        // strike, so the K_refs must span (and resolve) the queryable strike
        // range [90.9, 111.1]; a lone K_ref = 100 measures 0.59 on the final
        // validation and is refused by the viability gate (spec D9).
        .kref_config = {.K_refs = {91.0, 100.0, 111.0}},
    };

    auto m_domain = to_log_m({0.9, 0.95, 1.0, 1.05, 1.1});
    std::vector<double> v_domain = {0.10, 0.20, 0.30};
    std::vector<double> r_domain = {0.03, 0.05};

    auto result = build_adaptive_chebyshev_segmented(
        params, seg_config, {m_domain, v_domain, r_domain});
    ASSERT_TRUE(result.has_value())
        << "build_adaptive_chebyshev_segmented failed with nearly-coincident dividends";

    double p = result->surface.price(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_TRUE(std::isfinite(p));
    EXPECT_GT(p, 0.0);
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

// Regression: narrow real segment between two close dividends must not
// produce zero prices.
// Bug: Width-based gap detection treated narrow real segments as gaps,
// giving them zero tensors. Queries inside the narrow real interval
// got stuck on the zero leaf because both neighbors were also gaps.
TEST(AdaptiveGridBuilderTest, SegmentedChebyshevNarrowRealSegment) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.01;
    params.max_iter = 1;
    // 16, not 8: a schedule entry at or beyond the queried tau makes
    // `solve_american_option` refuse, so every sample below the first
    // dividend date loses its reference -- roughly half the tau range
    // here.  Eight samples would leave the validation set sitting
    // exactly on the `max(4, n/4)` floor.
    params.validation_samples = 16;

    // Two dividends 5 days apart. With ε=5e-4 gap half-width:
    //   div1 at cal_time=0.48 → tau_split=0.52, gap [0.5195, 0.5205]
    //   div2 at cal_time=0.50 → tau_split=0.50, gap [0.4995, 0.5005]
    // Real segment between gaps: [0.5005, 0.5195] — width 0.019 > kMinSegmentWidth
    // But with closer dividends (2 days apart):
    //   div1 at cal_time=0.494 → tau_split=0.506, gap [0.5055, 0.5065]
    //   div2 at cal_time=0.500 → tau_split=0.500, gap [0.4995, 0.5005]
    // Real segment between gaps: [0.5005, 0.5055] — width 0.005 < kMinSegmentWidth
    // This narrow real segment would be misclassified as a gap.
    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {
            Dividend{.calendar_time = 0.494, .amount = 1.0},
            Dividend{.calendar_time = 0.500, .amount = 1.0},
        },
        .maturity = 1.0,
        // The assembled surface blends K_ref-struck prices linearly in
        // strike, so the K_refs must span (and resolve) the queryable strike
        // range [90.9, 111.1]; a lone K_ref = 100 measures 0.59 on the final
        // validation and is refused by the viability gate (spec D9).
        .kref_config = {.K_refs = {91.0, 100.0, 111.0}},
    };

    auto m_domain = to_log_m({0.9, 0.95, 1.0, 1.05, 1.1});
    std::vector<double> v_domain = {0.10, 0.20, 0.30};
    std::vector<double> r_domain = {0.03, 0.05};

    auto result = build_adaptive_chebyshev_segmented(
        params, seg_config, {m_domain, v_domain, r_domain});
    ASSERT_TRUE(result.has_value())
        << "build_adaptive_chebyshev_segmented failed";

    // Query inside the narrow real segment between the two gaps.
    // tau=0.503 is between the two gap bands.
    double p = result->surface.price(100.0, 100.0, 0.503, 0.20, 0.05);
    EXPECT_TRUE(std::isfinite(p)) << "Price not finite in narrow real segment";
    EXPECT_GT(p, 0.5)
        << "Price " << p << " is near-zero in narrow real segment — "
        << "likely hitting a zero-tensor leaf";

    // Also verify prices at tau values in the wide segments on either
    // side are reasonable for comparison.
    double p_before = result->surface.price(100.0, 100.0, 0.40, 0.20, 0.05);
    double p_after  = result->surface.price(100.0, 100.0, 0.60, 0.20, 0.05);
    EXPECT_GT(p_before, 0.5);
    EXPECT_GT(p_after, 0.5);

    // The narrow segment price should be in the same order of magnitude
    // as the wide segment prices (within 5x).
    EXPECT_GT(p, p_before * 0.2)
        << "Narrow segment price " << p << " is far too low vs "
        << "left-side price " << p_before;
}

// ===========================================================================
// Tests for make_tau_split_from_segments
// ===========================================================================


// ===========================================================================
// Equivalence tests: typed vs type-erased Chebyshev segmented paths
// ===========================================================================

// SplitSurface composition gives same result as manual leaf evaluation
TEST(ChebyshevSegmentedEquivalence, CompositionMatchesManualLeafEval) {
    // Build pieces for a single K_ref with fixed CGL nodes
    std::vector<Dividend> divs = {Dividend{.calendar_time = 0.5, .amount = 2.0}};
    auto [seg_bounds, seg_is_gap] = compute_segment_boundaries(divs, 1.0, 0.01, 1.0);

    // Use cc_level_nodes for reproducible grids
    auto m_nodes = cc_level_nodes(4, -0.4, 0.4);
    std::vector<double> tau_nodes;
    for (size_t s = 0; s + 1 < seg_bounds.size(); ++s) {
        if (seg_is_gap[s]) continue;
        for (double t : cc_level_nodes(3, seg_bounds[s], seg_bounds[s + 1]))
            tau_nodes.push_back(t);
    }
    std::sort(tau_nodes.begin(), tau_nodes.end());
    tau_nodes.erase(std::unique(tau_nodes.begin(), tau_nodes.end(),
        [](double a, double b) { return std::abs(a - b) < 1e-10; }),
        tau_nodes.end());
    auto sigma_nodes = cc_level_nodes(2, 0.08, 0.35);
    auto rate_nodes = cc_level_nodes(1, 0.02, 0.06);

    double K_ref = 100.0;
    auto pieces = build_chebyshev_segmented_pieces(
        K_ref, OptionType::PUT, 0.02, divs,
        seg_bounds, seg_is_gap,
        m_nodes, tau_nodes, sigma_nodes, rate_nodes);
    ASSERT_TRUE(pieces.has_value()) << "build_chebyshev_segmented_pieces failed";

    // Compose into ChebyshevTauSegmented
    ChebyshevTauSegmented composite(
        std::move(pieces->leaves), std::move(pieces->tau_split));

    // Re-build fresh pieces for manual leaf evaluation
    auto pieces2 = build_chebyshev_segmented_pieces(
        K_ref, OptionType::PUT, 0.02, divs,
        seg_bounds, seg_is_gap,
        m_nodes, tau_nodes, sigma_nodes, rate_nodes);
    ASSERT_TRUE(pieces2.has_value());

    // Query at several points and verify composite matches
    // The composite (SplitSurface<Leaf, TauSegmentSplit>) should produce the
    // same result as: find segment, compute local tau, call leaf.price(), scale.
    std::vector<double> test_taus = {0.1, 0.3, 0.7, 0.9};

    for (double tau : test_taus) {
        double spot = 100.0;
        double sigma = 0.20;
        double rate = 0.04;

        double p_composite = composite.price(spot, K_ref, tau, sigma, rate);

        EXPECT_TRUE(std::isfinite(p_composite))
            << "Composite price not finite at tau=" << tau;
        EXPECT_GT(p_composite, 0.0)
            << "Composite price not positive at tau=" << tau;

        // Also verify vega is finite and positive (ATM put)
        double v_composite = composite.vega(spot, K_ref, tau, sigma, rate);
        EXPECT_TRUE(std::isfinite(v_composite))
            << "Composite vega not finite at tau=" << tau;
    }
}

TEST(ChebyshevSegmentedEquivalence, VegaReasonable) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 2;
    params.validation_samples = 8;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    };

    auto m_domain = to_log_m({0.8, 0.9, 1.0, 1.1, 1.2});
    IVGrid grid{m_domain, {0.10, 0.20, 0.30}, {0.03, 0.05}};

    auto result = build_adaptive_chebyshev_segmented(
        params, seg_config, grid);
    ASSERT_TRUE(result.has_value());

    // ATM put: vega should be positive and finite
    double vega = result->surface.vega(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_TRUE(std::isfinite(vega));
    EXPECT_GT(vega, 0.0);

    // Compare analytical vega vs FD vega (central diff)
    double eps = 1e-4;
    double p_up = result->surface.price(100.0, 100.0, 0.5, 0.20 + eps, 0.05);
    double p_dn = result->surface.price(100.0, 100.0, 0.5, 0.20 - eps, 0.05);
    double fd_vega = (p_up - p_dn) / (2.0 * eps);

    // Analytical should agree with FD within 1%
    double rel_diff = std::abs(vega - fd_vega) / std::max(std::abs(vega), 1e-6);
    EXPECT_LT(rel_diff, 0.01)
        << "Analytical vega=" << vega << " vs FD vega=" << fd_vega;
}

// ===========================================================================
// Tests for resolve_k_refs
// ===========================================================================


// ===========================================================================
// Tests for build_chebyshev_segmented_manual (non-adaptive path)
// ===========================================================================

TEST(ChebyshevSegmentedManual, BasicPricing) {
    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    };

    auto m_domain = to_log_m({0.8, 0.9, 1.0, 1.1, 1.2});
    IVGrid grid{m_domain, {0.10, 0.20, 0.30}, {0.03, 0.05}};

    auto result = build_chebyshev_segmented_manual(seg_config, grid);
    ASSERT_TRUE(result.has_value()) << "Manual build failed";

    // ATM put: price should be positive and finite
    double p = result->price(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_TRUE(std::isfinite(p));
    EXPECT_GT(p, 0.0);

    // Vega should be positive
    double v = result->vega(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_TRUE(std::isfinite(v));
    EXPECT_GT(v, 0.0);
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
// Tests for expand_segmented_domain
// ===========================================================================


// ===========================================================================
// Chebyshev refiner contract (spec D6): exact axis, level cap, state rollback
// ===========================================================================

namespace {


}  // namespace


// Regression: the continuous Chebyshev path must return the surface that was
// built from the grids the loop actually picked.
// Bug risk: the caller used to rebuild unconditionally after run_refinement;
// it now consumes the loop's captured surface, so a drift between the picked
// candidate and the `last_surface` side channel would ship silently.  The node
// counts baked into the interpolant are the observable that pins it -- the
// axis *bounds* cannot, since the CC extension freezes them at seed time and
// every refinement level spans the same interval.
TEST(AdaptiveGridBuilderTest, ContinuousChebyshevSurfaceMatchesPickedGrids) {
    OptionGrid chain{
        .ticker = "TEST",
        .spot = 100.0,
        .strikes = {90.0, 100.0, 110.0},
        .maturities = {0.25, 1.0},
        .implied_vols = {0.20, 0.30},
        .rates = {0.03, 0.05},
        .dividend_yield = 0.0,
    };
    AdaptiveGridParams params{
        .target_iv_error = 3e-4,    // below the seed grid's error: forces one
                                    // refinement, so the hooks and the
                                    // pick-vs-last-build path are exercised
        .max_iter = 2,              // one refinement step exercises the hooks
        .validation_samples = 8,
    };

    auto result = build_adaptive_chebyshev(params, chain, OptionType::PUT);
    ASSERT_TRUE(result.has_value()) << "build_adaptive_chebyshev failed";
    ASSERT_NE(result->surface, nullptr);
    ASSERT_FALSE(result->iterations.empty());

    // The last recorded build is always a successful one (a failed trial is
    // followed by the loop's final rebuild), and it is the build whose grids
    // the loop returned.
    // The seed grid cannot meet this target, so a refinement trial always
    // runs; without one the invariant under test would be trivial.
    ASSERT_GE(result->iterations.size(), 2u) << "no refinement was attempted";
    bool refined_an_axis = false;
    for (const auto& it : result->iterations) {
        if (it.refined_dim >= 0) refined_an_axis = true;
    }
    EXPECT_TRUE(refined_an_axis);

    const auto& last = result->iterations.back();
    ASSERT_FALSE(last.build_failed);

    const auto& interp = result->surface->inner().interpolant();
    EXPECT_EQ(interp.num_pts(), last.grid_sizes)
        << "returned surface was built from grids other than the picked ones";

    // Published bounds are the measurement domain (spec D2/AC2), *not* the
    // node span: the CC extension is interpolation support the validation
    // never sampled, so it must not be advertised as queryable.
    const auto& sb = result->sample_bounds;
    EXPECT_DOUBLE_EQ(result->surface->m_min(), sb.m_min);
    EXPECT_DOUBLE_EQ(result->surface->m_max(), sb.m_max);
    EXPECT_DOUBLE_EQ(result->surface->tau_min(), sb.tau_min);
    EXPECT_DOUBLE_EQ(result->surface->tau_max(), sb.tau_max);
    EXPECT_DOUBLE_EQ(result->surface->sigma_min(), sb.sigma_min);
    EXPECT_DOUBLE_EQ(result->surface->sigma_max(), sb.sigma_max);
    EXPECT_DOUBLE_EQ(result->surface->rate_min(), sb.rate_min);
    EXPECT_DOUBLE_EQ(result->surface->rate_max(), sb.rate_max);

    // And the sample domain is strictly inside the node span it was fit on.
    const auto& dom = interp.domain();
    EXPECT_GT(sb.m_min, dom.lo[0]);
    EXPECT_LT(sb.m_max, dom.hi[0]);
    EXPECT_GT(sb.sigma_min, dom.lo[2]);
    EXPECT_LT(sb.sigma_max, dom.hi[2]);

    // Every CC level is nested (2^l + 1 nodes), so a refined axis stays so.
    for (size_t d = 0; d < 4; ++d) {
        size_t n = interp.num_pts()[d];
        EXPECT_GE(n, 3u) << "axis " << d;
        EXPECT_EQ((n - 1) & (n - 2), 0u)
            << "axis " << d << " has " << n << " nodes, not 2^l + 1";
    }

    // And it prices.
    double px = result->surface->price(100.0, 100.0, 0.5, 0.25, 0.04);
    EXPECT_TRUE(std::isfinite(px));
    EXPECT_GT(px, 0.0);
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

// The segmented Chebyshev path gained a mandatory final gate: the assembled
// all-K_ref surface is measured, and its numbers -- not the single-K_ref
// sizing loop's -- are what the result reports.
TEST(SegmentedFinalContract, ChebyshevReportsAssembledSurfaceNumbers) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.01;
    params.max_iter = 1;
    // 16, not 8: a schedule entry at or beyond the queried tau makes
    // `solve_american_option` refuse, so every sample below the first
    // dividend date loses its reference -- roughly half the tau range
    // here.  Eight samples would leave the validation set sitting
    // exactly on the `max(4, n/4)` floor.
    params.validation_samples = 16;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        // The assembled surface blends K_ref-struck prices linearly in
        // strike, so the K_refs must span (and resolve) the queryable strike
        // range [90.9, 111.1]; a lone K_ref = 100 measures 0.59 on the final
        // validation and is refused by the viability gate (spec D9).
        .kref_config = {.K_refs = {91.0, 100.0, 111.0}},
    };

    auto m_domain = to_log_m({0.9, 0.95, 1.0, 1.05, 1.1});
    std::vector<double> v_domain = {0.10, 0.20, 0.30};
    std::vector<double> r_domain = {0.03, 0.05};
    IVGrid domain{m_domain, v_domain, r_domain};

    auto result = build_adaptive_chebyshev_segmented(params, seg_config, domain);
    ASSERT_TRUE(result.has_value())
        << "code " << static_cast<int>(result.error().code);

    EXPECT_GT(result->diagnostics.holdout_points, 0u)
        << "the assembled surface must be measured, not assumed";
    EXPECT_DOUBLE_EQ(result->diagnostics.achieved_max_error,
                     result->achieved_max_error);
    EXPECT_EQ(result->diagnostics.target_met, result->target_met);
    // The gate refuses anything above the viability bound, so a returned
    // surface is always within it.
    EXPECT_LE(result->achieved_max_error, kViabilityBound);
    EXPECT_TRUE(std::isfinite(result->achieved_max_error));

    // Re-score the surface we were handed on an independently reproduced
    // reference set: the reported numbers must be its own, not the
    // single-K_ref sizing loop's.
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
    auto refs_fn = make_fd_vega_refs_fn(
        params, make_validate_fn(seg_config.dividend_yield,
                                 seg_config.option_type,
                                 seg_config.discrete_dividends));
    auto points = detail::prepare_final_validation(params, ctx, refs_fn,
                                                   params.lhs_seed + 999);
    ASSERT_TRUE(points.has_value());

    const SurfaceHandle returned{
        .price = [&](double spot, double strike, double tau, double sigma,
                     double rate) {
            return result->surface.price(spot, strike, tau, sigma, rate);
        }};
    auto measured = detail::score_final_surface(
        points->points, returned,
        make_iv_score_fn(params, seg_config.option_type), ctx);

    EXPECT_EQ(measured.measured,
              result->diagnostics.holdout_points_measured);
    EXPECT_NEAR(measured.max_error, result->achieved_max_error, 1e-12);
    EXPECT_NEAR(measured.avg_error, result->achieved_avg_error, 1e-12);
}

// ===========================================================================
// Regression tests for the q0 bifurcation (issue #434)
// ===========================================================================

// Regression: adaptive refinement returned its catastrophically-degraded
// final iteration (issue #434); retention must return the best candidate
// and IV inversion must never return a spurious low root.
// Bug: the pre-fix loop returned the last built iteration unconditionally,
// measured error over an oversized headroom band, and had no query-time
// screen. Under the exact EEP projection (max(0, x)) this bifurcated a q=0
// PUT B-spline surface's sigma=30% region so badly that the diagnostic
// `interp_iv_safety --path=q0` regressed from 7.3-8.7 bps to 289.3 bps RMS,
// and interpolated IV inversion near K/S=0.8, T=30d could converge to a
// spurious low root instead of the true 30% vol. Fixed-holdout retention
// (D5), user-domain measurement (D2/D3), and the query-time multi-root
// screen (D8) together bound both failure modes.
TEST(AdaptiveRegressionTest, Q0BifurcationRetainedAndScreened) {
    IVSolverFactoryConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .dividend_yield = 0.0,
        .grid = IVGrid{
            // Upper bound widened to 1.30 (vs. the brief's 1.2 sketch) so
            // the wrong-root probe below (S/K = 100/80 = 1.25) falls inside
            // the surface's published bounds instead of being rejected.
            .moneyness = {0.8, 0.9, 1.0, 1.15, 1.3},
            .vol = {0.10, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.05, 0.08},
        },
        .adaptive = AdaptiveGridParams{
            .target_iv_error = 2e-5,
            .max_iter = 4,
            .min_moneyness_points = 10,  // keep build under the test budget
            .validation_samples = 16,
        },
        .backend = BSplineBackend{
            .maturity_grid = {0.05, 0.1, 0.3, 0.6, 1.0},
        },
    };

    auto solver_result = make_interpolated_iv_solver(config);
    ASSERT_TRUE(solver_result.has_value());
    auto solver = std::move(*solver_result);

    auto diag = solver.build_diagnostics();
    ASSERT_TRUE(diag.has_value());
    EXPECT_LE(diag->achieved_max_error, 0.01);  // 100 bps sanity bound

    // Wrong-root region probe: sigma=0.30, T=30d, K/S=0.8 put -- the corner
    // of the surface where the pre-fix loop's degraded candidate produced a
    // spurious low IV root.
    PricingParams params;
    params.spot = 100.0;
    params.strike = 80.0;
    params.maturity = 30.0 / 365.0;
    params.rate = 0.05;
    params.dividend_yield = 0.0;
    params.volatility = 0.30;
    params.option_type = OptionType::PUT;

    auto ref = solve_american_option(params);
    ASSERT_TRUE(ref.has_value());
    double market_price = ref->value_at(params.spot);

    IVQuery query(
        OptionSpec{.spot = 100.0,
                   .strike = 80.0,
                   .maturity = 30.0 / 365.0,
                   .rate = 0.05,
                   .dividend_yield = 0.0,
                   .option_type = OptionType::PUT},
        market_price);

    // `MultipleRoots` would also be a defended outcome here (the D8 screen
    // refusing to guess), but the retained candidate from this build
    // recovers the true root cleanly (implied_vol == 0.30034), so assert
    // that outright rather than accepting the weaker disjunction.
    auto iv_result = solver.solve(query);
    ASSERT_TRUE(iv_result.has_value());
    // Never a spurious low root: the pre-fix bug returned IVs well below
    // 0.15 in this region.
    EXPECT_GE(iv_result->implied_vol, 0.15);
    EXPECT_NEAR(iv_result->implied_vol, 0.30, 2e-2);
}

}  // namespace
}  // namespace mango
