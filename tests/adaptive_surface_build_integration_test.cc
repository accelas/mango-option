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
#include "mango/math/cubic_spline_solver.hpp"
#include "mango/option/grid_spec_types.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

namespace mango {
namespace {

// Regression #501: these placements originally failed validation in event
// gaps. Event-sided sampling now supports those neighborhoods themselves.
class SegmentedDividendPlacement
    : public testing::TestWithParam<std::pair<double, double>> {};

TEST_P(SegmentedDividendPlacement, BuildsAndPricesAcrossExactEventBoundaries) {
    const auto [first_days, maturity_days] = GetParam();
    SegmentedAdaptiveConfig config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {},
        .maturity = maturity_days / 365.0,
        .kref_config = {.K_refs = {90.0, 92.5, 95.0, 97.5, 100.0,
                                 102.5, 105.0, 107.5, 110.0}},
    };
    for (double day = first_days; day < maturity_days; day += 91.25) {
        config.discrete_dividends.push_back({day / 365.0, 0.50});
    }
    IVGrid domain{
        .moneyness = {std::log(0.92), std::log(0.95), 0.0,
                      std::log(1.05), std::log(1.08)},
        .vol = {0.10, 0.15, 0.20, 0.30},
        .rate = {0.02, 0.03, 0.05, 0.07},
    };
    // The one placement in this family the round-trip metric refuses -- the
    // first dividend at day 10 of a 60-day expiry -- is not instantiated
    // here; it is pinned on its own in
    // DayTenOfSixtyIsRefusedAsNearIntrinsic below.
    //
    // Budget (2026-09-21): two iterations and 16 samples, not the struct's
    // default eight and 64.  What this case asserts is structural -- the
    // surface covers every event boundary and prices finitely across it --
    // and that holds on the first candidate; the extra iterations only
    // sharpened an accuracy this case never reads.  At the default the seven
    // placements cost 1320 s of the target's 3106 s, because a reference
    // preparation is now six High-accuracy solves.
    auto result = build_adaptive_bspline_segmented(
        AdaptiveGridParams{.target_iv_error = 1e-3, .max_iter = 2,
                           .validation_samples = 16}, config, domain);
    ASSERT_TRUE(result.has_value())
        << "code " << static_cast<int>(result.error().code);
    EXPECT_TRUE(std::isfinite(result->achieved_max_error));
    // D4: accuracy no longer gates admissibility; Task 10 re-measures this.
    EXPECT_EQ(result->diagnostics.surface_failures, 0u);
    EXPECT_GT(result->diagnostics.holdout_points_measured, 0u);
    EXPECT_EQ(result->diagnostics.holdout_points_invalid, 0u);

    for (const auto& dividend : config.discrete_dividends) {
        const double event_tau = config.maturity - dividend.calendar_time;
        EXPECT_TRUE(result->surface.contains_maturity(event_tau));
        EXPECT_TRUE(std::isfinite(result->surface.price(
            100.0, 100.0, event_tau, 0.20, 0.03)));
        for (double tau : {event_tau - 0.001, event_tau + 0.001}) {
            EXPECT_TRUE(result->surface.contains_maturity(tau));
            EXPECT_TRUE(std::isfinite(result->surface.price(
                100.0, 100.0, tau, 0.20, 0.03)));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Issue501, SegmentedDividendPlacement,
    testing::Values(std::pair{10.0, 30.0}, std::pair{60.0, 180.0},
                    std::pair{75.0, 365.0}, std::pair{10.0, 14.0},
                    std::pair{20.0, 30.0}, std::pair{45.0, 60.0},
                    std::pair{75.0, 90.0}));

// Regression: the (day 10, 60-day) placement is refused, and the refusal is
// the measured outcome, not a gap misclassification.
// Bug: this schedule puts a quarter of the holdout on deep-ITM samples with
// essentially no time value, where the shipped inversion cannot recover a
// volatility however accurate the price is.  Measured 2026-09-21 over 8
// candidates: SurfaceAmbiguous 25, SurfaceNoRoot 0 -- every failure is the
// 17-point screen reporting MultipleRoots on a price that is flat in sigma,
// and one such point is enough to make a candidate non-viable under D4.
// Representative coordinate: K = 106.6849, tau = 0.04912, sigma0 = 0.17117,
// reference 6.6855678 against an intrinsic of 6.6849043 -- TV/K = 6.2e-6 --
// which the surface reproduces to 2.2e-6 of strike with a vega of 0.428.
// The surface is accurate; the metric cannot measure it here, and the
// acceptance band is not what is missing.  The event-boundary pricing this
// suite exists for is asserted on the seven placements above.
TEST(SegmentedDividendPlacement, DayTenOfSixtyIsRefusedAsNearIntrinsic) {
    SegmentedAdaptiveConfig config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {},
        .maturity = 60.0 / 365.0,
        .kref_config = {.K_refs = {90.0, 92.5, 95.0, 97.5, 100.0,
                                 102.5, 105.0, 107.5, 110.0}},
    };
    for (double day = 10.0; day < 60.0; day += 91.25) {
        config.discrete_dividends.push_back({day / 365.0, 0.50});
    }
    IVGrid domain{
        .moneyness = {std::log(0.92), std::log(0.95), 0.0,
                      std::log(1.05), std::log(1.08)},
        .vol = {0.10, 0.15, 0.20, 0.30},
        .rate = {0.02, 0.03, 0.05, 0.07},
    };
    // Budget: the sample count is the struct's default 64 and stays there.
    // The refusal is about one near-intrinsic coordinate, so it is a
    // property of the draw: at 16 samples this configuration builds.  Only
    // the iteration count is cut, from eight to two, and the refusal was
    // re-measured at that budget on 2026-09-21: still NoViableSurface.  The
    // candidate counts quoted above were measured at eight iterations.
    auto result = build_adaptive_bspline_segmented(
        AdaptiveGridParams{.target_iv_error = 1e-3, .max_iter = 2},
        config, domain);
    ASSERT_FALSE(result.has_value())
        << "near-intrinsic samples must not be certified as measured";
    EXPECT_EQ(result.error().code, PriceTableErrorCode::NoViableSurface);
}

TEST(SegmentedShortMaturity, RecoversOneDayAtmVolatility) {
    SegmentedAdaptiveConfig config{
        .spot = 100.0, .option_type = OptionType::PUT, .dividend_yield = 0.02,
        .discrete_dividends = {{10.0 / 365, 0.50}}, .maturity = 30.0 / 365,
        .kref_config = {.K_refs = {90, 92.5, 95, 97.5, 100, 102.5, 105, 107.5, 110}},
    };
    IVGrid domain{
        .moneyness = {std::log(0.92), std::log(0.95), 0, std::log(1.05), std::log(1.08)},
        .vol = {0.10, 0.15, 0.20, 0.30}, .rate = {0.02, 0.03, 0.05, 0.07},
    };
    // Budget: the struct's defaults, deliberately.  This case reads an
    // accuracy -- the one-day ATM volatility below, to 1e-3 -- and at two
    // iterations and 16 samples it recovers 0.2237 against 0.225, so the
    // budget is what buys the assertion and cannot be cut.
    auto built = build_adaptive_bspline_segmented(
        AdaptiveGridParams{.target_iv_error = 1e-3}, config, domain);
    ASSERT_TRUE(built.has_value());
    PricingParams query(OptionSpec{.spot = 100, .strike = 100,
        .maturity = 1.0 / 365, .rate = 0.04, .dividend_yield = 0.02,
        .option_type = OptionType::PUT}, 0.225);
    auto reference_solver = AmericanOptionSolver::create(query,
        PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::Ultra)});
    ASSERT_TRUE(reference_solver.has_value());
    auto reference = reference_solver->solve();
    ASSERT_TRUE(reference.has_value());
    BSplineMultiKRefSurface table(built->surface, built->sample_bounds,
        OptionType::PUT, 0.02);
    auto solver = InterpolatedIVSolver<BSplineMultiKRefSurface>::create(
        table, {}, config.discrete_dividends);
    ASSERT_TRUE(solver.has_value());
    auto iv = solver->solve(IVQuery(query, reference->value()));
    ASSERT_TRUE(iv.has_value());
    EXPECT_NEAR(iv->implied_vol, 0.225, 1e-3)
        << "price " << table.price(100, 100, query.maturity, 0.225, 0.04)
        << " reference " << reference->value();
}

/// Convert S/K moneyness to log-moneyness for internal builder APIs.
std::vector<double> to_log_m(std::initializer_list<double> sk) {
    std::vector<double> v;
    v.reserve(sk.size());
    for (double m : sk) v.push_back(std::log(m));
    return v;
}

// Budget note (2026-09-21), for the three knobs raised below.  At the
// original budget -- three vol seeds, a 51-point / 200-step PDE grid, two
// iterations, eight samples -- this chain's fit left 72-133 bps of error at
// the top of its sigma domain ([0.15, 0.25]).  That put the round trip's
// root 21-145 bps *outside* the fit range, where a B-spline has no support
// at all, so the shipped inversion reported SurfaceNoRoot and no candidate
// was viable.  Resolution is what was missing, not tolerance; the target
// below is unchanged.
TEST(AdaptiveGridBuilderTest, BuildsWithSyntheticChain) {
    // Create a minimal synthetic chain
    OptionGrid chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;

    // Add strikes and maturities
    chain.strikes = {90.0, 95.0, 100.0, 105.0, 110.0};
    chain.maturities = {0.25, 0.5, 1.0};
    chain.implied_vols = {0.16, 0.18, 0.20, 0.22, 0.24};  // five seeds, not three
    chain.rates = {0.04, 0.05, 0.06};

    AdaptiveGridParams params;
    params.target_iv_error = 0.002;  // 20 bps - relaxed for test speed
    params.max_iter = 5;             // was 2
    params.validation_samples = 16;  // was 8

    // The explicit grid is what this test supplies, so it is what gets
    // refined: 201 points and 400 steps, was 51 and 200.
    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0).value();
    auto result = build_adaptive_bspline(params, chain,
        PDEGridConfig{grid_spec, 400, {}}, OptionType::PUT);

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
    // max_iter 4, not 1: a single iteration leaves ~117 bps of sigma-axis fit
    // error at the top of [0.15, 0.25], which puts the round trip's root 89
    // bps outside the domain.  A B-spline has no support beyond its fit
    // range, so the acceptance band cannot reach it and the inversion
    // reports SurfaceNoRoot.  Refining is the fix; the tolerance is not.
    params.max_iter = 4;
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
    chain1.implied_vols = {0.17, 0.19, 0.21, 0.23};  // see BuildsWithSyntheticChain
    chain1.rates = {0.04, 0.05};

    OptionGrid chain2 = chain1;
    // A different rate set, not a different spot: the cache keys on the whole
    // chain, so any difference exercises it.  Moving the spot instead (90
    // against strikes of 90-110) makes every holdout point a deep-ITM put
    // whose reference stencil barely separates and whose surface price falls
    // below intrinsic, so the build refuses under D2/D4 -- a real contract,
    // but not the one this test is about.  Measured on 2026-09-21 at spot 90:
    // K = 109.9658, tau = 0.85237, sigma0 = 0.21304, reference 20.11304
    // against an intrinsic of 19.96583, surface 19.94375 -- below intrinsic --
    // with a surface vega of 0.816.
    chain2.rates = {0.045, 0.055};

    AdaptiveGridParams params;
    // max_iter 4, not 1: see RegressionSingleValueAxes -- one iteration
    // leaves ~79 bps of sigma-axis error at the domain edge, which the round
    // trip cannot invert from inside the fit range.
    params.max_iter = 4;
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

// Regression: a sparse reference-strike pair preserves moneyness while
// interpolating the normalized cash amounts.
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
    ASSERT_TRUE(result.has_value());
    // D4: accuracy no longer gates admissibility; Task 10 re-measures this.
    EXPECT_EQ(result->diagnostics.surface_failures, 0u);
    EXPECT_EQ(result->diagnostics.holdout_points_invalid, 0u);
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
// With no cash dividends, preserving moneyness makes this configuration
// homogeneous in strike even outside the tightly clustered reference span.
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
    ASSERT_TRUE(result.has_value());
    // D4: accuracy no longer gates admissibility; Task 10 re-measures this.
    EXPECT_EQ(result->diagnostics.surface_failures, 0u);
    EXPECT_EQ(result->diagnostics.holdout_points_invalid, 0u);
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

// A single automatic reference can pass validation after spot remapping.
// Its normalized cash amount remains an approximation away from that strike.
// Regression: one auto reference strike cannot serve S/K in [0.7, 1.3].
// Bug: the surface preserves moneyness and blends normalized prices in
// inverse strike, so a single K_ref (K_ref_count = 1, span 0.3 => K_ref =
// 100) has nothing to blend against at the ends of a +-30 % moneyness
// domain.  Measured on 2026-09-21 at K = 135.4196 (S/K = 0.738), tau =
// 0.5939, sigma0 = 0.2769: the reference is 36.9412 (intrinsic 35.4196,
// time value 1.52) while the surface's *entire* attainable range over
// sigma in [0.1, 0.3] is [37.5051, 37.7440] -- a 37 % time-value error,
// $0.56 per $135 of strike, with no root anywhere in the domain, i.e.
// ~1266 bps below sigma_min.  The other five measured points come in at
// 39.8 bps against a 50 bps target, so it is this one coordinate, not the
// fit budget, that is wrong.  CLAUDE.md's K_ref/moneyness coherence rule
// says the K_refs must span and resolve the strike range the moneyness
// grid implies; here they do not, and the metric refuses.
TEST(AdaptiveGridBuilderTest, BuildSegmentedSingleAutoKRefRefusesWideMoneyness) {
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
        << "a single K_ref must not certify a +-30 % moneyness domain";
    EXPECT_EQ(result.error().code, PriceTableErrorCode::NoViableSurface);
}

// Short-maturity construction retains a positive measurement domain while
// fitting from the exact expiry payoff and resolving dividend regimes.
TEST(AdaptiveGridBuilderTest, BuildSegmentedVeryShortMaturity) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 1;
    // 24, not 8: at a 0.05 y maturity most sampled options are worth 1e-7 or
    // less, and a +-50 bps sigma bump moves the price by less than the
    // reference's own two-grid uncertainty, so the point cannot resolve (D2).
    // Only 3 of 8 resolved against a floor of 4.
    params.validation_samples = 24;

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

    // The IV measurement domain remains positive and bounded by maturity.
    auto bounds = expand_segmented_domain(
        {m, v, r}, seg_config.maturity, seg_config.dividend_yield,
        seg_config.discrete_dividends, 90.0);
    ASSERT_TRUE(bounds.has_value());
    EXPECT_LE(bounds->tau_max, seg_config.maturity);
    EXPECT_GT(bounds->tau_min, 0.0);

    // Regression: at a 0.05 y maturity the adaptive path refuses, and the
    // domain assertions above are what this test is really for.
    // Bug: the sampled options carry almost no time value, so the shipped
    // inversion cannot recover a volatility from the reference price on that
    // surface however accurate the surface's own price is.  Measured on 2026-09-21 at K =
    // 106.2027, tau = 0.01846, sigma0 = 0.2093: reference 6.2032083 against
    // an intrinsic of 6.2027205 -- a time value of 4.9e-4, i.e. TV/K =
    // 4.6e-6 -- with a surface price residual of 1.1e-5 of strike and a
    // surface vega of 0.222.  The 17-point screen reports MultipleRoots on a
    // price that is flat in sigma, which is a surface failure under D4 and
    // leaves no candidate viable.  Raising the sample count fixed the
    // separate D2 coverage refusal (24 samples now clear the floor); it
    // cannot make a near-intrinsic price invertible.
    auto result = build_adaptive_bspline_segmented(params, seg_config, {m, v, r});
    ASSERT_FALSE(result.has_value())
        << "a near-intrinsic 18-day sample cannot be round-tripped";
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

    // Re-measured 2026-09-21 under the round-trip metric's acceptance band
    // (spec D3, rev 5): this configuration builds again.  Under the earlier
    // exact-bracket rule the segmented Chebyshev surface was refused on a
    // single edge-adjacent sample, which said nothing about the segment
    // classification this test exists for.  The Chebyshev segmented fit
    // domain carries sigma headroom beyond the sampled range, so the
    // tolerance band is not clipped away and the edge roots are measured.
    auto adaptive = build_adaptive_chebyshev_segmented(
        params, seg_config, {m_domain, v_domain, r_domain});
    ASSERT_TRUE(adaptive.has_value())
        << "the segments must build and be measured; a gap misclassification "
           "would surface as a build error instead: code "
        << static_cast<int>(adaptive.error().code);
    // What the metric now certifies, pinned as measured rather than assumed:
    // no candidate point defeated the shipped inversion, and the holdout was
    // large enough to say so.  The achieved accuracy itself is not pinned --
    // a 7-day option has near-zero vega, so any price error divides into a
    // large IV error, and that is a property of the maturity, not of segment
    // classification.
    EXPECT_EQ(adaptive->diagnostics.surface_failures, 0u);
    EXPECT_GT(adaptive->diagnostics.holdout_points_measured, 0u);
    // Every prepared holdout point ends in exactly one of four outcomes --
    // measured, unresolved, surface failure, or a non-finite evaluation
    // ("skipped") -- but `skipped` is never reported on its own: it is
    // folded into `holdout_points_invalid` together with the preparations
    // that failed and so never entered the prepared set at all
    // (chebyshev_adaptive.cpp: `invalid + final_score.skipped`).  The three
    // separable outcomes can therefore only under-count the prepared set,
    // and adding the folded counter can only over-count it; an exact
    // identity is not expressible from the public fields.
    // Bug: the previous `measured + unresolved + unsupported + invalid ==
    // holdout_points` was not an identity.  `holdout_points` is already the
    // prepared set, so it excludes the unsupported and invalid samples the
    // assertion subtracted a second time, and on the segmented Chebyshev
    // path it mixed two draws: `holdout_points_unsupported` comes from the
    // sizing loop's fixed holdout while `holdout_points` is the final
    // validation set.
    EXPECT_LE(adaptive->diagnostics.holdout_points_measured
                  + adaptive->diagnostics.holdout_points_unresolved
                  + adaptive->diagnostics.surface_failures,
              adaptive->diagnostics.holdout_points);
    EXPECT_GE(adaptive->diagnostics.holdout_points_measured
                  + adaptive->diagnostics.holdout_points_unresolved
                  + adaptive->diagnostics.surface_failures
                  + adaptive->diagnostics.holdout_points_invalid,
              adaptive->diagnostics.holdout_points);

    auto surface = build_chebyshev_segmented_manual(
        seg_config, {m_domain, v_domain, r_domain});
    // Narrow real segments should build successfully, not be rejected as gaps
    ASSERT_TRUE(surface.has_value())
        << "Narrow real segments should produce valid prices, not errors";

    // The exact event is unsupported; the real narrow segment still prices.
    EXPECT_FALSE(surface->contains_maturity(0.01));
    EXPECT_FALSE(std::isfinite(surface->price(100.0, 100.0, 0.01, 0.20, 0.05)));
    double p = surface->price(100.0, 100.0, 0.012, 0.20, 0.05);
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
    // 1e-4 (1 bp), not 1e-6: the target must be unreachable by the *fit*
    // (so the retry path runs) while staying resolvable by the *reference*.
    // At 1e-6 a sigma bump moves the price by 7e-6..2.7e-5 against a
    // High-profile two-grid estimate of 5e-5..5e-4 -- 10x to 138x the
    // signal -- so no point resolved and the build refused at validation
    // (ValidationFailed) before scoring any surface.  The subject here is
    // loop mechanics, not the smallest representable tolerance.
    params.target_iv_error = 1e-4;  // unreachable by the fit; resolvable by the oracle
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
    // D4: accuracy no longer gates admissibility; Task 10 re-measures this.
    EXPECT_EQ(result->diagnostics.surface_failures, 0u);

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
    const ReferenceOracle oracle{
        .dividend_yield = seg_config.dividend_yield,
        .option_type = seg_config.option_type,
        .discrete_dividends = seg_config.discrete_dividends,
        .reference_maturity = seg_config.maturity,
        .accuracy = make_grid_accuracy(kReferenceAccuracy),
    };
    auto refs_fn = make_stencil_refs_fn(
        params, oracle, std::make_shared<ReferenceSolveCounter>());
    auto points = detail::prepare_final_validation(params, ctx, refs_fn,
                                                   params.lhs_seed + 999);
    ASSERT_TRUE(points.has_value());

    const SurfaceHandle returned{
        .price = [&](double spot, double strike, double tau, double sigma,
                     double rate) {
            return result->surface.price(spot, strike, tau, sigma, rate);
        },
        .vega = [&](double spot, double strike, double tau, double sigma,
                    double rate) {
            return result->surface.vega(spot, strike, tau, sigma, rate);
        }};
    auto measured = detail::score_final_surface(
        points->points, returned,
        make_round_trip_score_fn(params, ctx, seg_config.option_type), ctx);

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
    const size_t r_bump = result->used_retry ? 1 : 0;

    EXPECT_EQ(result->grid.moneyness.size(),
              std::min(seeded.moneyness.size() + m_bump,
                       params.max_points_per_dim))
        << "reported moneyness grid does not describe the returned surface"
        << " (used_retry = " << result->used_retry << ")";
    const auto& first_leaf = result->surface.pieces().front().pieces().front();
    EXPECT_EQ(result->grid.vol, first_leaf.interpolant().get().grid(2));
    EXPECT_EQ(result->grid.rate.size(),
              std::min(seeded.rate.size() + r_bump, params.max_points_per_dim));
    size_t max_segment_size = 0;
    for (const auto& segment : result->surface.pieces().front().pieces()) {
        max_segment_size = std::max(max_segment_size, segment.interpolant().get().grid(1).size());
    }
    EXPECT_EQ(result->tau_points_per_segment, max_segment_size);
    EXPECT_FALSE(result->tau_grid.empty());
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
    params.max_iter = 4;  // see AutomaticGridCoversMoneynessTailsWithFixedBudget
    // 16, not 8: see AutomaticGridCoversMoneynessTailsWithFixedBudget -- the
    // same 60-140 strike range at tau <= 0.1 leaves most samples unresolvable.
    params.validation_samples = 16;

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

// Regression (#437/#487): adaptive missing slices must retain the entire
// requested moneyness range. This fixture previously supplied 17 points but
// depended on silent replacement by 101; request its automatic grid explicitly.
TEST(AdaptiveGridBuilderTest, AutomaticGridCoversMoneynessTailsWithFixedBudget) {
    OptionGrid chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;
    chain.strikes = {60.0, 80.0, 100.0, 120.0, 140.0};
    chain.maturities = {0.05, 0.1};
    chain.implied_vols = {0.10, 0.15, 0.20};
    chain.rates = {0.03, 0.05};

    AdaptiveGridParams params;
    params.target_iv_error = 0.002;
    // max_iter 4, not 2: the seeded sigma axis left ~10 bps of fit error at
    // the top of [0.1, 0.2], enough to push the round trip's root just
    // outside the fit range where the B-spline has no support.
    params.max_iter = 4;
    // 16, not 8: strikes 60-140 at tau <= 0.1 put much of the sample domain
    // where the reference is exactly 0 or exactly intrinsic, so its stencil
    // cannot separate and the point does not resolve (D2).  At 8 samples the
    // floor is max(4, 2) = 4, i.e. 50 % must resolve; at 16 it is 25 %.
    params.validation_samples = 16;

    GridAccuracyParams accuracy;
    accuracy.min_spatial_points = accuracy.max_spatial_points = 101;
    accuracy.max_time_steps = 200;
    accuracy.alpha = 2.0;
    auto result = build_adaptive_bspline(params, chain, accuracy, OptionType::PUT);
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

    // Preserve the existing quote-unit tolerance: the pre-coverage bug
    // produced .0569 error, compared with this .001 acceptance limit.
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
// with FDM to <= 6.65e-09 and the user-strike queries to <= 0.01759; the
// two classes are therefore asserted at different tolerances below.
// (Re-measured after the boundary-clearance change of spec D11, which
// widens this chain's covering half-width from ~1.20 to ~1.65 and clamps
// Nx at the Ultra 5,000-point cap: both classes moved by less than one
// part in a thousand of their own size.)
TEST(AdaptiveGridBuilderTest, ChebyshevNodesMatchFdmAtExtremeMoneyness) {
    OptionGrid chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;
    chain.strikes = {40.0, 60.0, 100.0, 160.0, 250.0};
    chain.maturities = {0.05, 0.1};
    chain.implied_vols = {0.10};
    chain.rates = {0.03, 0.05};

    AdaptiveGridParams params;
    params.target_iv_error = 0.002;
    params.max_iter = 2;
    params.validation_samples = 16;

    // Regression: this chain cannot be measured, and the refusal is the
    // outcome, pinned rather than worked around.
    // Bug: at strikes 40-250 with tau <= 0.1 the reference price is exactly 0
    // (deep OTM) or exactly K - S (deep ITM) at almost every sampled point.
    // Implied volatility is undefined there, so the six-solve stencil cannot
    // separate and the point does not resolve (D2).  Measured 2026-09-21: 16
    // prepared, 2 resolved, against a floor of max(4, 16/4) = 4.  More
    // samples do not help -- the domain is degenerate, not undersampled: at
    // 32 samples the count is 3 resolved against a floor of 8, the same
    // fraction and the same refusal, so 16 is kept as the cheaper measurement.
    // Widening the vol axis to {0.10, 0.60} only trades the D2 refusal for a
    // D4 one: the samples then resolve, but the Chebyshev polynomial cannot
    // represent a deep-OTM price of 1.7e-4 (its minimum over the acceptance
    // band is 16x that), so the inversion reports MultipleRoots.
    //
    // FOLLOW-UP(#500-remainder): re-home #480 S1's node-level FDM agreement
    // (pre-fix 27.85 at m_lo = -1.088095, sigma-independent to eight
    // significant figures, against post-fix 6.65e-09 on the node class and
    // 0.01759 on the user-strike class); it needs a manual builder for the
    // continuous Chebyshev path, which does not exist today.
    auto result = build_adaptive_chebyshev(params, chain, OptionType::PUT);
    ASSERT_FALSE(result.has_value())
        << "a domain of exactly-intrinsic references must not be certified";
    EXPECT_EQ(result.error().code, PriceTableErrorCode::ValidationFailed);
}

// Direct pricing oracle for the contract at the query valuation point.
// The anchor-maturity regression below has no schedule roll; shorter-maturity
// tests must supply their independently rolled fixed-expiry dividends.

// Regression (#480, S2): the segmented Chebyshev build solved its dividend
// batch gridless.  A batch with discrete dividends is normalized-ineligible,
// so it solved on the batch-union grid: half-width 5 * sigma_hi * sqrt(T)
// with T = 1.01 * 0.25 and sigma_hi = 0.15 + 0.075 (CC headroom on the
// [0.05, 0.15] sample range) ~= 0.57, left edge nudged by the dividend
// extension.  The moneyness nodes span the user range [ln 0.49, ln 2]
// (dividend-widened) plus 3/32 of headroom ~= [-0.85, 0.83], so every
// endpoint node was extrapolated; the queried S = 50 and S = 200 are
// themselves outside the old domain.
// Pre-fix max abs error on this branch's parent (coverage oracle):
// 6.31e+10 at S=50, sigma=0.05 -- the surface returned
// 63,086,864,583.46 for a put whose reference value is 50.51.  The other
// two failing cases show the same signature at different scales: S=50 /
// sigma=0.15 returned exactly 0 against a reference of 50.5089 (error
// 50.51), and S=200 / sigma=0.05 returned 2,991,753.61 against a
// reference of 7.5e-163 (error 2.99e+06).  Only S=200 / sigma=0.15
// happened to land close enough to pass.  Runaway magnitudes of that
// kind, and an exact 0 where the put is 50 in the money, are the
// signature of a cubic spline evaluated far outside its data range, not
// of interpolation error.

}  // namespace
}  // namespace mango

namespace mango {
namespace {

// ===========================================================================
// Regression tests for issue #461: segmented probe aggregation
// ===========================================================================

// Regression: the segmented adaptive builder rebuilt its final grids as
// `linspace` at the probes' maximum sizes, so every knot position the probe
// loops (and the user's seed) had chosen was discarded.
// Bug: `aggregate_max_sizes` carried `.size()` per axis, not the vectors.
//
// With `max_iter = 1` the probes return their seeds, and the seeds are the
// user's knots.  The non-uniform vol seed {0.10, 0.15, 0.20, 0.30} cannot
// survive `linspace(0.10, 0.30, 4)`, so its presence in the returned grid is
// the signature of position-preserving aggregation.
TEST(SegmentedKnotRetention, ReturnedGridKeepsSeedKnotPositions) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 1;
    params.validation_samples = 8;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {},
        .maturity = 1.0,
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    };

    auto m_domain = to_log_m({0.85, 0.9, 1.0, 1.1, 1.2});
    std::vector<double> v_domain = {0.10, 0.15, 0.20, 0.30};
    std::vector<double> r_domain = {0.02, 0.03, 0.05, 0.07};

    auto result = build_adaptive_bspline_segmented(
        params, seg_config, {m_domain, v_domain, r_domain});
    ASSERT_TRUE(result.has_value())
        << "code " << static_cast<int>(result.error().code);

    const auto contains = [](const std::vector<double>& grid, double x) {
        return std::ranges::any_of(grid, [x](double g) {
            return std::abs(g - x) < 1e-12; });
    };
    for (double v : v_domain) {
        EXPECT_TRUE(contains(result->grid.vol, v))
            << "vol seed knot " << v << " was lost by aggregation";
    }
    for (double r : r_domain) {
        EXPECT_TRUE(contains(result->grid.rate, r))
            << "rate seed knot " << r << " was lost by aggregation";
    }
    for (double m : m_domain) {
        EXPECT_TRUE(contains(result->grid.moneyness, m))
            << "moneyness seed knot " << m << " was lost by aggregation";
    }
}

// ===========================================================================
// Segmented Chebyshev sizing loop: probe contract and maturity support
// ===========================================================================

// One off-ATM reference-strike pair and one cash dividend, shared by the two
// tests below.  No K_ref sits at the spot, so a sizing handle scored on the
// user's contract would carry the dividend-scaling residual described in
// spec D1/L6.
SegmentedAdaptiveConfig probe_contract_config() {
    return SegmentedAdaptiveConfig{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 1.50}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {95.0, 105.0}},
    };
}

IVGrid probe_contract_domain() {
    return IVGrid{
        .moneyness = {std::log(100.0 / 108.0), std::log(100.0 / 92.0)},
        .vol = {0.15, 0.35},
        .rate = {0.02, 0.06},
    };
}

// The sizing loop must get through an off-ATM reference-strike pair with a
// cash dividend: no K_ref sits at the spot here, so every sample is scored on
// a scaled probe.  What the scaling does to the references is pinned directly
// by `ProbeScaledRefs.*` in reference_oracle_test.cc; this covers the build
// end to end, including that the stencil actually ran.
TEST(SegmentedChebyshevAdaptive, SizingLoopBuildsWithOffAtmKrefAndDividend) {
    AdaptiveGridParams params{.target_iv_error = 1e-3, .max_iter = 2,
                              .validation_samples = 16};
    const auto cfg = probe_contract_config();
    const auto domain = probe_contract_domain();

    auto result = build_adaptive_chebyshev_segmented(params, cfg, domain);
    ASSERT_TRUE(result.has_value())
        << "code " << static_cast<int>(result.error().code);
    EXPECT_EQ(result->diagnostics.surface_failures, 0u);
    EXPECT_GT(result->diagnostics.holdout_points_measured, 0u);
    // The stencil runs six solves per preparation, so a wired-up builder
    // reports reference solves on both grid levels.
    EXPECT_GT(result->diagnostics.reference_solves_fine, 0u);
    EXPECT_GT(result->diagnostics.reference_solves_coarse, 0u);
}

// Spec D4: event-gap maturities are excluded before preparation, not scored
// as defects.  The dividend at calendar time 0.5 of a 1y contract puts the
// event at tau = 0.5, and `compute_segment_boundaries` opens a +-5e-4 gap
// there; the sizing loop's holdout draw must skip whatever lands inside it.
//
// The seed is pinned, not arbitrary: the holdout draw is
// `latin_hypercube_4d(16, lhs_seed ^ 0x484F4C44)` scaled to the sample
// domain's tau range [0.01, 1.0], and seed 27 is the first seed whose draw
// puts exactly one sample inside the gap.  The observed count is that one
// sample.
TEST(SegmentedChebyshevAdaptive, EventGapSamplesAreUnsupportedNotFailures) {
    AdaptiveGridParams params{.target_iv_error = 1e-3, .max_iter = 2,
                              .validation_samples = 16, .lhs_seed = 27};
    const auto cfg = probe_contract_config();
    const auto domain = probe_contract_domain();

    auto result = build_adaptive_chebyshev_segmented(params, cfg, domain);
    ASSERT_TRUE(result.has_value())
        << "code " << static_cast<int>(result.error().code);
    EXPECT_EQ(result->diagnostics.surface_failures, 0u);
    // An excluded maturity is neither a reference nor a defect.
    EXPECT_EQ(result->diagnostics.holdout_points_unsupported, 1u);
    EXPECT_GT(result->diagnostics.holdout_points_measured, 0u);
}

}  // namespace
}  // namespace mango
