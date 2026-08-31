// SPDX-License-Identifier: MIT
//
// Heavy end-to-end adaptive surface builds: cases that each spend tens of
// seconds of PDE solves to construct one scenario (segmented Chebyshev edge
// handling, numeric equivalence, accuracy regressions).  They stress
// computation results rather than exposing cheap software invariants, so
// they run in the nightly slow suite (tag `slow`), not per-PR CI.  The fast
// wiring and regression invariants stay in
// adaptive_surface_build_integration_test.cc.
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

