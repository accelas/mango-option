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

TEST(ReferenceStrikeValidation, BothBuildersRejectInvalidExplicitSetsBeforeSolving) {
    const IVGrid domain{to_log_m({0.7, 0.9, 1.0, 1.3}),
        {0.1, 0.2, 0.3, 0.4}, {0.01, 0.03, 0.05, 0.07}};
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (const auto& refs : {std::vector{70.0, nan, 150.0}, std::vector{70.0, inf, 150.0},
             std::vector{0.0, 100.0, 150.0}, std::vector{-10.0, 100.0, 150.0},
             std::vector{70.0, 100.0, 100.0, 150.0}, std::vector{90.0, 100.0, 110.0}}) {
        SegmentedAdaptiveConfig config{.spot = 100.0, .option_type = OptionType::PUT,
            .dividend_yield = 0.0, .discrete_dividends = {}, .maturity = 1.0,
            .kref_config = {.K_refs = refs}};
        auto bspline = BSplineSegmentedBuilder::create(config, domain);
        auto chebyshev = ChebyshevSegmentedBuilder::create(config, domain);
        EXPECT_FALSE(bspline.has_value());
        EXPECT_FALSE(chebyshev.has_value());
    }
}

// Helper to create a dummy AmericanOptionResult for cache testing
std::shared_ptr<AmericanOptionResult> make_dummy_result() {
    PricingParams params;
    params.spot = 100.0;
    params.strike = 100.0;
    params.maturity = 1.0;
    params.volatility = 0.20;
    params.rate = 0.05;
    params.dividend_yield = 0.0;
    params.option_type = OptionType::PUT;

    auto result = solve_american_option(params);
    if (result.has_value()) {
        return std::make_shared<AmericanOptionResult>(std::move(result.value()));
    }
    return nullptr;
}


TEST(AdaptiveGridBuilderTest, EmptyChainReturnsError) {
    AdaptiveGridParams params;

    OptionGrid chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;
    // No options added

    auto grid_spec = GridSpec<double>::uniform(-3.0, 3.0, 51).value();
    auto result = build_adaptive_bspline(params, chain,
        PDEGridConfig{grid_spec, 100, {}}, OptionType::PUT);

    // Should return error for empty chain
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}

// ===========================================================================
// BSplinePDECache unit tests
// ===========================================================================

TEST(BSplinePDECacheTest, AddAndRetrieve) {
    BSplinePDECache cache;

    // Create a result using the auto solver
    auto result_ptr = make_dummy_result();
    ASSERT_NE(result_ptr, nullptr);
    cache.add(0.20, 0.05, result_ptr);

    // Retrieve
    auto retrieved = cache.get(0.20, 0.05);
    EXPECT_NE(retrieved, nullptr);

    // Miss on different key
    auto missed = cache.get(0.25, 0.05);
    EXPECT_EQ(missed, nullptr);
}

TEST(BSplinePDECacheTest, ContainsCheck) {
    BSplinePDECache cache;

    EXPECT_FALSE(cache.contains(0.20, 0.05));

    auto result_ptr = make_dummy_result();
    ASSERT_NE(result_ptr, nullptr);
    cache.add(0.20, 0.05, result_ptr);

    EXPECT_TRUE(cache.contains(0.20, 0.05));
    EXPECT_FALSE(cache.contains(0.25, 0.05));
}

TEST(BSplinePDECacheTest, GetMissingIndices) {
    BSplinePDECache cache;

    // Add some pairs
    auto dummy = make_dummy_result();
    ASSERT_NE(dummy, nullptr);
    cache.add(0.20, 0.05, dummy);
    cache.add(0.25, 0.05, dummy);

    std::vector<std::pair<double, double>> all_pairs = {
        {0.20, 0.05},  // cached
        {0.25, 0.05},  // cached
        {0.30, 0.05},  // missing
        {0.20, 0.06},  // missing (different rate)
    };

    auto missing = cache.get_missing_indices(all_pairs);

    EXPECT_EQ(missing.size(), 2);
    EXPECT_EQ(missing[0], 2);  // Index of (0.30, 0.05)
    EXPECT_EQ(missing[1], 3);  // Index of (0.20, 0.06)
}

TEST(BSplinePDECacheTest, InvalidateOnTauChange) {
    BSplinePDECache cache;

    auto dummy = make_dummy_result();
    ASSERT_NE(dummy, nullptr);
    cache.add(0.20, 0.05, dummy);
    cache.add(0.25, 0.05, dummy);

    EXPECT_EQ(cache.size(), 2);

    // Set initial tau grid
    std::vector<double> tau1 = {0.25, 0.5, 1.0};
    cache.set_tau_grid(tau1);

    // Same tau grid - should NOT invalidate
    cache.invalidate_if_tau_changed(tau1);
    EXPECT_EQ(cache.size(), 2);

    // Different tau grid - should invalidate
    std::vector<double> tau2 = {0.25, 0.5, 0.75, 1.0};
    cache.invalidate_if_tau_changed(tau2);
    EXPECT_EQ(cache.size(), 0);
}

TEST(BSplinePDECacheTest, CachePreservedOnMChange) {
    BSplinePDECache cache;

    auto dummy = make_dummy_result();
    ASSERT_NE(dummy, nullptr);
    cache.add(0.20, 0.05, dummy);
    cache.add(0.25, 0.05, dummy);

    // Set tau grid
    std::vector<double> tau = {0.25, 0.5, 1.0};
    cache.set_tau_grid(tau);

    EXPECT_EQ(cache.size(), 2);

    // Moneyness grid changes don't affect cache directly
    // (m changes are handled by extract_tensor interpolation)
    // Cache should still contain the (σ,r) pairs
    EXPECT_TRUE(cache.contains(0.20, 0.05));
    EXPECT_TRUE(cache.contains(0.25, 0.05));
}

TEST(BSplinePDECacheTest, Clear) {
    BSplinePDECache cache;

    auto dummy = make_dummy_result();
    ASSERT_NE(dummy, nullptr);
    cache.add(0.20, 0.05, dummy);

    EXPECT_EQ(cache.size(), 1);

    cache.clear();

    EXPECT_EQ(cache.size(), 0);
    EXPECT_FALSE(cache.contains(0.20, 0.05));
}

// ===========================================================================
// ErrorBins unit tests
// ===========================================================================

TEST(ErrorBinsTest, RecordAndWorstDimension) {
    ErrorBins bins;

    // Record errors concentrated ONLY in dimension 0 (moneyness)
    // For dim 0: all in bin 0 (concentration 1.0)
    // For dims 1-3: scattered across different bins
    std::array<double, 4> pos1 = {{0.05, 0.1, 0.3, 0.7}};
    std::array<double, 4> pos2 = {{0.08, 0.5, 0.6, 0.2}};
    std::array<double, 4> pos3 = {{0.03, 0.9, 0.8, 0.4}};

    double threshold = 0.001;
    bins.record_error(pos1, 0.005, threshold);
    bins.record_error(pos2, 0.004, threshold);
    bins.record_error(pos3, 0.003, threshold);

    // Dimension 0 has all errors in bin 0 (concentration 1.0)
    // Other dimensions have errors scattered (concentration ~0.33-0.67)
    // So dimension 0 should have highest score
    size_t worst = bins.worst_dimension();
    EXPECT_EQ(worst, 0);
}

TEST(ErrorBinsTest, ProblematicBins) {
    ErrorBins bins;

    // Record multiple errors in bin 0 of dimension 1 (tau)
    double threshold = 0.001;
    std::array<double, 4> pos1 = {{0.5, 0.05, 0.5, 0.5}};  // Low tau (bin 0)
    std::array<double, 4> pos2 = {{0.5, 0.08, 0.5, 0.5}};  // Low tau (bin 0)
    std::array<double, 4> pos3 = {{0.5, 0.95, 0.5, 0.5}};  // High tau (bin 4)

    bins.record_error(pos1, 0.005, threshold);
    bins.record_error(pos2, 0.004, threshold);
    bins.record_error(pos3, 0.003, threshold);

    auto problematic = bins.problematic_bins(1, 2);  // dim 1, min_count 2

    // Bin 0 should be problematic (2 errors)
    bool found_bin0 = std::find(problematic.begin(), problematic.end(), 0) != problematic.end();
    EXPECT_TRUE(found_bin0);
}

// ===========================================================================
// Regression tests for bugs found during code review
// ===========================================================================


// ===========================================================================
// Probe measurement bands (spec D2/D9)
//
// The assembled multi-K_ref surface routes each query to the K_ref nearest
// its strike, so each probe is measured only over the strike band it serves:
// the geometric midpoints to its neighbours, clipped to the user's own
// strike range.  These two tests pin the band's degenerate cases.
// ===========================================================================


// ===========================================================================
// Coverage gap tests — Priority 1 (Critical)
// ===========================================================================

// Coverage: an automatic reference ceiling must be positive
TEST(AdaptiveGridBuilderTest, BuildSegmentedRejectsZeroReferenceCeiling) {
    AdaptiveGridParams params;
    params.max_iter = 1;
    params.validation_samples = 8;  // spec D3 minimum

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {}, .max_references = 0},
    };

    auto m = to_log_m({0.7, 0.9, 1.0, 1.1, 1.3});
    std::vector<double> v = {0.10, 0.20, 0.30, 0.40};
    std::vector<double> r = {0.02, 0.05, 0.07, 0.10};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m, v, r});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}

// Coverage: at least one selection round is required
TEST(AdaptiveGridBuilderTest, BuildSegmentedRejectsZeroSelectionRounds) {
    AdaptiveGridParams params;
    params.max_iter = 1;
    params.validation_samples = 8;  // spec D3 minimum

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {}, .max_selection_rounds = 0},
    };

    auto m = to_log_m({0.7, 0.9, 1.0, 1.1, 1.3});
    std::vector<double> v = {0.10, 0.20, 0.30, 0.40};
    std::vector<double> r = {0.02, 0.05, 0.07, 0.10};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m, v, r});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}

// ===========================================================================
// Coverage gap tests — Priority 2 (High)
// ===========================================================================


// ===========================================================================
// Coverage gap tests — Priority 3 (Medium)
// ===========================================================================


// Coverage: Probe failure propagation — validation_samples=0 makes
// run_refinement fail, which build_segmented should propagate cleanly
TEST(AdaptiveGridBuilderTest, BuildSegmentedProbeFailurePropagation) {
    AdaptiveGridParams params;
    params.max_iter = 1;
    params.validation_samples = 0;  // Triggers InvalidConfig inside run_refinement

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    };

    auto m = to_log_m({0.7, 0.9, 1.0, 1.1, 1.3});
    std::vector<double> v = {0.10, 0.20, 0.30, 0.40};
    std::vector<double> r = {0.02, 0.05, 0.07, 0.10};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m, v, r});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}

// Coverage: the requested absolute-strike interval must be ordered
TEST(AdaptiveGridBuilderTest, BuildSegmentedRejectsInvertedStrikeInterval) {
    AdaptiveGridParams params;
    params.max_iter = 1;
    params.validation_samples = 8;  // spec D3 minimum

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        .kref_config = {}, .strike_bounds = StrikeBounds{110.0, 90.0},
    };

    auto m = to_log_m({0.7, 0.9, 1.0, 1.1, 1.3});
    std::vector<double> v = {0.10, 0.20, 0.30, 0.40};
    std::vector<double> r = {0.02, 0.05, 0.07, 0.10};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m, v, r});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}


// ===========================================================================
// Regression tests for segmented Chebyshev dividend edge cases
// ===========================================================================


// ===========================================================================
// Tests for make_tau_split_from_segments
// ===========================================================================

TEST(MakeTauSplitTest, SingleDividendAbsorbsGap) {
    std::vector<double> bounds = {0.01, 0.4995, 0.5005, 1.0};
    std::vector<bool> is_gap = {false, true, false};

    auto split = make_tau_split_from_segments(bounds, is_gap, 100.0);

    auto br_left = split.bracket(100.0, 100.0, 0.3, 0.2, 0.05);
    EXPECT_EQ(br_left.count, 1u);
    EXPECT_EQ(br_left.entries[0].index, 0u);

    auto br_right = split.bracket(100.0, 100.0, 0.7, 0.2, 0.05);
    EXPECT_EQ(br_right.count, 1u);
    EXPECT_EQ(br_right.entries[0].index, 1u);

    auto br_gap_left = split.bracket(100.0, 100.0, 0.4999, 0.2, 0.05);
    EXPECT_EQ(br_gap_left.count, 1u);
    EXPECT_EQ(br_gap_left.entries[0].index, 0u);

    auto br_gap_right = split.bracket(100.0, 100.0, 0.5001, 0.2, 0.05);
    EXPECT_EQ(br_gap_right.count, 1u);
    EXPECT_EQ(br_gap_right.entries[0].index, 1u);
}

TEST(MakeTauSplitTest, TwoDividendsTwoGaps) {
    std::vector<double> bounds = {0.01, 0.2495, 0.2505, 0.4995, 0.5005, 1.0};
    std::vector<bool> is_gap = {false, true, false, true, false};

    auto split = make_tau_split_from_segments(bounds, is_gap, 100.0);

    auto br0 = split.bracket(100.0, 100.0, 0.15, 0.2, 0.05);
    EXPECT_EQ(br0.entries[0].index, 0u);

    auto br1 = split.bracket(100.0, 100.0, 0.375, 0.2, 0.05);
    EXPECT_EQ(br1.entries[0].index, 1u);

    auto br2 = split.bracket(100.0, 100.0, 0.75, 0.2, 0.05);
    EXPECT_EQ(br2.entries[0].index, 2u);
}

TEST(MakeTauSplitTest, NoGaps) {
    std::vector<double> bounds = {0.01, 1.0};
    std::vector<bool> is_gap = {false};

    auto split = make_tau_split_from_segments(bounds, is_gap, 100.0);

    auto br = split.bracket(100.0, 100.0, 0.5, 0.2, 0.05);
    EXPECT_EQ(br.count, 1u);
    EXPECT_EQ(br.entries[0].index, 0u);
}

// ===========================================================================
// compute_segment_boundaries unit tests: dividend dates -> tau segments.
// These pin the segmentation key points directly at the seam; the full-build
// counterparts (SegmentedChebyshevDuplicateDividends and
// SegmentedChebyshevNearlyCoincidentDividends in
// adaptive_surface_build_integration_test.cc) keep end-to-end smoke coverage.
// ===========================================================================

// Regression companion: duplicate dividend dates must merge to ONE gap
// Bug risk: two dividends at the same date creating two overlapping gap
// segments with non-monotonic boundaries.
TEST(ComputeSegmentBoundariesTest, DuplicateDividendDatesMergeToOneGap) {
    std::vector<Dividend> divs = {
        Dividend{.calendar_time = 0.5, .amount = 1.0},
        Dividend{.calendar_time = 0.5, .amount = 1.5},
    };

    auto seg = compute_segment_boundaries(divs, 1.0, 0.01, 1.0);

    // One merged split at tau = 1.0 - 0.5, bracketed by the 5e-4 inset
    ASSERT_EQ(seg.bounds.size(), 4u);
    ASSERT_EQ(seg.is_gap.size(), 3u);
    EXPECT_DOUBLE_EQ(seg.bounds[0], 0.01);
    EXPECT_NEAR(seg.bounds[1], 0.4995, 1e-12);
    EXPECT_NEAR(seg.bounds[2], 0.5005, 1e-12);
    EXPECT_DOUBLE_EQ(seg.bounds[3], 1.0);
    EXPECT_FALSE(seg.is_gap[0]);
    EXPECT_TRUE(seg.is_gap[1]);
    EXPECT_FALSE(seg.is_gap[2]);
}

// Regression companion: nearly-coincident dividends must not create
// overlapping gaps
// Bug risk: splits closer than the gap width producing non-monotonic
// boundaries (gap A's right edge past gap B's left edge).
TEST(ComputeSegmentBoundariesTest, NearlyCoincidentDividendsKeepMonotonicBounds) {
    // Tau splits 1e-3 apart -- closer than 4 * kInset (2e-3), so they must
    // collapse into a single gap at the cluster midpoint.
    std::vector<Dividend> divs = {
        Dividend{.calendar_time = 0.499, .amount = 1.0},
        Dividend{.calendar_time = 0.500, .amount = 1.0},
    };

    auto seg = compute_segment_boundaries(divs, 1.0, 0.01, 1.0);

    ASSERT_EQ(seg.bounds.size(), 4u);
    EXPECT_NEAR(seg.bounds[1], 0.5000, 1e-12);
    EXPECT_NEAR(seg.bounds[2], 0.5010, 1e-12);
    for (size_t i = 0; i + 1 < seg.bounds.size(); ++i) {
        EXPECT_LT(seg.bounds[i], seg.bounds[i + 1])
            << "boundaries must be strictly increasing at index " << i;
    }
}

// Sanity: well-separated dividends alternate real, gap, real, gap, real
TEST(ComputeSegmentBoundariesTest, SeparatedDividendsProduceOneGapEach) {
    std::vector<Dividend> divs = {
        Dividend{.calendar_time = 0.25, .amount = 1.0},
        Dividend{.calendar_time = 0.50, .amount = 1.0},
    };

    auto seg = compute_segment_boundaries(divs, 1.0, 0.01, 1.0);

    ASSERT_EQ(seg.bounds.size(), 6u);
    ASSERT_EQ(seg.is_gap.size(), 5u);
    EXPECT_FALSE(seg.is_gap[0]);
    EXPECT_TRUE(seg.is_gap[1]);   // around tau = 0.5
    EXPECT_FALSE(seg.is_gap[2]);
    EXPECT_TRUE(seg.is_gap[3]);   // around tau = 0.75
    EXPECT_FALSE(seg.is_gap[4]);
}

// Splits too close to the tau range edges are dropped, not clamped
TEST(ComputeSegmentBoundariesTest, SplitNearRangeEdgeIsDropped) {
    // tau split = 1.0 - 0.9995 = 0.0005, inside tau_min + 2 * kInset
    std::vector<Dividend> divs = {
        Dividend{.calendar_time = 0.9995, .amount = 1.0},
    };

    auto seg = compute_segment_boundaries(divs, 1.0, 0.01, 1.0);

    ASSERT_EQ(seg.bounds.size(), 2u);
    ASSERT_EQ(seg.is_gap.size(), 1u);
    EXPECT_FALSE(seg.is_gap[0]);
}

// ===========================================================================
// Tests for resolve_k_refs
// ===========================================================================

TEST(ResolveKRefsTest, ExplicitValuesArePreservedAndSorted) {
    MultiKRefConfig config{.K_refs = {120.0, 80.0, 100.0}};
    auto result = resolve_k_refs(config, StrikeBounds{85.0, 115.0});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, (std::vector{80.0, 100.0, 120.0}));
}

TEST(ResolveKRefsTest, SingletonDomainUsesExactlyOneReference) {
    for (auto config : {MultiKRefConfig{}, MultiKRefConfig{.K_refs = {100.0}}}) {
        auto result = resolve_k_refs(config, StrikeBounds{100.0, 100.0});
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(*result, (std::vector{100.0}));
    }
}

TEST(ResolveKRefsTest, AutomaticSeedCoversTheRequestedDomainWithinCeiling) {
    MultiKRefConfig config;
    const StrikeBounds bounds{100.0 / 1.3, 100.0 / 0.7};
    for (size_t cap : {2u, 3u, 65u}) {
        config.max_references = cap;
        auto result = resolve_k_refs(config, bounds);
        ASSERT_TRUE(result.has_value());
        EXPECT_DOUBLE_EQ(result->front(), bounds.min);
        EXPECT_DOUBLE_EQ(result->back(), bounds.max);
        EXPECT_LE(result->size(), cap);
        for (size_t i = 1; i < result->size(); ++i) EXPECT_LT((*result)[i - 1], (*result)[i]);
    }
}

TEST(ResolveKRefsTest, RejectsImpossibleLimitsAndInvalidDomains) {
    for (auto config : {MultiKRefConfig{.max_references = 0},
                        MultiKRefConfig{.max_references = 1},
                        MultiKRefConfig{.max_selection_rounds = 0}}) {
        EXPECT_FALSE(resolve_k_refs(config, StrikeBounds{80.0, 120.0}).has_value());
    }
    for (auto domain : {StrikeBounds{120.0, 80.0}, StrikeBounds{0.0, 120.0},
                        StrikeBounds{80.0, std::numeric_limits<double>::infinity()}}) {
        EXPECT_FALSE(resolve_k_refs({}, domain).has_value());
    }
}

TEST(ResolveKRefsTest, ExplicitCeilingNeverTruncatesTheRequestedSet) {
    MultiKRefConfig config{.K_refs = {80.0, 90.0, 100.0, 120.0}, .max_references = 3};
    EXPECT_FALSE(resolve_k_refs(config, StrikeBounds{85.0, 115.0}).has_value());
    EXPECT_EQ(config.K_refs, (std::vector{80.0, 90.0, 100.0, 120.0}));
}

// ===========================================================================
// Tests for build_chebyshev_segmented_manual (non-adaptive path)
// ===========================================================================


// ===========================================================================
// Tests for expand_segmented_domain
// ===========================================================================

// Support widening must be an enclosure, including requests below numerical
// padding floors. These assertions do not prescribe a padding formula.
TEST(ExpandSegmentedDomainTest, PreservesRequestedEndpointsBelowPaddingFloors) {
    for (double sigma_min : {0.005, 1e-8}) {
        IVGrid domain{.moneyness = to_log_m({0.001, 0.005, 0.02}),
            .vol = {sigma_min, 0.02}, .rate = {-0.10, -0.06}};
        auto support = expand_segmented_domain(domain, .25, 0.0, {{.1, 2.0}}, 100.0);
        ASSERT_TRUE(support);
        EXPECT_LE(support->m_min, domain.moneyness.front());
        EXPECT_GE(support->m_max, domain.moneyness.back());
        EXPECT_GT(support->sigma_min, 0.0);
        EXPECT_LE(support->sigma_min, domain.vol.front());
        EXPECT_GE(support->sigma_max, domain.vol.back());
        EXPECT_LE(support->rate_min, domain.rate.front());
        EXPECT_GE(support->rate_max, domain.rate.back());
    }
}

TEST(ChebyshevSupportBounds, PaddingContainsEveryRequestedEndpoint) {
    for (double tau_min : {0.0, 1e-7, .1}) {
        for (double sigma_min : {1e-8, .005, .1}) {
            const SurfaceBounds requested{-7.0, -4.0, tau_min, .25,
                sigma_min, .3, -.1, -.06};
            const auto support = detail::chebyshev_support_bounds(requested, {5, 3, 2, 1});
            EXPECT_LE(support.m_min, requested.m_min);
            EXPECT_GE(support.m_max, requested.m_max);
            EXPECT_LE(support.tau_min, requested.tau_min);
            EXPECT_GE(support.tau_max, requested.tau_max);
            EXPECT_GE(support.tau_min, 0.0);
            EXPECT_LE(support.sigma_min, requested.sigma_min);
            EXPECT_GE(support.sigma_max, requested.sigma_max);
            EXPECT_GT(support.sigma_min, 0.0);
            EXPECT_LE(support.rate_min, requested.rate_min);
            EXPECT_GE(support.rate_max, requested.rate_max);
        }
    }
}

TEST(ExpandSegmentedDomainTest, InvalidVolatilityIsNotRepairedByPadding) {
    for (double sigma : {0.0, -.01, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
        IVGrid domain{.moneyness = {-.1, .1}, .vol = {sigma, .2}, .rate = {-.1, -.06}};
        auto support = expand_segmented_domain(domain, .25, 0.0, {}, 100.0);
        EXPECT_FALSE(support);
        if (!support) {
            EXPECT_EQ(support.error().code, PriceTableErrorCode::InvalidConfig);
        }
        SegmentedAdaptiveConfig config{.spot = 100.0, .option_type = OptionType::PUT,
            .dividend_yield = 0.0, .maturity = .25,
            .kref_config = {.K_refs = {100.0}}, .strike_bounds = StrikeBounds{100.0, 100.0}};
        EXPECT_FALSE(BSplineSegmentedBuilder::create(config, domain));
        EXPECT_FALSE(ChebyshevSegmentedBuilder::create(config, domain));
    }
}

TEST(ExpandSegmentedDomainTest, NoDividends) {
    IVGrid domain{
        .moneyness = to_log_m({0.8, 1.0, 1.2}),
        .vol = {0.20, 0.30},
        .rate = {0.03, 0.05}
    };
    double maturity = 1.0;
    std::vector<Dividend> divs;
    auto result = expand_segmented_domain(domain, maturity, 0.02, divs, 100.0);
    ASSERT_TRUE(result.has_value());
    // Verify expansion by standard spreads
    // log(0.8) ~ -0.223, log(1.2) ~ 0.182 → width > 0.10, so expand_domain_bounds
    // just ensures spread >= 0.10 (already satisfied), no extra push
    EXPECT_LE(result->m_min, std::log(0.8));
    EXPECT_GE(result->m_max, std::log(1.2));
    // Vol: [0.20, 0.30] width=0.10, exactly spread → no extra push
    EXPECT_LE(result->sigma_min, 0.20);
    EXPECT_GE(result->sigma_max, 0.30);
    // Rate: [0.03, 0.05] width=0.02 < 0.04 → should expand to 0.04 wide
    EXPECT_LT(result->rate_min, 0.03);
    EXPECT_GT(result->rate_max, 0.05);
    // Numerical support includes the exact analytical payoff row.
    EXPECT_EQ(result->tau_max, maturity);
    EXPECT_EQ(result->tau_min, 0.0);
}

TEST(ExpandSegmentedDomainTest, WithDividends) {
    IVGrid domain{
        .moneyness = to_log_m({0.8, 1.0, 1.2}),
        .vol = {0.20},
        .rate = {0.05}
    };
    std::vector<Dividend> divs = {Dividend{.calendar_time = 0.25, .amount = 2.0}};
    auto result = expand_segmented_domain(domain, 1.0, 0.0, divs, 100.0);
    ASSERT_TRUE(result.has_value());
    // Dividend expansion: total_div=2.0, min_K_ref=100.0 → expansion=0.02
    // min_m starts at log(0.8), expanded by shifting exp(log(0.8))-0.02 = 0.78
    // min_m becomes log(0.78) < log(0.8)
    double orig_min_m = std::log(0.8);
    EXPECT_LT(result->m_min, orig_min_m);
}

TEST(ExpandSegmentedDomainTest, EmptyDomain) {
    IVGrid empty_domain{.moneyness = {}, .vol = {}, .rate = {}};
    auto result = expand_segmented_domain(empty_domain, 1.0, 0.0, {}, 100.0);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}

TEST(ExpandSegmentedDomainTest, TauCappedAtMaturity) {
    // Short maturity: expansion should not exceed maturity
    IVGrid domain{
        .moneyness = to_log_m({0.9, 1.0, 1.1}),
        .vol = {0.20, 0.30},
        .rate = {0.03, 0.05}
    };
    double maturity = 0.05;
    auto result = expand_segmented_domain(domain, maturity, 0.0, {}, 100.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_LE(result->tau_max, maturity);
}

TEST(ExpandSegmentedDomainTest, LargeDividendClamps) {
    // Large dividend relative to K_ref should clamp moneyness floor at 0.01
    IVGrid domain{
        .moneyness = to_log_m({0.5, 1.0, 1.5}),
        .vol = {0.20},
        .rate = {0.05}
    };
    // total_div = 50, min_K_ref = 50 → expansion = 1.0
    // exp(log(0.5)) = 0.5, 0.5 - 1.0 = -0.5, clamped to 0.01
    std::vector<Dividend> divs = {Dividend{.calendar_time = 0.25, .amount = 50.0}};
    auto result = expand_segmented_domain(domain, 1.0, 0.0, divs, 50.0);
    ASSERT_TRUE(result.has_value());
    // min_m should be log(0.01) after clamping
    EXPECT_GE(result->m_min, std::log(0.01) - 0.1);
}

// ===========================================================================
// Chebyshev refiner contract (spec D6): exact axis, level cap, state rollback
// ===========================================================================

namespace {

/// Deliberately anisotropic levels: moneyness sits 4 levels above rate, so the
/// removed balance rule ("refuse a dimension more than 2 levels ahead of the
/// minimum, fall back to the lowest") would have redirected a moneyness
/// request to the rate axis.
detail::ChebyshevRefinementState make_cheb_state() {
    return detail::ChebyshevRefinementState{
        .m_level = 5, .tau_level = 3, .sigma_level = 2, .rate_level = 1,
        .max_level = 7,
        .m_lo = -0.5, .m_hi = 0.5,
        .tau_lo = 0.05, .tau_hi = 1.5,
        .sigma_lo = 0.10, .sigma_hi = 0.50,
        .rate_lo = 0.0, .rate_hi = 0.08,
    };
}

/// The four working grids the refiner mutates, seeded at the state's levels.
struct ChebGrids {
    std::vector<double> moneyness, tau, vol, rate;

    static ChebGrids seed(const detail::ChebyshevRefinementState& s) {
        return ChebGrids{
            .moneyness = cc_level_nodes(s.m_level, s.m_lo, s.m_hi),
            .tau = cc_level_nodes(s.tau_level, s.tau_lo, s.tau_hi),
            .vol = cc_level_nodes(s.sigma_level, s.sigma_lo, s.sigma_hi),
            .rate = cc_level_nodes(s.rate_level, s.rate_lo, s.rate_hi),
        };
    }

    RefineOutcome refine(const RefineFn& fn, size_t dim) {
        return fn(dim, {}, moneyness, tau, vol, rate);
    }

    std::array<size_t, 4> sizes() const {
        return {moneyness.size(), tau.size(), vol.size(), rate.size()};
    }

    bool operator==(const ChebGrids&) const = default;
};

std::array<size_t, 4> levels_of(const detail::ChebyshevRefinementState& s) {
    return {s.m_level, s.tau_level, s.sigma_level, s.rate_level};
}

}  // namespace

// Regression: the refiner must advance EXACTLY the requested axis (spec D6).
// Bug: a balance rule redirected any request for a dimension more than 2
// levels above the minimum to the lowest-level dimension, so the coordinate
// descent walk could never actually test the axis it picked.
TEST(ChebyshevRefineFn, HonorsRequestedAxisDespiteLevelSpread) {
    auto state = make_cheb_state();
    auto refine = detail::make_chebyshev_refine_fn(state);
    auto grids = ChebGrids::seed(state);

    // Moneyness is 4 levels above the minimum: the old rule redirected here.
    auto outcome = grids.refine(refine, 0);

    EXPECT_TRUE(outcome.changed);
    EXPECT_EQ(outcome.changed_dim, 0);
    EXPECT_EQ(levels_of(state), (std::array<size_t, 4>{6, 3, 2, 1}));
    EXPECT_EQ(grids.sizes(), (std::array<size_t, 4>{65, 9, 5, 3}));
}

// Every axis, however far ahead, advances on request.
TEST(ChebyshevRefineFn, HonorsEveryRequestedAxis) {
    for (size_t dim = 0; dim < 4; ++dim) {
        auto state = make_cheb_state();
        auto refine = detail::make_chebyshev_refine_fn(state);
        auto grids = ChebGrids::seed(state);
        auto before = levels_of(state);

        auto outcome = grids.refine(refine, dim);

        ASSERT_TRUE(outcome.changed) << "dim " << dim;
        EXPECT_EQ(outcome.changed_dim, static_cast<int>(dim));
        auto after = levels_of(state);
        for (size_t d = 0; d < 4; ++d) {
            EXPECT_EQ(after[d], before[d] + (d == dim ? 1u : 0u))
                << "dim " << dim << " moved level " << d;
        }
    }
}

// An axis at its level cap reports changed=false instead of redirecting.
TEST(ChebyshevRefineFn, AxisAtCapReportsNoChange) {
    auto state = make_cheb_state();
    state.m_level = state.max_level;
    auto refine = detail::make_chebyshev_refine_fn(state);
    auto grids = ChebGrids::seed(state);
    auto before = grids;

    auto outcome = grids.refine(refine, 0);

    EXPECT_FALSE(outcome.changed);
    EXPECT_EQ(outcome.changed_dim, -1);
    EXPECT_EQ(levels_of(state), (std::array<size_t, 4>{7, 3, 2, 1}))
        << "no other axis may be bumped in place of the capped one";
    EXPECT_TRUE(grids == before);
}

// Snapshot/restore must round-trip the level counters exactly, so a rejected
// trial can be re-run from the same base and land on the same grids.
TEST(ChebyshevStateHooks, RestoreMakesRefinementRepeatable) {
    auto state = make_cheb_state();
    auto refine = detail::make_chebyshev_refine_fn(state);
    auto hooks = detail::make_chebyshev_state_hooks(state);
    ASSERT_TRUE(hooks.snapshot != nullptr);
    ASSERT_TRUE(hooks.restore != nullptr);

    auto base_grids = ChebGrids::seed(state);
    auto snap = hooks.snapshot();

    auto first = base_grids;
    ASSERT_TRUE(first.refine(refine, 0).changed);
    auto first_levels = levels_of(state);

    hooks.restore(snap);
    EXPECT_EQ(levels_of(state), (std::array<size_t, 4>{5, 3, 2, 1}));

    auto second = base_grids;
    ASSERT_TRUE(second.refine(refine, 0).changed);

    EXPECT_EQ(levels_of(state), first_levels);
    EXPECT_TRUE(second == first);
}

// Spec-pinned scenario (D6): axis 0 rejected, axis 1 accepted (walk restart),
// axis 0 retried.  The retry must start from the accepted state, so it lands
// exactly where a fresh single refinement of axis 0 from that state lands --
// no double advance from the rejected trial.
TEST(ChebyshevStateHooks, RetryAfterBacktrackMatchesFreshRefinement) {
    auto state = make_cheb_state();
    auto refine = detail::make_chebyshev_refine_fn(state);
    auto hooks = detail::make_chebyshev_state_hooks(state);

    auto base_grids = ChebGrids::seed(state);
    auto base_snap = hooks.snapshot();

    // Trial 1: axis 0 -- rejected (no holdout improvement).
    auto trial = base_grids;
    ASSERT_TRUE(trial.refine(refine, 0).changed);

    // Backtrack to the exploration base, then trial 2: axis 1 -- accepted.
    hooks.restore(base_snap);
    auto accepted = base_grids;
    ASSERT_TRUE(accepted.refine(refine, 1).changed);
    auto accepted_levels = levels_of(state);
    auto accepted_snap = hooks.snapshot();

    // Walk restarts; axis 0 is retried from the accepted base.
    hooks.restore(accepted_snap);
    auto retry = accepted;
    ASSERT_TRUE(retry.refine(refine, 0).changed);
    auto retry_levels = levels_of(state);
    auto retry_grids = retry;

    // Reference: a fresh single refinement of axis 0 from the accepted state.
    detail::ChebyshevRefinementState fresh_state = make_cheb_state();
    auto fresh_refine = detail::make_chebyshev_refine_fn(fresh_state);
    auto fresh_grids = ChebGrids::seed(fresh_state);
    ASSERT_TRUE(fresh_grids.refine(fresh_refine, 1).changed);
    ASSERT_EQ(levels_of(fresh_state), accepted_levels);
    ASSERT_TRUE(fresh_grids.refine(fresh_refine, 0).changed);

    EXPECT_EQ(retry_levels, levels_of(fresh_state));
    EXPECT_EQ(retry_levels, (std::array<size_t, 4>{6, 4, 2, 1}));
    EXPECT_TRUE(retry_grids == fresh_grids);
}


// The segmented refiner carries the same contract, with per-segment tau nodes.
TEST(SegmentedChebyshevRefineFn, HonorsRequestedAxisAndCap) {
    auto state = make_cheb_state();
    state.seg_boundaries = {0.05, 0.5, 0.55, 1.5};
    state.seg_is_gap = {false, true, false};
    auto refine = detail::make_segmented_chebyshev_refine_fn(state);
    auto grids = ChebGrids::seed(state);
    grids.tau = detail::generate_segmented_tau_nodes(
        state.tau_level, state.seg_boundaries, state.seg_is_gap);

    // Moneyness sits far above the minimum level: no redirection.
    auto outcome = grids.refine(refine, 0);
    EXPECT_TRUE(outcome.changed);
    EXPECT_EQ(outcome.changed_dim, 0);
    EXPECT_EQ(levels_of(state), (std::array<size_t, 4>{6, 3, 2, 1}));

    // Tau refinement regenerates per-segment nodes (gaps skipped).
    size_t tau_before = grids.tau.size();
    outcome = grids.refine(refine, 1);
    EXPECT_TRUE(outcome.changed);
    EXPECT_EQ(outcome.changed_dim, 1);
    EXPECT_GT(grids.tau.size(), tau_before);
    EXPECT_GE(grids.tau.front(), state.seg_boundaries.front() - 1e-12);
    EXPECT_LE(grids.tau.back(), state.seg_boundaries.back() + 1e-12);

    // Cap: rate at max_level reports no change rather than redirecting.
    state.rate_level = state.max_level;
    auto before = grids;
    outcome = grids.refine(refine, 3);
    EXPECT_FALSE(outcome.changed);
    EXPECT_EQ(outcome.changed_dim, -1);
    EXPECT_TRUE(grids == before);
}

// ===========================================================================
// Segmented final-surface contracts (spec D9)
//
// The segmented builders assemble their final surface outside the refinement
// loop, so it gets its own references, its own score, and its own viability
// gate.  These tests pin the selection arithmetic directly (no PDE solves)
// and then check the assembled B-spline path reports its *returned* surface.
// ===========================================================================

/// Synthetic zero-price references let the passthrough proxy equal absolute
/// price error; the two channels remain independently counted.
std::vector<detail::ValidationPoint> make_points(size_t n) {
    std::vector<detail::ValidationPoint> pts;
    pts.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        pts.push_back(detail::ValidationPoint{
            .coords = {0.0, 0.5, 0.20 + 0.01 * static_cast<double>(i), 0.05},
            .strike = 100.0, .spot = 100.0,
            .refs = {.ref_price = 0.0, .vega = 1.0}});
    }
    return pts;
}

RefinementContext make_score_ctx() {
    return RefinementContext{
        .spot = 100.0,
        .dividend_yield = 0.0,
        .option_type = OptionType::PUT,
        .bounds = {.m_min = -0.4, .m_max = 0.4, .tau_min = 0.05,
                   .tau_max = 1.0, .sigma_min = 0.1, .sigma_max = 0.5,
                   .rate_min = 0.0, .rate_max = 0.1},
        .sample_bounds = {.m_min = -0.3, .m_max = 0.3, .tau_min = 0.05,
                          .tau_max = 1.0, .sigma_min = 0.1, .sigma_max = 0.5,
                          .rate_min = 0.0, .rate_max = 0.1},
    };
}

/// A ScoreErrorFn that returns |interp| verbatim, so a test can dictate the
/// exact error at every point through the surface handle.
ScoreErrorFn passthrough_score() {
    return [](double interp, const ErrorRefs&, double, double, double,
              double, double) -> std::optional<double> { return interp; };
}

/// Like `passthrough_score`, but skips every `period`-th point the way the
/// TV/K and vega-floor filters do (spec D4: nullopt, not zero).
ScoreErrorFn filtering_score(size_t period,
                             const std::shared_ptr<size_t>& calls) {
    return [period, calls](double interp, const ErrorRefs&, double, double,
                           double, double, double) -> std::optional<double> {
        if ((*calls)++ % period == 0) return std::nullopt;
        return interp;
    };
}

detail::FinalScore score_of(double max_error, bool all_finite = true,
                            size_t measured = 8) {
    detail::FinalScore s;
    s.max_error = max_error;
    s.avg_error = max_error;
    s.measured = measured;
    s.all_finite = all_finite;
    s.price_errors = {.measured = 8, .max_error = 0.0, .mean_error = 0.0};
    return s;
}

TEST(SegmentedFinalContract, PriceMissSurvivesExactIvProxyForRetrySelection) {
    auto original = score_of(0.0);
    original.price_errors = {.measured = 8, .max_error = .05, .mean_error = .05};
    const auto retry = score_of(0.0);
    EXPECT_TRUE(detail::needs_final_retry(original, 1e-4));
    EXPECT_EQ(detail::select_final_surface(original, retry), detail::FinalPick::Retry);
}

// Regression: `if (err > 0.0) valid++` counted only *nonzero* errors, so a
// surface that reproduced every reference exactly reported valid == 0 -- a
// zero average denominator and a `target_met` that came from the
// "nothing measured" branch rather than from convergence.
TEST(SegmentedFinalContract, PerfectSurfaceCountsEveryScoredPoint) {
    const auto pts = make_points(8);
    const auto ctx = make_score_ctx();
    // Interpolation reproduces the reference exactly => score 0 everywhere.
    const SurfaceHandle exact{
        .price = [](double, double, double, double, double) { return 0.0; }};

    auto s = detail::score_final_surface(pts, exact, passthrough_score(), ctx);

    EXPECT_EQ(s.measured, pts.size())
        << "zero error is a measurement, not a gap";
    EXPECT_EQ(s.skipped, 0u);
    EXPECT_DOUBLE_EQ(s.max_error, 0.0);
    EXPECT_DOUBLE_EQ(s.avg_error, 0.0);
    EXPECT_TRUE(s.viable());
    // target_met now comes from a real measurement of 8 points.
    EXPECT_FALSE(detail::needs_final_retry(s, 1e-5));
}

// Half the points measure, half are non-finite: the average denominator is
// the measured count, and any non-finite evaluation disqualifies the
// surface.
TEST(SegmentedFinalContract, NonFiniteEvaluationIsNotViable) {
    const auto pts = make_points(8);
    const auto ctx = make_score_ctx();
    size_t calls = 0;
    const SurfaceHandle flaky{
        .price = [&calls](double, double, double, double, double) {
            return (calls++ % 2 == 0)
                ? 0.001
                : std::numeric_limits<double>::quiet_NaN();
        }};

    auto s = detail::score_final_surface(pts, flaky, passthrough_score(), ctx);

    EXPECT_EQ(s.measured, 4u);
    EXPECT_EQ(s.skipped, 4u);
    EXPECT_FALSE(s.all_finite);
    EXPECT_FALSE(s.viable());
    EXPECT_TRUE(std::isnan(s.max_error))
        << "a surface that produced NaN must not report a rosy max";
    EXPECT_TRUE(detail::needs_final_retry(s, 1.0))
        << "non-viable must retry even under a target it nominally meets";
}

// A validation set that cannot measure cannot certify the surface (D4/D9.1).
TEST(SegmentedFinalContract, SparseReferencesFailValidation) {
    AdaptiveGridParams params;
    params.validation_samples = 16;  // min valid = max(4, 16/4) = 4
    const auto ctx = make_score_ctx();

    size_t calls = 0;
    PrepareRefsFn mostly_failing =
        [&calls](double, double, double, double, double)
        -> std::expected<ErrorRefs, SolverError> {
        if (calls++ < 3) return ErrorRefs{.ref_price = 10.0, .vega = 1.0};
        return std::unexpected(SolverError{SolverErrorCode::ConvergenceFailure});
    };

    auto set = detail::prepare_final_validation(params, ctx, mostly_failing,
                                                params.lhs_seed + 999);
    ASSERT_FALSE(set.has_value());
    EXPECT_EQ(set.error().code, PriceTableErrorCode::ValidationFailed);

    // One more valid point clears the bar.
    calls = 0;
    PrepareRefsFn four_ok = [&calls](double, double, double, double, double)
        -> std::expected<ErrorRefs, SolverError> {
        if (calls++ < 4) return ErrorRefs{.ref_price = 10.0, .vega = 1.0};
        return std::unexpected(SolverError{SolverErrorCode::ConvergenceFailure});
    };
    auto ok = detail::prepare_final_validation(params, ctx, four_ok,
                                               params.lhs_seed + 999);
    ASSERT_TRUE(ok.has_value());
    EXPECT_EQ(ok->points.size(), 4u);
    EXPECT_EQ(ok->invalid, 12u);
}

// Selection returns the lowest-error *viable* surface -- the retry is not
// preferred just because it was built.
TEST(SegmentedFinalContract, SelectionKeepsOriginalWhenRetryIsWorse) {
    const auto orig = score_of(0.01);
    const auto retry = score_of(0.05);

    EXPECT_EQ(detail::select_final_surface(orig, retry),
              detail::FinalPick::Original);
    EXPECT_EQ(detail::select_final_surface(orig, std::nullopt),
              detail::FinalPick::Original);
    // Equal accuracy keeps the smaller surface.
    EXPECT_EQ(detail::select_final_surface(orig, score_of(0.01)),
              detail::FinalPick::Original);
    // And a genuine improvement is taken.
    EXPECT_EQ(detail::select_final_surface(orig, score_of(0.001)),
              detail::FinalPick::Retry);
}

// A non-viable original is never returned, even when the retry is worse than
// the target -- viability, not accuracy, decides admissibility.
TEST(SegmentedFinalContract, SelectionPrefersViableOverLowerError) {
    const auto garbage = score_of(5.0);          // > kViabilityBound
    const auto mediocre = score_of(0.05);        // viable, misses a tight target

    EXPECT_EQ(detail::select_final_surface(garbage, mediocre),
              detail::FinalPick::Retry);
    EXPECT_EQ(detail::select_final_surface(mediocre, garbage),
              detail::FinalPick::Original);
    EXPECT_EQ(detail::select_final_surface(garbage, score_of(6.0)),
              detail::FinalPick::None)
        << "both non-viable must refuse, not return the lesser garbage";
    EXPECT_EQ(detail::select_final_surface(garbage, std::nullopt),
              detail::FinalPick::None);
}

// A loose target does not admit a garbage surface: the retry is still tried
// even though the original's max error is nominally "under target".
TEST(SegmentedFinalContract, LooseTargetStillRetriesNonViableOriginal) {
    const auto orig = score_of(0.30);  // <= 0.5 target, > 0.20 viability bound
    EXPECT_LE(orig.max_error, 0.5);
    EXPECT_FALSE(orig.viable());
    EXPECT_TRUE(detail::needs_final_retry(orig, 0.5));

    // And a strict target retries a perfectly viable surface too.
    const auto good = score_of(0.001);
    EXPECT_TRUE(good.viable());
    EXPECT_TRUE(detail::needs_final_retry(good, 1e-5));
    EXPECT_FALSE(detail::needs_final_retry(good, 0.01));
}

// Zero measured points cannot certify anything, whatever the max says.
TEST(SegmentedFinalContract, NoMeasuredPointsIsNotViable) {
    EXPECT_FALSE(score_of(0.0, true, 0).viable());
    EXPECT_EQ(detail::select_final_surface(score_of(0.0, true, 0),
                                           std::nullopt),
              detail::FinalPick::None);
}

// Regression: filtered points are not measurements.
// Bug: the score fn returned 0.0 where the TV/K or vega-floor filter fired,
// so a filtered point entered the average as a perfect score and counted
// toward "at least one measurement".  A surface filtered everywhere reported
// max 0 / avg 0 and passed the viability gate having been measured nowhere
// (final-review amendment 2026-08-29).
TEST(SegmentedFinalContract, FilteredIvStillRetainsPriceStatistics) {
    const auto pts = make_points(8);
    const auto ctx = make_score_ctx();
    // Every point would score 0.10; half of them are filtered out.
    const SurfaceHandle flat{
        .price = [](double, double, double, double, double) { return 0.10; }};

    auto calls = std::make_shared<size_t>(0);
    auto s = detail::score_final_surface(pts, flat, filtering_score(2, calls),
                                         ctx);

    EXPECT_EQ(s.measured, 4u);
    EXPECT_EQ(s.filtered, 4u);
    EXPECT_EQ(s.price_errors.measured, pts.size());
    ASSERT_TRUE(s.price_errors.max_error);
    EXPECT_DOUBLE_EQ(*s.price_errors.max_error, .10);
    EXPECT_EQ(s.skipped, 0u);
    EXPECT_TRUE(s.all_finite);
    // Averaged over the measured points only -- a filtered point pulled the
    // average toward zero before.
    EXPECT_DOUBLE_EQ(s.max_error, 0.10);
    EXPECT_DOUBLE_EQ(s.avg_error, 0.10);
    EXPECT_TRUE(s.viable());
}

// And with *every* point filtered there is nothing to certify.
TEST(SegmentedFinalContract, FullyFilteredSurfaceIsNotViable) {
    const auto pts = make_points(8);
    const auto ctx = make_score_ctx();
    const SurfaceHandle flat{
        .price = [](double, double, double, double, double) { return 0.10; }};

    auto calls = std::make_shared<size_t>(0);
    auto s = detail::score_final_surface(pts, flat, filtering_score(1, calls),
                                         ctx);

    EXPECT_EQ(s.measured, 0u);
    EXPECT_EQ(s.filtered, pts.size());
    EXPECT_DOUBLE_EQ(s.max_error, 0.0);
    EXPECT_FALSE(s.viable())
        << "a max of zero over an empty measurement set certifies nothing";
    EXPECT_EQ(detail::select_final_surface(s, std::nullopt),
              detail::FinalPick::None);
}


// ===========================================================================
// Regression tests for the q0 bifurcation (issue #434)
// ===========================================================================


}  // namespace
}  // namespace mango
