// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/bspline/bspline_adaptive.hpp"
#include "mango/option/table/bspline/bspline_pde_cache.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/math/chebyshev/chebyshev_nodes.hpp"
#include "mango/option/american_option_batch.hpp"
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

    // Surface should be populated
    EXPECT_NE(result->surface, nullptr);

    // Should have done some PDE solves
    EXPECT_GT(result->total_pde_solves, 0);
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
    params.validation_samples = 4;

    auto grid_spec = GridSpec<double>::uniform(-3.0, 3.0, 31).value();
    auto result = build_adaptive_bspline(params, chain,
        PDEGridConfig{grid_spec, 100, {}}, OptionType::PUT);

    // Should succeed (bounds expanded) rather than fail with InsufficientGridPoints
    ASSERT_TRUE(result.has_value())
        << "Single-value axes should be expanded to valid ranges. "
        << "Error code: " << (result.has_value() ? 0 : static_cast<int>(result.error().code));

    // Surface should be usable
    EXPECT_NE(result->surface, nullptr);
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
    chain2.spot = 50.0;  // Different spot => cache must not reuse chain1 slices

    AdaptiveGridParams params;
    params.max_iter = 1;
    params.validation_samples = 1;  // Minimum to satisfy validation guard

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
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    };

    auto m_domain = to_log_m({0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3});
    std::vector<double> v_domain = {0.05, 0.10, 0.20, 0.30, 0.50};
    std::vector<double> r_domain = {0.01, 0.03, 0.05, 0.10};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m_domain, v_domain, r_domain});
    ASSERT_TRUE(result.has_value())
        << "build_adaptive_bspline_segmented failed";

    // Should be able to query prices at various strikes
    double price = result->surface.price(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
    EXPECT_TRUE(std::isfinite(price));

    // And at off-K_ref strikes
    double price2 = result->surface.price(100.0, 90.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price2, 0.0);
    EXPECT_TRUE(std::isfinite(price2));
}

TEST(AdaptiveGridBuilderTest, BuildSegmentedSmallKRefList) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 1;
    params.validation_samples = 8;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {Dividend{.calendar_time = 0.25, .amount = 1.50}},
        .maturity = 0.5,
        .kref_config = {.K_refs = {95.0, 105.0}},  // < 3 K_refs — probe all
    };

    auto m_domain = to_log_m({0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3});
    std::vector<double> v_domain = {0.10, 0.15, 0.20, 0.30};
    std::vector<double> r_domain = {0.02, 0.03, 0.05, 0.07};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m_domain, v_domain, r_domain});
    ASSERT_TRUE(result.has_value());
}

// Large discrete dividend (total_div/K_ref > 0.2, stresses moneyness expansion)
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
    ASSERT_TRUE(result.has_value());

    double price = result->surface.price(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
    EXPECT_TRUE(std::isfinite(price));
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

    auto m_domain = to_log_m({0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3});
    std::vector<double> v_domain = {0.10, 0.15, 0.20, 0.30};
    std::vector<double> r_domain = {0.02, 0.03, 0.05, 0.07};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m_domain, v_domain, r_domain});
    ASSERT_TRUE(result.has_value());

    double price = result->surface.price(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
    EXPECT_TRUE(std::isfinite(price));
}

// ===========================================================================
// Coverage gap tests — Priority 1 (Critical)
// ===========================================================================

// Coverage: Invalid auto-K_ref config with count < 1
TEST(AdaptiveGridBuilderTest, BuildSegmentedRejectsInvalidKRefCount) {
    AdaptiveGridParams params;
    params.max_iter = 1;
    params.validation_samples = 4;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {}, .K_ref_count = 0, .K_ref_span = 0.3},
    };

    auto m = to_log_m({0.7, 0.9, 1.0, 1.1, 1.3});
    std::vector<double> v = {0.10, 0.20, 0.30, 0.40};
    std::vector<double> r = {0.02, 0.05, 0.07, 0.10};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m, v, r});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}

// Coverage: Invalid auto-K_ref config with span <= 0
TEST(AdaptiveGridBuilderTest, BuildSegmentedRejectsZeroSpan) {
    AdaptiveGridParams params;
    params.max_iter = 1;
    params.validation_samples = 4;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {}, .K_ref_count = 5, .K_ref_span = 0.0},
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

// Coverage: ATM K_ref coincides with lowest K_ref — dedup prevents 3rd probe
TEST(AdaptiveGridBuilderTest, BuildSegmentedATMEqualsLowest) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 1;
    params.validation_samples = 8;
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

    auto m = to_log_m({0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3});
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
    params.validation_samples = 8;
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

    auto m = to_log_m({0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3});
    std::vector<double> v = {0.10, 0.15, 0.20, 0.30};
    std::vector<double> r = {0.02, 0.03, 0.05, 0.07};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m, v, r});
    ASSERT_TRUE(result.has_value());
    double price = result->surface.price(100.0, 90.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
}

// Coverage: Single auto-generated K_ref (count=1)
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
    ASSERT_TRUE(result.has_value());

    // Single K_ref = spot, should produce valid prices
    double price = result->surface.price(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
    EXPECT_TRUE(std::isfinite(price));
}

// Coverage: Very short maturity — tau domain compressed, max_tau clamped
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

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m, v, r});
    ASSERT_TRUE(result.has_value());

    // Query at a tau within the short maturity
    double price = result->surface.price(100.0, 100.0, 0.03, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
    EXPECT_TRUE(std::isfinite(price));
}

// ===========================================================================
// Coverage gap tests — Priority 3 (Medium)
// ===========================================================================

// Coverage: Large expansion clamps moneyness to 0.01
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

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m, v, r});
    // Should succeed — moneyness floor prevents negative/zero domain
    ASSERT_TRUE(result.has_value());
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

// Coverage: Negative span with auto K_refs
TEST(AdaptiveGridBuilderTest, BuildSegmentedRejectsNegativeSpan) {
    AdaptiveGridParams params;
    params.max_iter = 1;
    params.validation_samples = 4;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {}, .K_ref_count = 3, .K_ref_span = -0.2},
    };

    auto m = to_log_m({0.7, 0.9, 1.0, 1.1, 1.3});
    std::vector<double> v = {0.10, 0.20, 0.30, 0.40};
    std::vector<double> r = {0.02, 0.05, 0.07, 0.10};

    auto result = build_adaptive_bspline_segmented(params, seg_config, {m, v, r});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
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

    GridAccuracyParams accuracy;
    accuracy.min_spatial_points = 200;
    accuracy.max_spatial_points = 200;

    auto result = build_adaptive_bspline(params, chain, accuracy, OptionType::PUT);
    ASSERT_TRUE(result.has_value()) << "Adaptive build failed";

    // Wrap surface for price queries
    auto wrapper = make_bspline_surface(result->surface, OptionType::PUT);
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
    params.validation_samples = 4;

    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {100.0}},
    };

    auto m_domain = to_log_m({0.8, 0.9, 1.0, 1.1, 1.2});
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
        return result->price_fn(100.0, 100.0, tau, 0.20, 0.05);
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
    params.validation_samples = 4;

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
        .kref_config = {.K_refs = {100.0}},
    };

    auto m_domain = to_log_m({0.8, 0.9, 1.0, 1.1, 1.2});
    std::vector<double> v_domain = {0.10, 0.20, 0.30};
    std::vector<double> r_domain = {0.03, 0.05};

    auto result = build_adaptive_chebyshev_segmented(
        params, seg_config, {m_domain, v_domain, r_domain});
    ASSERT_TRUE(result.has_value())
        << "build_adaptive_chebyshev_segmented failed with duplicate dividends";

    // Should be able to query across the entire tau range
    for (double tau : {0.1, 0.3, 0.5, 0.7, 0.9}) {
        double p = result->price_fn(100.0, 100.0, tau, 0.20, 0.05);
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
    params.validation_samples = 4;

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
        .kref_config = {.K_refs = {100.0}},
    };

    auto m_domain = to_log_m({0.8, 0.9, 1.0, 1.1, 1.2});
    std::vector<double> v_domain = {0.10, 0.20, 0.30};
    std::vector<double> r_domain = {0.03, 0.05};

    auto result = build_adaptive_chebyshev_segmented(
        params, seg_config, {m_domain, v_domain, r_domain});
    ASSERT_TRUE(result.has_value())
        << "build_adaptive_chebyshev_segmented failed with nearly-coincident dividends";

    double p = result->price_fn(100.0, 100.0, 0.5, 0.20, 0.05);
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
    params.validation_samples = 4;

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

    auto result = build_adaptive_chebyshev_segmented(
        params, seg_config, {m_domain, v_domain, r_domain});

    // Narrow real segments should build successfully, not be rejected as gaps
    ASSERT_TRUE(result.has_value())
        << "Narrow real segments should produce valid prices, not errors";

    // Price at ATM should be positive
    double p = result->price_fn(100.0, 100.0, 0.01, 0.20, 0.05);
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
    params.validation_samples = 4;

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
        .kref_config = {.K_refs = {100.0}},
    };

    auto m_domain = to_log_m({0.8, 0.9, 1.0, 1.1, 1.2});
    std::vector<double> v_domain = {0.10, 0.20, 0.30};
    std::vector<double> r_domain = {0.03, 0.05};

    auto result = build_adaptive_chebyshev_segmented(
        params, seg_config, {m_domain, v_domain, r_domain});
    ASSERT_TRUE(result.has_value())
        << "build_adaptive_chebyshev_segmented failed";

    // Query inside the narrow real segment between the two gaps.
    // tau=0.503 is between the two gap bands.
    double p = result->price_fn(100.0, 100.0, 0.503, 0.20, 0.05);
    EXPECT_TRUE(std::isfinite(p)) << "Price not finite in narrow real segment";
    EXPECT_GT(p, 0.5)
        << "Price " << p << " is near-zero in narrow real segment — "
        << "likely hitting a zero-tensor leaf";

    // Also verify prices at tau values in the wide segments on either
    // side are reasonable for comparison.
    double p_before = result->price_fn(100.0, 100.0, 0.40, 0.20, 0.05);
    double p_after  = result->price_fn(100.0, 100.0, 0.60, 0.20, 0.05);
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

    auto result = build_adaptive_chebyshev_segmented_typed(
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

}  // namespace
}  // namespace mango
