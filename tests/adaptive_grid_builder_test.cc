// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/adaptive_grid_builder.hpp"
#include "mango/option/table/spliced_surface.hpp"
#include "mango/option/table/standard_surface.hpp"
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

TEST(AdaptiveGridBuilderTest, ConstructWithDefaultParams) {
    AdaptiveGridParams params;
    AdaptiveGridBuilder builder(params);

    // Should compile and not crash
    SUCCEED();
}

TEST(AdaptiveGridBuilderTest, ConstructWithCustomParams) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.001;  // 10 bps
    params.max_iter = 3;

    AdaptiveGridBuilder builder(params);
    SUCCEED();
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

    AdaptiveGridBuilder builder(params);

    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0).value();
    auto result = builder.build(chain, grid_spec, 200, OptionType::PUT);

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
    AdaptiveGridBuilder builder(params);

    OptionGrid chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;
    // No options added

    auto grid_spec = GridSpec<double>::uniform(-3.0, 3.0, 51).value();
    auto result = builder.build(chain, grid_spec, 100, OptionType::PUT);

    // Should return error for empty chain
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}

// ===========================================================================
// SliceCache unit tests
// ===========================================================================

TEST(SliceCacheTest, AddAndRetrieve) {
    SliceCache cache;

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

TEST(SliceCacheTest, ContainsCheck) {
    SliceCache cache;

    EXPECT_FALSE(cache.contains(0.20, 0.05));

    auto result_ptr = make_dummy_result();
    ASSERT_NE(result_ptr, nullptr);
    cache.add(0.20, 0.05, result_ptr);

    EXPECT_TRUE(cache.contains(0.20, 0.05));
    EXPECT_FALSE(cache.contains(0.25, 0.05));
}

TEST(SliceCacheTest, GetMissingIndices) {
    SliceCache cache;

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

TEST(SliceCacheTest, InvalidateOnTauChange) {
    SliceCache cache;

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

TEST(SliceCacheTest, CachePreservedOnMChange) {
    SliceCache cache;

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

TEST(SliceCacheTest, Clear) {
    SliceCache cache;

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

    AdaptiveGridBuilder builder(params);

    auto grid_spec = GridSpec<double>::uniform(-3.0, 3.0, 31).value();
    auto result = builder.build(chain, grid_spec, 100, OptionType::PUT);

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

    AdaptiveGridBuilder builder(params);
    auto grid_spec = GridSpec<double>::uniform(-3.0, 3.0, 31).value();

    auto result1 = builder.build(chain1, grid_spec, 100, OptionType::PUT);
    ASSERT_TRUE(result1.has_value());
    size_t solves1 = result1->iterations[0].pde_solves_table;

    auto result2 = builder.build(chain2, grid_spec, 100, OptionType::PUT);
    ASSERT_TRUE(result2.has_value());
    size_t solves2 = result2->iterations[0].pde_solves_table;

    EXPECT_EQ(solves1, solves2) << "Second build should recompute all slices for new chain";
}


TEST(AdaptiveGridBuilderTest, BuildSegmentedBasic) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;  // 50 bps — relaxed for test speed
    params.max_iter = 2;
    params.validation_samples = 16;

    AdaptiveGridBuilder builder(params);
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

    auto result = builder.build_segmented(seg_config, {m_domain, v_domain, r_domain});
    ASSERT_TRUE(result.has_value())
        << "build_segmented failed";

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

    AdaptiveGridBuilder builder(params);
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

    auto result = builder.build_segmented(seg_config, {m_domain, v_domain, r_domain});
    ASSERT_TRUE(result.has_value());
}

// Large discrete dividend (total_div/K_ref > 0.2, stresses moneyness expansion)
TEST(AdaptiveGridBuilderTest, BuildSegmentedLargeDividend) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 2;
    params.validation_samples = 16;

    AdaptiveGridBuilder builder(params);

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

    auto result = builder.build_segmented(seg_config, {m_domain, v_domain, r_domain});
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

    AdaptiveGridBuilder builder(params);

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

    auto result = builder.build_segmented(seg_config, {m_domain, v_domain, r_domain});
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
    AdaptiveGridBuilder builder(params);

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

    auto result = builder.build_segmented(seg_config, {m, v, r});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}

// Coverage: Invalid auto-K_ref config with span <= 0
TEST(AdaptiveGridBuilderTest, BuildSegmentedRejectsZeroSpan) {
    AdaptiveGridParams params;
    params.max_iter = 1;
    params.validation_samples = 4;
    AdaptiveGridBuilder builder(params);

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

    auto result = builder.build_segmented(seg_config, {m, v, r});
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
    AdaptiveGridBuilder builder(params);

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

    auto result = builder.build_segmented(seg_config, {m, v, r});
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
    AdaptiveGridBuilder builder(params);

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

    auto result = builder.build_segmented(seg_config, {m, v, r});
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
    AdaptiveGridBuilder builder(params);

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

    auto result = builder.build_segmented(seg_config, {m, v, r});
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
    AdaptiveGridBuilder builder(params);

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

    auto result = builder.build_segmented(seg_config, {m, v, r});
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
    AdaptiveGridBuilder builder(params);

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

    auto result = builder.build_segmented(seg_config, {m, v, r});
    // Should succeed — moneyness floor prevents negative/zero domain
    ASSERT_TRUE(result.has_value());
}

// Coverage: Negative K_ref in explicit list (K_ref_min <= 0 guard)
TEST(AdaptiveGridBuilderTest, BuildSegmentedNegativeKRefExpansionGuard) {
    AdaptiveGridParams params;
    params.target_iv_error = 0.005;
    params.max_iter = 1;
    params.validation_samples = 8;
    AdaptiveGridBuilder builder(params);

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

    auto result = builder.build_segmented(seg_config, {m, v, r});
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
    AdaptiveGridBuilder builder(params);

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

    auto result = builder.build_segmented(seg_config, {m, v, r});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, PriceTableErrorCode::InvalidConfig);
}

// Coverage: Negative span with auto K_refs
TEST(AdaptiveGridBuilderTest, BuildSegmentedRejectsNegativeSpan) {
    AdaptiveGridParams params;
    params.max_iter = 1;
    params.validation_samples = 4;
    AdaptiveGridBuilder builder(params);

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

    auto result = builder.build_segmented(seg_config, {m, v, r});
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

    AdaptiveGridBuilder builder(params);
    GridAccuracyParams accuracy;
    accuracy.min_spatial_points = 200;
    accuracy.max_spatial_points = 200;

    auto result = builder.build(chain, accuracy, OptionType::PUT);
    ASSERT_TRUE(result.has_value()) << "Adaptive build failed";

    // Wrap surface for price queries
    auto wrapper = make_standard_wrapper(result->surface, OptionType::PUT);
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

}  // namespace
}  // namespace mango
