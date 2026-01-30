// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/table/adaptive_grid_builder.hpp"
#include "src/option/american_option_batch.hpp"
#include <algorithm>
#include <iostream>

namespace mango {
namespace {

// Helper to create a dummy AmericanOptionResult for cache testing
std::shared_ptr<AmericanOptionResult> make_dummy_result() {
    PricingParams params;
    params.spot = 100.0;
    params.strike = 100.0;
    params.maturity = 1.0;
    params.volatility = 0.20;
    params.rate = 0.05;
    params.dividend_yield = 0.0;
    params.type = OptionType::PUT;

    auto result = solve_american_option_auto(params);
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
    params.max_iterations = 3;

    AdaptiveGridBuilder builder(params);
    SUCCEED();
}

TEST(AdaptiveGridBuilderTest, BuildsWithSyntheticChain) {
    // Create a minimal synthetic chain
    OptionChain chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.0;

    // Add strikes and maturities
    chain.strikes = {90.0, 95.0, 100.0, 105.0, 110.0};
    chain.maturities = {0.25, 0.5, 1.0};
    chain.implied_vols = {0.18, 0.20, 0.22};  // Some variation
    chain.rates = {0.04, 0.05, 0.06};

    AdaptiveGridParams params;
    params.target_iv_error = 0.002;  // 20 bps - relaxed for test speed
    params.max_iterations = 2;
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

    OptionChain chain;
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
    OptionChain chain;
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
    params.max_iterations = 1;
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
    OptionChain chain1;
    chain1.spot = 100.0;
    chain1.dividend_yield = 0.0;
    chain1.strikes = {90.0, 100.0, 110.0};
    chain1.maturities = {0.25, 0.5, 1.0};
    chain1.implied_vols = {0.18, 0.22};
    chain1.rates = {0.04, 0.05};

    OptionChain chain2 = chain1;
    chain2.spot = 50.0;  // Different spot => cache must not reuse chain1 slices

    AdaptiveGridParams params;
    params.max_iterations = 1;
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

}  // namespace
}  // namespace mango
