// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/bspline/bspline_pde_cache.hpp"
#include "mango/pde/core/grid.hpp"
#include "mango/pde/core/time_domain.hpp"
#include "mango/option/option_spec.hpp"

namespace mango {
namespace {

// Helper to create a minimal AmericanOptionResult for testing
std::shared_ptr<AmericanOptionResult> make_mock_result(double sigma, double rate) {
    // Create minimal grid spec
    auto grid_spec = GridSpec<double>::uniform(-3.0, 3.0, 11).value();
    auto time_domain = TimeDomain::from_n_steps(0.0, 1.0, 100);

    // Create grid using Grid::create which returns shared_ptr
    auto grid = Grid<double>::create(grid_spec, time_domain).value();

    // Fill with some values
    auto solution = grid->solution();
    for (size_t i = 0; i < solution.size(); ++i) {
        solution[i] = static_cast<double>(i);
    }

    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = rate, .option_type = OptionType::PUT}, sigma);

    return std::make_shared<AmericanOptionResult>(grid, params);
}

TEST(BSplinePDECacheTest, AddAndRetrieve) {
    BSplinePDECache cache;
    auto result = make_mock_result(0.20, 0.05);

    cache.add(0.20, 0.05, result);

    auto retrieved = cache.get(0.20, 0.05);
    EXPECT_NE(retrieved, nullptr);
    EXPECT_DOUBLE_EQ(retrieved->volatility(), 0.20);
}

TEST(BSplinePDECacheTest, MissingKeyReturnsNullptr) {
    BSplinePDECache cache;
    auto retrieved = cache.get(0.30, 0.04);
    EXPECT_EQ(retrieved, nullptr);
}

TEST(BSplinePDECacheTest, InvalidateOnTauChange) {
    BSplinePDECache cache;
    auto result = make_mock_result(0.20, 0.05);

    cache.set_tau_grid({0.1, 0.5, 1.0});
    cache.add(0.20, 0.05, result);

    // Same tau grid - should still have result
    cache.invalidate_if_tau_changed({0.1, 0.5, 1.0});
    EXPECT_NE(cache.get(0.20, 0.05), nullptr);

    // Different tau grid - should invalidate
    cache.invalidate_if_tau_changed({0.1, 0.25, 0.5, 1.0});
    EXPECT_EQ(cache.get(0.20, 0.05), nullptr);
}

TEST(BSplinePDECacheTest, GetMissingPairs) {
    BSplinePDECache cache;
    cache.add(0.20, 0.05, make_mock_result(0.20, 0.05));
    cache.add(0.30, 0.05, make_mock_result(0.30, 0.05));

    std::vector<std::pair<double, double>> all_pairs = {
        {0.20, 0.05},  // exists
        {0.30, 0.05},  // exists
        {0.25, 0.05},  // missing
        {0.20, 0.04},  // missing
    };

    auto missing = cache.get_missing_pairs(all_pairs);
    EXPECT_EQ(missing.size(), 2);
}

TEST(BSplinePDECacheTest, GetMissingIndices) {
    BSplinePDECache cache;
    cache.add(0.20, 0.05, make_mock_result(0.20, 0.05));

    std::vector<std::pair<double, double>> all_pairs = {
        {0.20, 0.05},  // exists - index 0
        {0.25, 0.05},  // missing - index 1
        {0.30, 0.05},  // missing - index 2
    };

    auto missing_indices = cache.get_missing_indices(all_pairs);
    ASSERT_EQ(missing_indices.size(), 2);
    EXPECT_EQ(missing_indices[0], 1);
    EXPECT_EQ(missing_indices[1], 2);
}

TEST(BSplinePDECacheTest, ContainsMethod) {
    BSplinePDECache cache;
    cache.add(0.20, 0.05, make_mock_result(0.20, 0.05));

    EXPECT_TRUE(cache.contains(0.20, 0.05));
    EXPECT_FALSE(cache.contains(0.30, 0.05));
}

TEST(BSplinePDECacheTest, SizeAndClear) {
    BSplinePDECache cache;
    EXPECT_EQ(cache.size(), 0);

    cache.add(0.20, 0.05, make_mock_result(0.20, 0.05));
    cache.add(0.30, 0.05, make_mock_result(0.30, 0.05));
    EXPECT_EQ(cache.size(), 2);

    cache.clear();
    EXPECT_EQ(cache.size(), 0);
    EXPECT_EQ(cache.get(0.20, 0.05), nullptr);
}

TEST(BSplinePDECacheTest, FloatingPointKeyRounding) {
    BSplinePDECache cache;
    cache.add(0.200000001, 0.050000001, make_mock_result(0.20, 0.05));

    // Should find even with slight floating point differences
    auto retrieved = cache.get(0.199999999, 0.049999999);
    EXPECT_NE(retrieved, nullptr);
}

}  // namespace
}  // namespace mango
