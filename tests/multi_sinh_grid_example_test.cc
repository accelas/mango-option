// SPDX-License-Identifier: MIT
/**
 * @file multi_sinh_grid_example_test.cc
 * @brief Tests for multi-sinh grid generation (converted from example_multi_sinh_grid.cc)
 *
 * Validates:
 * - Single cluster multi-sinh grid generation
 * - Dual cluster grids (ATM + ITM)
 * - Triple cluster grids (deep ITM + ATM + OTM)
 * - Grid spacing properties (concentration near cluster centers)
 */

#include "mango/pde/core/grid.hpp"
#include <gtest/gtest.h>
#include <limits>

using namespace mango;

TEST(MultiSinhGridTest, SingleCluster) {
    std::vector<MultiSinhCluster<double>> clusters = {
        {.center_x = 0.0, .alpha = 2.0, .weight = 1.0}
    };

    auto result = GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 11, clusters);
    ASSERT_TRUE(result.has_value());

    auto grid = result.value().generate();
    EXPECT_EQ(grid.size(), 11u);
    EXPECT_DOUBLE_EQ(grid[0], -3.0);
    EXPECT_DOUBLE_EQ(grid[10], 3.0);

    // Grid should be monotonically increasing
    for (size_t i = 1; i < grid.size(); ++i) {
        EXPECT_GT(grid[i], grid[i-1]);
    }
}

TEST(MultiSinhGridTest, DualClusters) {
    std::vector<MultiSinhCluster<double>> clusters = {
        {.center_x = 0.0, .alpha = 2.5, .weight = 2.0},
        {.center_x = -0.2, .alpha = 2.0, .weight = 1.0}
    };

    auto result = GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 21, clusters);
    ASSERT_TRUE(result.has_value());

    auto grid = result.value().generate();
    EXPECT_EQ(grid.size(), 21u);

    // Monotonically increasing
    for (size_t i = 1; i < grid.size(); ++i) {
        EXPECT_GT(grid[i], grid[i-1]);
    }

    // Spacing near cluster centers should be smaller than at boundaries
    // Find spacing near center (x~0) vs near boundary (x~3)
    double min_dx = std::numeric_limits<double>::max();
    double max_dx = 0.0;
    for (size_t i = 1; i < grid.size(); ++i) {
        double dx = grid[i] - grid[i-1];
        min_dx = std::min(min_dx, dx);
        max_dx = std::max(max_dx, dx);
    }
    // Concentration ratio should be non-trivial
    EXPECT_GT(max_dx / min_dx, 1.5);
}

TEST(MultiSinhGridTest, TripleClusters) {
    std::vector<MultiSinhCluster<double>> clusters = {
        {.center_x = -1.5, .alpha = 1.8, .weight = 1.0},
        {.center_x = 0.0, .alpha = 2.5, .weight = 2.0},
        {.center_x = 1.5, .alpha = 1.8, .weight = 1.0}
    };

    auto result = GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 31, clusters);
    ASSERT_TRUE(result.has_value());

    auto grid = result.value().generate();
    EXPECT_EQ(grid.size(), 31u);
    EXPECT_DOUBLE_EQ(grid[0], -3.0);
    EXPECT_DOUBLE_EQ(grid[30], 3.0);

    // Spacing statistics
    double min_dx = std::numeric_limits<double>::max();
    double max_dx = 0.0;
    for (size_t i = 1; i < grid.size(); ++i) {
        double dx = grid[i] - grid[i-1];
        min_dx = std::min(min_dx, dx);
        max_dx = std::max(max_dx, dx);
    }

    EXPECT_GT(min_dx, 0.0);
    EXPECT_GT(max_dx / min_dx, 2.0);  // Non-trivial concentration
}

TEST(MultiSinhGridTest, SingleClusterSufficesForNearbyStrikes) {
    // When strikes differ by only a few percent, single cluster works
    std::vector<MultiSinhCluster<double>> clusters = {
        {.center_x = 0.0, .alpha = 2.0, .weight = 1.0}
    };

    auto result = GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 21, clusters);
    ASSERT_TRUE(result.has_value());

    auto grid = result.value().generate();
    EXPECT_EQ(grid.size(), 21u);

    // Grid should still have concentration at center
    // Middle spacing should be smaller than boundary spacing
    double center_dx = grid[11] - grid[10];
    double boundary_dx = grid[1] - grid[0];
    EXPECT_LT(center_dx, boundary_dx);
}
