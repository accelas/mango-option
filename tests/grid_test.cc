#include "src/cpp/grid.hpp"
#include <gtest/gtest.h>

TEST(GridSpecTest, UniformGridGeneration) {
    auto spec = mango::GridSpec<>::uniform(0.0, 1.0, 11);
    auto grid = spec.generate();

    EXPECT_EQ(grid.size(), 11);
    EXPECT_DOUBLE_EQ(grid[0], 0.0);
    EXPECT_DOUBLE_EQ(grid[10], 1.0);
    EXPECT_DOUBLE_EQ(grid[5], 0.5);  // Check midpoint
}

TEST(GridSpecTest, UniformGridSpacing) {
    auto spec = mango::GridSpec<>::uniform(0.0, 10.0, 6);
    auto grid = spec.generate();

    // Points should be: 0, 2, 4, 6, 8, 10
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(grid[i], static_cast<double>(i * 2));
    }
}

TEST(GridSpecTest, LogSpacedGridGeneration) {
    auto spec = mango::GridSpec<>::log_spaced(1.0, 100.0, 5);
    auto grid = spec.generate();

    EXPECT_EQ(grid.size(), 5);
    EXPECT_DOUBLE_EQ(grid[0], 1.0);
    EXPECT_DOUBLE_EQ(grid[4], 100.0);
    // Geometric spacing: midpoint in log space should be sqrt(1*100) = 10
    EXPECT_NEAR(grid[2], 10.0, 1e-10);
}

TEST(GridSpecTest, SinhSpacedGridGeneration) {
    auto spec = mango::GridSpec<>::sinh_spaced(0.0, 1.0, 11, 2.0);
    auto grid = spec.generate();

    EXPECT_EQ(grid.size(), 11);
    EXPECT_DOUBLE_EQ(grid[0], 0.0);
    EXPECT_DOUBLE_EQ(grid[10], 1.0);
    EXPECT_DOUBLE_EQ(grid[5], 0.5);  // Center point should be at midpoint
}
