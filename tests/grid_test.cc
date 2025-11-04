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

TEST(GridViewTest, ViewFromBuffer) {
    auto spec = mango::GridSpec<>::uniform(0.0, 10.0, 6);
    auto grid = spec.generate();
    auto view = grid.view();

    EXPECT_EQ(view.size(), 6);
    EXPECT_DOUBLE_EQ(view[0], 0.0);
    EXPECT_DOUBLE_EQ(view[5], 10.0);
}

TEST(GridViewTest, ViewIsCheapToCopy) {
    auto spec = mango::GridSpec<>::uniform(0.0, 1.0, 11);
    auto grid = spec.generate();
    auto view1 = grid.view();
    auto view2 = view1;  // Copy view (cheap)

    EXPECT_EQ(view2.size(), 11);
    EXPECT_DOUBLE_EQ(view2[0], 0.0);
}

TEST(GridViewTest, ViewFromSpan) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto view = mango::GridView<>(std::span<const double>(data));

    EXPECT_EQ(view.size(), 5);
    EXPECT_DOUBLE_EQ(view[2], 3.0);
}
