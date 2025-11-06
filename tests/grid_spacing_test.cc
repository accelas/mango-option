#include "src/operators/grid_spacing.hpp"
#include "src/grid.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(GridSpacingTest, UniformGridSpacing) {
    // Create uniform grid [0, 10] with 11 points (dx = 1.0)
    std::vector<double> x = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto grid = mango::GridView<double>(x);
    auto spacing = mango::operators::GridSpacing<double>(grid);

    EXPECT_TRUE(spacing.is_uniform());
    EXPECT_DOUBLE_EQ(spacing.spacing(), 1.0);
    EXPECT_DOUBLE_EQ(spacing.spacing_inv(), 1.0);
    EXPECT_DOUBLE_EQ(spacing.spacing_inv_sq(), 1.0);
}

TEST(GridSpacingTest, LeftSpacingUniform) {
    std::vector<double> x = {0, 1, 2, 3, 4, 5};
    auto grid = mango::GridView<double>(x);
    auto spacing = mango::operators::GridSpacing<double>(grid);

    // left_spacing(i) = x[i] - x[i-1]
    EXPECT_DOUBLE_EQ(spacing.left_spacing(1), 1.0);  // x[1] - x[0] = 1 - 0
    EXPECT_DOUBLE_EQ(spacing.left_spacing(3), 1.0);  // x[3] - x[2] = 3 - 2
    EXPECT_DOUBLE_EQ(spacing.left_spacing(5), 1.0);  // x[5] - x[4] = 5 - 4
}

TEST(GridSpacingTest, RightSpacingUniform) {
    std::vector<double> x = {0, 1, 2, 3, 4, 5};
    auto grid = mango::GridView<double>(x);
    auto spacing = mango::operators::GridSpacing<double>(grid);

    // right_spacing(i) = x[i+1] - x[i]
    EXPECT_DOUBLE_EQ(spacing.right_spacing(0), 1.0);  // x[1] - x[0] = 1 - 0
    EXPECT_DOUBLE_EQ(spacing.right_spacing(2), 1.0);  // x[3] - x[2] = 3 - 2
    EXPECT_DOUBLE_EQ(spacing.right_spacing(4), 1.0);  // x[5] - x[4] = 5 - 4
}

TEST(GridSpacingTest, LeftSpacingNonUniform) {
    // Non-uniform grid with variable spacing
    std::vector<double> x = {0.0, 0.5, 1.0, 2.0, 4.0};
    auto grid = mango::GridView<double>(x);
    auto spacing = mango::operators::GridSpacing<double>(grid);

    EXPECT_FALSE(spacing.is_uniform());
    EXPECT_DOUBLE_EQ(spacing.left_spacing(1), 0.5);  // x[1] - x[0] = 0.5 - 0.0
    EXPECT_DOUBLE_EQ(spacing.left_spacing(2), 0.5);  // x[2] - x[1] = 1.0 - 0.5
    EXPECT_DOUBLE_EQ(spacing.left_spacing(3), 1.0);  // x[3] - x[2] = 2.0 - 1.0
    EXPECT_DOUBLE_EQ(spacing.left_spacing(4), 2.0);  // x[4] - x[3] = 4.0 - 2.0
}

TEST(GridSpacingTest, RightSpacingNonUniform) {
    std::vector<double> x = {0.0, 0.5, 1.0, 2.0, 4.0};
    auto grid = mango::GridView<double>(x);
    auto spacing = mango::operators::GridSpacing<double>(grid);

    EXPECT_DOUBLE_EQ(spacing.right_spacing(0), 0.5);  // x[1] - x[0] = 0.5 - 0.0
    EXPECT_DOUBLE_EQ(spacing.right_spacing(1), 0.5);  // x[2] - x[1] = 1.0 - 0.5
    EXPECT_DOUBLE_EQ(spacing.right_spacing(2), 1.0);  // x[3] - x[2] = 2.0 - 1.0
    EXPECT_DOUBLE_EQ(spacing.right_spacing(3), 2.0);  // x[4] - x[3] = 4.0 - 2.0
}

TEST(GridSpacingTest, MinStencilSize) {
    EXPECT_EQ(mango::operators::GridSpacing<double>::min_stencil_size(), 3);
}

TEST(GridSpacingTest, DegenerateGridTooSmall) {
    std::vector<double> x = {0.0, 1.0};  // n = 2 < min_stencil_size()
    auto grid = mango::GridView<double>(x);
    auto spacing = mango::operators::GridSpacing<double>(grid);

    // Grid is too small for 3-point stencil
    EXPECT_EQ(spacing.size(), 2);
    EXPECT_LT(spacing.size(), spacing.min_stencil_size());
}
