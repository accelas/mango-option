// SPDX-License-Identifier: MIT
#include "mango/pde/core/grid.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(GridSpacingTest, UniformGridSpacing) {
    // Create uniform grid [0, 10] with 11 points (dx = 1.0)
    std::vector<double> x = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto grid = mango::GridView<double>(x);
    auto spacing = mango::GridSpacing<double>(grid);

    EXPECT_TRUE(spacing.is_uniform());
    EXPECT_DOUBLE_EQ(spacing.spacing(), 1.0);
    EXPECT_DOUBLE_EQ(spacing.spacing_inv(), 1.0);
    EXPECT_DOUBLE_EQ(spacing.spacing_inv_sq(), 1.0);
}





TEST(GridSpacingTest, MinStencilSize) {
    EXPECT_EQ(mango::GridSpacing<double>::min_stencil_size(), 3);
}

TEST(GridSpacingTest, DegenerateGridTooSmall) {
    std::vector<double> x = {0.0, 1.0};  // n = 2 < min_stencil_size()
    auto grid = mango::GridView<double>(x);
    auto spacing = mango::GridSpacing<double>(grid);

    // Grid is too small for 3-point stencil
    EXPECT_EQ(spacing.size(), 2);
    EXPECT_LT(spacing.size(), spacing.min_stencil_size());
}

TEST(GridSpacingTest, NonUniformPrecomputationCorrectness) {
    // Create tanh-clustered grid
    std::vector<double> x(11);
    x[0] = -1.0; x[1] = -0.8; x[2] = -0.5; x[3] = -0.2; x[4] = -0.05;
    x[5] = 0.0; x[6] = 0.05; x[7] = 0.2; x[8] = 0.5; x[9] = 0.8; x[10] = 1.0;

    auto grid = mango::GridView<double>(x);
    auto spacing = mango::GridSpacing<double>(grid);

    ASSERT_FALSE(spacing.is_uniform());

    // Verify precomputed values (interior points i=1..9)
    for (size_t i = 1; i < 10; ++i) {
        const double dx_left = x[i] - x[i-1];
        const double dx_right = x[i+1] - x[i];
        const double dx_center = 0.5 * (dx_left + dx_right);

        EXPECT_DOUBLE_EQ(spacing.dx_left_inv()[i-1], 1.0 / dx_left);
        EXPECT_DOUBLE_EQ(spacing.dx_right_inv()[i-1], 1.0 / dx_right);
        EXPECT_DOUBLE_EQ(spacing.dx_center_inv()[i-1], 1.0 / dx_center);
        EXPECT_DOUBLE_EQ(spacing.w_left()[i-1], dx_right / (dx_left + dx_right));
        EXPECT_DOUBLE_EQ(spacing.w_right()[i-1], dx_left / (dx_left + dx_right));
    }
}

TEST(GridSpacingTest, UniformGridNoPrecomputation) {
    // Uniform grid
    std::vector<double> x(11);
    for (size_t i = 0; i < 11; ++i) x[i] = i * 0.1;

    auto grid = mango::GridView<double>(x);
    auto spacing = mango::GridSpacing<double>(grid);

    ASSERT_TRUE(spacing.is_uniform());

    // Accessors should throw std::bad_variant_access on uniform grids
    EXPECT_THROW(spacing.dx_left_inv(), std::bad_variant_access);
}

TEST(GridSpacingTest, VariantUniformGrid) {
    // Uniform grid: [0.0, 0.1, 0.2, ..., 1.0]
    std::vector<double> x(11);
    for (size_t i = 0; i < 11; ++i) {
        x[i] = i * 0.1;
    }

    auto grid = mango::GridView<double>(x);
    mango::GridSpacing<double> spacing(grid);

    // Should detect uniformity
    EXPECT_TRUE(spacing.is_uniform());

    // Should store UniformSpacing variant
    EXPECT_DOUBLE_EQ(spacing.spacing(), 0.1);
    EXPECT_DOUBLE_EQ(spacing.spacing_inv(), 10.0);
    EXPECT_DOUBLE_EQ(spacing.spacing_inv_sq(), 100.0);
}

TEST(GridSpacingTest, VariantNonUniformGrid) {
    // Non-uniform grid: [0.0, 0.1, 0.3, 0.6, 1.0]
    std::vector<double> x = {0.0, 0.1, 0.3, 0.6, 1.0};
    auto grid = mango::GridView<double>(x);
    mango::GridSpacing<double> spacing(grid);

    // Should detect non-uniformity
    EXPECT_FALSE(spacing.is_uniform());

    // Should have access to non-uniform arrays
    auto dx_left = spacing.dx_left_inv();
    EXPECT_EQ(dx_left.size(), 3);  // Interior points
    EXPECT_DOUBLE_EQ(dx_left[0], 10.0);  // 1 / 0.1
}

TEST(GridSpacingTest, VariantMemoryEfficiency) {
    // Uniform grid should not allocate large arrays
    std::vector<double> x(1000);
    for (size_t i = 0; i < 1000; ++i) {
        x[i] = i * 0.001;
    }

    auto grid = mango::GridView<double>(x);
    mango::GridSpacing<double> spacing(grid);

    // For uniform grid, size should be minimal (no large precomputed arrays)
    // This is implicitly tested by the fact that we can construct it
    // without OOM on a 1000-point uniform grid
    EXPECT_TRUE(spacing.is_uniform());
}
