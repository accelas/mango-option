#include "src/pde/core/grid_spacing_data.hpp"
#include <gtest/gtest.h>

TEST(UniformSpacingTest, ConstructionAndAccessors) {
    mango::UniformSpacing<double> spacing(0.1, 101);

    EXPECT_EQ(spacing.n, 101);
    EXPECT_DOUBLE_EQ(spacing.dx, 0.1);
    EXPECT_DOUBLE_EQ(spacing.dx_inv, 10.0);
    EXPECT_DOUBLE_EQ(spacing.dx_inv_sq, 100.0);
}

TEST(UniformSpacingTest, NegativeSpacingHandled) {
    // Should handle negative dx (reversed grid)
    mango::UniformSpacing<double> spacing(-0.05, 50);

    EXPECT_DOUBLE_EQ(spacing.dx, -0.05);
    EXPECT_DOUBLE_EQ(spacing.dx_inv, -20.0);
    EXPECT_DOUBLE_EQ(spacing.dx_inv_sq, 400.0);
}
