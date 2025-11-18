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

TEST(NonUniformSpacingTest, ConstructionAndPrecomputation) {
    // Simple non-uniform grid: [0.0, 0.1, 0.3, 0.6, 1.0]
    std::vector<double> x = {0.0, 0.1, 0.3, 0.6, 1.0};
    mango::NonUniformSpacing<double> spacing(x);

    EXPECT_EQ(spacing.n, 5);
    EXPECT_EQ(spacing.dx_left_inv().size(), 3);  // Interior points: 1, 2, 3
    EXPECT_EQ(spacing.dx_right_inv().size(), 3);

    // Verify first interior point (i=1): left=0.1, right=0.2
    auto dx_left = spacing.dx_left_inv();
    auto dx_right = spacing.dx_right_inv();

    EXPECT_DOUBLE_EQ(dx_left[0], 10.0);   // 1 / 0.1
    EXPECT_DOUBLE_EQ(dx_right[0], 5.0);   // 1 / 0.2
}

TEST(NonUniformSpacingTest, ZeroCopySpanAccess) {
    std::vector<double> x = {0.0, 0.5, 1.5, 3.0, 5.0};
    mango::NonUniformSpacing<double> spacing(x);

    // Spans should point into precomputed buffer (zero-copy)
    auto dx_left = spacing.dx_left_inv();
    auto dx_right = spacing.dx_right_inv();
    auto dx_center = spacing.dx_center_inv();
    auto w_left = spacing.w_left();
    auto w_right = spacing.w_right();

    // All should have size = n-2 = 3
    EXPECT_EQ(dx_left.size(), 3);
    EXPECT_EQ(dx_right.size(), 3);
    EXPECT_EQ(dx_center.size(), 3);
    EXPECT_EQ(w_left.size(), 3);
    EXPECT_EQ(w_right.size(), 3);

    // Verify pointers are into same underlying buffer (contiguous)
    const double* base = dx_left.data();
    EXPECT_EQ(dx_right.data(), base + 3);
    EXPECT_EQ(dx_center.data(), base + 6);
    EXPECT_EQ(w_left.data(), base + 9);
    EXPECT_EQ(w_right.data(), base + 12);
}
