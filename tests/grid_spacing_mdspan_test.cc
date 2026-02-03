// SPDX-License-Identifier: MIT
#include "mango/pde/core/grid.hpp"
#include <gtest/gtest.h>

namespace mango {
namespace {

TEST(GridSpacingMdspan, NonUniformSectionView) {
    // Create non-uniform grid
    std::vector<double> x{0.0, 0.1, 0.3, 0.7, 1.0};
    auto grid = GridView<double>(x);
    auto spacing = GridSpacing<double>(grid);

    ASSERT_FALSE(spacing.is_uniform());

    // Access sections via span (existing API)
    auto dx_left = spacing.dx_left_inv();
    auto dx_right = spacing.dx_right_inv();

    EXPECT_EQ(dx_left.size(), 3);  // n - 2 = 5 - 2
    EXPECT_EQ(dx_right.size(), 3);

    // Values should match manual calculation
    EXPECT_NEAR(dx_left[0], 1.0 / 0.1, 1e-10);
    EXPECT_NEAR(dx_right[0], 1.0 / 0.2, 1e-10);
}

TEST(GridSpacingMdspan, SectionLayout) {
    std::vector<double> x{0.0, 1.0, 3.0, 4.0};
    auto grid = GridView<double>(x);
    auto spacing = GridSpacing<double>(grid);

    // Non-uniform spacing has 5 sections
    // Each section has (n-2) elements
    auto dx_left = spacing.dx_left_inv();
    auto w_left = spacing.w_left();

    EXPECT_EQ(dx_left.size(), 2);  // n-2 = 4-2
    EXPECT_EQ(w_left.size(), 2);
}

}  // namespace
}  // namespace mango
