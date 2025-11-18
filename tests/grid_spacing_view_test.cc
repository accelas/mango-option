#include <gtest/gtest.h>
#include "src/pde/core/grid_spacing.hpp"
#include <vector>

namespace mango {
namespace {

TEST(GridSpacingViewTest, CreateFromSpans) {
    std::vector<double> grid = {0.0, 0.1, 0.2, 0.3};
    std::vector<double> dx = {0.1, 0.1, 0.1};

    auto spacing = GridSpacing::create(grid, dx);
    ASSERT_TRUE(spacing.has_value());

    EXPECT_TRUE(spacing->is_uniform());
    EXPECT_NEAR(spacing->spacing(), 0.1, 1e-14);
}

TEST(GridSpacingViewTest, ValueTypeSemantics) {
    std::vector<double> grid = {0.0, 0.1, 0.2};
    std::vector<double> dx = {0.1, 0.1};

    auto spacing1 = GridSpacing::create(grid, dx).value();
    auto spacing2 = spacing1;  // Copy

    EXPECT_TRUE(spacing2.is_uniform());
    EXPECT_NEAR(spacing2.spacing(), 0.1, 1e-14);
}

TEST(GridSpacingViewTest, NonUniformSpacing) {
    std::vector<double> grid = {0.0, 0.05, 0.15, 0.3};
    std::vector<double> dx = {0.05, 0.10, 0.15};

    auto spacing = GridSpacing::create(grid, dx);
    ASSERT_TRUE(spacing.has_value());

    EXPECT_FALSE(spacing->is_uniform());
    auto dx_left = spacing->dx_left_inv();
    EXPECT_GT(dx_left.size(), 0);
}

}  // namespace
}  // namespace mango
