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

TEST(GridSpacingViewTest, LifetimeSafety) {
    // Test that GridSpacing works after source vectors are destroyed
    GridSpacing spacing = []() {
        std::vector<double> grid = {0.0, 0.05, 0.15, 0.3};
        std::vector<double> dx = {0.05, 0.10, 0.15};

        auto result = GridSpacing::create(grid, dx);
        EXPECT_TRUE(result.has_value());

        return result.value();
        // grid and dx go out of scope here
    }();

    // GridSpacing should still be valid because it owns its data
    EXPECT_FALSE(spacing.is_uniform());

    // Verify we can access the precomputed arrays
    auto dx_left = spacing.dx_left_inv();
    auto dx_right = spacing.dx_right_inv();
    auto dx_center = spacing.dx_center_inv();
    auto w_left = spacing.w_left();
    auto w_right = spacing.w_right();

    EXPECT_EQ(dx_left.size(), 2);
    EXPECT_EQ(dx_right.size(), 2);
    EXPECT_EQ(dx_center.size(), 2);
    EXPECT_EQ(w_left.size(), 2);
    EXPECT_EQ(w_right.size(), 2);

    // Verify values are correct
    EXPECT_NEAR(dx_left[0], 1.0 / 0.05, 1e-14);
    EXPECT_NEAR(dx_left[1], 1.0 / 0.10, 1e-14);
    EXPECT_NEAR(dx_right[0], 1.0 / 0.10, 1e-14);
    EXPECT_NEAR(dx_right[1], 1.0 / 0.15, 1e-14);
}

}  // namespace
}  // namespace mango
