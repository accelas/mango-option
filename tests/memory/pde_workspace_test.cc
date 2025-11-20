#include "src/pde/core/pde_workspace.hpp"
#include "src/pde/core/grid.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <memory_resource>

TEST(PDEWorkspaceTest, BasicConstruction) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::PDEWorkspace workspace(101, grid.span(), std::pmr::get_default_resource());

    EXPECT_EQ(workspace.logical_size(), 101);
    EXPECT_EQ(workspace.padded_size(), 104);  // Rounded to 8

    // All array accessors should return valid spans
    EXPECT_EQ(workspace.u_current().size(), 101);
    EXPECT_EQ(workspace.u_next().size(), 101);
    EXPECT_EQ(workspace.u_stage().size(), 101);
    EXPECT_EQ(workspace.rhs().size(), 101);
    EXPECT_EQ(workspace.lu().size(), 101);
    EXPECT_EQ(workspace.psi_buffer().size(), 101);
}

TEST(PDEWorkspaceTest, PaddedAccessors) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::PDEWorkspace workspace(101, grid.span(), std::pmr::get_default_resource());

    // After revert, there are no separate _padded() accessors
    // The padding is internal - accessors return logical size only
    EXPECT_EQ(workspace.u_current().size(), 101);
    EXPECT_EQ(workspace.u_next().size(), 101);
    EXPECT_EQ(workspace.lu().size(), 101);

    // Padded size is accessible but padding is internal implementation detail
    EXPECT_EQ(workspace.padded_size(), 104);
}

TEST(PDEWorkspaceTest, GridSpacing) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 10.0, 6);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();  // 0, 2, 4, 6, 8, 10

    mango::PDEWorkspace workspace(6, grid.span(), std::pmr::get_default_resource());

    auto dx = workspace.dx();
    EXPECT_EQ(dx.size(), 5);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(dx[i], 2.0);
    }

    // After revert, there's no dx_padded() accessor
    // Padding is internal implementation detail
}

TEST(PDEWorkspaceTest, ArraysAreIndependent) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 10);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::PDEWorkspace workspace(10, grid.span(), std::pmr::get_default_resource());

    // Write to one array
    auto u = workspace.u_current();
    std::fill(u.begin(), u.end(), 1.0);

    // Other arrays should remain zero
    auto v = workspace.u_next();
    EXPECT_DOUBLE_EQ(v[0], 0.0);
}

// After revert: PDEWorkspace doesn't have reset() method
// TEST(PDEWorkspaceTest, ResetInvalidatesSpans) {
//     auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 10);
//     ASSERT_TRUE(grid_result.has_value());
//     auto grid = grid_result.value().generate();
//
//     mango::PDEWorkspace workspace(10, grid.span(), std::pmr::get_default_resource());
//
//     auto u_before = workspace.u_current();
//     u_before[0] = 999.0;
//
//     workspace.reset();
//
//     // Must re-acquire span after reset
//     auto u_after = workspace.u_current();
//     EXPECT_DOUBLE_EQ(u_after[0], 0.0);  // Freshly allocated
// }

