#include "src/pde/memory/pde_workspace.hpp"
#include "src/pde/core/grid.hpp"
#include <gtest/gtest.h>
#include <algorithm>

TEST(PDEWorkspaceTest, BasicConstruction) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::PDEWorkspace workspace(101, grid.span());

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

    mango::PDEWorkspace workspace(101, grid.span());

    EXPECT_EQ(workspace.u_current_padded().size(), 104);
    EXPECT_EQ(workspace.u_next_padded().size(), 104);
    EXPECT_EQ(workspace.lu_padded().size(), 104);

    // Padding should be zero-initialized
    auto u_padded = workspace.u_current_padded();
    EXPECT_DOUBLE_EQ(u_padded[101], 0.0);
    EXPECT_DOUBLE_EQ(u_padded[102], 0.0);
    EXPECT_DOUBLE_EQ(u_padded[103], 0.0);
}

TEST(PDEWorkspaceTest, GridSpacing) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 10.0, 6);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();  // 0, 2, 4, 6, 8, 10

    mango::PDEWorkspace workspace(6, grid.span());

    auto dx = workspace.dx();
    EXPECT_EQ(dx.size(), 5);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(dx[i], 2.0);
    }

    // Padded dx (size 5 → 8)
    auto dx_padded = workspace.dx_padded();
    EXPECT_EQ(dx_padded.size(), 8);
    EXPECT_DOUBLE_EQ(dx_padded[5], 0.0);  // Zero-padded tail
}

TEST(PDEWorkspaceTest, ArraysAreIndependent) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 10);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::PDEWorkspace workspace(10, grid.span());

    // Write to one array
    auto u = workspace.u_current();
    std::fill(u.begin(), u.end(), 1.0);

    // Other arrays should remain zero
    auto v = workspace.u_next();
    EXPECT_DOUBLE_EQ(v[0], 0.0);
}

TEST(PDEWorkspaceTest, ResetInvalidatesSpans) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 10);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::PDEWorkspace workspace(10, grid.span());

    auto u_before = workspace.u_current();
    u_before[0] = 999.0;

    workspace.reset();

    // Must re-acquire span after reset
    auto u_after = workspace.u_current();
    EXPECT_DOUBLE_EQ(u_after[0], 0.0);  // Freshly allocated
}

TEST(PDEWorkspaceTest, TileMetadata) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 100);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::PDEWorkspace workspace(100, grid.span());

    // 100 elements into 3 tiles: 34, 33, 33
    auto tile0 = workspace.tile_info(0, 3);
    EXPECT_EQ(tile0.tile_start, 0);
    EXPECT_EQ(tile0.tile_size, 34);
}
