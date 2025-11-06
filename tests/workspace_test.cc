#include "src/workspace.hpp"
#include "src/grid.hpp"
#include <gtest/gtest.h>

TEST(WorkspaceStorageTest, SmallGridSingleBlock) {
    auto spec = mango::GridSpec<>::uniform(0.0, 1.0, 100);
    auto grid = spec.generate();

    mango::WorkspaceStorage workspace(grid.size(), grid.span());

    // Small grid should use single block
    EXPECT_EQ(workspace.cache_config().n_blocks, 1);
    EXPECT_EQ(workspace.cache_config().block_size, 100);
    EXPECT_EQ(workspace.cache_config().overlap, 1);
}

TEST(WorkspaceStorageTest, LargeGridMultipleBlocks) {
    auto spec = mango::GridSpec<>::uniform(0.0, 100.0, 10000);
    auto grid = spec.generate();

    mango::WorkspaceStorage workspace(grid.size(), grid.span());

    // Large grid should use L1-blocked strategy
    EXPECT_GT(workspace.cache_config().n_blocks, 1);
    EXPECT_LT(workspace.cache_config().block_size, 10000);
}

TEST(WorkspaceStorageTest, PreComputedDxArray) {
    auto spec = mango::GridSpec<>::uniform(0.0, 10.0, 6);
    auto grid = spec.generate();  // Points: 0, 2, 4, 6, 8, 10

    mango::WorkspaceStorage workspace(grid.size(), grid.span());

    // dx array should have size n-1
    EXPECT_EQ(workspace.dx().size(), 5);

    // All dx values should be 2.0 for uniform grid
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(workspace.dx()[i], 2.0);
    }
}

TEST(WorkspaceStorageTest, GetBlockInterior) {
    auto spec = mango::GridSpec<>::uniform(0.0, 10.0, 20);
    auto grid = spec.generate();

    mango::WorkspaceStorage workspace(grid.size(), grid.span());
    workspace.cache_config() = mango::CacheBlockConfig{10, 2, 1};  // Force 2 blocks for testing

    // Block 0: indices 0-9, interior 1-9 (skip boundary at 0)
    auto [start, end] = workspace.get_block_interior_range(0);
    EXPECT_EQ(start, 1);
    EXPECT_EQ(end, 10);

    // Block 1: indices 10-19, interior 10-18 (skip boundary at 19)
    auto [start2, end2] = workspace.get_block_interior_range(1);
    EXPECT_EQ(start2, 10);
    EXPECT_EQ(end2, 19);
}

TEST(WorkspaceStorageTest, GetBlockWithHalo) {
    auto spec = mango::GridSpec<>::uniform(0.0, 1.0, 20);
    auto grid = spec.generate();

    mango::WorkspaceStorage workspace(grid.size(), grid.span());
    workspace.cache_config() = mango::CacheBlockConfig{10, 2, 1};  // Force 2 blocks for testing

    // Initialize u_current with index values for testing
    for (size_t i = 0; i < 20; ++i) {
        workspace.u_current()[i] = static_cast<double>(i);
    }

    // Block with halo should include overlap points
    auto block_info = workspace.get_block_with_halo(workspace.u_current(), 1);

    EXPECT_GT(block_info.halo_left, 0);
    EXPECT_EQ(block_info.data[block_info.halo_left], workspace.u_current()[block_info.interior_start]);
}

TEST(WorkspaceStorageTest, ThresholdControlsBlocking) {
    // Grid below threshold: single block
    auto spec_small = mango::GridSpec<>::uniform(0.0, 1.0, 100);
    auto small_grid = spec_small.generate();

    mango::WorkspaceStorage ws_small(small_grid.size(), small_grid.span(), 5000);
    EXPECT_EQ(ws_small.cache_config().n_blocks, 1);
    EXPECT_EQ(ws_small.cache_config().overlap, 1);

    // Grid above threshold: multiple blocks
    auto spec_large = mango::GridSpec<>::uniform(0.0, 1.0, 5001);
    auto large_grid = spec_large.generate();

    mango::WorkspaceStorage ws_large(large_grid.size(), large_grid.span(), 5000);
    EXPECT_GT(ws_large.cache_config().n_blocks, 1);
    EXPECT_EQ(ws_large.cache_config().overlap, 1);
}

TEST(WorkspaceStorageTest, PsiBufferAvailable) {
    const size_t n = 100;
    auto spec = mango::GridSpec<>::uniform(0.0, 1.0, n);
    auto grid = spec.generate();

    mango::WorkspaceStorage ws(grid.size(), grid.span());

    auto psi = ws.psi_buffer();
    EXPECT_EQ(psi.size(), n);

    // Verify writable
    for (size_t i = 0; i < n; ++i) {
        psi[i] = static_cast<double>(i);
    }

    // Verify no overlap with other buffers
    auto u_current = ws.u_current();
    u_current[0] = 999.0;
    EXPECT_NE(psi[0], 999.0);
}
