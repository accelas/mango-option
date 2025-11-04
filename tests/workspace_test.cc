#include "src/cpp/workspace.hpp"
#include "src/cpp/grid.hpp"
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
