#include "src/pde/memory/workspace_base.hpp"
#include <gtest/gtest.h>

TEST(WorkspaceBaseTest, TileMetadataGeneration) {
    // Divide 100 elements into 3 tiles: 34, 33, 33
    auto tile0 = mango::WorkspaceBase::tile_info(100, 0, 3);
    EXPECT_EQ(tile0.tile_start, 0);
    EXPECT_EQ(tile0.tile_size, 34);
    EXPECT_EQ(tile0.padded_size, 40);  // Rounded to SIMD_WIDTH=8
    EXPECT_EQ(tile0.alignment, 64);

    auto tile1 = mango::WorkspaceBase::tile_info(100, 1, 3);
    EXPECT_EQ(tile1.tile_start, 34);
    EXPECT_EQ(tile1.tile_size, 33);
    EXPECT_EQ(tile1.padded_size, 40);

    auto tile2 = mango::WorkspaceBase::tile_info(100, 2, 3);
    EXPECT_EQ(tile2.tile_start, 67);
    EXPECT_EQ(tile2.tile_size, 33);
    EXPECT_EQ(tile2.padded_size, 40);
}

TEST(WorkspaceBaseTest, SIMDPadding) {
    EXPECT_EQ(mango::WorkspaceBase::pad_to_simd(1), 8);
    EXPECT_EQ(mango::WorkspaceBase::pad_to_simd(8), 8);
    EXPECT_EQ(mango::WorkspaceBase::pad_to_simd(9), 16);
    EXPECT_EQ(mango::WorkspaceBase::pad_to_simd(16), 16);
    EXPECT_EQ(mango::WorkspaceBase::pad_to_simd(17), 24);
}

TEST(WorkspaceBaseTest, TileInfoBoundsChecking) {
    // Debug mode should assert on invalid inputs
    #ifndef NDEBUG
    EXPECT_DEATH(mango::WorkspaceBase::tile_info(100, 0, 0), "num_tiles must be positive");
    EXPECT_DEATH(mango::WorkspaceBase::tile_info(100, 5, 3), "tile_idx out of bounds");
    #endif
}

TEST(WorkspaceBaseTest, BytesAllocatedTracking) {
    mango::WorkspaceBase workspace(1024);
    EXPECT_EQ(workspace.bytes_allocated(), 0);
}
