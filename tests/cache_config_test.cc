#include "src/cpp/cache_config.hpp"
#include <gtest/gtest.h>

TEST(CacheBlockConfigTest, SmallGridSingleBlock) {
    auto config = mango::CacheBlockConfig::adaptive(100);

    EXPECT_EQ(config.n_blocks, 1);
    EXPECT_EQ(config.block_size, 100);
    EXPECT_EQ(config.overlap, 1);  // Still need halo for stencil
}

TEST(CacheBlockConfigTest, LargeGridMultipleBlocks) {
    auto config = mango::CacheBlockConfig::adaptive(10000);

    EXPECT_GT(config.n_blocks, 1);  // Should split into blocks
    EXPECT_LT(config.block_size, 10000);  // Blocks smaller than full grid
    EXPECT_EQ(config.overlap, 1);  // Stencil width
}

TEST(CacheBlockConfigTest, L1BlockingSize) {
    auto config = mango::CacheBlockConfig::l1_blocked(10000);

    // L1 cache: 32 KB / 24 bytes = ~1333 points
    // But should be reasonable (between 500 and 2000)
    EXPECT_GT(config.block_size, 500);
    EXPECT_LT(config.block_size, 2000);
}

TEST(CacheBlockConfigTest, BlocksCoverEntireGrid) {
    const size_t n = 10000;
    auto config = mango::CacheBlockConfig::adaptive(n);

    // Verify blocks cover the grid
    // Last block may be smaller, but total should cover n points
    size_t covered = (config.n_blocks - 1) * config.block_size + config.block_size;
    EXPECT_GE(covered, n);
}
