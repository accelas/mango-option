#include <gtest/gtest.h>
#include "src/support/memory/aligned_arena.hpp"

namespace mango {
namespace memory {
namespace {

TEST(AlignedArenaTest, CreateValidArena) {
    auto result = AlignedArena::create(1024, 64);
    ASSERT_TRUE(result.has_value());

    auto arena = result.value();
    EXPECT_NE(arena, nullptr);
}

TEST(AlignedArenaTest, AllocateAligned) {
    auto arena = AlignedArena::create(1024, 64).value();

    double* ptr = arena->allocate(10);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);  // Check 64-byte alignment
}

TEST(AlignedArenaTest, ShareArena) {
    auto arena1 = AlignedArena::create(1024, 64).value();
    auto arena2 = arena1->share();

    EXPECT_EQ(arena1.use_count(), 2);
}

} // namespace
} // namespace memory
} // namespace mango
