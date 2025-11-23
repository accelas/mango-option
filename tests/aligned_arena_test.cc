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

TEST(AlignedArenaTest, RejectsZeroSize) {
    auto result = AlignedArena::create(0, 64);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Arena size must be positive");
}

TEST(AlignedArenaTest, RejectsZeroAlignment) {
    auto result = AlignedArena::create(1024, 0);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Alignment must be a power of 2");
}

TEST(AlignedArenaTest, RejectsNonPowerOfTwoAlignment) {
    auto result = AlignedArena::create(1024, 63);  // 63 is not a power of 2
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Alignment must be a power of 2");
}

TEST(AlignedArenaTest, HandlesArenaExhaustion) {
    auto arena = AlignedArena::create(128, 64).value();

    // First allocation should succeed (64 bytes aligned + 10*8=80 bytes)
    double* ptr1 = arena->allocate(10);
    EXPECT_NE(ptr1, nullptr);

    // Second allocation should fail (not enough space for 64-byte alignment + data)
    double* ptr2 = arena->allocate(10);
    EXPECT_EQ(ptr2, nullptr);
}

TEST(AlignedArenaTest, MultipleAllocationsStayAligned) {
    auto arena = AlignedArena::create(4096, 64).value();

    // Allocate multiple blocks
    double* ptr1 = arena->allocate(7);   // 56 bytes (not 64-byte multiple)
    double* ptr2 = arena->allocate(13);  // 104 bytes (not 64-byte multiple)
    double* ptr3 = arena->allocate(5);   // 40 bytes (not 64-byte multiple)

    // All should be non-null
    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr3, nullptr);

    // All should be 64-byte aligned
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr1) % 64, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr2) % 64, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr3) % 64, 0);
}

} // namespace
} // namespace memory
} // namespace mango
