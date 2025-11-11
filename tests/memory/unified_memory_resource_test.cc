#include "src/memory/unified_memory_resource.hpp"
#include <gtest/gtest.h>

TEST(UnifiedMemoryResourceTest, BasicAllocation) {
    mango::memory::UnifiedMemoryResource resource(1024);

    void* ptr = resource.allocate(64, 64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);  // 64-byte aligned
    EXPECT_EQ(resource.bytes_allocated(), 64);
}

TEST(UnifiedMemoryResourceTest, MultipleAllocations) {
    mango::memory::UnifiedMemoryResource resource(1024);

    void* ptr1 = resource.allocate(32, 64);
    void* ptr2 = resource.allocate(32, 64);

    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);
    EXPECT_NE(ptr1, ptr2);
    EXPECT_EQ(resource.bytes_allocated(), 64);
}

TEST(UnifiedMemoryResourceTest, ResetClearsMemory) {
    mango::memory::UnifiedMemoryResource resource(1024);

    resource.allocate(128, 64);
    EXPECT_EQ(resource.bytes_allocated(), 128);

    resource.reset();
    EXPECT_EQ(resource.bytes_allocated(), 0);

    // Can allocate again after reset
    void* ptr = resource.allocate(64, 64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(resource.bytes_allocated(), 64);
}
