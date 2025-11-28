// tests/thread_workspace_test.cc
#include "src/support/thread_workspace.hpp"
#include <gtest/gtest.h>
#include <cstdint>

using namespace mango;

TEST(ThreadWorkspaceBufferTest, BasicAllocation) {
    ThreadWorkspaceBuffer buffer(1024);

    EXPECT_GE(buffer.size(), 1024u);
    EXPECT_EQ(buffer.size() % 64, 0u);  // 64-byte aligned size
}

TEST(ThreadWorkspaceBufferTest, Alignment64Byte) {
    ThreadWorkspaceBuffer buffer(100);

    auto bytes = buffer.bytes();
    auto addr = reinterpret_cast<std::uintptr_t>(bytes.data());
    EXPECT_EQ(addr % 64, 0u) << "Base address must be 64-byte aligned";
}

TEST(ThreadWorkspaceBufferTest, ByteSpanStability) {
    ThreadWorkspaceBuffer buffer(512);

    auto span1 = buffer.bytes();
    auto span2 = buffer.bytes();

    EXPECT_EQ(span1.data(), span2.data());
    EXPECT_EQ(span1.size(), span2.size());
}

TEST(ThreadWorkspaceBufferTest, PMRResourceAccessible) {
    ThreadWorkspaceBuffer buffer(1024);

    std::pmr::memory_resource& resource = buffer.resource();

    // Should be able to allocate from the resource
    void* p = resource.allocate(64, 8);
    EXPECT_NE(p, nullptr);
    resource.deallocate(p, 64, 8);
}

TEST(ThreadWorkspaceBufferTest, MoveSemantics) {
    ThreadWorkspaceBuffer buffer1(512);
    auto* original_data = buffer1.bytes().data();

    ThreadWorkspaceBuffer buffer2(std::move(buffer1));

    EXPECT_EQ(buffer2.bytes().data(), original_data);
}

TEST(ThreadWorkspaceBufferTest, FallbackAllocation) {
    // Small buffer to trigger fallback
    ThreadWorkspaceBuffer buffer(128);

    std::pmr::memory_resource& resource = buffer.resource();

    // Allocate beyond buffer capacity to trigger fallback to unsynchronized_pool
    void* p1 = resource.allocate(100, 8);  // Fits in buffer
    void* p2 = resource.allocate(100, 8);  // Should trigger fallback

    EXPECT_NE(p1, nullptr);
    EXPECT_NE(p2, nullptr);

    // Note: Don't deallocate - monotonic_buffer_resource doesn't support it
    // and we're testing that fallback works, not deallocation
}
