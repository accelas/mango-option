// SPDX-License-Identifier: MIT
// tests/thread_workspace_test.cc
#include "mango/support/thread_workspace.hpp"
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

TEST(ThreadWorkspaceBufferTest, MoveSemantics) {
    ThreadWorkspaceBuffer buffer1(512);
    auto* original_data = buffer1.bytes().data();

    ThreadWorkspaceBuffer buffer2(std::move(buffer1));

    EXPECT_EQ(buffer2.bytes().data(), original_data);
}
