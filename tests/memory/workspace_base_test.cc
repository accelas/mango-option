#include "src/pde/memory/workspace_base.hpp"
#include <gtest/gtest.h>

TEST(WorkspaceBaseTest, SIMDPadding) {
    EXPECT_EQ(mango::WorkspaceBase::pad_to_simd(1), 8);
    EXPECT_EQ(mango::WorkspaceBase::pad_to_simd(8), 8);
    EXPECT_EQ(mango::WorkspaceBase::pad_to_simd(9), 16);
    EXPECT_EQ(mango::WorkspaceBase::pad_to_simd(16), 16);
    EXPECT_EQ(mango::WorkspaceBase::pad_to_simd(17), 24);
}

TEST(WorkspaceBaseTest, BytesAllocatedTracking) {
    mango::WorkspaceBase workspace(1024);
    EXPECT_EQ(workspace.bytes_allocated(), 0);
}
