#include "src/support/memory/workspace_base.hpp"
#include <gtest/gtest.h>
#include <memory_resource>

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

TEST(WorkspaceBaseTest, PMRResourceAccessor) {
    // Test PMR resource accessor
    std::pmr::monotonic_buffer_resource custom_resource(4096);
    mango::WorkspaceBase workspace(1024, &custom_resource);

    // Test that we can get the PMR resource
    std::pmr::memory_resource* resource = workspace.pmr_resource();
    ASSERT_NE(resource, nullptr);
    EXPECT_NE(resource, std::pmr::get_default_resource());

    // Test const accessor
    const mango::WorkspaceBase& const_workspace = workspace;
    const std::pmr::memory_resource* const_resource = const_workspace.pmr_resource();
    ASSERT_NE(const_resource, nullptr);
    EXPECT_EQ(const_resource, resource);
}
