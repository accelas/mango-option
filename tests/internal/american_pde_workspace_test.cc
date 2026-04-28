// SPDX-License-Identifier: MIT
// tests/american_pde_workspace_test.cc

#include "mango/pde/core/american_pde_workspace.hpp"
#include "mango/support/thread_workspace.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace mango;

TEST(AmericanPDEWorkspaceTest, RequiredBytesMatchesPDEWorkspace) {
    const size_t n = 101;

    size_t pde_doubles = PDEWorkspace::required_size(n);
    size_t american_bytes = AmericanPDEWorkspace::required_bytes(n);

    // Should be at least pde_doubles * sizeof(double)
    EXPECT_GE(american_bytes, pde_doubles * sizeof(double));
    // Should be 64-byte aligned
    EXPECT_EQ(american_bytes % 64, 0u);
}

TEST(AmericanPDEWorkspaceTest, FromBytesSuccess) {
    const size_t n = 101;
    ThreadWorkspaceBuffer buffer(AmericanPDEWorkspace::required_bytes(n));

    auto result = AmericanPDEWorkspace::from_bytes(buffer.bytes(), n);

    ASSERT_TRUE(result.has_value()) << result.error();
    auto& ws = result.value();

    EXPECT_EQ(ws.size(), n);
    EXPECT_EQ(ws.workspace().dx().size(), n - 1);
    EXPECT_EQ(ws.workspace().u_stage().size(), n);
    EXPECT_EQ(ws.workspace().rhs().size(), n);
}

TEST(AmericanPDEWorkspaceTest, BufferTooSmall) {
    const size_t n = 101;
    std::vector<std::byte> small(100);  // Way too small

    auto result = AmericanPDEWorkspace::from_bytes(std::span(small), n);

    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("Buffer too small"), std::string::npos);
}

TEST(AmericanPDEWorkspaceTest, GridSizeTooSmall) {
    ThreadWorkspaceBuffer buffer(1024);

    auto result = AmericanPDEWorkspace::from_bytes(buffer.bytes(), 1);

    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("Grid size must be at least 2"), std::string::npos);
}

TEST(AmericanPDEWorkspaceTest, WorkspaceAccessorReturnsInner) {
    const size_t n = 30;
    ThreadWorkspaceBuffer buffer(AmericanPDEWorkspace::required_bytes(n));

    auto ws = AmericanPDEWorkspace::from_bytes(buffer.bytes(), n).value();

    // Access inner workspace
    PDEWorkspace& inner = ws.workspace();
    const PDEWorkspace& const_inner = const_cast<const AmericanPDEWorkspace&>(ws).workspace();

    EXPECT_EQ(inner.size(), n);
    EXPECT_EQ(const_inner.size(), n);

    // Write and read through inner workspace
    for (size_t i = 0; i < n; ++i) {
        inner.rhs()[i] = static_cast<double>(i * 5);
    }
    for (size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(inner.rhs()[i], static_cast<double>(i * 5));
    }
}

TEST(AmericanPDEWorkspaceTest, MultipleWorkspacesFromDifferentBuffers) {
    const size_t n1 = 50;
    const size_t n2 = 100;

    ThreadWorkspaceBuffer buffer1(AmericanPDEWorkspace::required_bytes(n1));
    ThreadWorkspaceBuffer buffer2(AmericanPDEWorkspace::required_bytes(n2));

    auto ws1 = AmericanPDEWorkspace::from_bytes(buffer1.bytes(), n1).value();
    auto ws2 = AmericanPDEWorkspace::from_bytes(buffer2.bytes(), n2).value();

    EXPECT_EQ(ws1.size(), n1);
    EXPECT_EQ(ws2.size(), n2);

    // Write to both through inner workspace
    for (size_t i = 0; i < n1; ++i) {
        ws1.workspace().u_stage()[i] = static_cast<double>(i);
    }
    for (size_t i = 0; i < n2; ++i) {
        ws2.workspace().u_stage()[i] = static_cast<double>(i * 2);
    }

    // Verify independent
    for (size_t i = 0; i < n1; ++i) {
        EXPECT_DOUBLE_EQ(ws1.workspace().u_stage()[i], static_cast<double>(i));
    }
    for (size_t i = 0; i < n2; ++i) {
        EXPECT_DOUBLE_EQ(ws2.workspace().u_stage()[i], static_cast<double>(i * 2));
    }
}

TEST(AmericanPDEWorkspaceTest, AlignmentRequirement) {
    // Verify required_bytes always returns 64-byte aligned sizes
    for (size_t n = 2; n <= 200; ++n) {
        size_t bytes = AmericanPDEWorkspace::required_bytes(n);
        EXPECT_EQ(bytes % 64, 0u) << "Not aligned for n=" << n;
    }
}
