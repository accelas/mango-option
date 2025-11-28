// tests/american_pde_workspace_test.cc

#include "src/pde/core/american_pde_workspace.hpp"
#include "src/support/thread_workspace.hpp"
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
    EXPECT_EQ(ws.dx().size(), n - 1);
    EXPECT_EQ(ws.u_stage().size(), n);
    EXPECT_EQ(ws.rhs().size(), n);
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

TEST(AmericanPDEWorkspaceTest, AccessorsWorkCorrectly) {
    const size_t n = 50;
    ThreadWorkspaceBuffer buffer(AmericanPDEWorkspace::required_bytes(n));

    auto ws = AmericanPDEWorkspace::from_bytes(buffer.bytes(), n).value();

    // Write to spans
    for (size_t i = 0; i < n; ++i) {
        ws.u_stage()[i] = static_cast<double>(i);
        ws.rhs()[i] = static_cast<double>(i * 2);
    }

    // Read back
    for (size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(ws.u_stage()[i], static_cast<double>(i));
        EXPECT_DOUBLE_EQ(ws.rhs()[i], static_cast<double>(i * 2));
    }
}

TEST(AmericanPDEWorkspaceTest, AllAccessorsSizesCorrect) {
    const size_t n = 75;
    ThreadWorkspaceBuffer buffer(AmericanPDEWorkspace::required_bytes(n));

    auto ws = AmericanPDEWorkspace::from_bytes(buffer.bytes(), n).value();

    // Arrays with size n
    EXPECT_EQ(ws.u_stage().size(), n);
    EXPECT_EQ(ws.rhs().size(), n);
    EXPECT_EQ(ws.lu().size(), n);
    EXPECT_EQ(ws.psi().size(), n);
    EXPECT_EQ(ws.jacobian_diag().size(), n);
    EXPECT_EQ(ws.residual().size(), n);
    EXPECT_EQ(ws.delta_u().size(), n);
    EXPECT_EQ(ws.newton_u_old().size(), n);
    EXPECT_EQ(ws.u_next().size(), n);

    // Arrays with size n-1
    EXPECT_EQ(ws.dx().size(), n - 1);
    EXPECT_EQ(ws.jacobian_upper().size(), n - 1);
    EXPECT_EQ(ws.jacobian_lower().size(), n - 1);

    // tridiag_workspace with size 2n
    EXPECT_EQ(ws.tridiag_workspace().size(), 2 * n);
}

TEST(AmericanPDEWorkspaceTest, JacobianAccessorWorks) {
    const size_t n = 50;
    ThreadWorkspaceBuffer buffer(AmericanPDEWorkspace::required_bytes(n));

    auto ws = AmericanPDEWorkspace::from_bytes(buffer.bytes(), n).value();

    // Get Jacobian view
    auto J = ws.jacobian();

    // Verify it references the correct spans
    EXPECT_EQ(J.lower().size(), n - 1);
    EXPECT_EQ(J.diag().size(), n);
    EXPECT_EQ(J.upper().size(), n - 1);

    // Write through Jacobian view
    for (size_t i = 0; i < n; ++i) {
        J.diag()[i] = static_cast<double>(i);
    }
    for (size_t i = 0; i < n - 1; ++i) {
        J.lower()[i] = static_cast<double>(i + 100);
        J.upper()[i] = static_cast<double>(i + 200);
    }

    // Read back through workspace accessors
    for (size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(ws.jacobian_diag()[i], static_cast<double>(i));
    }
    for (size_t i = 0; i < n - 1; ++i) {
        EXPECT_DOUBLE_EQ(ws.jacobian_lower()[i], static_cast<double>(i + 100));
        EXPECT_DOUBLE_EQ(ws.jacobian_upper()[i], static_cast<double>(i + 200));
    }
}

TEST(AmericanPDEWorkspaceTest, ConstAccessorsWork) {
    const size_t n = 40;
    ThreadWorkspaceBuffer buffer(AmericanPDEWorkspace::required_bytes(n));

    auto ws = AmericanPDEWorkspace::from_bytes(buffer.bytes(), n).value();

    // Write some data
    for (size_t i = 0; i < n; ++i) {
        ws.u_stage()[i] = static_cast<double>(i * 3);
    }

    // Read through const reference
    const auto& const_ws = ws;
    for (size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(const_ws.u_stage()[i], static_cast<double>(i * 3));
    }
    EXPECT_EQ(const_ws.size(), n);
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

    // Modifications through inner should be visible through wrapper
    for (size_t i = 0; i < n; ++i) {
        inner.rhs()[i] = static_cast<double>(i * 5);
    }
    for (size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(ws.rhs()[i], static_cast<double>(i * 5));
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

    // Write to both
    for (size_t i = 0; i < n1; ++i) {
        ws1.u_stage()[i] = static_cast<double>(i);
    }
    for (size_t i = 0; i < n2; ++i) {
        ws2.u_stage()[i] = static_cast<double>(i * 2);
    }

    // Verify independent
    for (size_t i = 0; i < n1; ++i) {
        EXPECT_DOUBLE_EQ(ws1.u_stage()[i], static_cast<double>(i));
    }
    for (size_t i = 0; i < n2; ++i) {
        EXPECT_DOUBLE_EQ(ws2.u_stage()[i], static_cast<double>(i * 2));
    }
}

TEST(AmericanPDEWorkspaceTest, AlignmentRequirement) {
    // Verify required_bytes always returns 64-byte aligned sizes
    for (size_t n = 2; n <= 200; ++n) {
        size_t bytes = AmericanPDEWorkspace::required_bytes(n);
        EXPECT_EQ(bytes % 64, 0u) << "Not aligned for n=" << n;
    }
}
