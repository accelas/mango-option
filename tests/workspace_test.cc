#include "src/workspace.hpp"
#include "src/grid.hpp"
#include <gtest/gtest.h>

TEST(WorkspaceStorageTest, PreComputedDxArray) {
    auto result = mango::GridSpec<>::uniform(0.0, 10.0, 6);
    ASSERT_TRUE(result.has_value());
    auto grid = result.value().generate();  // Points: 0, 2, 4, 6, 8, 10

    mango::WorkspaceStorage workspace(grid.size(), grid.span());

    // dx array should have size n-1
    EXPECT_EQ(workspace.dx().size(), 5);

    // All dx values should be 2.0 for uniform grid
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(workspace.dx()[i], 2.0);
    }
}

TEST(WorkspaceStorageTest, PsiBufferAvailable) {
    const size_t n = 100;
    auto result = mango::GridSpec<>::uniform(0.0, 1.0, n);
    ASSERT_TRUE(result.has_value());
    auto grid = result.value().generate();

    mango::WorkspaceStorage ws(grid.size(), grid.span());

    auto psi = ws.psi_buffer();
    EXPECT_EQ(psi.size(), n);

    // Verify writable
    for (size_t i = 0; i < n; ++i) {
        psi[i] = static_cast<double>(i);
    }

    // Verify no overlap with other buffers
    auto u_current = ws.u_current();
    u_current[0] = 999.0;
    EXPECT_NE(psi[0], 999.0);
}
