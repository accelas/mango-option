#include "src/newton_workspace.hpp"
#include "src/workspace.hpp"
#include "src/grid.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(NewtonWorkspaceTest, CorrectAllocationSizes) {
    const size_t n = 101;
    std::vector<double> grid_data(n);
    for (size_t i = 0; i < n; ++i) {
        grid_data[i] = static_cast<double>(i) / (n - 1);
    }

    mango::WorkspaceStorage pde_ws(n, std::span{grid_data});
    mango::NewtonWorkspace newton_ws(n, pde_ws);

    // Owned arrays
    EXPECT_EQ(newton_ws.jacobian_diag().size(), n);
    EXPECT_EQ(newton_ws.jacobian_lower().size(), n - 1);
    EXPECT_EQ(newton_ws.jacobian_upper().size(), n - 1);
    EXPECT_EQ(newton_ws.residual().size(), n);
    EXPECT_EQ(newton_ws.delta_u().size(), n);
    EXPECT_EQ(newton_ws.u_old().size(), n);
    EXPECT_EQ(newton_ws.tridiag_workspace().size(), 2 * n);  // CRITICAL: 2n

    // Borrowed arrays
    EXPECT_EQ(newton_ws.Lu().size(), n);
    EXPECT_EQ(newton_ws.u_perturb().size(), n);
    EXPECT_EQ(newton_ws.Lu_perturb().size(), n);
}

TEST(NewtonWorkspaceTest, BorrowedArraysPointToWorkspace) {
    const size_t n = 101;
    std::vector<double> grid_data(n);
    for (size_t i = 0; i < n; ++i) {
        grid_data[i] = static_cast<double>(i) / (n - 1);
    }

    mango::WorkspaceStorage pde_ws(n, std::span{grid_data});
    mango::NewtonWorkspace newton_ws(n, pde_ws);

    // Verify borrowed arrays point to correct workspace arrays
    EXPECT_EQ(newton_ws.Lu().data(), pde_ws.lu().data());
    EXPECT_EQ(newton_ws.u_perturb().data(), pde_ws.u_stage().data());
    EXPECT_EQ(newton_ws.Lu_perturb().data(), pde_ws.rhs().data());
}

TEST(NewtonWorkspaceTest, OwnedArraysAreDistinct) {
    const size_t n = 101;
    std::vector<double> grid_data(n);
    for (size_t i = 0; i < n; ++i) {
        grid_data[i] = static_cast<double>(i) / (n - 1);
    }

    mango::WorkspaceStorage pde_ws(n, std::span{grid_data});
    mango::NewtonWorkspace newton_ws(n, pde_ws);

    // Owned arrays should not overlap
    EXPECT_NE(newton_ws.jacobian_diag().data(), newton_ws.residual().data());
    EXPECT_NE(newton_ws.residual().data(), newton_ws.delta_u().data());
    EXPECT_NE(newton_ws.delta_u().data(), newton_ws.u_old().data());
    EXPECT_NE(newton_ws.u_old().data(), newton_ws.tridiag_workspace().data());
}
