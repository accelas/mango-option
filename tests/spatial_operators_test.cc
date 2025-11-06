#include "src/spatial_operators.hpp"
#include "src/workspace.hpp"
#include "src/grid.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(BlackScholesOperatorTest, EquityOperatorBasic) {
    // Create operator for equity option
    // Parameters: r=0.05, sigma=0.2, no dividends
    mango::EquityBlackScholesOperator op(0.05, 0.2);

    // Create simple grid
    auto spec = mango::GridSpec<>::uniform(80.0, 120.0, 41);
    auto grid = spec.generate();
    mango::WorkspaceStorage workspace(grid.size(), grid.span());

    // Test input: linear function u(S) = S (delta = 1, gamma = 0)
    auto u = workspace.u_current();
    for (size_t i = 0; i < 41; ++i) {
        u[i] = grid[i];
    }

    auto Lu = workspace.lu();

    // Apply operator with pre-computed dx
    op.apply(0.0, grid.span(), u, Lu, workspace.dx());

    // For u(S) = S, the Black-Scholes operator gives:
    // L(u) = r*S*du/dS - r*u = r*S*1 - r*S = 0
    // (Middle points should be approximately zero)
    EXPECT_NEAR(Lu[20], 0.0, 0.01);  // S=100
}

TEST(BlackScholesOperatorTest, EquityOperatorParabolic) {
    // Test with u(S) = S^2 (delta = 2S, gamma = 2)
    mango::EquityBlackScholesOperator op(0.05, 0.2);

    auto spec = mango::GridSpec<>::uniform(90.0, 110.0, 21);
    auto grid = spec.generate();
    mango::WorkspaceStorage workspace(grid.size(), grid.span());

    auto u = workspace.u_current();
    for (size_t i = 0; i < 21; ++i) {
        u[i] = grid[i] * grid[i];
    }

    auto Lu = workspace.lu();
    op.apply(0.0, grid.span(), u, Lu, workspace.dx());

    // For u(S) = S^2:
    // du/dS = 2S, d2u/dS2 = 2
    // L(u) = 0.5*sigma^2*S^2*2 + r*S*2S - r*S^2
    //      = sigma^2*S^2 + 2*r*S^2 - r*S^2
    //      = sigma^2*S^2 + r*S^2
    double S = grid[10];  // Middle point
    double expected = 0.2*0.2*S*S + 0.05*S*S;
    EXPECT_NEAR(Lu[10], expected, 0.1);
}

TEST(BlackScholesOperatorTest, IndexOperatorWithDividend) {
    // Create operator with dividend yield q=0.03
    mango::IndexBlackScholesOperator op(0.05, 0.2, 0.03);

    auto spec = mango::GridSpec<>::uniform(80.0, 120.0, 41);
    auto grid = spec.generate();
    mango::WorkspaceStorage workspace(grid.size(), grid.span());

    // Test with u(S) = S (delta = 1, gamma = 0)
    auto u = workspace.u_current();
    for (size_t i = 0; i < 41; ++i) {
        u[i] = grid[i];
    }

    auto Lu = workspace.lu();
    op.apply(0.0, grid.span(), u, Lu, workspace.dx());

    // For u(S) = S with dividend:
    // du/dS = 1, d2u/dS2 = 0
    // L(u) = 0 + (r - q)*S*1 - r*S = (r - q - r)*S = -q*S
    double S = grid[20];  // S = 100
    double expected = -0.03 * S;  // -q*S
    EXPECT_NEAR(Lu[20], expected, 0.01);
}

TEST(BlackScholesOperatorTest, IndexVsEquityDifference) {
    // Verify dividend yield affects operator output
    double r = 0.05, sigma = 0.2, q = 0.03;

    mango::EquityBlackScholesOperator equity_op(r, sigma);
    mango::IndexBlackScholesOperator index_op(r, sigma, q);

    auto spec = mango::GridSpec<>::uniform(90.0, 110.0, 21);
    auto grid = spec.generate();
    mango::WorkspaceStorage workspace(grid.size(), grid.span());

    auto u = workspace.u_current();
    for (size_t i = 0; i < 21; ++i) {
        u[i] = grid[i];  // u(S) = S
    }

    auto Lu_equity = workspace.lu();
    std::vector<double> Lu_index_buffer(21);
    std::span<double> Lu_index(Lu_index_buffer);

    equity_op.apply(0.0, grid.span(), u, Lu_equity, workspace.dx());
    index_op.apply(0.0, grid.span(), u, Lu_index, workspace.dx());

    // For u(S) = S:
    // Equity: L(u) = r*S - r*S = 0
    // Index:  L(u) = (r-q)*S - r*S = -q*S
    // Difference should be -q*S
    double S = grid[10];
    EXPECT_NEAR(Lu_equity[10], 0.0, 0.01);
    EXPECT_NEAR(Lu_index[10], -q * S, 0.01);
    EXPECT_NEAR(Lu_index[10] - Lu_equity[10], -q * S, 0.01);
}

TEST(BlackScholesOperatorTest, OperatorUsesPrecomputedDx) {
    // Create non-uniform grid to ensure dx matters
    auto spec = mango::GridSpec<>::sinh_spaced(50.0, 150.0, 21, 1.5);
    auto grid = spec.generate();

    // Create workspace with pre-computed dx
    mango::WorkspaceStorage workspace(grid.size(), grid.span());

    // Initialize test function u(S) = S
    auto u = workspace.u_current();
    for (size_t i = 0; i < grid.size(); ++i) {
        u[i] = grid[i];
    }

    // Apply index operator WITH dx parameter
    mango::IndexBlackScholesOperator op(0.05, 0.2, 0.03);
    auto Lu = workspace.lu();

    // NEW SIGNATURE: pass pre-computed dx
    op.apply(0.0, grid.span(), u, Lu, workspace.dx());

    // For u(S) = S: L(u) = -q*S
    double S_mid = grid[10];  // Middle of grid
    double expected = -0.03 * S_mid;
    EXPECT_NEAR(Lu[10], expected, 0.01);
}

TEST(LaplacianOperatorTest, ApplyBlockMiddleBlock) {
    mango::LaplacianOperator op(1.0);  // D = 1.0

    // Grid: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    std::vector<double> grid = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
    std::vector<double> u = {0.0, 0.01, 0.04, 0.09, 0.16, 0.25};  // u = x^2

    // Pre-compute dx
    std::vector<double> dx(grid.size() - 1);
    for (size_t i = 0; i < dx.size(); ++i) {
        dx[i] = grid[i+1] - grid[i];
    }

    // Block: base_idx=2, halo_left=1, halo_right=1
    // x_with_halo: [0.1, 0.2, 0.3, 0.4] (4 elements)
    // u_with_halo: [0.01, 0.04, 0.09, 0.16] (4 elements)
    std::span<const double> x_halo(grid.data() + 1, 4);
    std::span<const double> u_halo(u.data() + 1, 4);

    // Lu_interior: [Lu2, Lu3] (2 elements)
    std::vector<double> Lu_interior(2);

    op.apply_block(0.0, 2, 1, 1, x_halo, u_halo,
                   std::span{Lu_interior}, std::span{dx});

    // For u = x^2, d2u/dx2 = 2, so Lu = D * 2 = 2.0
    EXPECT_NEAR(Lu_interior[0], 2.0, 1e-10);  // Lu[2]
    EXPECT_NEAR(Lu_interior[1], 2.0, 1e-10);  // Lu[3]
}

TEST(LaplacianOperatorTest, ApplyBlockFirstBlock) {
    // Test first block with halo_left=1, halo_right=1
    // Interior starts at global index 1
    mango::LaplacianOperator op(1.0);

    std::vector<double> grid = {0.0, 0.1, 0.2, 0.3};
    std::vector<double> u = {0.0, 0.01, 0.04, 0.09};

    std::vector<double> dx(grid.size() - 1);
    for (size_t i = 0; i < dx.size(); ++i) {
        dx[i] = grid[i+1] - grid[i];
    }

    // Block: base_idx=1, interior=[1,2], halo=[0,1,2,3]
    std::span<const double> x_halo(grid.data(), 4);
    std::span<const double> u_halo(u.data(), 4);
    std::vector<double> Lu_interior(2);

    op.apply_block(0.0, 1, 1, 1, x_halo, u_halo,
                   std::span{Lu_interior}, std::span{dx});

    EXPECT_NEAR(Lu_interior[0], 2.0, 1e-10);
    EXPECT_NEAR(Lu_interior[1], 2.0, 1e-10);
}

TEST(LaplacianOperatorTest, ApplyBlockLastBlock) {
    // Test last block with halo_left=1, halo_right=1
    mango::LaplacianOperator op(1.0);

    std::vector<double> grid = {0.0, 0.1, 0.2, 0.3};
    std::vector<double> u = {0.0, 0.01, 0.04, 0.09};

    std::vector<double> dx(grid.size() - 1);
    for (size_t i = 0; i < dx.size(); ++i) {
        dx[i] = grid[i+1] - grid[i];
    }

    // Block: base_idx=2, interior=[2], halo=[1,2,3]
    std::span<const double> x_halo(grid.data() + 1, 3);
    std::span<const double> u_halo(u.data() + 1, 3);
    std::vector<double> Lu_interior(1);

    op.apply_block(0.0, 2, 1, 1, x_halo, u_halo,
                   std::span{Lu_interior}, std::span{dx});

    EXPECT_NEAR(Lu_interior[0], 2.0, 1e-10);
}

TEST(LaplacianOperatorTest, ApplyBlockSmallerLastBlock) {
    // Test last block with fewer than block_size points
    mango::LaplacianOperator op(0.5);

    std::vector<double> grid = {0.0, 0.1, 0.2, 0.3, 0.4};
    std::vector<double> u = {0.0, 0.01, 0.04, 0.09, 0.16};

    std::vector<double> dx(grid.size() - 1);
    for (size_t i = 0; i < dx.size(); ++i) {
        dx[i] = grid[i+1] - grid[i];
    }

    // Last block with only 1 interior point
    std::span<const double> x_halo(grid.data() + 2, 3);
    std::span<const double> u_halo(u.data() + 2, 3);
    std::vector<double> Lu_interior(1);

    op.apply_block(0.0, 3, 1, 1, x_halo, u_halo,
                   std::span{Lu_interior}, std::span{dx});

    EXPECT_NEAR(Lu_interior[0], 1.0, 1e-10);  // D * 2 = 0.5 * 2
}

TEST(EquityBlackScholesOperatorTest, ApplyBlockMiddleBlock) {
    mango::EquityBlackScholesOperator op(0.05, 0.20);  // r=5%, sigma=20%

    std::vector<double> grid = {80.0, 90.0, 100.0, 110.0, 120.0};
    std::vector<double> u = {20.0, 15.0, 10.0, 6.0, 3.0};  // Call prices

    std::vector<double> dx(grid.size() - 1);
    for (size_t i = 0; i < dx.size(); ++i) {
        dx[i] = grid[i+1] - grid[i];
    }

    // Block: base_idx=2, interior=[2], halo=[1,2,3]
    std::span<const double> x_halo(grid.data() + 1, 3);
    std::span<const double> u_halo(u.data() + 1, 3);
    std::vector<double> Lu_interior(1);

    op.apply_block(0.0, 2, 1, 1, x_halo, u_halo,
                   std::span{Lu_interior}, std::span{dx});

    // Just verify it computes something reasonable (non-zero)
    EXPECT_NE(Lu_interior[0], 0.0);
}

TEST(IndexBlackScholesOperatorTest, ApplyBlockMiddleBlock) {
    mango::IndexBlackScholesOperator op(0.05, 0.20, 0.02);  // r=5%, sigma=20%, q=2%

    std::vector<double> grid = {80.0, 90.0, 100.0, 110.0, 120.0};
    std::vector<double> u = {20.0, 15.0, 10.0, 6.0, 3.0};

    std::vector<double> dx(grid.size() - 1);
    for (size_t i = 0; i < dx.size(); ++i) {
        dx[i] = grid[i+1] - grid[i];
    }

    std::span<const double> x_halo(grid.data() + 1, 3);
    std::span<const double> u_halo(u.data() + 1, 3);
    std::vector<double> Lu_interior(1);

    op.apply_block(0.0, 2, 1, 1, x_halo, u_halo,
                   std::span{Lu_interior}, std::span{dx});

    EXPECT_NE(Lu_interior[0], 0.0);
}

TEST(SpatialOperatorsTest, FirstDerivativeParabola) {
    // Test ∂/∂x(x²) = 2x on uniform grid
    const size_t n = 5;
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, n).generate();
    mango::WorkspaceStorage ws(n, grid.span(), 10000);

    std::vector<double> u(n);
    for (size_t i = 0; i < n; ++i) {
        double x = grid.span()[i];
        u[i] = x * x;  // u = x²
    }

    std::vector<double> du(n);
    mango::LaplacianOperator op(1.0);
    op.compute_first_derivative(grid.span(), std::span{u}, std::span{du}, ws.dx());

    // Check interior points: du/dx = 2x
    for (size_t i = 1; i < n - 1; ++i) {
        double x = grid.span()[i];
        double expected = 2.0 * x;
        EXPECT_NEAR(du[i], expected, 1e-10) << "at i=" << i;
    }

    // Boundaries use one-sided differences (first-order accurate, O(dx) error)
    EXPECT_NEAR(du[0], 0.0, 0.3);      // x=0: 2x=0, one-sided has O(dx)=0.25 error
    EXPECT_NEAR(du[n-1], 2.0, 0.3);    // x=1: 2x=2, one-sided has O(dx)=0.25 error
}

TEST(SpatialOperatorsTest, SecondDerivativeParabola) {
    // Test ∂²/∂x²(x²) = 2 on uniform grid
    const size_t n = 5;
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, n).generate();
    mango::WorkspaceStorage ws(n, grid.span(), 10000);

    std::vector<double> u(n);
    for (size_t i = 0; i < n; ++i) {
        double x = grid.span()[i];
        u[i] = x * x;
    }

    std::vector<double> d2u(n);
    mango::LaplacianOperator op(1.0);
    op.compute_second_derivative(grid.span(), std::span{u}, std::span{d2u}, ws.dx());

    // Check interior points: d²u/dx² = 2
    for (size_t i = 1; i < n - 1; ++i) {
        EXPECT_NEAR(d2u[i], 2.0, 1e-10) << "at i=" << i;
    }

    // Boundaries set to zero
    EXPECT_DOUBLE_EQ(d2u[0], 0.0);
    EXPECT_DOUBLE_EQ(d2u[n-1], 0.0);
}
