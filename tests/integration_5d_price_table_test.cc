#include "src/core/multigrid.hpp"
#include "src/memory/pde_workspace.hpp"
#include "src/core/spatial_operators.hpp"
#include "src/core/grid.hpp"
#include <gtest/gtest.h>

/// Integration test: 5D price table grid setup
///
/// Verifies MultiGridBuffer works with spatial operators and workspace storage.
/// This simulates the grid setup phase of price table precomputation.
TEST(Integration5DPriceTableTest, GridSetupWithDividendDimension) {
    // Step 1: Create 5D price table grid
    mango::MultiGridBuffer price_table_grid;

    auto moneyness_spec = mango::GridSpec<>::log_spaced(0.7, 1.3, 10);
    ASSERT_TRUE(moneyness_spec.has_value());
    auto result1 = price_table_grid.add_axis(mango::GridAxis::Moneyness, *moneyness_spec);
    EXPECT_TRUE(result1.has_value());

    auto maturity_spec = mango::GridSpec<>::uniform(0.027, 2.0, 8);
    ASSERT_TRUE(maturity_spec.has_value());
    auto result2 = price_table_grid.add_axis(mango::GridAxis::Maturity, *maturity_spec);
    EXPECT_TRUE(result2.has_value());

    auto volatility_spec = mango::GridSpec<>::uniform(0.10, 0.50, 5);
    ASSERT_TRUE(volatility_spec.has_value());
    auto result3 = price_table_grid.add_axis(mango::GridAxis::Volatility, *volatility_spec);
    EXPECT_TRUE(result3.has_value());

    auto rate_spec = mango::GridSpec<>::uniform(0.0, 0.10, 4);
    ASSERT_TRUE(rate_spec.has_value());
    auto result4 = price_table_grid.add_axis(mango::GridAxis::Rate, *rate_spec);
    EXPECT_TRUE(result4.has_value());

    auto dividend_spec = mango::GridSpec<>::uniform(0.0, 0.05, 3);
    ASSERT_TRUE(dividend_spec.has_value());
    auto result5 = price_table_grid.add_axis(mango::GridAxis::Dividend, *dividend_spec);
    EXPECT_TRUE(result5.has_value());

    // Verify grid dimensions
    EXPECT_EQ(price_table_grid.n_axes(), 5);
    size_t expected_combinations = 10 * 8 * 5 * 4 * 3;  // 4,800 parameter combinations
    EXPECT_EQ(price_table_grid.total_points(), expected_combinations);

    // Step 2: Extract dividend axis and verify spacing
    auto dividend_axis_result = price_table_grid.axis_view(mango::GridAxis::Dividend);
    EXPECT_TRUE(dividend_axis_result.has_value());
    auto dividend_axis = dividend_axis_result.value();
    EXPECT_EQ(dividend_axis.size(), 3);
    EXPECT_DOUBLE_EQ(dividend_axis[0], 0.0);    // No dividend
    EXPECT_DOUBLE_EQ(dividend_axis[1], 0.025);  // 2.5% yield
    EXPECT_DOUBLE_EQ(dividend_axis[2], 0.05);   // 5% yield

    // Step 3: Create PDE solver spatial grid (for a single parameter combination)
    // This would be created once per parameter combination during precompute
    auto pde_grid_spec = mango::GridSpec<>::log_spaced(50.0, 150.0, 101);
    ASSERT_TRUE(pde_grid_spec.has_value());
    auto pde_grid = pde_grid_spec->generate();

    // Step 4: Create workspace for this PDE solve
    mango::PDEWorkspace workspace(pde_grid.size(), pde_grid.span());

    // Verify workspace has pre-computed dx
    EXPECT_EQ(workspace.dx().size(), 100);  // n-1 spacing values

    // Step 5: Create index operator with dividend from price table
    double dividend_yield = dividend_axis[1];  // Use q=0.025 from table
    mango::IndexBlackScholesOperator op(0.05, 0.3, dividend_yield);

    // Step 6: Initialize test solution and apply operator
    auto u = workspace.u_current();
    for (size_t i = 0; i < pde_grid.size(); ++i) {
        u[i] = pde_grid[i];  // u(S) = S
    }

    auto Lu = workspace.lu();
    op.apply(0.0, pde_grid.span(), u, Lu, workspace.dx());

    // Step 7: Verify operator used dividend correctly: L(S) = -q*S
    double S_mid = pde_grid[50];
    double expected_Lu = -dividend_yield * S_mid;
    EXPECT_NEAR(Lu[50], expected_Lu, 0.01);

    // This confirms the full pipeline:
    // 5D grid → dividend axis → index operator → workspace → dx → result
}

/// Test: Verify dividend dimension affects operator output
TEST(Integration5DPriceTableTest, DividendAffectsPriceCalculation) {
    // Create small 2D slice: dividend × rate
    mango::MultiGridBuffer grid;
    auto dividend_spec = mango::GridSpec<>::uniform(0.0, 0.04, 3);
    ASSERT_TRUE(dividend_spec.has_value());
    auto result1 = grid.add_axis(mango::GridAxis::Dividend, *dividend_spec);
    EXPECT_TRUE(result1.has_value());

    auto rate_spec = mango::GridSpec<>::uniform(0.02, 0.06, 3);
    ASSERT_TRUE(rate_spec.has_value());
    auto result2 = grid.add_axis(mango::GridAxis::Rate,     *rate_spec);
    EXPECT_TRUE(result2.has_value());

    // PDE solver grid (same for all parameter combinations)
    auto pde_spec = mango::GridSpec<>::uniform(80.0, 120.0, 41);
    ASSERT_TRUE(pde_spec.has_value());
    auto pde_grid = pde_spec->generate();
    mango::PDEWorkspace workspace(pde_grid.size(), pde_grid.span());

    auto div_axis_result = grid.axis_view(mango::GridAxis::Dividend);
    EXPECT_TRUE(div_axis_result.has_value());
    auto div_axis = div_axis_result.value();

    auto rate_axis_result = grid.axis_view(mango::GridAxis::Rate);
    EXPECT_TRUE(rate_axis_result.has_value());
    auto rate_axis = rate_axis_result.value();

    // Test two parameter combinations with different dividends
    double r = rate_axis[1];  // r = 0.04
    double q1 = div_axis[0];  // q = 0.0
    double q2 = div_axis[2];  // q = 0.04

    // Create operators
    mango::IndexBlackScholesOperator op1(r, 0.2, q1);
    mango::IndexBlackScholesOperator op2(r, 0.2, q2);

    // Initialize u(S) = S
    auto u = workspace.u_current();
    for (size_t i = 0; i < pde_grid.size(); ++i) {
        u[i] = pde_grid[i];
    }

    // Apply both operators
    auto Lu1 = workspace.lu();
    op1.apply(0.0, pde_grid.span(), u, Lu1, workspace.dx());

    std::vector<double> Lu2_buffer(pde_grid.size());
    std::span<double> Lu2(Lu2_buffer);
    op2.apply(0.0, pde_grid.span(), u, Lu2, workspace.dx());

    // Verify difference is due to dividend: Lu1 - Lu2 = (q2 - q1)*S
    double S_mid = pde_grid[20];
    double expected_diff = (q2 - q1) * S_mid;  // 0.04 * S
    double actual_diff = Lu1[20] - Lu2[20];

    EXPECT_NEAR(actual_diff, expected_diff, 0.01);

    // This confirms dividend dimension properly affects PDE operator
}
