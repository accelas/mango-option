/**
 * @file slice_solver_workspace_test.cc
 * @brief Tests for SliceSolverWorkspace and workspace-mode AmericanOptionSolver
 */

#include "src/slice_solver_workspace.hpp"
#include "src/american_option.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace mango {
namespace {

TEST(SliceSolverWorkspaceTest, BasicConstruction) {
    // Create workspace
    SliceSolverWorkspace workspace(-3.0, 3.0, 101);

    // Verify grid parameters
    EXPECT_EQ(workspace.x_min(), -3.0);
    EXPECT_EQ(workspace.x_max(), 3.0);
    EXPECT_EQ(workspace.n_space(), 101);

    // Verify grid span
    auto grid_span = workspace.grid_span();
    EXPECT_EQ(grid_span.size(), 101);
    EXPECT_DOUBLE_EQ(grid_span.front(), -3.0);
    EXPECT_DOUBLE_EQ(grid_span.back(), 3.0);
}

TEST(SliceSolverWorkspaceTest, GridSpacingReuse) {
    SliceSolverWorkspace workspace(-3.0, 3.0, 101);

    // Get GridSpacing (should be same instance)
    auto spacing1 = workspace.grid_spacing();
    auto spacing2 = workspace.grid_spacing();

    // Same shared_ptr instance
    EXPECT_EQ(spacing1.get(), spacing2.get());
}

TEST(AmericanOptionSolverTest, WorkspaceModeConstruction) {
    // Setup common parameters
    AmericanOptionGrid grid_config;
    grid_config.x_min = -3.0;
    grid_config.x_max = 3.0;
    grid_config.n_space = 101;
    grid_config.n_time = 100;

    // Create workspace with shared_ptr for proper lifetime management
    auto workspace = std::make_shared<SliceSolverWorkspace>(
        grid_config.x_min, grid_config.x_max, grid_config.n_space);

    // Create solver with workspace
    AmericanOptionParams params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.2,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .discrete_dividends = {}
    };

    EXPECT_NO_THROW({
        AmericanOptionSolver solver(params, grid_config, workspace);
    });
}

TEST(AmericanOptionSolverTest, WorkspaceModeMismatchedGrid) {
    // Create workspace with specific grid
    auto workspace = std::make_shared<SliceSolverWorkspace>(-3.0, 3.0, 101);

    // Try to create solver with mismatched grid
    AmericanOptionGrid grid_config;
    grid_config.x_min = -2.0;  // MISMATCH
    grid_config.x_max = 3.0;
    grid_config.n_space = 101;

    AmericanOptionParams params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.2,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .discrete_dividends = {}
    };

    // Should throw because grid parameters don't match workspace
    EXPECT_THROW({
        AmericanOptionSolver solver(params, grid_config, workspace);
    }, std::invalid_argument);
}

TEST(AmericanOptionSolverTest, WorkspaceModeNullWorkspace) {
    // Test that passing nullptr workspace throws
    AmericanOptionGrid grid_config;
    grid_config.x_min = -3.0;
    grid_config.x_max = 3.0;
    grid_config.n_space = 101;

    AmericanOptionParams params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.2,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .discrete_dividends = {}
    };

    // Should throw because workspace is null
    EXPECT_THROW({
        AmericanOptionSolver solver(params, grid_config, nullptr);
    }, std::invalid_argument);
}

TEST(AmericanOptionSolverTest, WorkspaceModeMultipleSolvers) {
    // Setup common grid
    AmericanOptionGrid grid_config;
    grid_config.x_min = -3.0;
    grid_config.x_max = 3.0;
    grid_config.n_space = 51;  // Smaller for faster test
    grid_config.n_time = 100;

    // Create workspace once with shared_ptr
    auto workspace = std::make_shared<SliceSolverWorkspace>(
        grid_config.x_min, grid_config.x_max, grid_config.n_space);

    // Create multiple solvers with different parameters
    std::vector<double> volatilities = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rates = {0.03, 0.05, 0.07};

    for (double vol : volatilities) {
        for (double rate : rates) {
            AmericanOptionParams params{
                .strike = 100.0,
                .spot = 100.0,
                .maturity = 1.0,
                .volatility = vol,
                .rate = rate,
                .continuous_dividend_yield = 0.02,
                .option_type = OptionType::PUT,
                .discrete_dividends = {}
            };

            // Create solver with shared workspace
            AmericanOptionSolver solver(params, grid_config, workspace);

            // Solve (verify it doesn't crash)
            auto result = solver.solve();
            EXPECT_TRUE(result.has_value());
            if (result.has_value()) {
                EXPECT_TRUE(result->converged);
                EXPECT_GT(result->value, 0.0);  // Put should have positive value
            }
        }
    }
}

TEST(AmericanOptionSolverTest, WorkspaceModeVsStandaloneConsistency) {
    // Setup grid
    AmericanOptionGrid grid_config;
    grid_config.x_min = -3.0;
    grid_config.x_max = 3.0;
    grid_config.n_space = 51;
    grid_config.n_time = 100;

    AmericanOptionParams params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.2,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .discrete_dividends = {}
    };

    // Solve in standalone mode
    AmericanOptionSolver solver_standalone(params, grid_config);
    auto result_standalone = solver_standalone.solve();
    ASSERT_TRUE(result_standalone.has_value());

    // Solve in workspace mode
    auto workspace = std::make_shared<SliceSolverWorkspace>(
        grid_config.x_min, grid_config.x_max, grid_config.n_space);
    AmericanOptionSolver solver_workspace(params, grid_config, workspace);
    auto result_workspace = solver_workspace.solve();
    ASSERT_TRUE(result_workspace.has_value());

    // Results should be identical (within numerical tolerance)
    EXPECT_NEAR(result_standalone->value, result_workspace->value, 1e-10);
    EXPECT_NEAR(result_standalone->delta, result_workspace->delta, 1e-10);
    EXPECT_NEAR(result_standalone->gamma, result_workspace->gamma, 1e-10);
}

}  // namespace
}  // namespace mango
