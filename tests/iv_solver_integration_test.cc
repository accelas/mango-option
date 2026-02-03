// SPDX-License-Identifier: MIT
/**
 * @file iv_solver_integration_test.cc
 * @brief End-to-end integration tests for IV solver std::expected API
 */

#include <gtest/gtest.h>
#include "mango/option/iv_solver.hpp"
#include <string>

using namespace mango;

// ============================================================================
// End-to-End Workflow Tests
// ============================================================================

TEST(IVSolverIntegration, FullWorkflowSuccess) {
    // Setup: Real-world scenario - ATM put
    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 10.0);

    IVSolverConfig config{
        .root_config = RootFindingConfig{
            .max_iter = 100,
            .tolerance = 1e-6
        },

    };

    IVSolver solver(config);
    auto result = solver.solve(query);

    // Verify success
    ASSERT_TRUE(result.has_value()) << "Solver should converge for valid query";

    // Check solution quality
    EXPECT_GT(result->implied_vol, 0.05) << "IV should be at least 5%";
    EXPECT_LT(result->implied_vol, 0.80) << "IV should be at most 80%";
    EXPECT_GT(result->iterations, 0) << "Should take at least 1 iteration";
    EXPECT_LT(result->iterations, 50) << "Should converge in < 50 iterations";
    EXPECT_LT(result->final_error, 1e-4) << "Final error should be small";
}

TEST(IVSolverIntegration, ValidationErrorPath) {
    // Test validation error handling
    IVQuery query(
        OptionSpec{.spot = -100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 10.0);

    IVSolver solver(IVSolverConfig{});
    auto result = solver.solve(query);

    // Verify failure with correct error code
    ASSERT_FALSE(result.has_value()) << "Should fail validation";
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeSpot);
    EXPECT_EQ(result.error().iterations, 0) << "No iterations on validation error";
}

TEST(IVSolverIntegration, ArbitrageErrorPath) {
    // Test arbitrage violation detection
    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 150.0);

    IVSolver solver(IVSolverConfig{});
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::ArbitrageViolation);
}

TEST(IVSolverIntegration, BatchProcessingWorkflow) {
    // Test batch API with mixed results
    std::vector<IVQuery> queries;

    // Valid query 1
    queries.push_back(IVQuery(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 10.0));

    // Invalid query (negative spot)
    queries.push_back(IVQuery(
        OptionSpec{.spot = -100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 10.0));

    // Valid query 2
    queries.push_back(IVQuery(
        OptionSpec{.spot = 100.0, .strike = 110.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 15.0));

    IVSolver solver(IVSolverConfig{});
    auto batch = solver.solve_batch(queries);

    // Check batch statistics
    EXPECT_EQ(batch.results.size(), 3);
    EXPECT_EQ(batch.failed_count, 1);
    EXPECT_FALSE(batch.all_succeeded());

    // Check individual results
    ASSERT_TRUE(batch.results[0].has_value());
    ASSERT_FALSE(batch.results[1].has_value());
    EXPECT_EQ(batch.results[1].error().code, IVErrorCode::NegativeSpot);
    ASSERT_TRUE(batch.results[2].has_value());
}

TEST(IVSolverIntegration, MonadicErrorPropagation) {
    // Test that validation errors short-circuit properly
    IVQuery query(
        OptionSpec{.spot = -100.0, .strike = -50.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 10.0);

    IVSolver solver(IVSolverConfig{});
    auto result = solver.solve(query);

    // Should stop at first validation error (spot)
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeSpot);
    // Should NOT report NegativeStrike - monadic chain short-circuits
}

TEST(IVSolverIntegration, ITMOTMScenarios) {
    IVSolver solver(IVSolverConfig{});

    // ITM put (K > S)
    auto itm_result = solver.solve(IVQuery(OptionSpec{.spot = 100.0, .strike = 110.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 15.0));
    ASSERT_TRUE(itm_result.has_value());

    // OTM put (K < S)
    auto otm_result = solver.solve(IVQuery(OptionSpec{.spot = 100.0, .strike = 90.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 3.0));
    ASSERT_TRUE(otm_result.has_value());

    // Different volatilities expected
    EXPECT_NE(itm_result->implied_vol, otm_result->implied_vol);
}
