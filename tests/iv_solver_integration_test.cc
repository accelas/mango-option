// SPDX-License-Identifier: MIT
/**
 * @file iv_solver_integration_test.cc
 * @brief End-to-end integration tests for IV solver std::expected API
 */

#include <gtest/gtest.h>
#include "src/option/iv_solver_fdm.hpp"
#include <string>

using namespace mango;

// ============================================================================
// End-to-End Workflow Tests
// ============================================================================

TEST(IVSolverIntegration, FullWorkflowSuccess) {
    // Setup: Real-world scenario - ATM put
    IVQuery query(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::PUT,
        10.0    // market_price
    );

    IVSolverFDMConfig config{
        .root_config = RootFindingConfig{
            .max_iter = 100,
            .tolerance = 1e-6
        }
    };

    IVSolverFDM solver(config);
    auto result = solver.solve_impl(query);

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
        -100.0,  // spot (invalid!)
        100.0,   // strike
        1.0,     // maturity
        0.05,    // rate
        0.02,    // dividend_yield
        OptionType::PUT,
        10.0     // market_price
    );

    IVSolverFDM solver(IVSolverFDMConfig{});
    auto result = solver.solve_impl(query);

    // Verify failure with correct error code
    ASSERT_FALSE(result.has_value()) << "Should fail validation";
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeSpot);
    EXPECT_EQ(result.error().iterations, 0) << "No iterations on validation error";
}

TEST(IVSolverIntegration, ArbitrageErrorPath) {
    // Test arbitrage violation detection
    IVQuery query(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::PUT,
        150.0   // market_price (put price > strike!)
    );

    IVSolverFDM solver(IVSolverFDMConfig{});
    auto result = solver.solve_impl(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::ArbitrageViolation);
}

TEST(IVSolverIntegration, BatchProcessingWorkflow) {
    // Test batch API with mixed results
    std::vector<IVQuery> queries;

    // Valid query 1
    queries.emplace_back(
        100.0, 100.0, 1.0, 0.05, 0.02, OptionType::PUT, 10.0
    );

    // Invalid query (negative spot)
    queries.emplace_back(
        -100.0, 100.0, 1.0, 0.05, 0.02, OptionType::PUT, 10.0
    );

    // Valid query 2
    queries.emplace_back(
        100.0, 110.0, 1.0, 0.05, 0.02, OptionType::PUT, 15.0
    );

    IVSolverFDM solver(IVSolverFDMConfig{});
    auto batch = solver.solve_batch_impl(queries);

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
        -100.0,  // spot (first error)
        -50.0,   // strike (would be second error)
        1.0,     // maturity
        0.05,    // rate
        0.02,    // dividend_yield
        OptionType::PUT,
        10.0     // market_price
    );

    IVSolverFDM solver(IVSolverFDMConfig{});
    auto result = solver.solve_impl(query);

    // Should stop at first validation error (spot)
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeSpot);
    // Should NOT report NegativeStrike - monadic chain short-circuits
}

TEST(IVSolverIntegration, ITMOTMScenarios) {
    IVSolverFDM solver(IVSolverFDMConfig{});

    // ITM put (K > S)
    auto itm_result = solver.solve_impl(IVQuery(
        100.0, 110.0, 1.0, 0.05, 0.02, OptionType::PUT, 15.0
    ));
    ASSERT_TRUE(itm_result.has_value());

    // OTM put (K < S)
    auto otm_result = solver.solve_impl(IVQuery(
        100.0, 90.0, 1.0, 0.05, 0.02, OptionType::PUT, 3.0
    ));
    ASSERT_TRUE(otm_result.has_value());

    // Different volatilities expected
    EXPECT_NE(itm_result->implied_vol, otm_result->implied_vol);
}
