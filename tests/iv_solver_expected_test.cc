// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/iv_solver.hpp"
#include "src/option/iv_result.hpp"
#include "src/support/error_types.hpp"

using namespace mango;

// Test for new std::expected signature (Task 2.1)
// NOTE: This test calls solve() directly to verify the new signature,
// bypassing the base class which still returns IVResult (will be updated in Task 3).
TEST(IVSolverFDMExpected, ReturnsExpectedType) {
    // Simple test to verify std::expected signature compiles
    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 10.0);

    IVSolverConfig config;
    IVSolver solver(config);

    // Call solve() directly (not solve() which goes through base class)
    auto result = solver.solve(query);

    // Verify it compiles and returns expected type
    static_assert(std::is_same_v<decltype(result), std::expected<IVSuccess, IVError>>);

    // Verify Brent solver returns success with real IV calculation
    EXPECT_TRUE(result.has_value());
    if (result.has_value()) {
        // Brent solver should produce reasonable volatility (not placeholder)
        EXPECT_GT(result->implied_vol, 0.01);
        EXPECT_LT(result->implied_vol, 1.0);
        EXPECT_GT(result->iterations, 0);  // Did actual work
    }
}

// Arbitrage and boundary validation tests (unique to std::expected API)

TEST(IVSolverFDMExpected, ValidationArbitrageCallExceedsSpot) {
    // Call price cannot exceed spot price
    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::CALL}, 150.0);

    IVSolverConfig config;
    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::ArbitrageViolation);
    // Error code checked above
}

TEST(IVSolverFDMExpected, ValidationArbitragePutExceedsStrike) {
    // Put price cannot exceed strike price
    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 150.0);

    IVSolverConfig config;
    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::ArbitrageViolation);
    // Error code checked above
}

TEST(IVSolverFDMExpected, ValidationPriceBelowIntrinsicCall) {
    // Market price must be >= intrinsic value
    // For call: intrinsic = max(S - K, 0) = max(110 - 100, 0) = 10
    IVQuery query(
        OptionSpec{.spot = 110.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::CALL}, 5.0);

    IVSolverConfig config;
    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::ArbitrageViolation);
    // Error code checked above
}

TEST(IVSolverFDMExpected, ValidationPriceBelowIntrinsicPut) {
    // Market price must be >= intrinsic value
    // For put: intrinsic = max(K - S, 0) = max(110 - 100, 0) = 10
    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 110.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 5.0);

    IVSolverConfig config;
    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::ArbitrageViolation);
    // Error code checked above
}

TEST(IVSolverFDMExpected, ValidationZeroSpot) {
    IVQuery query(
        OptionSpec{.spot = 0.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 10.0);

    IVSolverConfig config;
    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeSpot);
    // Error code checked above
}

TEST(IVSolverFDMExpected, ValidationZeroStrike) {
    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 0.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 10.0);

    IVSolverConfig config;
    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeStrike);
    // Error code checked above
}

// Task 2.3 Brent Solver Integration Tests

TEST(IVSolverFDMExpected, SolvesATMPut) {
    // ATM American put - typical case
    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 10.0);

    IVSolverConfig config;
    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->implied_vol, 0.01);  // Reasonable vol
    EXPECT_LT(result->implied_vol, 1.0);   // Not too high
    EXPECT_GT(result->iterations, 0);      // Did some work
    EXPECT_LT(result->final_error, 1e-4);  // Converged
}

TEST(IVSolverFDMExpected, SolvesITMPut) {
    // ITM put - K > S
    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 110.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 15.0);

    IVSolverConfig config;
    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->implied_vol, 0.01);
    EXPECT_LT(result->implied_vol, 1.0);
}

TEST(IVSolverFDMExpected, SolvesOTMPut) {
    // OTM put - K < S
    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 90.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 3.0);

    IVSolverConfig config;
    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->implied_vol, 0.01);
    EXPECT_LT(result->implied_vol, 1.0);
}

TEST(IVSolverFDMExpected, ConvergenceFailureMaxIterations) {
    // Artificially low max_iter to force failure
    IVSolverConfig config;
    config.root_config.max_iter = 2;  // Too few iterations

    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 10.0);

    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::MaxIterationsExceeded);
    EXPECT_GT(result.error().iterations, 0);
}

TEST(IVSolverFDMExpected, RealisticVolatilityValues) {
    // Test with known market scenario
    IVQuery query(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.5, .rate = 0.03, .option_type = OptionType::PUT}, 5.0);

    IVSolverConfig config;
    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_TRUE(result.has_value());

    // Volatility should be in reasonable range
    EXPECT_GE(result->implied_vol, 0.05);  // At least 5%
    EXPECT_LE(result->implied_vol, 0.80);  // At most 80%

    // Should converge quickly for well-behaved case
    EXPECT_LE(result->iterations, 50);

    // Error should be small
    EXPECT_LE(result->final_error, 0.01);  // Within 1 cent
}

// Task 3.2 Batch Solver Tests

TEST(IVSolverFDMExpected, BatchSolveAllSuccess) {
    // Batch of 3 valid queries
    std::vector<IVQuery> queries = {
        IVQuery(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 10.0),
        IVQuery(OptionSpec{.spot = 100.0, .strike = 110.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 15.0),
        IVQuery(OptionSpec{.spot = 100.0, .strike = 90.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 3.0)
    };

    IVSolverConfig config;
    IVSolver solver(config);
    auto batch_result = solver.solve_batch(queries);

    EXPECT_TRUE(batch_result.all_succeeded());
    EXPECT_EQ(batch_result.failed_count, 0);
    EXPECT_EQ(batch_result.results.size(), 3);

    // Check each result
    for (size_t i = 0; i < batch_result.results.size(); ++i) {
        SCOPED_TRACE("Query index: " + std::to_string(i));
        ASSERT_TRUE(batch_result.results[i].has_value());
        EXPECT_GT(batch_result.results[i]->implied_vol, 0.0);
        EXPECT_LT(batch_result.results[i]->implied_vol, 1.0);
    }
}

TEST(IVSolverFDMExpected, BatchSolveMixedResults) {
    // Mix of valid and invalid queries
    std::vector<IVQuery> queries = {
        // Valid query
        IVQuery(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 10.0),
        // Invalid: negative spot
        IVQuery(OptionSpec{.spot = -100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 10.0),
        // Valid query
        IVQuery(OptionSpec{.spot = 100.0, .strike = 90.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 3.0)
    };

    IVSolverConfig config;
    IVSolver solver(config);
    auto batch_result = solver.solve_batch(queries);

    EXPECT_FALSE(batch_result.all_succeeded());
    EXPECT_EQ(batch_result.failed_count, 1);
    EXPECT_EQ(batch_result.results.size(), 3);

    // Check specific results
    ASSERT_TRUE(batch_result.results[0].has_value());

    ASSERT_FALSE(batch_result.results[1].has_value());
    EXPECT_EQ(batch_result.results[1].error().code, IVErrorCode::NegativeSpot);

    ASSERT_TRUE(batch_result.results[2].has_value());
}

TEST(IVSolverFDMExpected, BatchSolveEmptyBatch) {
    std::vector<IVQuery> queries;  // Empty

    IVSolverConfig config;
    IVSolver solver(config);
    auto batch_result = solver.solve_batch(queries);

    EXPECT_TRUE(batch_result.all_succeeded());
    EXPECT_EQ(batch_result.failed_count, 0);
    EXPECT_EQ(batch_result.results.size(), 0);
}
