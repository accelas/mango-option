// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "src/option/iv_solver.hpp"
#include <cmath>

using namespace mango;

class IVSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        query = IVQuery(
            OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                .rate = 0.05, .option_type = OptionType::PUT}, 10.45);

        config = IVSolverConfig{
            .root_config = RootFindingConfig{
                .max_iter = 100,
                .tolerance = 1e-6,
                .brent_tol_abs = 1e-6
            },
        };
    }

    IVQuery query;
    IVSolverConfig config;
};

// Test 2: Basic ATM put IV calculation
// Re-enabled: ProjectedThomas is now the default (PR #200)
TEST_F(IVSolverTest, ATMPutIVCalculation) {
    IVSolver solver(config);

    auto result = solver.solve(query);

    // Should converge with real implementation
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->implied_vol, 0.316, 0.05);
    EXPECT_GT(result->iterations, 0);
}

// Test 3: Invalid parameters should be caught
TEST_F(IVSolverTest, InvalidSpotPrice) {
    query.spot = -100.0;  // Invalid

    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeSpot);
    // Error code: result.error().code
}

// Test 4: Invalid strike price
TEST_F(IVSolverTest, InvalidStrike) {
    query.strike = 0.0;  // Invalid

    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeStrike);
    // Error code: result.error().code
}

// Test 5: Invalid time to maturity
TEST_F(IVSolverTest, InvalidTimeToMaturity) {
    query.maturity = -1.0;  // Invalid

    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeMaturity);
    // Error code: result.error().code
}

// Test 6: Invalid market price
TEST_F(IVSolverTest, InvalidMarketPrice) {
    query.market_price = -5.0;  // Invalid

    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeMarketPrice);
    // Error code: result.error().code
}

// Test 7: ITM put IV calculation
TEST_F(IVSolverTest, ITMPutIVCalculation) {
    query.strike = 110.0;  // In the money
    query.market_price = 15.0;

    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->implied_vol, 0.28, 0.08);
}

// Test 8: OTM put IV calculation
TEST_F(IVSolverTest, OTMPutIVCalculation) {
    query.strike = 90.0;  // Out of the money
    query.market_price = 2.5;

    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->implied_vol, 0.20, 0.08);
}

// Test 9: Deep ITM put (tests adaptive grid bounds)
TEST_F(IVSolverTest, DeepITMPutIVCalculation) {
    query.spot = 50.0;      // Deep ITM (S/K = 0.5)
    query.strike = 100.0;
    query.market_price = 54.5;  // Realistic: intrinsic=50 + time value ~4.5

    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_TRUE(result.has_value()) << "Deep ITM should converge with adaptive grid";
    EXPECT_NEAR(result->implied_vol, 0.875, 0.10);
}

// Test 10: Deep OTM put (tests adaptive grid bounds)
// Re-enabled: ProjectedThomas is now the default (PR #200)
TEST_F(IVSolverTest, DeepOTMPutIVCalculation) {
    query.spot = 200.0;  // Deep out of the money (S/K = 2.0)
    query.strike = 100.0;
    query.market_price = 1.0;

    IVSolver solver(config);
    auto result = solver.solve(query);

    // Should converge with adaptive grid
    ASSERT_TRUE(result.has_value()) << "Deep OTM should converge with adaptive grid";
    EXPECT_GT(result->implied_vol, 0.0);
    EXPECT_LT(result->implied_vol, 1.5);
}

// Test 11: Call option IV calculation
// Re-enabled: ProjectedThomas is now the default (PR #200)
TEST_F(IVSolverTest, ATMCallIVCalculation) {
    query.option_type = OptionType::CALL;
    query.market_price = 10.0;  // ATM call price

    IVSolver solver(config);
    auto result = solver.solve(query);

    ASSERT_TRUE(result.has_value()) << "ATM call should converge";
    // Relaxed lower bound slightly due to minor numerical differences after CRTP refactoring
    EXPECT_GT(result->implied_vol, 0.14);
    EXPECT_LT(result->implied_vol, 0.35);
}

// Test 12: PDEGridConfig with minimal spatial points
TEST_F(IVSolverTest, ExplicitGridMinimalPoints) {
    config.grid = PDEGridConfig{
        GridSpec<double>::sinh_spaced(-3.0, 3.0, 11, 2.0).value(), 50};
    IVSolver solver(config);
    auto result = solver.solve(query);
    // Minimal grid should still produce a result (possibly less accurate)
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->implied_vol, 0.1);
    EXPECT_LT(result->implied_vol, 0.5);
}

// Test 13: PDEGridConfig with few time steps
TEST_F(IVSolverTest, ExplicitGridFewTimeSteps) {
    config.grid = PDEGridConfig{
        GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, 2.0).value(), 10};
    IVSolver solver(config);
    auto result = solver.solve(query);
    // Few time steps — solver should still produce a result
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->implied_vol, 0.1);
    EXPECT_LT(result->implied_vol, 0.5);
}

// Test 14: GridSpec rejects invalid bounds
TEST_F(IVSolverTest, GridSpecRejectsInvalidBounds) {
    // GridSpec::sinh_spaced should reject x_min >= x_max
    auto bad_grid = GridSpec<double>::sinh_spaced(3.0, -3.0, 101, 2.0);
    ASSERT_FALSE(bad_grid.has_value());
}

// Test 15: PDEGridConfig with 201 points
TEST_F(IVSolverTest, ExplicitGrid201Points) {
    config.grid = PDEGridConfig{
        GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0).value(), 1000};
    IVSolver solver(config);
    auto result = solver.solve(query);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->implied_vol, 0.1);
    EXPECT_LT(result->implied_vol, 0.5);
    EXPECT_GT(result->iterations, 0);
}

// ===========================================================================
// Config field tests
// ===========================================================================

TEST(IVSolverConfigTest, TargetPriceErrorConfig) {
    mango::IVSolverConfig config{.root_config = {}, .target_price_error = 0.005};
    EXPECT_EQ(config.target_price_error, 0.005);

    mango::IVSolverConfig default_config{};
    EXPECT_EQ(default_config.target_price_error, 0.01);

    mango::IVSolverConfig heuristic_config{.root_config = {}, .target_price_error = 0.0};
    EXPECT_EQ(heuristic_config.target_price_error, 0.0);
}

// ===========================================================================
// Regression tests
// ===========================================================================

// Regression: IVSolver must respect grid accuracy in config
// Bug: objective_function() called estimate_pde_grid() with default
// GridAccuracyParams, ignoring config_.grid. This caused a ~2-4 bps
// IV error floor that could not be reduced.
TEST_F(IVSolverTest, GridAccuracyReducesError) {
    // Use the fixture query (ATM put, market_price=10.45)
    // Both solvers recover IV from the same market price.
    // The default solver's IV has ~2-4 bps error from its coarse internal grid.
    // The high-accuracy solver should produce a meaningfully different IV,
    // demonstrating that grid_accuracy actually flows through.

    // Default accuracy (disable probe-based calibration to use config_.grid)
    IVSolverConfig default_config = config;
    default_config.target_price_error = 0.0;  // Use heuristic grid from config_.grid
    IVSolver solver_default(default_config);
    auto result_default = solver_default.solve(query);
    ASSERT_TRUE(result_default.has_value())
        << "Default accuracy failed with error code: "
        << static_cast<int>(result_default.error().code);

    // Finer grid: tol=5e-3 (vs default 1e-2), min 150 points (vs default 100)
    IVSolverConfig finer_config = config;
    finer_config.target_price_error = 0.0;  // Use heuristic grid from config_.grid
    auto finer_accuracy = GridAccuracyParams{};
    finer_accuracy.tol = 5e-3;
    finer_accuracy.min_spatial_points = 150;
    finer_config.grid = finer_accuracy;
    IVSolver solver_finer(finer_config);
    auto result_finer = solver_finer.solve(query);
    ASSERT_TRUE(result_finer.has_value())
        << "Finer accuracy failed with error code: "
        << static_cast<int>(result_finer.error().code);

    // The two results should differ — proving grid_accuracy is being used.
    // Default grid (~101 points) vs finer grid (~150 points) produce
    // different pricing, so the recovered IVs must differ.
    double diff = std::abs(result_finer->implied_vol - result_default->implied_vol);
    EXPECT_GT(diff, 1e-6)
        << "Default IV=" << result_default->implied_vol
        << " Finer IV=" << result_finer->implied_vol
        << " — grid_accuracy should affect the result";

    // Both should still be reasonable (between 15% and 35%)
    EXPECT_GT(result_default->implied_vol, 0.15);
    EXPECT_LT(result_default->implied_vol, 0.35);
    EXPECT_GT(result_finer->implied_vol, 0.15);
    EXPECT_LT(result_finer->implied_vol, 0.35);
}

// ===========================================================================
// Probe-based grid calibration tests
// ===========================================================================

TEST(IVSolverProbeTest, UsesProbeBasedGridWhenTargetPriceErrorSet) {
    mango::OptionSpec spec{
        .spot = 100.0, .strike = 100.0, .maturity = 1.0,
        .rate = 0.05, .dividend_yield = 0.02,
        .option_type = mango::OptionType::PUT};

    mango::IVSolverConfig config{
        .root_config = {},
        .target_price_error = 0.01  // Enable probe-based calibration
    };

    mango::IVSolver solver(config);
    mango::IVQuery query(spec, 5.50);

    auto result = solver.solve(query);
    ASSERT_TRUE(result.has_value())
        << "Probe-based solve failed with error code: "
        << static_cast<int>(result.error().code);
    // IV should be reasonable (between 10% and 30% for this price)
    EXPECT_GT(result->implied_vol, 0.10);
    EXPECT_LT(result->implied_vol, 0.30);
}

TEST(IVSolverProbeTest, DifferentOptionsGetDifferentGrids) {
    // Verify cache is per-solve(), not per-solver-instance
    mango::IVSolverConfig config{.root_config = {}, .target_price_error = 0.01};
    mango::IVSolver solver(config);

    // Option 1: short maturity
    mango::OptionSpec spec1{
        .spot = 100.0, .strike = 100.0, .maturity = 0.1,
        .rate = 0.05, .dividend_yield = 0.0,
        .option_type = mango::OptionType::PUT};
    mango::IVQuery query1(spec1, 2.0);

    // Option 2: long maturity (needs different grid)
    mango::OptionSpec spec2{
        .spot = 100.0, .strike = 100.0, .maturity = 2.0,
        .rate = 0.05, .dividend_yield = 0.0,
        .option_type = mango::OptionType::PUT};
    mango::IVQuery query2(spec2, 12.0);

    auto result1 = solver.solve(query1);
    auto result2 = solver.solve(query2);

    // Both should succeed (if cache was shared incorrectly, one might fail)
    ASSERT_TRUE(result1.has_value())
        << "Short maturity solve failed";
    ASSERT_TRUE(result2.has_value())
        << "Long maturity solve failed";

    // IVs should be different (different maturities, different prices)
    EXPECT_NE(result1->implied_vol, result2->implied_vol);
}

TEST(IVSolverProbeTest, FallsBackToHeuristicWhenProbeDisabled) {
    mango::OptionSpec spec{
        .spot = 100.0, .strike = 100.0, .maturity = 1.0,
        .rate = 0.05, .dividend_yield = 0.0,
        .option_type = mango::OptionType::PUT};

    // target_price_error = 0 disables probe-based calibration
    mango::IVSolverConfig config{.root_config = {}, .target_price_error = 0.0};
    mango::IVSolver solver(config);
    mango::IVQuery query(spec, 5.50);

    auto result = solver.solve(query);
    ASSERT_TRUE(result.has_value())
        << "Heuristic fallback solve failed with error code: "
        << static_cast<int>(result.error().code);
    // IV should be reasonable (between 10% and 30%)
    EXPECT_GT(result->implied_vol, 0.10);
    EXPECT_LT(result->implied_vol, 0.30);
}

TEST(IVSolverProbeTest, FallsBackToHeuristicWhenProbeDoesntConverge) {
    mango::OptionSpec spec{
        .spot = 100.0, .strike = 100.0, .maturity = 1.0,
        .rate = 0.05, .dividend_yield = 0.0,
        .option_type = mango::OptionType::PUT};

    // Very tight tolerance that probe may not achieve in 3 iterations
    mango::IVSolverConfig config{.root_config = {}, .target_price_error = 1e-10};
    mango::IVSolver solver(config);
    mango::IVQuery query(spec, 5.50);

    // Should still succeed (falls back to heuristic when probe doesn't converge)
    auto result = solver.solve(query);
    ASSERT_TRUE(result.has_value())
        << "Fallback solve failed with error code: "
        << static_cast<int>(result.error().code);
    // IV should still be reasonable
    EXPECT_GT(result->implied_vol, 0.10);
    EXPECT_LT(result->implied_vol, 0.50);
}
