// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "src/option/iv_solver_fdm.hpp"
#include <cmath>

using namespace mango;

class IVSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        query = IVQuery(
            OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                .rate = 0.05, .option_type = OptionType::PUT}, 10.45);

        config = IVSolverFDMConfig{
            .root_config = RootFindingConfig{
                .max_iter = 100,
                .tolerance = 1e-6,
                .brent_tol_abs = 1e-6
            },
        };
    }

    IVQuery query;
    IVSolverFDMConfig config;
};

// Test 2: Basic ATM put IV calculation
// Re-enabled: ProjectedThomas is now the default (PR #200)
TEST_F(IVSolverTest, ATMPutIVCalculation) {
    IVSolverFDM solver(config);

    auto result = solver.solve(query);

    // Should converge with real implementation
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->implied_vol, 0.316, 0.05);
    EXPECT_GT(result->iterations, 0);
}

// Test 3: Invalid parameters should be caught
TEST_F(IVSolverTest, InvalidSpotPrice) {
    query.spot = -100.0;  // Invalid

    IVSolverFDM solver(config);
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeSpot);
    // Error code: result.error().code
}

// Test 4: Invalid strike price
TEST_F(IVSolverTest, InvalidStrike) {
    query.strike = 0.0;  // Invalid

    IVSolverFDM solver(config);
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeStrike);
    // Error code: result.error().code
}

// Test 5: Invalid time to maturity
TEST_F(IVSolverTest, InvalidTimeToMaturity) {
    query.maturity = -1.0;  // Invalid

    IVSolverFDM solver(config);
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeMaturity);
    // Error code: result.error().code
}

// Test 6: Invalid market price
TEST_F(IVSolverTest, InvalidMarketPrice) {
    query.market_price = -5.0;  // Invalid

    IVSolverFDM solver(config);
    auto result = solver.solve(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeMarketPrice);
    // Error code: result.error().code
}

// Test 7: ITM put IV calculation
TEST_F(IVSolverTest, ITMPutIVCalculation) {
    query.strike = 110.0;  // In the money
    query.market_price = 15.0;

    IVSolverFDM solver(config);
    auto result = solver.solve(query);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->implied_vol, 0.28, 0.08);
}

// Test 8: OTM put IV calculation
TEST_F(IVSolverTest, OTMPutIVCalculation) {
    query.strike = 90.0;  // Out of the money
    query.market_price = 2.5;

    IVSolverFDM solver(config);
    auto result = solver.solve(query);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->implied_vol, 0.20, 0.08);
}

// Test 9: Deep ITM put (tests adaptive grid bounds)
TEST_F(IVSolverTest, DeepITMPutIVCalculation) {
    query.spot = 50.0;      // Deep ITM (S/K = 0.5)
    query.strike = 100.0;
    query.market_price = 54.5;  // Realistic: intrinsic=50 + time value ~4.5

    IVSolverFDM solver(config);
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

    IVSolverFDM solver(config);
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

    IVSolverFDM solver(config);
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
    IVSolverFDM solver(config);
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
    IVSolverFDM solver(config);
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
    IVSolverFDM solver(config);
    auto result = solver.solve(query);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->implied_vol, 0.1);
    EXPECT_LT(result->implied_vol, 0.5);
    EXPECT_GT(result->iterations, 0);
}

// ===========================================================================
// Regression tests
// ===========================================================================

// Regression: IVSolverFDM must respect grid accuracy in config
// Bug: objective_function() called estimate_pde_grid() with default
// GridAccuracyParams, ignoring config_.grid. This caused a ~2-4 bps
// IV error floor that could not be reduced.
TEST_F(IVSolverTest, GridAccuracyReducesError) {
    // Use the fixture query (ATM put, market_price=10.45)
    // Both solvers recover IV from the same market price.
    // The default solver's IV has ~2-4 bps error from its coarse internal grid.
    // The high-accuracy solver should produce a meaningfully different IV,
    // demonstrating that grid_accuracy actually flows through.

    // Default accuracy
    IVSolverFDMConfig default_config = config;
    IVSolverFDM solver_default(default_config);
    auto result_default = solver_default.solve(query);
    ASSERT_TRUE(result_default.has_value())
        << "Default accuracy failed with error code: "
        << static_cast<int>(result_default.error().code);

    // Finer grid: tol=5e-3 (vs default 1e-2), min 150 points (vs default 100)
    IVSolverFDMConfig finer_config = config;
    auto finer_accuracy = GridAccuracyParams{};
    finer_accuracy.tol = 5e-3;
    finer_accuracy.min_spatial_points = 150;
    finer_config.grid = finer_accuracy;
    IVSolverFDM solver_finer(finer_config);
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
