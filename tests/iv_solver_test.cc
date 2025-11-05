#include <gtest/gtest.h>
#include "src/cpp/iv_solver.hpp"
#include <cmath>

using namespace mango;

class IVSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default parameters for testing
        params = IVParams{
            .spot_price = 100.0,
            .strike = 100.0,
            .time_to_maturity = 1.0,
            .risk_free_rate = 0.05,
            .market_price = 10.45,
            .is_call = false  // American put
        };

        config = IVConfig{
            .root_config = RootFindingConfig{
                .max_iter = 100,
                .tolerance = 1e-6,
                .brent_tol_abs = 1e-6
            },
            .grid_n_space = 101,
            .grid_n_time = 1000,
            .grid_s_max = 200.0
        };
    }

    IVParams params;
    IVConfig config;
};

// Test 1: Construction should succeed (TDD - this should fail initially)
TEST_F(IVSolverTest, ConstructionSucceeds) {
    // Create solver - should compile but not implement solve() yet
    IVSolver solver(params, config);

    // Construction itself should succeed
    SUCCEED();
}

// Test 2: Basic ATM put IV calculation (should converge now)
TEST_F(IVSolverTest, ATMPutIVCalculation) {
    IVSolver solver(params, config);

    IVResult result = solver.solve();

    // Should converge with real implementation
    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.15);
    EXPECT_LT(result.implied_vol, 0.35);
    EXPECT_GT(result.iterations, 0);
}

// Test 3: Invalid parameters should be caught
TEST_F(IVSolverTest, InvalidSpotPrice) {
    params.spot_price = -100.0;  // Invalid

    IVSolver solver(params, config);
    IVResult result = solver.solve();

    EXPECT_FALSE(result.converged);
    EXPECT_TRUE(result.failure_reason.has_value());
}

// Test 4: Invalid strike price
TEST_F(IVSolverTest, InvalidStrike) {
    params.strike = 0.0;  // Invalid

    IVSolver solver(params, config);
    IVResult result = solver.solve();

    EXPECT_FALSE(result.converged);
    EXPECT_TRUE(result.failure_reason.has_value());
}

// Test 5: Invalid time to maturity
TEST_F(IVSolverTest, InvalidTimeToMaturity) {
    params.time_to_maturity = -1.0;  // Invalid

    IVSolver solver(params, config);
    IVResult result = solver.solve();

    EXPECT_FALSE(result.converged);
    EXPECT_TRUE(result.failure_reason.has_value());
}

// Test 6: Invalid market price
TEST_F(IVSolverTest, InvalidMarketPrice) {
    params.market_price = -5.0;  // Invalid

    IVSolver solver(params, config);
    IVResult result = solver.solve();

    EXPECT_FALSE(result.converged);
    EXPECT_TRUE(result.failure_reason.has_value());
}

// Test 7: ITM put IV calculation
TEST_F(IVSolverTest, ITMPutIVCalculation) {
    params.strike = 110.0;  // In the money
    params.market_price = 15.0;

    IVSolver solver(params, config);
    IVResult result = solver.solve();

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.0);
    EXPECT_LT(result.implied_vol, 1.0);
}

// Test 8: OTM put IV calculation
TEST_F(IVSolverTest, OTMPutIVCalculation) {
    params.strike = 90.0;  // Out of the money
    params.market_price = 2.5;

    IVSolver solver(params, config);
    IVResult result = solver.solve();

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.0);
    EXPECT_LT(result.implied_vol, 1.0);
}
