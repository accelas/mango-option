#include <gtest/gtest.h>
#include "src/option/iv_solver_fdm.hpp"
#include "src/option/iv_result.hpp"
#include "src/support/error_types.hpp"

using namespace mango;

// Test for new std::expected signature (Task 2.1)
// NOTE: This test calls solve_impl() directly to verify the new signature,
// bypassing the base class which still returns IVResult (will be updated in Task 3).
TEST(IVSolverFDMExpected, ReturnsExpectedType) {
    // Simple test to verify std::expected signature compiles
    IVQuery query{
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.0,    // dividend_yield
        OptionType::PUT,
        10.0    // market_price
    };

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);

    // Call solve_impl() directly (not solve() which goes through base class)
    auto result = solver.solve_impl(query);

    // Verify it compiles and returns expected type
    static_assert(std::is_same_v<decltype(result), std::expected<IVSuccess, IVError>>);

    // Verify placeholder implementation returns success
    EXPECT_TRUE(result.has_value());
    if (result.has_value()) {
        EXPECT_EQ(result->implied_vol, 0.20);  // Placeholder value
        EXPECT_EQ(result->iterations, 0);
    }
}

// Task 2.2 Validation Tests: These tests verify that solve_impl() returns
// appropriate IVError codes for invalid inputs

TEST(IVSolverFDMExpected, ValidationNegativeSpot) {
    IVQuery query{
        -100.0,  // Invalid: negative spot
        100.0,
        1.0,
        0.05,
        0.0,
        OptionType::PUT,
        10.0
    };

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);
    auto result = solver.solve_impl(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeSpot);
    EXPECT_FALSE(result.error().message.empty());
}

TEST(IVSolverFDMExpected, ValidationNegativeStrike) {
    IVQuery query{
        100.0,
        -100.0,  // Invalid: negative strike
        1.0,
        0.05,
        0.0,
        OptionType::PUT,
        10.0
    };

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);
    auto result = solver.solve_impl(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeStrike);
    EXPECT_FALSE(result.error().message.empty());
}

TEST(IVSolverFDMExpected, ValidationNegativeMaturity) {
    IVQuery query{
        100.0,
        100.0,
        -1.0,  // Invalid: negative maturity
        0.05,
        0.0,
        OptionType::PUT,
        10.0
    };

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);
    auto result = solver.solve_impl(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeMaturity);
    EXPECT_FALSE(result.error().message.empty());
}

TEST(IVSolverFDMExpected, ValidationNegativeMarketPrice) {
    IVQuery query{
        100.0,
        100.0,
        1.0,
        0.05,
        0.0,
        OptionType::PUT,
        -10.0  // Invalid: negative market price
    };

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);
    auto result = solver.solve_impl(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeMarketPrice);
    EXPECT_FALSE(result.error().message.empty());
}

TEST(IVSolverFDMExpected, ValidationArbitrageCallExceedsSpot) {
    // Call price cannot exceed spot price
    IVQuery query{
        100.0,
        100.0,
        1.0,
        0.05,
        0.0,
        OptionType::CALL,
        150.0  // Invalid: call price > spot (arbitrage)
    };

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);
    auto result = solver.solve_impl(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::ArbitrageViolation);
    EXPECT_FALSE(result.error().message.empty());
}

TEST(IVSolverFDMExpected, ValidationArbitragePutExceedsStrike) {
    // Put price cannot exceed strike price
    IVQuery query{
        100.0,
        100.0,
        1.0,
        0.05,
        0.0,
        OptionType::PUT,
        150.0  // Invalid: put price > strike (arbitrage)
    };

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);
    auto result = solver.solve_impl(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::ArbitrageViolation);
    EXPECT_FALSE(result.error().message.empty());
}

TEST(IVSolverFDMExpected, ValidationPriceBelowIntrinsicCall) {
    // Market price must be >= intrinsic value
    // For call: intrinsic = max(S - K, 0) = max(110 - 100, 0) = 10
    IVQuery query{
        110.0,  // spot
        100.0,  // strike
        1.0,
        0.05,
        0.0,
        OptionType::CALL,
        5.0  // Invalid: market price < intrinsic (5 < 10)
    };

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);
    auto result = solver.solve_impl(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::ArbitrageViolation);
    EXPECT_FALSE(result.error().message.empty());
}

TEST(IVSolverFDMExpected, ValidationPriceBelowIntrinsicPut) {
    // Market price must be >= intrinsic value
    // For put: intrinsic = max(K - S, 0) = max(110 - 100, 0) = 10
    IVQuery query{
        100.0,  // spot
        110.0,  // strike
        1.0,
        0.05,
        0.0,
        OptionType::PUT,
        5.0  // Invalid: market price < intrinsic (5 < 10)
    };

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);
    auto result = solver.solve_impl(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::ArbitrageViolation);
    EXPECT_FALSE(result.error().message.empty());
}

TEST(IVSolverFDMExpected, ValidationZeroSpot) {
    IVQuery query{
        0.0,  // Invalid: zero spot
        100.0,
        1.0,
        0.05,
        0.0,
        OptionType::PUT,
        10.0
    };

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);
    auto result = solver.solve_impl(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeSpot);
    EXPECT_FALSE(result.error().message.empty());
}

TEST(IVSolverFDMExpected, ValidationZeroStrike) {
    IVQuery query{
        100.0,
        0.0,  // Invalid: zero strike
        1.0,
        0.05,
        0.0,
        OptionType::PUT,
        10.0
    };

    IVSolverFDMConfig config;
    IVSolverFDM solver(config);
    auto result = solver.solve_impl(query);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IVErrorCode::NegativeStrike);
    EXPECT_FALSE(result.error().message.empty());
}
