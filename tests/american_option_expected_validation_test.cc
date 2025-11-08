/**
 * @file american_option_expected_validation_test.cc
 * @brief Tests for expected-based validation of AmericanOption constructors
 */

#include "src/american_option.hpp"
#include <gtest/gtest.h>
#include <string>

namespace mango {
namespace {

// Test fixture for expected validation tests
class AmericanOptionExpectedValidationTest : public ::testing::Test {
protected:
    AmericanOptionParams valid_params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.2,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::CALL
    };

    AmericanOptionGrid valid_grid{};  // Use defaults
};

// Test valid parameters pass validation
TEST_F(AmericanOptionExpectedValidationTest, ValidParametersPass) {
    auto result = AmericanOptionParams::validate_expected(valid_params);
    EXPECT_TRUE(result.has_value());
}

// Test invalid strike (negative)
TEST_F(AmericanOptionExpectedValidationTest, InvalidStrikeNegative) {
    auto params = valid_params;
    params.strike = -100.0;

    auto result = AmericanOptionParams::validate_expected(params);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("Strike must be positive") != std::string::npos);
}

// Test invalid strike (zero)
TEST_F(AmericanOptionExpectedValidationTest, InvalidStrikeZero) {
    auto params = valid_params;
    params.strike = 0.0;

    auto result = AmericanOptionParams::validate_expected(params);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("Strike must be positive") != std::string::npos);
}

// Test invalid spot (negative)
TEST_F(AmericanOptionExpectedValidationTest, InvalidSpotNegative) {
    auto params = valid_params;
    params.spot = -50.0;

    auto result = AmericanOptionParams::validate_expected(params);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("Spot must be positive") != std::string::npos);
}

// Test invalid spot (zero)
TEST_F(AmericanOptionExpectedValidationTest, InvalidSpotZero) {
    auto params = valid_params;
    params.spot = 0.0;

    auto result = AmericanOptionParams::validate_expected(params);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("Spot must be positive") != std::string::npos);
}

// Test invalid maturity (negative)
TEST_F(AmericanOptionExpectedValidationTest, InvalidMaturityNegative) {
    auto params = valid_params;
    params.maturity = -1.0;

    auto result = AmericanOptionParams::validate_expected(params);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("Maturity must be positive") != std::string::npos);
}

// Test invalid maturity (zero)
TEST_F(AmericanOptionExpectedValidationTest, InvalidMaturityZero) {
    auto params = valid_params;
    params.maturity = 0.0;

    auto result = AmericanOptionParams::validate_expected(params);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("Maturity must be positive") != std::string::npos);
}

// Test invalid volatility (negative)
TEST_F(AmericanOptionExpectedValidationTest, InvalidVolatilityNegative) {
    auto params = valid_params;
    params.volatility = -0.2;

    auto result = AmericanOptionParams::validate_expected(params);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("Volatility must be positive") != std::string::npos);
}

// Test invalid volatility (zero)
TEST_F(AmericanOptionExpectedValidationTest, InvalidVolatilityZero) {
    auto params = valid_params;
    params.volatility = 0.0;

    auto result = AmericanOptionParams::validate_expected(params);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("Volatility must be positive") != std::string::npos);
}

// Test negative rate (should be allowed)
TEST_F(AmericanOptionExpectedValidationTest, NegativeRateAllowed) {
    auto params = valid_params;
    params.rate = -0.02;  // Negative rate (valid for EUR, JPY, CHF markets)

    auto result = AmericanOptionParams::validate_expected(params);
    EXPECT_TRUE(result.has_value());
}

// Test zero rate (should be allowed)
TEST_F(AmericanOptionExpectedValidationTest, ZeroRateAllowed) {
    auto params = valid_params;
    params.rate = 0.0;

    auto result = AmericanOptionParams::validate_expected(params);
    EXPECT_TRUE(result.has_value());
}

// Test invalid continuous dividend yield (negative)
TEST_F(AmericanOptionExpectedValidationTest, InvalidContinuousDividendYieldNegative) {
    auto params = valid_params;
    params.continuous_dividend_yield = -0.02;

    auto result = AmericanOptionParams::validate_expected(params);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("Continuous dividend yield must be non-negative") != std::string::npos);
}

// Test zero continuous dividend yield (should be allowed)
TEST_F(AmericanOptionExpectedValidationTest, ZeroContinuousDividendYieldAllowed) {
    auto params = valid_params;
    params.continuous_dividend_yield = 0.0;

    auto result = AmericanOptionParams::validate_expected(params);
    EXPECT_TRUE(result.has_value());
}

// Test valid discrete dividends
TEST_F(AmericanOptionExpectedValidationTest, ValidDiscreteDividends) {
    auto params = valid_params;
    params.discrete_dividends = {
        {0.5, 2.0},   // Valid: time in [0, maturity], amount positive
        {0.25, 1.5}   // Valid: time in [0, maturity], amount positive
    };

    auto result = AmericanOptionParams::validate_expected(params);
    EXPECT_TRUE(result.has_value());
}

// Test invalid discrete dividend time (negative)
TEST_F(AmericanOptionExpectedValidationTest, InvalidDiscreteDividendTimeNegative) {
    auto params = valid_params;
    params.discrete_dividends = {
        {-0.1, 2.0}  // Invalid: negative time
    };

    auto result = AmericanOptionParams::validate_expected(params);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("Discrete dividend time must be in [0, maturity]") != std::string::npos);
}

// Test invalid discrete dividend time (exceeds maturity)
TEST_F(AmericanOptionExpectedValidationTest, InvalidDiscreteDividendTimeExceedsMaturity) {
    auto params = valid_params;
    params.discrete_dividends = {
        {1.5, 2.0}  // Invalid: time > maturity (1.0)
    };

    auto result = AmericanOptionParams::validate_expected(params);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("Discrete dividend time must be in [0, maturity]") != std::string::npos);
}

// Test invalid discrete dividend amount (negative)
TEST_F(AmericanOptionExpectedValidationTest, InvalidDiscreteDividendAmountNegative) {
    auto params = valid_params;
    params.discrete_dividends = {
        {0.5, -2.0}  // Invalid: negative amount
    };

    auto result = AmericanOptionParams::validate_expected(params);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("Discrete dividend amount must be non-negative") != std::string::npos);
}

// Test zero discrete dividend amount (should be allowed)
TEST_F(AmericanOptionExpectedValidationTest, ZeroDiscreteDividendAmountAllowed) {
    auto params = valid_params;
    params.discrete_dividends = {
        {0.5, 0.0}  // Valid: zero amount
    };

    auto result = AmericanOptionParams::validate_expected(params);
    EXPECT_TRUE(result.has_value());
}

// Test multiple validation errors (should report first error)
TEST_F(AmericanOptionExpectedValidationTest, MultipleErrorsReportsFirst) {
    auto params = valid_params;
    params.strike = -100.0;  // Invalid
    params.spot = 0.0;       // Also invalid
    params.maturity = -1.0;  // Also invalid

    auto result = AmericanOptionParams::validate_expected(params);
    EXPECT_FALSE(result.has_value());
    // Should report the first error encountered (strike validation)
    EXPECT_TRUE(result.error().find("Strike must be positive") != std::string::npos);
}

// Test AmericanOptionGrid validation - valid grid
TEST_F(AmericanOptionExpectedValidationTest, ValidGrid) {
    auto result = AmericanOptionGrid::validate_expected(valid_grid);
    EXPECT_TRUE(result.has_value());
}

// Test invalid grid n_space (too small)
TEST_F(AmericanOptionExpectedValidationTest, InvalidGridNSpaceTooSmall) {
    auto grid = valid_grid;
    grid.n_space = 5;  // Too small

    auto result = AmericanOptionGrid::validate_expected(grid);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("n_space must be >= 10") != std::string::npos);
}

// Test invalid grid n_time (too small)
TEST_F(AmericanOptionExpectedValidationTest, InvalidGridNTimeTooSmall) {
    auto grid = valid_grid;
    grid.n_time = 5;  // Too small

    auto result = AmericanOptionGrid::validate_expected(grid);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("n_time must be >= 10") != std::string::npos);
}

// Test invalid grid bounds (x_min >= x_max)
TEST_F(AmericanOptionExpectedValidationTest, InvalidGridBounds) {
    auto grid = valid_grid;
    grid.x_min = 3.0;
    grid.x_max = -3.0;  // x_min >= x_max

    auto result = AmericanOptionGrid::validate_expected(grid);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("x_min must be < x_max") != std::string::npos);
}

// Test factory method with valid parameters
TEST_F(AmericanOptionExpectedValidationTest, FactoryMethodValid) {
    auto result = AmericanOptionSolver::create(valid_params, valid_grid);
    EXPECT_TRUE(result.has_value());

    // Should be able to solve
    auto solution = result.value().solve();
    EXPECT_TRUE(solution.has_value());
}

// Test factory method with invalid parameters
TEST_F(AmericanOptionExpectedValidationTest, FactoryMethodInvalid) {
    auto params = valid_params;
    params.strike = -100.0;  // Invalid

    auto result = AmericanOptionSolver::create(params, valid_grid);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("Strike must be positive") != std::string::npos);
}

// Test backward compatibility - existing constructor still throws
TEST_F(AmericanOptionExpectedValidationTest, BackwardCompatibilityConstructorThrows) {
    auto params = valid_params;
    params.strike = -100.0;  // Invalid

    EXPECT_THROW({
        AmericanOptionSolver solver(params, valid_grid);
    }, std::invalid_argument);
}

// Test backward compatibility - existing constructor works with valid params
TEST_F(AmericanOptionExpectedValidationTest, BackwardCompatibilityConstructorValid) {
    EXPECT_NO_THROW({
        AmericanOptionSolver solver(valid_params, valid_grid);
    });
}

// Test workspace factory method with valid parameters
TEST_F(AmericanOptionExpectedValidationTest, WorkspaceFactoryMethodValid) {
    auto workspace = std::make_shared<SliceSolverWorkspace>(valid_grid.x_min, valid_grid.x_max, valid_grid.n_space);

    auto result = AmericanOptionSolver::create_with_workspace(valid_params, valid_grid, workspace);
    EXPECT_TRUE(result.has_value());

    // Should be able to solve
    auto solution = result.value().solve();
    EXPECT_TRUE(solution.has_value());
}

// Test workspace factory method with invalid parameters
TEST_F(AmericanOptionExpectedValidationTest, WorkspaceFactoryMethodInvalid) {
    auto workspace = std::make_shared<SliceSolverWorkspace>(valid_grid.x_min, valid_grid.x_max, valid_grid.n_space);

    auto params = valid_params;
    params.volatility = -0.2;  // Invalid

    auto result = AmericanOptionSolver::create_with_workspace(params, valid_grid, workspace);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("Volatility must be positive") != std::string::npos);
}

}  // namespace
}  // namespace mango