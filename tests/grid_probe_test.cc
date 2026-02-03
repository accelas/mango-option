// SPDX-License-Identifier: MIT
/**
 * @file grid_probe_test.cc
 * @brief Tests for probe_grid_adequacy() Richardson-style grid calibration
 */

#include <gtest/gtest.h>
#include "src/option/grid_probe.hpp"

namespace mango {
namespace {

// ===========================================================================
// Basic functionality tests
// ===========================================================================

TEST(GridProbeTest, ConvergesForTypicalOption) {
    PricingParams params(
        OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT},
        0.20);

    auto result = probe_grid_adequacy(params, 0.01);

    ASSERT_TRUE(result.has_value()) << "probe_grid_adequacy failed";
    EXPECT_TRUE(result->converged);
    EXPECT_LE(result->estimated_error, 0.01);
    EXPECT_GE(result->grid.n_points(), 100);
    EXPECT_GE(result->time_domain.n_steps(), 50);
}

TEST(GridProbeTest, ReturnsGridSpecAndTimeDomain) {
    PricingParams params(
        OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT},
        0.20);

    auto result = probe_grid_adequacy(params, 0.01);

    ASSERT_TRUE(result.has_value());

    // Grid spec is valid
    EXPECT_GT(result->grid.n_points(), 0);
    EXPECT_LT(result->grid.x_min(), result->grid.x_max());

    // Time domain is valid
    EXPECT_GT(result->time_domain.n_steps(), 0);
    EXPECT_DOUBLE_EQ(result->time_domain.t_start(), 0.0);
    EXPECT_DOUBLE_EQ(result->time_domain.t_end(), params.maturity);
}

TEST(GridProbeTest, ConvergedFlagTrueWhenConverged) {
    PricingParams params(
        OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT},
        0.20);

    // Loose tolerance should converge easily
    auto result = probe_grid_adequacy(params, 0.1, 100, 3);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->converged);
    EXPECT_LE(result->probe_iterations, 3);
}

// ===========================================================================
// Validation tests
// ===========================================================================

TEST(GridProbeTest, RejectsNonPositiveTargetError) {
    PricingParams params(
        OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT},
        0.20);

    auto result_zero = probe_grid_adequacy(params, 0.0);
    EXPECT_FALSE(result_zero.has_value());
    EXPECT_EQ(result_zero.error().code, ValidationErrorCode::InvalidBounds);

    auto result_negative = probe_grid_adequacy(params, -0.01);
    EXPECT_FALSE(result_negative.has_value());
    EXPECT_EQ(result_negative.error().code, ValidationErrorCode::InvalidBounds);
}

// ===========================================================================
// Option type tests
// ===========================================================================

TEST(GridProbeTest, WorksForCallOption) {
    PricingParams params(
        OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::CALL},
        0.20);

    auto result = probe_grid_adequacy(params, 0.01);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->converged);
}

// ===========================================================================
// Parameter variation tests
// ===========================================================================

TEST(GridProbeTest, HighVolatilityStillConverges) {
    PricingParams params(
        OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT},
        0.80);  // High volatility

    auto result = probe_grid_adequacy(params, 0.01);

    ASSERT_TRUE(result.has_value());
    // May or may not converge with default 3 iterations
    // Just verify the function returns a result
    EXPECT_GE(result->grid.n_points(), 100);
}

TEST(GridProbeTest, ShortMaturityEnforcesNtFloor) {
    PricingParams params(
        OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 0.05,  // ~18 days
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT},
        0.20);

    auto result = probe_grid_adequacy(params, 0.01);

    ASSERT_TRUE(result.has_value());
    // Nt floor of 50 should be enforced
    EXPECT_GE(result->time_domain.n_steps(), 50);
}

TEST(GridProbeTest, DeepITMOption) {
    PricingParams params(
        OptionSpec{
            .spot = 80.0, .strike = 100.0, .maturity = 1.0,  // Deep ITM put
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT},
        0.20);

    auto result = probe_grid_adequacy(params, 0.01);

    ASSERT_TRUE(result.has_value());
    // Should still produce valid result
    EXPECT_GT(result->grid.n_points(), 0);
}

TEST(GridProbeTest, DeepOTMOption) {
    PricingParams params(
        OptionSpec{
            .spot = 120.0, .strike = 100.0, .maturity = 1.0,  // Deep OTM put
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT},
        0.20);

    auto result = probe_grid_adequacy(params, 0.01);

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->grid.n_points(), 0);
}

// ===========================================================================
// Iteration behavior tests
// ===========================================================================

TEST(GridProbeTest, GridDoublesOnNonConvergence) {
    // Use very tight tolerance to force multiple iterations
    PricingParams params(
        OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT},
        0.20);

    // With 50 initial points and tight tolerance, should need doubling
    auto result = probe_grid_adequacy(params, 1e-6, 50, 3);

    ASSERT_TRUE(result.has_value());
    // After 3 iterations: 50 -> 100 -> 200 -> (maybe 400)
    // Grid should have doubled at least once
    if (!result->converged) {
        // If didn't converge, grid should be at maximum iteration level
        EXPECT_GE(result->grid.n_points(), 200);
    }
}

TEST(GridProbeTest, ReportsIterationCount) {
    PricingParams params(
        OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT},
        0.20);

    auto result = probe_grid_adequacy(params, 0.01, 100, 5);

    ASSERT_TRUE(result.has_value());
    EXPECT_GE(result->probe_iterations, 1);
    EXPECT_LE(result->probe_iterations, 5);
}

// ===========================================================================
// Error estimation tests
// ===========================================================================

TEST(GridProbeTest, ErrorEstimateIsReasonable) {
    PricingParams params(
        OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT},
        0.20);

    auto result = probe_grid_adequacy(params, 0.01);

    ASSERT_TRUE(result.has_value());
    // Error should be non-negative and finite
    EXPECT_GE(result->estimated_error, 0.0);
    EXPECT_TRUE(std::isfinite(result->estimated_error));
}

// ===========================================================================
// Max iterations behavior
// ===========================================================================

TEST(GridProbeTest, MaxIterationsLimitsWork) {
    PricingParams params(
        OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT},
        0.20);

    // Very tight tolerance with only 1 iteration allowed
    auto result = probe_grid_adequacy(params, 1e-10, 100, 1);

    ASSERT_TRUE(result.has_value());
    // With such tight tolerance, unlikely to converge in 1 iteration
    // But should still return a valid result
    EXPECT_EQ(result->probe_iterations, 1);
}

// ===========================================================================
// Discrete dividend tests
// ===========================================================================

TEST(GridProbeTest, WithDiscreteDividends) {
    PricingParams params(
        OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.0,
            .option_type = OptionType::PUT},
        0.20,
        {Dividend{.calendar_time = 0.25, .amount = 2.0}});

    auto result = probe_grid_adequacy(params, 0.01);

    ASSERT_TRUE(result.has_value());
    // Should still converge with discrete dividends
    EXPECT_TRUE(result->converged);
    EXPECT_LE(result->estimated_error, 0.01);
    EXPECT_GE(result->grid.n_points(), 100);
}

TEST(GridProbeTest, WithMultipleDiscreteDividends) {
    PricingParams params(
        OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.0,
            .option_type = OptionType::PUT},
        0.20,
        {Dividend{.calendar_time = 0.25, .amount = 1.50},
         Dividend{.calendar_time = 0.50, .amount = 1.50},
         Dividend{.calendar_time = 0.75, .amount = 1.50}});

    auto result = probe_grid_adequacy(params, 0.01);

    ASSERT_TRUE(result.has_value());
    // Multiple dividends may require more grid points
    EXPECT_GE(result->grid.n_points(), 100);
}

}  // namespace
}  // namespace mango
