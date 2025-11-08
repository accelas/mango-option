#include "src/grid.hpp"
#include "src/expected.hpp"
#include <gtest/gtest.h>

using namespace mango;

// Test uniform grid creation with expected pattern
TEST(GridSpecExpectedTest, UniformValid) {
    auto result = GridSpec<>::uniform(0.0, 1.0, 11);
    EXPECT_TRUE(result.has_value());

    auto spec = result.value();
    EXPECT_EQ(spec.type(), GridSpec<>::Type::Uniform);
    EXPECT_DOUBLE_EQ(spec.x_min(), 0.0);
    EXPECT_DOUBLE_EQ(spec.x_max(), 1.0);
    EXPECT_EQ(spec.n_points(), 11);

    // Generate and verify the actual grid
    auto grid = spec.generate();
    EXPECT_EQ(grid.size(), 11);
    EXPECT_DOUBLE_EQ(grid[0], 0.0);
    EXPECT_DOUBLE_EQ(grid[10], 1.0);
}

TEST(GridSpecExpectedTest, UniformInvalidTooFewPoints) {
    auto result = GridSpec<>::uniform(0.0, 1.0, 1);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Grid must have at least 2 points");
}

TEST(GridSpecExpectedTest, UniformInvalidMinMax) {
    auto result = GridSpec<>::uniform(1.0, 0.0, 10);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "x_min must be less than x_max");

    auto result_equal = GridSpec<>::uniform(1.0, 1.0, 10);
    EXPECT_FALSE(result_equal.has_value());
    EXPECT_EQ(result_equal.error(), "x_min must be less than x_max");
}

// Test log-spaced grid creation with expected pattern
TEST(GridSpecExpectedTest, LogSpacedValid) {
    auto result = GridSpec<>::log_spaced(1.0, 100.0, 5);
    EXPECT_TRUE(result.has_value());

    auto spec = result.value();
    EXPECT_EQ(spec.type(), GridSpec<>::Type::LogSpaced);
    EXPECT_DOUBLE_EQ(spec.x_min(), 1.0);
    EXPECT_DOUBLE_EQ(spec.x_max(), 100.0);
    EXPECT_EQ(spec.n_points(), 5);

    // Generate and verify the actual grid
    auto grid = spec.generate();
    EXPECT_EQ(grid.size(), 5);
    EXPECT_DOUBLE_EQ(grid[0], 1.0);
    EXPECT_DOUBLE_EQ(grid[4], 100.0);
    // Geometric spacing: midpoint in log space should be sqrt(1*100) = 10
    EXPECT_NEAR(grid[2], 10.0, 1e-10);
}

TEST(GridSpecExpectedTest, LogSpacedInvalidNonPositiveMin) {
    auto result_zero = GridSpec<>::log_spaced(0.0, 1.0, 10);
    EXPECT_FALSE(result_zero.has_value());
    EXPECT_EQ(result_zero.error(), "Log-spaced grid requires positive bounds");

    auto result_negative = GridSpec<>::log_spaced(-0.1, 1.0, 10);
    EXPECT_FALSE(result_negative.has_value());
    EXPECT_EQ(result_negative.error(), "Log-spaced grid requires positive bounds");
}

TEST(GridSpecExpectedTest, LogSpacedInvalidNonPositiveMax) {
    auto result = GridSpec<>::log_spaced(1.0, 0.0, 10);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Log-spaced grid requires positive bounds");
}

TEST(GridSpecExpectedTest, LogSpacedInvalidMinMax) {
    auto result = GridSpec<>::log_spaced(1.0, 0.5, 10);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "x_min must be less than x_max");
}

TEST(GridSpecExpectedTest, LogSpacedInvalidTooFewPoints) {
    auto result = GridSpec<>::log_spaced(1.0, 10.0, 1);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Grid must have at least 2 points");
}

// Test sinh-spaced grid creation with expected pattern
TEST(GridSpecExpectedTest, SinhSpacedValid) {
    auto result = GridSpec<>::sinh_spaced(0.0, 1.0, 11, 2.0);
    EXPECT_TRUE(result.has_value());

    auto spec = result.value();
    EXPECT_EQ(spec.type(), GridSpec<>::Type::SinhSpaced);
    EXPECT_DOUBLE_EQ(spec.x_min(), 0.0);
    EXPECT_DOUBLE_EQ(spec.x_max(), 1.0);
    EXPECT_EQ(spec.n_points(), 11);
    EXPECT_DOUBLE_EQ(spec.concentration(), 2.0);

    // Generate and verify the actual grid
    auto grid = spec.generate();
    EXPECT_EQ(grid.size(), 11);
    EXPECT_DOUBLE_EQ(grid[0], 0.0);
    EXPECT_DOUBLE_EQ(grid[10], 1.0);
    EXPECT_DOUBLE_EQ(grid[5], 0.5);  // Center point should be at midpoint
}

TEST(GridSpecExpectedTest, SinhSpacedDefaultConcentration) {
    auto result = GridSpec<>::sinh_spaced(0.0, 1.0, 11);
    EXPECT_TRUE(result.has_value());

    auto spec = result.value();
    EXPECT_DOUBLE_EQ(spec.concentration(), 1.0);  // Default value
}

TEST(GridSpecExpectedTest, SinhSpacedInvalidMinMax) {
    auto result = GridSpec<>::sinh_spaced(1.0, 0.0, 10);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "x_min must be less than x_max");

    auto result_equal = GridSpec<>::sinh_spaced(1.0, 1.0, 10);
    EXPECT_FALSE(result_equal.has_value());
    EXPECT_EQ(result_equal.error(), "x_min must be less than x_max");
}

TEST(GridSpecExpectedTest, SinhSpacedInvalidTooFewPoints) {
    auto result = GridSpec<>::sinh_spaced(0.0, 1.0, 1);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Grid must have at least 2 points");
}

TEST(GridSpecExpectedTest, SinhSpacedInvalidConcentration) {
    auto result = GridSpec<>::sinh_spaced(0.0, 1.0, 10, 0.0);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Concentration parameter must be positive");

    auto result_negative = GridSpec<>::sinh_spaced(0.0, 1.0, 10, -1.0);
    EXPECT_FALSE(result_negative.has_value());
    EXPECT_EQ(result_negative.error(), "Concentration parameter must be positive");
}

// Test error handling with multiple validation failures
TEST(GridSpecExpectedTest, MultipleErrorsUniform) {
    // Both n_points < 2 and x_min >= x_max - should report first error
    auto result = GridSpec<>::uniform(1.0, 0.0, 1);
    EXPECT_FALSE(result.has_value());
    // Should report the first validation error encountered
    EXPECT_TRUE(result.error().find("Grid must have at least 2 points") != std::string::npos ||
                result.error().find("x_min must be less than x_max") != std::string::npos);
}

TEST(GridSpecExpectedTest, MultipleErrorsLogSpaced) {
    // Both n_points < 2 and non-positive bounds - should report first error
    auto result = GridSpec<>::log_spaced(0.0, 1.0, 1);
    EXPECT_FALSE(result.has_value());
    // Should report the first validation error encountered
    EXPECT_TRUE(result.error().find("Grid must have at least 2 points") != std::string::npos ||
                result.error().find("Log-spaced grid requires positive bounds") != std::string::npos);
}

TEST(GridSpecExpectedTest, MultipleErrorsSinhSpaced) {
    // Both n_points < 2 and invalid concentration - should report first error
    auto result = GridSpec<>::sinh_spaced(0.0, 1.0, 1, 0.0);
    EXPECT_FALSE(result.has_value());
    // Should report the first validation error encountered
    EXPECT_TRUE(result.error().find("Grid must have at least 2 points") != std::string::npos ||
                result.error().find("Concentration parameter must be positive") != std::string::npos);
}

// Test backward compatibility - existing tests should still work
TEST(GridSpecExpectedTest, BackwardCompatibilityUniform) {
    // Test that the existing usage pattern still works with .value()
    auto result = GridSpec<>::uniform(0.0, 1.0, 11);
    ASSERT_TRUE(result.has_value());

    auto spec = result.value();
    auto grid = spec.generate();

    EXPECT_EQ(grid.size(), 11);
    EXPECT_DOUBLE_EQ(grid[0], 0.0);
    EXPECT_DOUBLE_EQ(grid[10], 1.0);
    EXPECT_DOUBLE_EQ(grid[5], 0.5);
}

TEST(GridSpecExpectedTest, BackwardCompatibilityLogSpaced) {
    auto result = GridSpec<>::log_spaced(1.0, 100.0, 5);
    ASSERT_TRUE(result.has_value());

    auto spec = result.value();
    auto grid = spec.generate();

    EXPECT_EQ(grid.size(), 5);
    EXPECT_DOUBLE_EQ(grid[0], 1.0);
    EXPECT_DOUBLE_EQ(grid[4], 100.0);
    EXPECT_NEAR(grid[2], 10.0, 1e-10);
}

TEST(GridSpecExpectedTest, BackwardCompatibilitySinhSpaced) {
    auto result = GridSpec<>::sinh_spaced(0.0, 1.0, 11, 2.0);
    ASSERT_TRUE(result.has_value());

    auto spec = result.value();
    auto grid = spec.generate();

    EXPECT_EQ(grid.size(), 11);
    EXPECT_DOUBLE_EQ(grid[0], 0.0);
    EXPECT_DOUBLE_EQ(grid[10], 1.0);
    EXPECT_DOUBLE_EQ(grid[5], 0.5);
}