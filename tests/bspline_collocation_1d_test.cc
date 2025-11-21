/**
 * @file bspline_collocation_1d_test.cc
 * @brief Unit tests for 1D B-spline collocation solver
 */

#include "src/bspline/bspline_fitter_4d.hpp"
#include "src/option/bspline_price_table.hpp"  // For evaluation
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace mango;

// Test fixture for collocation tests
class BSplineCollocation1DTest : public ::testing::Test {
protected:
    // Helper: Generate uniform grid
    std::vector<double> uniform_grid(double a, double b, size_t n) {
        std::vector<double> grid(n);
        for (size_t i = 0; i < n; ++i) {
            grid[i] = a + i * (b - a) / (n - 1);
        }
        return grid;
    }

    // Helper: Evaluate function on grid
    template<typename F>
    std::vector<double> evaluate(const std::vector<double>& grid, F func) {
        std::vector<double> values(grid.size());
        for (size_t i = 0; i < grid.size(); ++i) {
            values[i] = func(grid[i]);
        }
        return values;
    }

    // Helper: Max error between fitted spline and original function
    template<typename F>
    double max_error(const std::vector<double>& grid,
                    const std::vector<double>& knots,
                    const std::vector<double>& coeffs,
                    F func,
                    size_t n_test = 100) {
        double x_min = grid.front();
        double x_max = grid.back();
        double dx = (x_max - x_min) / (n_test - 1);

        double max_err = 0.0;
        for (size_t i = 0; i < n_test; ++i) {
            double x = x_min + i * dx;

            // Evaluate B-spline
            int span = find_span_cubic(knots, x);
            double basis[4];
            cubic_basis_nonuniform(knots, span, x, basis);

            double spline_val = 0.0;
            for (int k = 0; k < 4; ++k) {
                int idx = span - k;
                if (idx >= 0 && idx < static_cast<int>(coeffs.size())) {
                    spline_val += basis[k] * coeffs[idx];
                }
            }

            // Compare to true function
            double true_val = func(x);
            double err = std::abs(spline_val - true_val);
            max_err = std::max(max_err, err);
        }

        return max_err;
    }
};

// Test 1: Constant function (should be exact)
TEST_F(BSplineCollocation1DTest, ConstantFunction) {
    auto grid = uniform_grid(0.0, 1.0, 10);
    auto values = evaluate(grid, [](double) { return 5.0; });

    auto solver_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    auto result = solver.fit(values);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_LT(result.max_residual, 1e-9);

    // Verify fitted spline reproduces constant
    auto knots = clamped_knots_cubic(grid);
    double error = max_error(grid, knots, result.coefficients,
                            [](double) { return 5.0; });
    EXPECT_LT(error, 1e-9);
}

// Test 2: Linear function (should be exact for cubic splines)
TEST_F(BSplineCollocation1DTest, LinearFunction) {
    auto grid = uniform_grid(0.0, 1.0, 10);
    auto values = evaluate(grid, [](double x) { return 2.0 * x + 3.0; });

    auto solver_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    auto result = solver.fit(values);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_LT(result.max_residual, 1e-9);

    // Verify fitted spline reproduces linear function
    auto knots = clamped_knots_cubic(grid);
    double error = max_error(grid, knots, result.coefficients,
                            [](double x) { return 2.0 * x + 3.0; });
    EXPECT_LT(error, 1e-9);
}

// Test 3: Quadratic function (should be exact for cubic splines)
TEST_F(BSplineCollocation1DTest, QuadraticFunction) {
    auto grid = uniform_grid(0.0, 1.0, 10);
    auto values = evaluate(grid, [](double x) { return x * x - 2.0 * x + 1.0; });

    auto solver_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    auto result = solver.fit(values);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_LT(result.max_residual, 1e-9);

    // Verify fitted spline reproduces quadratic
    auto knots = clamped_knots_cubic(grid);
    double error = max_error(grid, knots, result.coefficients,
                            [](double x) { return x * x - 2.0 * x + 1.0; });
    EXPECT_LT(error, 1e-9);
}

// Test 4: Cubic function (should be exact)
TEST_F(BSplineCollocation1DTest, CubicFunction) {
    auto grid = uniform_grid(0.0, 1.0, 10);
    auto values = evaluate(grid, [](double x) { return x * x * x - x + 2.0; });

    auto solver_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    auto result = solver.fit(values);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_LT(result.max_residual, 1e-9);

    // Verify fitted spline reproduces cubic
    auto knots = clamped_knots_cubic(grid);
    double error = max_error(grid, knots, result.coefficients,
                            [](double x) { return x * x * x - x + 2.0; });
    EXPECT_LT(error, 1e-9);
}

// Test 5: Exponential function (approximation, not exact)
TEST_F(BSplineCollocation1DTest, ExponentialFunction) {
    auto grid = uniform_grid(0.0, 1.0, 20);  // More points for smooth approximation
    auto values = evaluate(grid, [](double x) { return std::exp(x); });

    auto solver_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    auto result = solver.fit(values);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_LT(result.max_residual, 1e-9);  // Grid point residuals should be tiny

    // Verify fitted spline approximates exp(x)
    // Error should be small (cubic spline approximation quality)
    auto knots = clamped_knots_cubic(grid);
    double error = max_error(grid, knots, result.coefficients,
                            [](double x) { return std::exp(x); });
    EXPECT_LT(error, 1e-4);  // Reasonable approximation error for 20 points
}

// Test 6: Sin function (smooth, periodic-ish)
TEST_F(BSplineCollocation1DTest, SinFunction) {
    auto grid = uniform_grid(0.0, M_PI, 30);
    auto values = evaluate(grid, [](double x) { return std::sin(x); });

    auto solver_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    auto result = solver.fit(values);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_LT(result.max_residual, 1e-9);

    // Verify approximation quality
    auto knots = clamped_knots_cubic(grid);
    double error = max_error(grid, knots, result.coefficients,
                            [](double x) { return std::sin(x); });
    EXPECT_LT(error, 1e-3);  // Good approximation with 30 points
}

// Test 7: Non-uniform grid
TEST_F(BSplineCollocation1DTest, NonUniformGrid) {
    // Create log-spaced grid (common for financial applications)
    std::vector<double> grid;
    for (size_t i = 0; i < 15; ++i) {
        double t = i / 14.0;  // [0, 1]
        grid.push_back(std::exp(t * std::log(10.0)));  // [1, 10] log-spaced
    }

    auto values = evaluate(grid, [](double x) { return std::log(x); });

    auto solver_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    auto result = solver.fit(values);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_LT(result.max_residual, 1e-9);

    // Verify fitted spline approximates log(x)
    auto knots = clamped_knots_cubic(grid);
    double error = max_error(grid, knots, result.coefficients,
                            [](double x) { return std::log(x); });
    EXPECT_LT(error, 1e-4);
}

// Test 8: Condition number monitoring
TEST_F(BSplineCollocation1DTest, ConditionNumberEstimate) {
    auto grid = uniform_grid(0.0, 1.0, 10);
    auto values = evaluate(grid, [](double x) { return x * x; });

    auto solver_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    auto result = solver.fit(values);

    ASSERT_TRUE(result.success);

    // Condition number should be reasonable for uniform grid
    EXPECT_LT(result.condition_estimate, 100.0);
    EXPECT_GT(result.condition_estimate, 1.0);
}

// Test 9: Minimum grid size (n=4)
TEST_F(BSplineCollocation1DTest, MinimumGridSize) {
    auto grid = uniform_grid(0.0, 1.0, 4);
    auto values = evaluate(grid, [](double x) { return x * x; });

    auto solver_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    auto result = solver.fit(values);

    ASSERT_TRUE(result.success);
    EXPECT_LT(result.max_residual, 1e-9);
}

// Test 10: Fail on too-small grid
TEST_F(BSplineCollocation1DTest, FailOnSmallGrid) {
    std::vector<double> grid = {0.0, 0.5, 1.0};  // Only 3 points

    auto result = BSplineCollocation1D<double>::create(grid);

    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("≥4 points"), std::string::npos);
}

// Test 11: Fail on unsorted grid
TEST_F(BSplineCollocation1DTest, FailOnUnsortedGrid) {
    std::vector<double> grid = {0.0, 1.0, 0.5, 0.75};  // Unsorted

    auto result = BSplineCollocation1D<double>::create(grid);

    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("sorted"), std::string::npos);
}

// Test 12: Mismatched value array size
TEST_F(BSplineCollocation1DTest, FailOnSizeMismatch) {
    auto grid = uniform_grid(0.0, 1.0, 10);
    std::vector<double> values(8);  // Wrong size

    auto solver_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    auto result = solver.fit(values);

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error_message.find("mismatch"), std::string::npos);
}

// Test 13: Duplicate grid points
TEST_F(BSplineCollocation1DTest, FailOnDuplicatePoints) {
    std::vector<double> grid = {0.0, 0.5, 0.5, 1.0};  // Duplicate at 0.5
    std::vector<double> values = {1.0, 2.0, 2.0, 3.0};

    auto result = BSplineCollocation1D<double>::create(grid);

    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("too close together"), std::string::npos);
}

// Test 14: Nearly duplicate points (ill-conditioned)
TEST_F(BSplineCollocation1DTest, IllConditionedNearDuplicates) {
    std::vector<double> grid = {0.0, 0.1, 0.1 + 1e-14, 0.5, 1.0};
    auto values = evaluate(grid, [](double x) { return x; });

    auto solver_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    auto result = solver.fit(values);

    // Should either fail or have extremely high condition number
    if (result.success) {
        EXPECT_GT(result.condition_estimate, 1e10);
    } else {
        EXPECT_NE(result.error_message.find("singular"), std::string::npos);
    }
}

// Test 15: NaN in input values
TEST_F(BSplineCollocation1DTest, FailOnNaNInput) {
    auto grid = uniform_grid(0.0, 1.0, 10);
    auto values = evaluate(grid, [](double x) { return x; });
    values[5] = std::numeric_limits<double>::quiet_NaN();

    auto solver_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    auto result = solver.fit(values);

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error_message.find("NaN"), std::string::npos);
}

// Test 16: Infinity in input values
TEST_F(BSplineCollocation1DTest, FailOnInfInput) {
    auto grid = uniform_grid(0.0, 1.0, 10);
    auto values = evaluate(grid, [](double x) { return x; });
    values[3] = std::numeric_limits<double>::infinity();

    auto solver_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    auto result = solver.fit(values);

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error_message.find("infinite"), std::string::npos);
}

// Test 17: Extremely ill-conditioned system (clustering at boundaries)
TEST_F(BSplineCollocation1DTest, ExtremelyClustered) {
    // Create grid with severe clustering at left boundary
    std::vector<double> grid;
    for (int i = 0; i < 8; ++i) {
        grid.push_back(i * 1e-10);  // 0, 1e-10, 2e-10, ..., 7e-10
    }
    grid.push_back(0.5);
    grid.push_back(1.0);

    auto values = evaluate(grid, [](double x) { return x; });

    auto solver_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(solver_result.has_value());
    auto& solver = solver_result.value();

    auto result = solver.fit(values);

    // Should either fail or have astronomical condition number
    if (result.success) {
        EXPECT_GT(result.condition_estimate, 1e12);
    } else {
        EXPECT_NE(result.error_message.find("singular"), std::string::npos);
    }
}

// Test 18: Zero-width grid
TEST_F(BSplineCollocation1DTest, FailOnZeroWidthGrid) {
    std::vector<double> grid = {1.0, 1.0, 1.0, 1.0};  // All same value

    auto result = BSplineCollocation1D<double>::create(grid);

    ASSERT_FALSE(result.has_value());
    // This will be caught by the "too close together" check since spacing = 0
    EXPECT_NE(result.error().find("too close together"), std::string::npos);
}

// Test 19: Factory pattern - successful creation with valid grid
TEST_F(BSplineCollocation1DTest, FactoryValidGrid) {
    auto grid = uniform_grid(0.0, 1.0, 10);

    auto result = BSplineCollocation1D<double>::create(grid);

    ASSERT_TRUE(result.has_value());
    EXPECT_NO_THROW({
        auto& solver = result.value();
        auto values = evaluate(grid, [](double x) { return x * x; });
        auto fit_result = solver.fit(values);
        EXPECT_TRUE(fit_result.success);
    });
}

// Test 20: Factory pattern - fail on too-small grid
TEST_F(BSplineCollocation1DTest, FactoryFailOnSmallGrid) {
    std::vector<double> grid = {0.0, 0.5, 1.0};  // Only 3 points

    auto result = BSplineCollocation1D<double>::create(grid);

    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("≥4 points"), std::string::npos);
}

// Test 21: Factory pattern - fail on unsorted grid
TEST_F(BSplineCollocation1DTest, FactoryFailOnUnsortedGrid) {
    std::vector<double> grid = {0.0, 1.0, 0.5, 0.75};  // Unsorted

    auto result = BSplineCollocation1D<double>::create(grid);

    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("sorted"), std::string::npos);
}

// Test 22: Factory pattern - duplicate grid points
TEST_F(BSplineCollocation1DTest, FactoryFailOnDuplicatePoints) {
    std::vector<double> grid = {0.0, 0.5, 0.5, 1.0};  // Duplicate at 0.5

    auto result = BSplineCollocation1D<double>::create(grid);

    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("too close together"), std::string::npos);
}

// Test 23: Factory pattern - zero-width grid
TEST_F(BSplineCollocation1DTest, FactoryFailOnZeroWidthGrid) {
    std::vector<double> grid = {1.0, 1.0, 1.0, 1.0};  // All same value

    auto result = BSplineCollocation1D<double>::create(grid);

    ASSERT_FALSE(result.has_value());
    // This will be caught by the "too close together" check since spacing = 0
    EXPECT_NE(result.error().find("too close together"), std::string::npos);
}

// Test 24: Factory pattern - nearly duplicate points
TEST_F(BSplineCollocation1DTest, FactoryFailOnNearlyDuplicatePoints) {
    std::vector<double> grid = {0.0, 0.5, 0.5 + 1e-15, 1.0};  // Nearly duplicate

    auto result = BSplineCollocation1D<double>::create(grid);

    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("too close together"), std::string::npos);
}

// Test 25: Factory pattern - minimum valid grid size
TEST_F(BSplineCollocation1DTest, FactoryMinimumGridSize) {
    auto grid = uniform_grid(0.0, 1.0, 4);  // Minimum 4 points

    auto result = BSplineCollocation1D<double>::create(grid);

    ASSERT_TRUE(result.has_value());
    auto& solver = result.value();
    auto values = evaluate(grid, [](double x) { return x * x; });
    auto fit_result = solver.fit(values);
    EXPECT_TRUE(fit_result.success);
}

// Test 26: Factory pattern - large grid
TEST_F(BSplineCollocation1DTest, FactoryLargeGrid) {
    auto grid = uniform_grid(0.0, 1.0, 200);  // Large but reasonable grid

    auto result = BSplineCollocation1D<double>::create(grid);

    ASSERT_TRUE(result.has_value());
    auto& solver = result.value();
    auto values = evaluate(grid, [](double x) { return std::sin(5*x); });  // Lower frequency for stability
    auto fit_result = solver.fit(values);
    EXPECT_TRUE(fit_result.success);
}

// Test 27: Factory pattern - non-uniform grid
TEST_F(BSplineCollocation1DTest, FactoryNonUniformGrid) {
    // Create log-spaced grid (common for financial applications)
    std::vector<double> grid;
    for (size_t i = 0; i < 15; ++i) {
        double t = i / 14.0;  // [0, 1]
        grid.push_back(std::exp(t * std::log(10.0)));  // [1, 10] log-spaced
    }

    auto result = BSplineCollocation1D<double>::create(grid);

    ASSERT_TRUE(result.has_value());
    auto& solver = result.value();
    auto values = evaluate(grid, [](double x) { return std::log(x); });
    auto fit_result = solver.fit(values);
    EXPECT_TRUE(fit_result.success);
}

// Test 28: Factory pattern - move semantics
TEST_F(BSplineCollocation1DTest, FactoryMoveSemantics) {
    auto grid = uniform_grid(0.0, 1.0, 10);

    auto result = BSplineCollocation1D<double>::create(std::move(grid));

    ASSERT_TRUE(result.has_value());
    // The grid should have been moved, not copied
    EXPECT_EQ(grid.size(), 0);  // Moved-from vector should be empty
}

// Test 29: Factory pattern - error message detail for small grid
TEST_F(BSplineCollocation1DTest, FactoryErrorMessageDetail) {
    std::vector<double> grid = {0.0, 1.0};  // Only 2 points

    auto result = BSplineCollocation1D<double>::create(grid);

    ASSERT_FALSE(result.has_value());
    auto error = result.error();
    EXPECT_NE(error.find("4"), std::string::npos);  // Should mention "4"
    EXPECT_NE(error.find("cubic"), std::string::npos);  // Should mention "cubic"
}

// Test 30: Factory pattern - error message detail for unsorted grid
TEST_F(BSplineCollocation1DTest, FactoryUnsortedErrorMessage) {
    std::vector<double> grid = {0.0, 1.0, 0.5, 0.75};  // Unsorted (4 points)

    auto result = BSplineCollocation1D<double>::create(grid);

    ASSERT_FALSE(result.has_value());
    auto error = result.error();
    EXPECT_NE(error.find("sorted"), std::string::npos);
}
