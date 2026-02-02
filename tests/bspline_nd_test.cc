// SPDX-License-Identifier: MIT
/**
 * @file bspline_nd_test.cc
 * @brief Tests for N-dimensional B-spline interpolation
 */

#include "src/math/bspline_nd.hpp"
#include "src/math/bspline_basis.hpp"
#include "src/math/bspline_nd_separable.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <cmath>
#include <numbers>

using namespace mango;

class BSplineNDTest : public ::testing::Test {
protected:
    /// Helper to create uniform grid
    static std::vector<double> create_uniform_grid(double xmin, double xmax, size_t n) {
        std::vector<double> grid(n);
        for (size_t i = 0; i < n; ++i) {
            grid[i] = xmin + (xmax - xmin) * i / (n - 1);
        }
        return grid;
    }

    /// Helper to create clamped knot vector
    static std::vector<double> create_clamped_knots(const std::vector<double>& grid) {
        return clamped_knots_cubic(grid);
    }
};

/// Test 1D B-spline (degenerate case, should work)
TEST_F(BSplineNDTest, OneDimensional) {
    // Create 1D grid
    auto grid = create_uniform_grid(0.0, 1.0, 10);
    auto knots = create_clamped_knots(grid);

    // Use constant coefficients for simple test
    std::vector<double> coeffs(grid.size(), 1.5);

    // Create 1D B-spline
    auto spline = BSplineND<double, 1>::create({grid}, {knots}, coeffs);
    ASSERT_TRUE(spline.has_value()) << "Failed to create 1D B-spline";

    // Test that evaluation works without crashing
    double val = spline->eval({0.5});
    EXPECT_TRUE(std::isfinite(val)) << "Evaluation should return finite value";

    // For constant coefficients, B-spline should approximate the constant
    EXPECT_NEAR(val, 1.5, 0.5) << "Should approximate constant value";
}

/// Test 3D B-spline
TEST_F(BSplineNDTest, ThreeDimensional) {
    // Create 3D grids
    auto grid_x = create_uniform_grid(0.0, 1.0, 6);
    auto grid_y = create_uniform_grid(0.0, 1.0, 5);
    auto grid_z = create_uniform_grid(0.0, 1.0, 4);

    auto knots_x = create_clamped_knots(grid_x);
    auto knots_y = create_clamped_knots(grid_y);
    auto knots_z = create_clamped_knots(grid_z);

    // Use constant coefficients for predictable behavior
    const size_t total_size = grid_x.size() * grid_y.size() * grid_z.size();
    std::vector<double> coeffs(total_size, 2.0);

    // Create 3D B-spline
    auto spline = BSplineND<double, 3>::create(
        {grid_x, grid_y, grid_z},
        {knots_x, knots_y, knots_z},
        coeffs
    );
    ASSERT_TRUE(spline.has_value()) << "Failed to create 3D B-spline";

    // Test dimensions
    auto dims = spline->dimensions();
    EXPECT_EQ(dims[0], 6);
    EXPECT_EQ(dims[1], 5);
    EXPECT_EQ(dims[2], 4);

    // Test that evaluation works at various points
    double val1 = spline->eval({0.0, 0.0, 0.0});
    double val2 = spline->eval({0.5, 0.5, 0.5});
    double val3 = spline->eval({1.0, 1.0, 1.0});

    EXPECT_TRUE(std::isfinite(val1));
    EXPECT_TRUE(std::isfinite(val2));
    EXPECT_TRUE(std::isfinite(val3));

    // For constant coefficients, should approximate the constant
    EXPECT_NEAR(val2, 2.0, 0.5) << "Should approximate constant value";
}

/// Test 4D B-spline (typical option pricing case)
TEST_F(BSplineNDTest, FourDimensional) {
    // Create 4D grids (smaller for testing)
    auto grid_m = create_uniform_grid(0.8, 1.2, 5);  // moneyness
    auto grid_t = create_uniform_grid(0.1, 2.0, 4);  // maturity
    auto grid_v = create_uniform_grid(0.1, 0.5, 4);  // volatility
    auto grid_r = create_uniform_grid(0.0, 0.1, 4);  // rate

    auto knots_m = create_clamped_knots(grid_m);
    auto knots_t = create_clamped_knots(grid_t);
    auto knots_v = create_clamped_knots(grid_v);
    auto knots_r = create_clamped_knots(grid_r);

    // Create test function: f(m,t,v,r) = m + t + v + r
    const size_t total_size = grid_m.size() * grid_t.size() * grid_v.size() * grid_r.size();
    std::vector<double> coeffs(total_size);

    for (size_t i = 0; i < grid_m.size(); ++i) {
        for (size_t j = 0; j < grid_t.size(); ++j) {
            for (size_t k = 0; k < grid_v.size(); ++k) {
                for (size_t l = 0; l < grid_r.size(); ++l) {
                    const size_t idx = ((i * grid_t.size() + j) * grid_v.size() + k) * grid_r.size() + l;
                    coeffs[idx] = grid_m[i] + grid_t[j] + grid_v[k] + grid_r[l];
                }
            }
        }
    }

    // Create 4D B-spline
    auto spline = BSplineND<double, 4>::create(
        {grid_m, grid_t, grid_v, grid_r},
        {knots_m, knots_t, knots_v, knots_r},
        coeffs
    );
    ASSERT_TRUE(spline.has_value()) << "Failed to create 4D B-spline";

    // Test grid access
    EXPECT_EQ(spline->grid(0).size(), 5);
    EXPECT_EQ(spline->grid(1).size(), 4);
    EXPECT_EQ(spline->grid(2).size(), 4);
    EXPECT_EQ(spline->grid(3).size(), 4);

    // Test interpolation at corner point
    double val = spline->eval({grid_m[0], grid_t[0], grid_v[0], grid_r[0]});
    double expected = grid_m[0] + grid_t[0] + grid_v[0] + grid_r[0];
    EXPECT_NEAR(val, expected, 1e-10) << "Interpolation error at corner";

    // Test evaluation at interior point
    val = spline->eval({1.0, 0.5, 0.2, 0.05});
    expected = 1.0 + 0.5 + 0.2 + 0.05;
    EXPECT_NEAR(val, expected, 0.05) << "Approximation error at interior point";
}

/// Test 5D B-spline (future extension with dividend)
TEST_F(BSplineNDTest, FiveDimensional) {
    // Create 5D grids (small for testing)
    auto grid_m = create_uniform_grid(0.8, 1.2, 4);  // moneyness
    auto grid_t = create_uniform_grid(0.1, 2.0, 4);  // maturity
    auto grid_v = create_uniform_grid(0.1, 0.5, 4);  // volatility
    auto grid_r = create_uniform_grid(0.0, 0.1, 4);  // rate
    auto grid_d = create_uniform_grid(0.0, 0.05, 4); // dividend

    auto knots_m = create_clamped_knots(grid_m);
    auto knots_t = create_clamped_knots(grid_t);
    auto knots_v = create_clamped_knots(grid_v);
    auto knots_r = create_clamped_knots(grid_r);
    auto knots_d = create_clamped_knots(grid_d);

    // Create test function: f = m + t + v + r + d
    const size_t total_size = grid_m.size() * grid_t.size() * grid_v.size() *
                             grid_r.size() * grid_d.size();
    std::vector<double> coeffs(total_size);

    for (size_t i = 0; i < grid_m.size(); ++i) {
        for (size_t j = 0; j < grid_t.size(); ++j) {
            for (size_t k = 0; k < grid_v.size(); ++k) {
                for (size_t l = 0; l < grid_r.size(); ++l) {
                    for (size_t p = 0; p < grid_d.size(); ++p) {
                        const size_t idx = (((i * grid_t.size() + j) * grid_v.size() + k) *
                                          grid_r.size() + l) * grid_d.size() + p;
                        coeffs[idx] = grid_m[i] + grid_t[j] + grid_v[k] + grid_r[l] + grid_d[p];
                    }
                }
            }
        }
    }

    // Create 5D B-spline
    auto spline = BSplineND<double, 5>::create(
        {grid_m, grid_t, grid_v, grid_r, grid_d},
        {knots_m, knots_t, knots_v, knots_r, knots_d},
        coeffs
    );
    ASSERT_TRUE(spline.has_value()) << "Failed to create 5D B-spline";

    // Test dimensions
    auto dims = spline->dimensions();
    EXPECT_EQ(dims[0], 4);
    EXPECT_EQ(dims[1], 4);
    EXPECT_EQ(dims[2], 4);
    EXPECT_EQ(dims[3], 4);
    EXPECT_EQ(dims[4], 4);

    // Test evaluation at corner point
    double val = spline->eval({grid_m[0], grid_t[0], grid_v[0], grid_r[0], grid_d[0]});
    double expected = grid_m[0] + grid_t[0] + grid_v[0] + grid_r[0] + grid_d[0];
    EXPECT_NEAR(val, expected, 1e-10) << "Interpolation error at corner";

    // Test evaluation at interior point
    val = spline->eval({1.0, 0.5, 0.2, 0.05, 0.02});
    expected = 1.0 + 0.5 + 0.2 + 0.05 + 0.02;
    EXPECT_NEAR(val, expected, 0.05) << "Approximation error at interior point";
}

/// Test validation errors
TEST_F(BSplineNDTest, ValidationErrors) {
    auto grid_x = create_uniform_grid(0.0, 1.0, 3);  // Too small (< 4)
    auto grid_y = create_uniform_grid(0.0, 1.0, 4);
    auto knots_x = create_clamped_knots(grid_x);
    auto knots_y = create_clamped_knots(grid_y);

    std::vector<double> coeffs(grid_x.size() * grid_y.size(), 0.0);

    // Should fail due to grid size < 4
    auto result = BSplineND<double, 2>::create(
        {grid_x, grid_y},
        {knots_x, knots_y},
        coeffs
    );
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::InterpolationErrorCode::InsufficientGridPoints);
}

/// Test coefficient size mismatch
TEST_F(BSplineNDTest, CoefficientSizeMismatch) {
    auto grid_x = create_uniform_grid(0.0, 1.0, 4);
    auto grid_y = create_uniform_grid(0.0, 1.0, 4);
    auto knots_x = create_clamped_knots(grid_x);
    auto knots_y = create_clamped_knots(grid_y);

    std::vector<double> coeffs(10);  // Wrong size (should be 4*4 = 16)

    auto result = BSplineND<double, 2>::create(
        {grid_x, grid_y},
        {knots_x, knots_y},
        coeffs
    );
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::InterpolationErrorCode::CoefficientSizeMismatch);
}

/// Test that NaN coefficients are rejected
TEST_F(BSplineNDTest, CreateRejectsNaNCoefficients) {
    auto grid = create_uniform_grid(0.0, 3.0, 4);
    auto knots = create_clamped_knots(grid);
    std::vector<double> coeffs = {1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), 4.0};

    auto result = BSplineND<double, 1>::create(
        {grid}, {knots}, std::move(coeffs));
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::InterpolationErrorCode::NaNInput);
}

/// Test that Inf coefficients are rejected
TEST_F(BSplineNDTest, CreateRejectsInfCoefficients) {
    auto grid = create_uniform_grid(0.0, 3.0, 4);
    auto knots = create_clamped_knots(grid);
    std::vector<double> coeffs = {1.0, std::numeric_limits<double>::infinity(), 3.0, 4.0};

    auto result = BSplineND<double, 1>::create(
        {grid}, {knots}, std::move(coeffs));
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::InterpolationErrorCode::NaNInput);
}

// ===========================================================================
// Second derivative tests
// ===========================================================================

/// Test 1D second derivative: f(x) = x² → f''(x) = 2
TEST_F(BSplineNDTest, SecondDerivativeQuadratic) {
    auto grid = create_uniform_grid(0.0, 2.0, 10);

    // Generate function values at grid points
    std::vector<double> values(grid.size());
    for (size_t i = 0; i < grid.size(); ++i) {
        values[i] = grid[i] * grid[i];
    }

    // Fit proper B-spline coefficients via collocation
    auto fitter = BSplineNDSeparable<double, 1>::create({grid});
    ASSERT_TRUE(fitter.has_value());
    auto fit = fitter->fit(values);
    ASSERT_TRUE(fit.has_value());

    auto knots = create_clamped_knots(grid);
    auto spline = BSplineND<double, 1>::create({grid}, {knots}, fit->coefficients);
    ASSERT_TRUE(spline.has_value());

    // Second derivative of x² should be 2.0 everywhere in the interior
    for (double x = 0.3; x <= 1.7; x += 0.2) {
        double d2f = spline->eval_second_partial(0, {x});
        EXPECT_NEAR(d2f, 2.0, 0.1) << "f''(x²) should be 2.0 at x=" << x;
    }
}

/// Test 1D second derivative: f(x) = x (linear) → f''(x) = 0
TEST_F(BSplineNDTest, SecondDerivativeLinear) {
    auto grid = create_uniform_grid(0.0, 1.0, 8);

    // Generate linear function values
    std::vector<double> values(grid.size());
    for (size_t i = 0; i < grid.size(); ++i) {
        values[i] = grid[i];
    }

    // Fit proper B-spline coefficients via collocation
    auto fitter = BSplineNDSeparable<double, 1>::create({grid});
    ASSERT_TRUE(fitter.has_value());
    auto fit = fitter->fit(values);
    ASSERT_TRUE(fit.has_value());

    auto knots = create_clamped_knots(grid);
    auto spline = BSplineND<double, 1>::create({grid}, {knots}, fit->coefficients);
    ASSERT_TRUE(spline.has_value());

    // Second derivative of linear function should be 0
    for (double x = 0.2; x <= 0.8; x += 0.2) {
        double d2f = spline->eval_second_partial(0, {x});
        EXPECT_NEAR(d2f, 0.0, 1e-8) << "f''(x) should be 0.0 at x=" << x;
    }
}

/// Test 2D second partial: f(x,y) = x²·y → ∂²f/∂x² = 2y
TEST_F(BSplineNDTest, SecondPartialDerivative2D) {
    auto grid_x = create_uniform_grid(0.0, 2.0, 8);
    auto grid_y = create_uniform_grid(0.0, 2.0, 8);

    // Generate function values at grid points
    std::vector<double> values(grid_x.size() * grid_y.size());
    for (size_t i = 0; i < grid_x.size(); ++i) {
        for (size_t j = 0; j < grid_y.size(); ++j) {
            values[i * grid_y.size() + j] = grid_x[i] * grid_x[i] * grid_y[j];
        }
    }

    // Fit proper B-spline coefficients via separable collocation
    auto fitter = BSplineNDSeparable<double, 2>::create({grid_x, grid_y});
    ASSERT_TRUE(fitter.has_value());
    auto fit = fitter->fit(values);
    ASSERT_TRUE(fit.has_value());

    auto knots_x = create_clamped_knots(grid_x);
    auto knots_y = create_clamped_knots(grid_y);
    auto spline = BSplineND<double, 2>::create(
        {grid_x, grid_y}, {knots_x, knots_y}, fit->coefficients);
    ASSERT_TRUE(spline.has_value());

    // ∂²f/∂x² at (1.0, 1.5) should be 2·1.5 = 3.0
    double d2f = spline->eval_second_partial(0, {1.0, 1.5});
    EXPECT_NEAR(d2f, 3.0, 0.3) << "∂²(x²·y)/∂x² should be 2y";
}

/// Test boundary clamping
TEST_F(BSplineNDTest, BoundaryClamping) {
    auto grid = create_uniform_grid(0.0, 1.0, 5);
    auto knots = create_clamped_knots(grid);
    std::vector<double> coeffs(grid.size(), 1.0);

    auto spline = BSplineND<double, 1>::create({grid}, {knots}, coeffs);
    ASSERT_TRUE(spline.has_value());

    // Query outside bounds should be clamped
    double val_below = spline->eval({-1.0});
    double val_above = spline->eval({2.0});

    // Should not crash and should return reasonable values
    EXPECT_TRUE(std::isfinite(val_below));
    EXPECT_TRUE(std::isfinite(val_above));
}
