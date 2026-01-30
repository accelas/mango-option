// SPDX-License-Identifier: MIT
/**
 * @file cubic_spline_2d_test.cc
 * @brief Comprehensive tests for CubicSpline2D template class
 */

#include "src/math/cubic_spline_solver.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <numbers>

using namespace mango;

// ========== Basic Functionality Tests ==========

TEST(CubicSpline2DTest, LinearFunction) {
    // Test bilinear function: z(x,y) = 2x + 3y
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> y = {0.0, 1.0, 2.0};

    // Row-major layout: z[i*ny + j] = z(x[i], y[j])
    std::vector<double> z;
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < y.size(); ++j) {
            z.push_back(2.0 * x[i] + 3.0 * y[j]);
        }
    }

    CubicSpline2D<double> spline;
    auto error = spline.build(std::span{x}, std::span{y}, std::span{z});

    ASSERT_FALSE(error.has_value()) << "Build failed: " << error.value();

    // Test interpolation at grid points
    EXPECT_NEAR(spline.eval(0.0, 0.0), 0.0, 1e-10);
    EXPECT_NEAR(spline.eval(1.0, 1.0), 5.0, 1e-10);
    EXPECT_NEAR(spline.eval(2.0, 2.0), 10.0, 1e-10);
    EXPECT_NEAR(spline.eval(3.0, 0.0), 6.0, 1e-10);

    // Test interpolation at off-grid points
    EXPECT_NEAR(spline.eval(0.5, 0.5), 2.5, 1e-9);
    EXPECT_NEAR(spline.eval(1.5, 1.5), 7.5, 1e-9);
    EXPECT_NEAR(spline.eval(2.5, 0.5), 6.5, 1e-9);
}

TEST(CubicSpline2DTest, QuadraticFunction) {
    // Test separable quadratic: z(x,y) = x² + y²
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {0.0, 1.0, 2.0, 3.0};

    std::vector<double> z;
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < y.size(); ++j) {
            z.push_back(x[i] * x[i] + y[j] * y[j]);
        }
    }

    CubicSpline2D<double> spline;
    auto error = spline.build(std::span{x}, std::span{y}, std::span{z});

    ASSERT_FALSE(error.has_value());

    // Grid points should be exact
    EXPECT_NEAR(spline.eval(0.0, 0.0), 0.0, 1e-10);
    EXPECT_NEAR(spline.eval(2.0, 3.0), 13.0, 1e-10);
    EXPECT_NEAR(spline.eval(4.0, 2.0), 20.0, 1e-10);

    // Off-grid points (cubic splines approximate quadratics well)
    EXPECT_NEAR(spline.eval(1.5, 1.5), 4.5, 0.2);
    EXPECT_NEAR(spline.eval(2.5, 2.5), 12.5, 0.3);
}

TEST(CubicSpline2DTest, SeparableSineProduct) {
    // Test separable function: z(x,y) = sin(πx) * sin(πy)
    const size_t nx = 9;
    const size_t ny = 7;

    std::vector<double> x(nx);
    std::vector<double> y(ny);
    std::vector<double> z(nx * ny);

    for (size_t i = 0; i < nx; ++i) {
        x[i] = static_cast<double>(i) / (nx - 1);
    }
    for (size_t j = 0; j < ny; ++j) {
        y[j] = static_cast<double>(j) / (ny - 1);
    }

    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            z[i * ny + j] = std::sin(std::numbers::pi * x[i]) *
                           std::sin(std::numbers::pi * y[j]);
        }
    }

    CubicSpline2D<double> spline;
    auto error = spline.build(std::span{x}, std::span{y}, std::span{z});

    ASSERT_FALSE(error.has_value());

    // Test at grid points
    EXPECT_NEAR(spline.eval(0.0, 0.0), 0.0, 1e-10);
    EXPECT_NEAR(spline.eval(0.5, 0.5), 1.0, 1e-10);
    EXPECT_NEAR(spline.eval(1.0, 1.0), 0.0, 1e-10);

    // Test interpolation accuracy at midpoints
    double z_mid = std::sin(std::numbers::pi * 0.25) *
                   std::sin(std::numbers::pi * 0.75);
    EXPECT_NEAR(spline.eval(0.25, 0.75), z_mid, 1e-3);
}

// ========== Edge Cases and Error Handling ==========

TEST(CubicSpline2DTest, MinimumGridSize) {
    // Minimum valid grid: 2x2
    std::vector<double> x = {0.0, 1.0};
    std::vector<double> y = {0.0, 1.0};
    std::vector<double> z = {0.0, 1.0, 1.0, 2.0};  // z = x + y

    CubicSpline2D<double> spline;
    auto error = spline.build(std::span{x}, std::span{y}, std::span{z});

    ASSERT_FALSE(error.has_value());

    // Should still interpolate (degenerates to linear)
    EXPECT_NEAR(spline.eval(0.0, 0.0), 0.0, 1e-10);
    EXPECT_NEAR(spline.eval(1.0, 1.0), 2.0, 1e-10);
    EXPECT_NEAR(spline.eval(0.5, 0.5), 1.0, 1e-9);
}

TEST(CubicSpline2DTest, SinglePointGrid) {
    // Invalid: single point grid
    std::vector<double> x = {1.0};
    std::vector<double> y = {2.0};
    std::vector<double> z = {3.0};

    CubicSpline2D<double> spline;
    auto error = spline.build(std::span{x}, std::span{y}, std::span{z});

    EXPECT_TRUE(error.has_value()) << "Should fail with single point grid";
}

TEST(CubicSpline2DTest, MismatchedDimensions) {
    // Invalid: z size doesn't match nx * ny
    std::vector<double> x = {0.0, 1.0, 2.0};
    std::vector<double> y = {0.0, 1.0};
    std::vector<double> z = {0.0, 1.0, 2.0};  // Should be 6 elements, not 3

    CubicSpline2D<double> spline;
    auto error = spline.build(std::span{x}, std::span{y}, std::span{z});

    EXPECT_TRUE(error.has_value()) << "Should fail with mismatched dimensions";
}

TEST(CubicSpline2DTest, NonMonotonicGrid) {
    // Invalid: non-monotonic x grid
    std::vector<double> x = {0.0, 2.0, 1.0, 3.0};  // Not monotonic!
    std::vector<double> y = {0.0, 1.0};
    std::vector<double> z(8, 0.0);

    CubicSpline2D<double> spline;
    auto error = spline.build(std::span{x}, std::span{y}, std::span{z});

    EXPECT_TRUE(error.has_value()) << "Should fail with non-monotonic grid";
}

TEST(CubicSpline2DTest, EmptyGrids) {
    // Invalid: empty grids
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;

    CubicSpline2D<double> spline;
    auto error = spline.build(std::span{x}, std::span{y}, std::span{z});

    EXPECT_TRUE(error.has_value()) << "Should fail with empty grids";
}

// ========== Extrapolation Behavior Tests ==========

TEST(CubicSpline2DTest, ExtrapolationBeyondXBounds) {
    // Test extrapolation beyond x domain
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y = {0.0, 1.0};
    std::vector<double> z = {1.0, 2.0,   // x=1
                             2.0, 3.0,   // x=2
                             3.0, 4.0};  // x=3

    CubicSpline2D<double> spline;
    auto error = spline.build(std::span{x}, std::span{y}, std::span{z});
    ASSERT_FALSE(error.has_value());

    // Extrapolate beyond x bounds (uses natural spline extrapolation)
    double val_low = spline.eval(0.5, 0.5);
    double val_high = spline.eval(3.5, 0.5);

    // Should return some value (not crash)
    EXPECT_TRUE(std::isfinite(val_low));
    EXPECT_TRUE(std::isfinite(val_high));
}

TEST(CubicSpline2DTest, ExtrapolationBeyondYBounds) {
    // Test extrapolation beyond y domain
    std::vector<double> x = {0.0, 1.0};
    std::vector<double> y = {1.0, 2.0, 3.0};
    std::vector<double> z = {1.0, 2.0, 3.0,
                             2.0, 3.0, 4.0};

    CubicSpline2D<double> spline;
    auto error = spline.build(std::span{x}, std::span{y}, std::span{z});
    ASSERT_FALSE(error.has_value());

    // Extrapolate beyond y bounds
    double val_low = spline.eval(0.5, 0.5);
    double val_high = spline.eval(0.5, 3.5);

    EXPECT_TRUE(std::isfinite(val_low));
    EXPECT_TRUE(std::isfinite(val_high));
}

// ========== Template Instantiation Tests ==========

TEST(CubicSpline2DTest, FloatPrecision) {
    // Test with single precision
    std::vector<float> x = {0.0f, 1.0f, 2.0f};
    std::vector<float> y = {0.0f, 1.0f};
    std::vector<float> z = {0.0f, 1.0f,
                            2.0f, 3.0f,
                            4.0f, 5.0f};

    CubicSpline2D<float> spline;
    auto error = spline.build(std::span{x}, std::span{y}, std::span{z});

    ASSERT_FALSE(error.has_value());

    EXPECT_NEAR(spline.eval(0.0f, 0.0f), 0.0f, 1e-5f);
    EXPECT_NEAR(spline.eval(1.0f, 1.0f), 3.0f, 1e-5f);
    EXPECT_NEAR(spline.eval(2.0f, 0.0f), 4.0f, 1e-5f);
}

// ========== Non-Uniform Grid Tests ==========

TEST(CubicSpline2DTest, NonUniformGrid) {
    // Test with non-uniform spacing
    std::vector<double> x = {0.0, 0.5, 1.5, 4.0};
    std::vector<double> y = {0.0, 1.0, 1.5, 3.0};

    std::vector<double> z;
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < y.size(); ++j) {
            z.push_back(x[i] + y[j]);
        }
    }

    CubicSpline2D<double> spline;
    auto error = spline.build(std::span{x}, std::span{y}, std::span{z});

    ASSERT_FALSE(error.has_value());

    // Grid points should be exact
    EXPECT_NEAR(spline.eval(0.5, 1.0), 1.5, 1e-10);
    EXPECT_NEAR(spline.eval(1.5, 1.5), 3.0, 1e-10);
    EXPECT_NEAR(spline.eval(4.0, 3.0), 7.0, 1e-10);
}

// ========== Accuracy Tests ==========

TEST(CubicSpline2DTest, SmoothFunctionAccuracy) {
    // Test with smooth function: z(x,y) = exp(-(x²+y²))
    const size_t nx = 11;
    const size_t ny = 11;

    std::vector<double> x(nx);
    std::vector<double> y(ny);
    std::vector<double> z(nx * ny);

    for (size_t i = 0; i < nx; ++i) {
        x[i] = -2.0 + 4.0 * i / (nx - 1);
    }
    for (size_t j = 0; j < ny; ++j) {
        y[j] = -2.0 + 4.0 * j / (ny - 1);
    }

    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            double r2 = x[i] * x[i] + y[j] * y[j];
            z[i * ny + j] = std::exp(-r2);
        }
    }

    CubicSpline2D<double> spline;
    auto error = spline.build(std::span{x}, std::span{y}, std::span{z});

    ASSERT_FALSE(error.has_value());

    // Test interpolation accuracy at off-grid points
    double x_test = 0.75;
    double y_test = -0.5;
    double r2_test = x_test * x_test + y_test * y_test;
    double z_exact = std::exp(-r2_test);
    double z_interp = spline.eval(x_test, y_test);

    // Should be accurate to within 1% for smooth functions
    EXPECT_NEAR(z_interp / z_exact, 1.0, 0.01);
}

// ========== State Management Tests ==========

TEST(CubicSpline2DTest, RebuildSpline) {
    // Test rebuilding a spline with different data
    std::vector<double> x1 = {0.0, 1.0};
    std::vector<double> y1 = {0.0, 1.0};
    std::vector<double> z1 = {0.0, 1.0, 1.0, 2.0};

    CubicSpline2D<double> spline;
    auto error1 = spline.build(std::span{x1}, std::span{y1}, std::span{z1});
    ASSERT_FALSE(error1.has_value());

    double val1 = spline.eval(0.5, 0.5);

    // Rebuild with different data
    std::vector<double> x2 = {0.0, 1.0, 2.0};
    std::vector<double> y2 = {0.0, 1.0};
    std::vector<double> z2 = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};

    auto error2 = spline.build(std::span{x2}, std::span{y2}, std::span{z2});
    ASSERT_FALSE(error2.has_value());

    double val2 = spline.eval(0.5, 0.5);

    // Values should be different after rebuild
    EXPECT_NE(val1, val2);
}

TEST(CubicSpline2DTest, EvalBeforeBuild) {
    // Test that eval returns 0.0 if called before build
    CubicSpline2D<double> spline;

    double val = spline.eval(1.0, 1.0);
    EXPECT_EQ(val, 0.0) << "Should return 0.0 when not built";
}

TEST(CubicSpline2DTest, IsBuiltCheck) {
    // Test is_built() method
    CubicSpline2D<double> spline;

    EXPECT_FALSE(spline.is_built()) << "Should not be built initially";

    std::vector<double> x = {0.0, 1.0};
    std::vector<double> y = {0.0, 1.0};
    std::vector<double> z = {0.0, 1.0, 1.0, 2.0};

    auto error = spline.build(std::span{x}, std::span{y}, std::span{z});
    ASSERT_FALSE(error.has_value());

    EXPECT_TRUE(spline.is_built()) << "Should be built after successful build";
}

// ========== Large Grid Performance Test ==========

TEST(CubicSpline2DTest, LargeGrid) {
    // Test with larger grid (stress test)
    const size_t nx = 50;
    const size_t ny = 30;

    std::vector<double> x(nx);
    std::vector<double> y(ny);
    std::vector<double> z(nx * ny);

    for (size_t i = 0; i < nx; ++i) {
        x[i] = static_cast<double>(i);
    }
    for (size_t j = 0; j < ny; ++j) {
        y[j] = static_cast<double>(j);
    }

    // Simple function for large grid
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            z[i * ny + j] = x[i] + y[j];
        }
    }

    CubicSpline2D<double> spline;
    auto error = spline.build(std::span{x}, std::span{y}, std::span{z});

    ASSERT_FALSE(error.has_value()) << "Large grid build should succeed";

    // Sample evaluation
    EXPECT_NEAR(spline.eval(25.0, 15.0), 40.0, 1e-9);
}
