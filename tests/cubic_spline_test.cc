#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" {
#include "../src/pde_solver.h"
}

// Test fixture for cubic spline tests
class CubicSplineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup
    }

    void TearDown() override {
        // Common cleanup
    }
};

// Test basic spline creation and destruction
TEST_F(CubicSplineTest, CreateAndDestroy) {
    const size_t n = 5;
    double x[] = {0.0, 0.25, 0.5, 0.75, 1.0};
    double y[] = {0.0, 0.25, 0.5, 0.75, 1.0};

    CubicSpline *spline = pde_spline_create(x, y, n);
    ASSERT_NE(spline, nullptr);
    EXPECT_EQ(spline->n_points, n);

    pde_spline_destroy(spline);
}

// Test spline reproduces input points exactly
TEST_F(CubicSplineTest, ReproducesInputPoints) {
    const size_t n = 11;
    double x[11], y[11];

    for (size_t i = 0; i < n; i++) {
        x[i] = i * 0.1;
        y[i] = std::sin(x[i]);
    }

    CubicSpline *spline = pde_spline_create(x, y, n);
    ASSERT_NE(spline, nullptr);

    // Spline should pass through all input points
    for (size_t i = 0; i < n; i++) {
        double result = pde_spline_eval(spline, x[i]);
        EXPECT_NEAR(result, y[i], 1e-12);
    }

    pde_spline_destroy(spline);
}

// Test cubic spline reproduces cubic polynomials exactly
TEST_F(CubicSplineTest, ReproducesCubicPolynomial) {
    // f(x) = x^3 - 2*x^2 + x
    const size_t n = 11;
    double x[11], y[11];

    for (size_t i = 0; i < n; i++) {
        x[i] = i * 0.1;
        y[i] = x[i] * x[i] * x[i] - 2.0 * x[i] * x[i] + x[i];
    }

    CubicSpline *spline = pde_spline_create(x, y, n);
    ASSERT_NE(spline, nullptr);

    // Test at off-grid points
    double test_points[] = {0.15, 0.27, 0.43, 0.58, 0.76, 0.92};
    for (size_t i = 0; i < sizeof(test_points) / sizeof(test_points[0]); i++) {
        double x_eval = test_points[i];
        double expected = x_eval * x_eval * x_eval - 2.0 * x_eval * x_eval + x_eval;
        double interpolated = pde_spline_eval(spline, x_eval);

        EXPECT_NEAR(interpolated, expected, 1e-10);
    }

    pde_spline_destroy(spline);
}

// Test derivative evaluation
TEST_F(CubicSplineTest, DerivativeEvaluation) {
    // f(x) = x^3 - 2*x^2 + x
    // f'(x) = 3*x^2 - 4*x + 1
    const size_t n = 21;
    double x[21], y[21];

    for (size_t i = 0; i < n; i++) {
        x[i] = i * 0.05;
        y[i] = x[i] * x[i] * x[i] - 2.0 * x[i] * x[i] + x[i];
    }

    CubicSpline *spline = pde_spline_create(x, y, n);
    ASSERT_NE(spline, nullptr);

    // Test derivative at various points
    double test_points[] = {0.12, 0.34, 0.56, 0.78, 0.95};
    for (size_t i = 0; i < sizeof(test_points) / sizeof(test_points[0]); i++) {
        double x_eval = test_points[i];
        double expected_deriv = 3.0 * x_eval * x_eval - 4.0 * x_eval + 1.0;
        double interpolated_deriv = pde_spline_eval_derivative(spline, x_eval);

        EXPECT_NEAR(interpolated_deriv, expected_deriv, 1e-8);
    }

    pde_spline_destroy(spline);
}

// Test on smooth non-polynomial function (Gaussian)
TEST_F(CubicSplineTest, GaussianFunction) {
    const size_t n = 51;
    double x[51], y[51];

    for (size_t i = 0; i < n; i++) {
        x[i] = i * 0.02;  // [0, 1]
        y[i] = std::exp(-std::pow(x[i] - 0.5, 2) / 0.02);
    }

    CubicSpline *spline = pde_spline_create(x, y, n);
    ASSERT_NE(spline, nullptr);

    // Test at multiple off-grid points
    std::vector<double> test_points = {0.117, 0.333, 0.485, 0.627, 0.854};

    for (double x_eval : test_points) {
        double expected = std::exp(-std::pow(x_eval - 0.5, 2) / 0.02);
        double interpolated = pde_spline_eval(spline, x_eval);

        // Within 1% error for smooth function with fine grid
        EXPECT_NEAR(interpolated / expected, 1.0, 0.01);
    }

    pde_spline_destroy(spline);
}

// Test boundary behavior
TEST_F(CubicSplineTest, BoundaryBehavior) {
    const size_t n = 11;
    double x[11], y[11];

    for (size_t i = 0; i < n; i++) {
        x[i] = i * 0.1;
        y[i] = std::sin(x[i]);
    }

    CubicSpline *spline = pde_spline_create(x, y, n);
    ASSERT_NE(spline, nullptr);

    // Test extrapolation at boundaries (uses nearest interval)
    double result_left = pde_spline_eval(spline, -0.1);
    double result_right = pde_spline_eval(spline, 1.1);

    // Should return reasonable values (not NaN or infinity)
    EXPECT_FALSE(std::isnan(result_left));
    EXPECT_FALSE(std::isinf(result_left));
    EXPECT_FALSE(std::isnan(result_right));
    EXPECT_FALSE(std::isinf(result_right));

    pde_spline_destroy(spline);
}

// Test quadratic function (special case)
TEST_F(CubicSplineTest, QuadraticFunction) {
    // f(x) = x^2 - x
    const size_t n = 11;
    double x[11], y[11];

    for (size_t i = 0; i < n; i++) {
        x[i] = i * 0.1;
        y[i] = x[i] * x[i] - x[i];
    }

    CubicSpline *spline = pde_spline_create(x, y, n);
    ASSERT_NE(spline, nullptr);

    // Cubic spline should reproduce quadratics exactly
    double test_points[] = {0.13, 0.47, 0.82};
    for (size_t i = 0; i < sizeof(test_points) / sizeof(test_points[0]); i++) {
        double x_eval = test_points[i];
        double expected = x_eval * x_eval - x_eval;
        double interpolated = pde_spline_eval(spline, x_eval);

        EXPECT_NEAR(interpolated, expected, 1e-12);
    }

    pde_spline_destroy(spline);
}

// Test monotonicity preservation (local property)
TEST_F(CubicSplineTest, MonotonicData) {
    // Create monotonically increasing data
    const size_t n = 11;
    double x[11], y[11];

    for (size_t i = 0; i < n; i++) {
        x[i] = i * 0.1;
        y[i] = std::sqrt(x[i]);  // Monotonically increasing
    }

    CubicSpline *spline = pde_spline_create(x, y, n);
    ASSERT_NE(spline, nullptr);

    // Check that interpolated values maintain reasonable ordering
    // (Note: cubic splines don't guarantee monotonicity, but should be reasonable)
    double prev_val = pde_spline_eval(spline, 0.0);
    for (int i = 1; i <= 20; i++) {
        double x_eval = i * 0.05;
        double curr_val = pde_spline_eval(spline, x_eval);
        // Allow small violations due to smoothness constraint
        EXPECT_GT(curr_val, prev_val - 0.01);
        prev_val = curr_val;
    }

    pde_spline_destroy(spline);
}

// Test with minimum number of points
TEST_F(CubicSplineTest, MinimumPoints) {
    const size_t n = 2;
    double x[] = {0.0, 1.0};
    double y[] = {0.0, 1.0};

    CubicSpline *spline = pde_spline_create(x, y, n);
    ASSERT_NE(spline, nullptr);

    // Should interpolate linearly between two points
    double mid = pde_spline_eval(spline, 0.5);
    EXPECT_NEAR(mid, 0.5, 1e-10);

    pde_spline_destroy(spline);
}

// Test non-uniform grid
TEST_F(CubicSplineTest, NonUniformGrid) {
    const size_t n = 7;
    double x[] = {0.0, 0.1, 0.15, 0.4, 0.6, 0.85, 1.0};
    double y[7];

    for (size_t i = 0; i < n; i++) {
        y[i] = std::exp(x[i]);
    }

    CubicSpline *spline = pde_spline_create(x, y, n);
    ASSERT_NE(spline, nullptr);

    // Test at several points
    double test_points[] = {0.05, 0.3, 0.7, 0.9};
    for (size_t i = 0; i < sizeof(test_points) / sizeof(test_points[0]); i++) {
        double x_eval = test_points[i];
        double expected = std::exp(x_eval);
        double interpolated = pde_spline_eval(spline, x_eval);

        // Within reasonable tolerance
        EXPECT_NEAR(interpolated / expected, 1.0, 0.05);
    }

    pde_spline_destroy(spline);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
