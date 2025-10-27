#include <gtest/gtest.h>
#include <cmath>
#include <limits>

extern "C" {
#include "../src/brent.h"
}

// Test fixture for Brent's method tests
class BrentTest : public ::testing::Test {
protected:
    static constexpr double tolerance = 1e-6;
    static constexpr int max_iter = 100;
};

// Simple linear function: f(x) = x - 1
static double linear_function(double x, [[maybe_unused]] void *user_data) {
    return x - 1.0;
}

TEST_F(BrentTest, LinearFunction) {
    BrentResult result = brent_find_root(linear_function, 0.0, 2.0, tolerance, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 1.0, tolerance);
    EXPECT_NEAR(result.f_root, 0.0, tolerance);
    EXPECT_LT(result.iterations, 10);  // Should converge very quickly
}

// Quadratic function: f(x) = x^2 - 4
static double quadratic_function(double x, [[maybe_unused]] void *user_data) {
    return x * x - 4.0;
}

TEST_F(BrentTest, QuadraticFunction) {
    // Root at x = 2
    BrentResult result = brent_find_root(quadratic_function, 0.0, 3.0, tolerance, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 2.0, tolerance);
    EXPECT_NEAR(result.f_root, 0.0, tolerance);
}

TEST_F(BrentTest, QuadraticFunctionNegativeRoot) {
    // Root at x = -2
    BrentResult result = brent_find_root(quadratic_function, -3.0, 0.0, tolerance, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, -2.0, tolerance);
    EXPECT_NEAR(result.f_root, 0.0, tolerance);
}

// Cubic function with inflection point: f(x) = x^3 - x - 2
static double cubic_function(double x, [[maybe_unused]] void *user_data) {
    return x * x * x - x - 2.0;
}

TEST_F(BrentTest, CubicFunction) {
    // Root at x ≈ 1.5214
    BrentResult result = brent_find_root(cubic_function, 0.0, 2.0, tolerance, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 1.5213797068, 1e-6);
    EXPECT_NEAR(result.f_root, 0.0, tolerance);
}

// Transcendental: f(x) = sin(x) with root at π
static double sine_function(double x, [[maybe_unused]] void *user_data) {
    return std::sin(x);
}

TEST_F(BrentTest, SineFunction) {
    // Root at x = π
    BrentResult result = brent_find_root(sine_function, 3.0, 4.0, tolerance, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, M_PI, tolerance);
    EXPECT_NEAR(result.f_root, 0.0, tolerance);
}

// f(x) = cos(x) with root at π/2
static double cosine_function(double x, [[maybe_unused]] void *user_data) {
    return std::cos(x);
}

TEST_F(BrentTest, CosineFunction) {
    // Root at x = π/2
    BrentResult result = brent_find_root(cosine_function, 0.0, 2.0, tolerance, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, M_PI / 2.0, tolerance);
    EXPECT_NEAR(result.f_root, 0.0, tolerance);
}

// Exponential: f(x) = e^x - 2 with root at ln(2)
static double exponential_function(double x, [[maybe_unused]] void *user_data) {
    return std::exp(x) - 2.0;
}

TEST_F(BrentTest, ExponentialFunction) {
    // Root at x = ln(2) ≈ 0.693147
    BrentResult result = brent_find_root(exponential_function, 0.0, 1.0, tolerance, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, std::log(2.0), tolerance);
    EXPECT_NEAR(result.f_root, 0.0, tolerance);
}

// Root at boundary (left endpoint)
static double root_at_left_boundary(double x, [[maybe_unused]] void *user_data) {
    return x;  // Root at x = 0
}

TEST_F(BrentTest, RootAtLeftBoundary) {
    BrentResult result = brent_find_root(root_at_left_boundary, 0.0, 1.0, tolerance, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 0.0, tolerance);
    EXPECT_NEAR(result.f_root, 0.0, tolerance);
}

// Root at boundary (right endpoint)
static double root_at_right_boundary(double x, [[maybe_unused]] void *user_data) {
    return x - 1.0;  // Root at x = 1
}

TEST_F(BrentTest, RootAtRightBoundary) {
    BrentResult result = brent_find_root(root_at_right_boundary, 0.0, 1.0, tolerance, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 1.0, tolerance);
    EXPECT_NEAR(result.f_root, 0.0, tolerance);
}

// Test with very small interval
TEST_F(BrentTest, SmallInterval) {
    BrentResult result = brent_find_root(linear_function, 0.999, 1.001, 1e-9, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 1.0, 1e-9);
}

// Test with very large interval
static double large_interval_function(double x, [[maybe_unused]] void *user_data) {
    return x - 1000.0;
}

TEST_F(BrentTest, LargeInterval) {
    BrentResult result = brent_find_root(large_interval_function, 0.0, 10000.0, tolerance, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 1000.0, tolerance);
}

// Test root not bracketed (should fail gracefully)
static double always_positive(double x, [[maybe_unused]] void *user_data) {
    return x * x + 1.0;  // Always positive, no real root
}

TEST_F(BrentTest, RootNotBracketed) {
    BrentResult result = brent_find_root(always_positive, 0.0, 1.0, tolerance, max_iter, nullptr);

    EXPECT_FALSE(result.converged);
}

// Test steep function (challenging numerically)
static double steep_function(double x, [[maybe_unused]] void *user_data) {
    return std::tanh(100.0 * (x - 0.5));  // Very steep at x = 0.5
}

TEST_F(BrentTest, SteepFunction) {
    BrentResult result = brent_find_root(steep_function, 0.0, 1.0, tolerance, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 0.5, tolerance);
    EXPECT_NEAR(result.f_root, 0.0, tolerance);
}

// Test nearly flat function (challenging for slope-based methods)
static double flat_function(double x, [[maybe_unused]] void *user_data) {
    return (x - 0.5) * (x - 0.5) * (x - 0.5);  // Inflection point at root
}

TEST_F(BrentTest, FlatFunction) {
    BrentResult result = brent_find_root(flat_function, 0.0, 1.0, tolerance, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 0.5, tolerance);
}

// Test with user data
struct ParabolicData {
    double a, b, c;
};

static double parabola_with_data(double x, void *user_data) {
    ParabolicData *data = (ParabolicData *)user_data;
    return data->a * x * x + data->b * x + data->c;
}

TEST_F(BrentTest, FunctionWithUserData) {
    // f(x) = x^2 - 5x + 6 = (x-2)(x-3), roots at 2 and 3
    ParabolicData data = {1.0, -5.0, 6.0};

    BrentResult result = brent_find_root(parabola_with_data, 1.0, 2.5, tolerance, max_iter, &data);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 2.0, tolerance);
}

// Test oscillating function - multiple roots in interval (finds closest to bisection)
static double oscillating(double x, [[maybe_unused]] void *user_data) {
    return std::sin(5.0 * x);  // Multiple roots in [0, 2π]
}

TEST_F(BrentTest, OscillatingFunction) {
    // First root after 0
    BrentResult result = brent_find_root(oscillating, 0.1, 1.0, tolerance, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, M_PI / 5.0, tolerance);  // First positive root
}

// Test high-order polynomial
static double high_order_polynomial(double x, [[maybe_unused]] void *user_data) {
    // (x-1)^5 = 0, root with multiplicity 5
    double dx = x - 1.0;
    return dx * dx * dx * dx * dx;
}

TEST_F(BrentTest, HighOrderRoot) {
    BrentResult result = brent_find_root(high_order_polynomial, 0.0, 2.0, tolerance, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 1.0, tolerance);
}

// Test convergence behavior with different tolerances
TEST_F(BrentTest, ConvergenceWithTightTolerance) {
    BrentResult result = brent_find_root(quadratic_function, 0.0, 3.0, 1e-12, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 2.0, 1e-12);
    EXPECT_LT(result.iterations, 50);
}

// Test iteration limit
static double slow_convergence(double x, [[maybe_unused]] void *user_data) {
    return std::atan(x);  // Root at x = 0
}

TEST_F(BrentTest, IterationLimit) {
    // Use very tight tolerance and low iteration limit
    BrentResult result = brent_find_root(slow_convergence, -0.1, 0.1, 1e-15, 5, nullptr);

    // May or may not converge in 5 iterations
    if (!result.converged) {
        EXPECT_EQ(result.iterations, 5);
    }
}

// Test with discontinuous function (step function)
static double step_function(double x, [[maybe_unused]] void *user_data) {
    return (x < 0.5) ? -1.0 : 1.0;
}

TEST_F(BrentTest, DiscontinuousFunction) {
    // Should find root near discontinuity
    BrentResult result = brent_find_root(step_function, 0.0, 1.0, tolerance, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 0.5, tolerance);
}

// Stress test: function with many scale changes
static double multi_scale_function(double x, [[maybe_unused]] void *user_data) {
    return std::sin(x) * std::exp(-x * x / 2.0);  // Root at π in [2, 4]
}

TEST_F(BrentTest, MultiScaleFunction) {
    BrentResult result = brent_find_root(multi_scale_function, 2.5, 4.0, tolerance, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, M_PI, tolerance);
}

// Numerical stability: very small function values
static double tiny_values(double x, [[maybe_unused]] void *user_data) {
    return 1e-10 * (x - 1.0);
}

TEST_F(BrentTest, TinyFunctionValues) {
    BrentResult result = brent_find_root(tiny_values, 0.0, 2.0, 1e-13, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 1.0, 1e-6);
}

// Numerical stability: very large function values
static double huge_values(double x, [[maybe_unused]] void *user_data) {
    return 1e10 * (x - 1.0);
}

TEST_F(BrentTest, HugeFunctionValues) {
    BrentResult result = brent_find_root(huge_values, 0.0, 2.0, 1e-6, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 1.0, 1e-6);
}

// Test simple convenience function
TEST_F(BrentTest, SimpleFunctionAPI) {
    BrentResult result = brent_find_root_simple(linear_function, 0.0, 2.0, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 1.0, 1e-6);  // Default tolerance
}

// Test pathological case: f(x) = x^(1/3) (non-smooth derivative at root)
static double cube_root(double x, [[maybe_unused]] void *user_data) {
    return (x >= 0) ? std::pow(x, 1.0/3.0) : -std::pow(-x, 1.0/3.0);
}

TEST_F(BrentTest, NonSmoothDerivative) {
    BrentResult result = brent_find_root(cube_root, -1.0, 1.0, tolerance, max_iter, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 0.0, tolerance);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
