#include "src/pde/core/root_finding.hpp"
#include <gtest/gtest.h>
#include <cmath>

// Test fixture for C++ Brent wrapper
class BrentCppTest : public ::testing::Test {
protected:
    static constexpr double tolerance = 1e-6;
    static constexpr size_t max_iter = 100;
};

// Simple polynomial: f(x) = x^2 - 4, root at x = 2
TEST_F(BrentCppTest, SimplePolynomial) {
    auto f = [](double x) { return x * x - 4.0; };

    mango::RootFindingConfig config{
        .max_iter = max_iter,
        .tolerance = tolerance,
        .brent_tol_abs = tolerance
    };

    auto result = mango::brent_find_root(f, 0.0, 3.0, config);

    EXPECT_TRUE(result.converged);
    ASSERT_TRUE(result.root.has_value());
    EXPECT_NEAR(result.root.value(), 2.0, tolerance);
    EXPECT_LT(result.iterations, 20);
    EXPECT_DOUBLE_EQ(result.final_error, std::abs(f(result.root.value())));
}

// Test with lambda capturing external data
TEST_F(BrentCppTest, LambdaWithCapture) {
    double target = 5.0;
    auto f = [target](double x) { return x * x - target; };

    mango::RootFindingConfig config{.max_iter = max_iter, .tolerance = tolerance};
    auto result = mango::brent_find_root(f, 0.0, 3.0, config);

    EXPECT_TRUE(result.converged);
    ASSERT_TRUE(result.root.has_value());
    EXPECT_NEAR(result.root.value(), std::sqrt(target), tolerance);
}

// Test with function object
struct Quadratic {
    double a, b, c;
    double operator()(double x) const {
        return a * x * x + b * x + c;
    }
};

TEST_F(BrentCppTest, FunctionObject) {
    // x^2 - 3x + 2 = (x-1)(x-2), root at x = 1
    Quadratic quad{1.0, -3.0, 2.0};

    mango::RootFindingConfig config{.max_iter = max_iter, .tolerance = tolerance};
    auto result = mango::brent_find_root(quad, 0.0, 1.5, config);

    EXPECT_TRUE(result.converged);
    ASSERT_TRUE(result.root.has_value());
    EXPECT_NEAR(result.root.value(), 1.0, tolerance);
}

// Test convergence failure (root not bracketed)
TEST_F(BrentCppTest, RootNotBracketed) {
    auto f = [](double x) { return x * x + 1.0; };  // No real root

    mango::RootFindingConfig config{.max_iter = max_iter, .tolerance = tolerance};
    auto result = mango::brent_find_root(f, 0.0, 2.0, config);

    EXPECT_FALSE(result.converged);
    EXPECT_TRUE(result.failure_reason.has_value());
    EXPECT_EQ(result.failure_reason.value(), "Root not bracketed");
}

// Test max iterations limit
TEST_F(BrentCppTest, MaxIterationsReached) {
    auto f = [](double x) { return std::atan(x); };

    mango::RootFindingConfig config{.max_iter = 2, .tolerance = 1e-15};
    auto result = mango::brent_find_root(f, -0.1, 0.1, config);

    // May or may not converge in 2 iterations with such tight tolerance
    if (!result.converged) {
        EXPECT_EQ(result.iterations, 2);
        EXPECT_TRUE(result.failure_reason.has_value());
        EXPECT_EQ(result.failure_reason.value(), "Max iterations reached");
    }
}

// Test transcendental function
TEST_F(BrentCppTest, TranscendentalFunction) {
    auto f = [](double x) { return std::sin(x); };

    mango::RootFindingConfig config{.max_iter = max_iter, .tolerance = tolerance};
    auto result = mango::brent_find_root(f, 3.0, 4.0, config);

    EXPECT_TRUE(result.converged);
    ASSERT_TRUE(result.root.has_value());
    EXPECT_NEAR(result.root.value(), M_PI, tolerance);
    EXPECT_LT(std::abs(result.final_error), tolerance);
}

// Test NaN handling (critical bug fix)
TEST_F(BrentCppTest, FunctionReturnsNaN) {
    // Function that always returns NaN (simulates PDE solver failure)
    auto f = [](double x) -> double {
        return std::numeric_limits<double>::quiet_NaN();
    };

    mango::RootFindingConfig config{.max_iter = max_iter, .tolerance = tolerance};
    auto result = mango::brent_find_root(f, 0.0, 3.0, config);

    // Should fail with descriptive error, not false convergence
    EXPECT_FALSE(result.converged);
    EXPECT_TRUE(result.failure_reason.has_value());
    EXPECT_EQ(result.failure_reason.value(), "Function returned non-finite value (NaN or Inf)");
    EXPECT_TRUE(std::isnan(result.final_error));
}

// Test Inf handling
TEST_F(BrentCppTest, FunctionReturnsInf) {
    // Function that always returns Inf
    auto f = [](double x) -> double {
        return std::numeric_limits<double>::infinity();
    };

    mango::RootFindingConfig config{.max_iter = max_iter, .tolerance = tolerance};
    auto result = mango::brent_find_root(f, 0.0, 3.0, config);

    // Should fail with descriptive error
    EXPECT_FALSE(result.converged);
    EXPECT_TRUE(result.failure_reason.has_value());
    EXPECT_EQ(result.failure_reason.value(), "Function returned non-finite value (NaN or Inf)");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
