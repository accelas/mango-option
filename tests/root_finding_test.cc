#include "src/math/root_finding.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(RootFindingConfigTest, DefaultValues) {
    mango::RootFindingConfig config;

    EXPECT_EQ(config.max_iter, 100);
    EXPECT_DOUBLE_EQ(config.tolerance, 1e-6);
    EXPECT_DOUBLE_EQ(config.jacobian_fd_epsilon, 1e-7);
    EXPECT_DOUBLE_EQ(config.brent_tol_abs, 1e-6);
}

TEST(RootFindingConfigTest, CustomValues) {
    mango::RootFindingConfig config{
        .max_iter = 50,
        .tolerance = 1e-8,
        .jacobian_fd_epsilon = 1e-9,
        .brent_tol_abs = 1e-8
    };

    EXPECT_EQ(config.max_iter, 50);
    EXPECT_DOUBLE_EQ(config.tolerance, 1e-8);
}

// ============================================================================
// Generic API Tests
// ============================================================================

TEST(GenericRootFindingTest, AutoDispatchBrent) {
    // Objective: x^2 - 2 = 0, root at x = sqrt(2)
    auto f = [](double x) { return x*x - 2.0; };

    mango::RootFindingConfig config{.max_iter = 100, .brent_tol_abs = 1e-6};

    // Automatically uses Brent (no derivative provided)
    auto result = mango::find_root(f, 0.0, 2.0, config);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->root, std::sqrt(2.0), 1e-6);
}

TEST(GenericRootFindingTest, AutoDispatchNewton) {
    // Objective: x^2 - 2 = 0, root at x = sqrt(2)
    auto f = [](double x) { return x*x - 2.0; };
    auto df = [](double x) { return 2.0*x; };  // Derivative

    mango::RootFindingConfig config{.max_iter = 100, .tolerance = 1e-9};

    // Automatically uses Newton (derivative provided)
    auto result = mango::find_root(f, df, 1.0, 0.0, 2.0, config);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->root, std::sqrt(2.0), 1e-9);

    // Newton should converge faster than Brent
    EXPECT_LT(result->iterations, 10);
}

TEST(GenericRootFindingTest, ExplicitBrentCall) {
    auto f = [](double x) { return x*x - 2.0; };

    mango::RootFindingConfig config{.max_iter = 100, .brent_tol_abs = 1e-6};

    // Explicit Brent call
    auto result = mango::find_root_bracketed(f, 0.0, 2.0, config);

    ASSERT_TRUE(result.has_value());
    
    EXPECT_NEAR(result->root, std::sqrt(2.0), 1e-6);
}

TEST(GenericRootFindingTest, ExplicitNewtonCall) {
    auto f = [](double x) { return x*x - 2.0; };
    auto df = [](double x) { return 2.0*x; };

    mango::RootFindingConfig config{.max_iter = 100, .tolerance = 1e-9};

    // Explicit Newton call
    auto result = mango::find_root_bounded(f, df, 1.0, 0.0, 2.0, config);

    ASSERT_TRUE(result.has_value());
    
    EXPECT_NEAR(result->root, std::sqrt(2.0), 1e-9);
}

TEST(GenericRootFindingTest, ComplexFunctionBrent) {
    // Transcendental equation: exp(x) - 3x = 0
    auto f = [](double x) { return std::exp(x) - 3.0*x; };

    mango::RootFindingConfig config{.max_iter = 100, .brent_tol_abs = 1e-6};

    // Use Brent (derivative-free), need to ensure bracket contains root
    // exp(x) - 3x = 0 has root around x ≈ 1.512
    auto result = mango::find_root(f, 1.0, 2.0, config);

    ASSERT_TRUE(result.has_value());
    

    // Verify: exp(x) ≈ 3x at root
    double x = result->root;
    EXPECT_NEAR(std::exp(x), 3.0*x, 1e-5);
}

TEST(GenericRootFindingTest, ComplexFunctionNewton) {
    // Transcendental equation: exp(x) - 3x = 0
    auto f = [](double x) { return std::exp(x) - 3.0*x; };
    auto df = [](double x) { return std::exp(x) - 3.0; };  // Derivative

    mango::RootFindingConfig config{.max_iter = 100, .tolerance = 1e-9};

    // Use Newton (with derivative)
    auto result = mango::find_root(f, df, 1.0, 0.0, 2.0, config);

    ASSERT_TRUE(result.has_value());
    

    // Verify: exp(x) ≈ 3x at root
    double x = result->root;
    EXPECT_NEAR(std::exp(x), 3.0*x, 1e-8);
}
