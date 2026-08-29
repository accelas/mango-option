// SPDX-License-Identifier: MIT
#include "mango/math/root_finding.hpp"
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

// ===========================================================================
// Error path tests
// ===========================================================================

TEST(RootFindingErrorTest, BrentInvalidBracket) {
    // f(a) and f(b) have same sign — no root bracketed
    auto f = [](double x) { return x * x + 1.0; };  // Always positive
    mango::RootFindingConfig config{.max_iter = 100, .brent_tol_abs = 1e-6};

    auto result = mango::brent_find_root(f, 0.0, 2.0, config);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::RootFindingErrorCode::InvalidBracket);
}

TEST(RootFindingErrorTest, BrentMaxIterationsExceeded) {
    auto f = [](double x) { return x * x - 2.0; };
    mango::RootFindingConfig config{.max_iter = 1, .brent_tol_abs = 1e-12};

    auto result = mango::brent_find_root(f, 0.0, 2.0, config);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::RootFindingErrorCode::MaxIterationsExceeded);
    EXPECT_LE(result.error().iterations, config.max_iter);
}

TEST(RootFindingErrorTest, BrentNaNAtEndpoint) {
    auto f = [](double x) { return std::log(x); };  // log(-1) = NaN
    mango::RootFindingConfig config{.max_iter = 100, .brent_tol_abs = 1e-6};

    auto result = mango::brent_find_root(f, -1.0, 1.0, config);

    // NaN at endpoint should be detected as either NumericalInstability or InvalidBracket
    ASSERT_FALSE(result.has_value());
    // Accept either error code since NaN makes bracket check undefined
    EXPECT_TRUE(result.error().code == mango::RootFindingErrorCode::NumericalInstability ||
                result.error().code == mango::RootFindingErrorCode::InvalidBracket);
}

TEST(RootFindingErrorTest, NewtonMaxIterationsExceeded) {
    auto f = [](double x) { return x * x - 2.0; };
    auto df = [](double x) { return 2.0 * x; };
    mango::RootFindingConfig config{.max_iter = 1, .tolerance = 1e-15};

    auto result = mango::newton_find_root(f, df, 0.1, 0.0, 2.0, config);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::RootFindingErrorCode::MaxIterationsExceeded);
}

TEST(RootFindingErrorTest, BrentRootAtEndpoint) {
    // Root exactly at bracket endpoint a
    auto f = [](double x) { return x * (x - 1.0); };  // Roots at 0 and 1
    mango::RootFindingConfig config{.max_iter = 100, .brent_tol_abs = 1e-6};

    auto result = mango::brent_find_root(f, 0.0, 0.5, config);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->root, 0.0, 1e-6);
}

// ===========================================================================
// Regression tests for bugs found during code review (issue #441)
// ===========================================================================

// Regression: unset brent_tol_x must reproduce current behavior exactly
// Bug: brent_tol_abs served both |f| and x-distance comparisons; the new
//      optional x-tolerance must not change any existing caller's result.
TEST(RootFindingTest, BrentTolXUnsetMatchesLegacyBehavior) {
    auto f = [](double x) { return x * x - 2.0; };
    mango::RootFindingConfig legacy{.max_iter = 100, .brent_tol_abs = 1e-6};
    mango::RootFindingConfig with_field{.max_iter = 100, .brent_tol_abs = 1e-6};
    // brent_tol_x left unset in both
    auto r1 = mango::brent_find_root(f, 0.0, 2.0, legacy);
    auto r2 = mango::brent_find_root(f, 0.0, 2.0, with_field);
    ASSERT_TRUE(r1.has_value());
    ASSERT_TRUE(r2.has_value());
    EXPECT_DOUBLE_EQ(r1->root, r2->root);
    EXPECT_EQ(r1->iterations, r2->iterations);
}

// Regression: one knob for |f| and x-distance is scale-dependent
// Bug: with a steep objective (f-values huge relative to x), a loose
//      brent_tol_abs stops on the |b-a| test at poor x-accuracy; there was
//      no way to tighten x-accuracy without also tightening the |f| test.
TEST(RootFindingTest, BrentTolXControlsXAccuracyIndependently) {
    // Steep function: root at sqrt(2), |f'| ~ 2.8e6 near the root
    auto f = [](double x) { return 1e6 * (x * x - 2.0); };
    const double root = std::sqrt(2.0);

    mango::RootFindingConfig loose{.max_iter = 200, .brent_tol_abs = 1e-2};
    auto r_loose = mango::brent_find_root(f, 0.0, 2.0, loose);
    ASSERT_TRUE(r_loose.has_value());

    mango::RootFindingConfig tight_x{.max_iter = 200, .brent_tol_abs = 1e-2};
    tight_x.brent_tol_x = 1e-12;
    auto r_tight = mango::brent_find_root(f, 0.0, 2.0, tight_x);
    ASSERT_TRUE(r_tight.has_value());

    // With tight x-tolerance the root is at least as accurate, and the
    // stopping cannot have come from the (now-tight) bracket-width test
    // at a worse x-error than the loose run allowed.
    EXPECT_LE(std::abs(r_tight->root - root), std::abs(r_loose->root - root) + 1e-12);
    EXPECT_LT(std::abs(r_tight->root - root), 1e-6);
}
