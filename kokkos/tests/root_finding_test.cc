#include <gtest/gtest.h>
#include "kokkos/src/math/root_finding.hpp"
#include <cmath>

namespace mango::kokkos::test {

// Global setup/teardown for Kokkos - once per test program
class KokkosEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        Kokkos::initialize();
    }
    void TearDown() override {
        Kokkos::finalize();
    }
};

// Register the global environment
[[maybe_unused]] static ::testing::Environment* const kokkos_env =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

class RootFindingTest : public ::testing::Test {
    // No per-test setup/teardown needed for Kokkos
};

// ============================================================================
// Brent's Method Tests (Host)
// ============================================================================

TEST_F(RootFindingTest, BrentFindsSqrtTwo) {
    // Find sqrt(2) by solving x^2 - 2 = 0
    auto f = [](double x) { return x * x - 2.0; };

    RootFindingConfig config;
    config.brent_tol_abs = 1e-10;

    auto result = brent_find_root(f, 1.0, 2.0, config);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->root, std::sqrt(2.0), 1e-10);
    EXPECT_LT(result->iterations, config.max_iter);
    EXPECT_LT(result->final_error, config.brent_tol_abs);
}

TEST_F(RootFindingTest, BrentFindsCubeRootEight) {
    // Find cube root of 8 by solving x^3 - 8 = 0
    auto f = [](double x) { return x * x * x - 8.0; };

    RootFindingConfig config;
    config.brent_tol_abs = 1e-10;

    auto result = brent_find_root(f, 1.0, 3.0, config);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->root, 2.0, 1e-10);
    EXPECT_LT(result->final_error, config.brent_tol_abs);
}

TEST_F(RootFindingTest, BrentDetectsInvalidBracket) {
    // Function with no root in [1, 2]
    auto f = [](double x) { return x * x + 1.0; };

    RootFindingConfig config;

    auto result = brent_find_root(f, 1.0, 2.0, config);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, RootFindingErrorCode::InvalidBracket);
}

TEST_F(RootFindingTest, BrentHandlesNaN) {
    // Function that returns NaN
    auto f = [](double x) { return (x == 1.5) ? std::numeric_limits<double>::quiet_NaN() : x - 1.0; };

    RootFindingConfig config;

    auto result = brent_find_root(f, 0.0, 2.0, config);

    // May or may not hit the NaN depending on iteration path
    // If it does, should return NumericalInstability
    if (!result.has_value()) {
        EXPECT_TRUE(result.error().code == RootFindingErrorCode::NumericalInstability ||
                   result.error().code == RootFindingErrorCode::MaxIterationsExceeded);
    }
}

TEST_F(RootFindingTest, BrentFindsRootAtEndpoint) {
    // Root is exactly at the bracket endpoint
    auto f = [](double x) { return x - 1.0; };

    RootFindingConfig config;

    auto result = brent_find_root(f, 1.0, 2.0, config);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->root, 1.0, 1e-10);
    EXPECT_EQ(result->iterations, 0);  // Found immediately at endpoint
}

TEST_F(RootFindingTest, BrentConvergenceRate) {
    // Test that Brent converges in reasonable iterations
    auto f = [](double x) { return x * x - 2.0; };

    RootFindingConfig config;
    config.brent_tol_abs = 1e-12;

    auto result = brent_find_root(f, 0.0, 2.0, config);

    ASSERT_TRUE(result.has_value());
    // Brent should converge in < 20 iterations for this smooth problem
    EXPECT_LT(result->iterations, 20);
}

// ============================================================================
// Newton's Method Tests (Host)
// ============================================================================

TEST_F(RootFindingTest, NewtonFindsSqrtTwo) {
    // Find sqrt(2) by solving x^2 - 2 = 0
    auto f = [](double x) { return x * x - 2.0; };
    auto df = [](double x) { return 2.0 * x; };  // Derivative

    RootFindingConfig config;
    config.tolerance = 1e-10;

    auto result = newton_find_root(f, df, 1.5, 0.0, 3.0, config);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->root, std::sqrt(2.0), 1e-10);
    EXPECT_LT(result->iterations, 10);  // Newton should converge fast
    EXPECT_LT(result->final_error, config.tolerance);
}

TEST_F(RootFindingTest, NewtonQuadraticConvergence) {
    // Test that Newton converges very quickly (quadratic rate)
    auto f = [](double x) { return x * x - 2.0; };
    auto df = [](double x) { return 2.0 * x; };

    RootFindingConfig config;
    config.tolerance = 1e-14;

    auto result = newton_find_root(f, df, 1.5, 0.0, 3.0, config);

    ASSERT_TRUE(result.has_value());
    // Newton should converge in < 6 iterations for quadratic convergence
    EXPECT_LT(result->iterations, 6);
}

TEST_F(RootFindingTest, NewtonDetectsFlatDerivative) {
    // Function with zero derivative at a point
    auto f = [](double x) { return (x - 1.0) * (x - 1.0); };
    auto df = [](double x) { return 2.0 * (x - 1.0); };

    RootFindingConfig config;

    // Start exactly at the root where derivative is zero
    auto result = newton_find_root(f, df, 1.0, 0.0, 2.0, config);

    // Should either converge immediately or detect flat derivative
    if (result.has_value()) {
        EXPECT_NEAR(result->root, 1.0, 1e-10);
    } else {
        EXPECT_EQ(result.error().code, RootFindingErrorCode::NoProgress);
    }
}

TEST_F(RootFindingTest, NewtonEnforcesBounds) {
    // Test that Newton respects bounds even if unbounded step would go outside
    auto f = [](double x) { return x - 10.0; };  // Root at x=10
    auto df = [](double) { return 1.0; };

    RootFindingConfig config;
    config.max_iter = 20;

    // Initial guess 0, root at 10, but upper bound at 5
    auto result = newton_find_root(f, df, 0.0, 0.0, 5.0, config);

    // Should fail because root is outside bounds
    EXPECT_FALSE(result.has_value());
    if (!result.has_value()) {
        // Should hit bounds repeatedly and give up
        EXPECT_TRUE(result.error().code == RootFindingErrorCode::NoProgress ||
                   result.error().code == RootFindingErrorCode::MaxIterationsExceeded);
    }
}

TEST_F(RootFindingTest, NewtonDetectsInvalidBounds) {
    auto f = [](double x) { return x - 1.0; };
    auto df = [](double) { return 1.0; };

    RootFindingConfig config;

    // Invalid bounds: x_min >= x_max
    auto result = newton_find_root(f, df, 1.0, 2.0, 1.0, config);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, RootFindingErrorCode::InvalidBracket);
}

TEST_F(RootFindingTest, NewtonClampsInitialGuess) {
    // Test that initial guess is clamped to bounds
    auto f = [](double x) { return x - 1.5; };
    auto df = [](double) { return 1.0; };

    RootFindingConfig config;

    // Initial guess outside bounds
    auto result = newton_find_root(f, df, 10.0, 1.0, 2.0, config);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->root, 1.5, 1e-10);
}

TEST_F(RootFindingTest, NewtonHandlesNaN) {
    // Function that returns NaN
    auto f = [](double) { return std::numeric_limits<double>::quiet_NaN(); };
    auto df = [](double) { return 1.0; };

    RootFindingConfig config;

    auto result = newton_find_root(f, df, 1.0, 0.0, 2.0, config);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, RootFindingErrorCode::NumericalInstability);
}

// ============================================================================
// Device Execution Tests
// ============================================================================

// Device-callable test functions
KOKKOS_INLINE_FUNCTION
double test_func_sqrt2(double x) {
    return x * x - 2.0;
}

KOKKOS_INLINE_FUNCTION
double test_func_sqrt2_deriv(double x) {
    return 2.0 * x;
}

TEST_F(RootFindingTest, BrentDeviceExecution) {
    // Test that Brent's method can execute on device in parallel
    const int num_problems = 100;

    Kokkos::View<double*, HostMemSpace> roots("roots", num_problems);
    Kokkos::View<size_t*, HostMemSpace> iterations("iterations", num_problems);
    Kokkos::View<int*, HostMemSpace> converged("converged", num_problems);

    RootFindingConfig config;
    config.brent_tol_abs = 1e-10;

    // Solve x^2 - 2 = 0 in parallel for different brackets
    Kokkos::parallel_for("brent_device_test", num_problems,
        KOKKOS_LAMBDA(const int i) {
            // Each problem uses slightly different bracket
            double a = 1.0 + 0.001 * i;
            double b = 2.0 + 0.001 * i;

            auto result = brent_find_root_device(test_func_sqrt2, a, b, config);

            if (result.has_value()) {
                roots(i) = result->root;
                iterations(i) = result->iterations;
                converged(i) = 1;
            } else {
                roots(i) = 0.0;
                iterations(i) = 0;
                converged(i) = 0;
            }
        });

    Kokkos::fence();

    // Verify all problems converged to sqrt(2)
    for (int i = 0; i < num_problems; ++i) {
        EXPECT_EQ(converged(i), 1) << "Problem " << i << " failed to converge";
        if (converged(i)) {
            EXPECT_NEAR(roots(i), std::sqrt(2.0), 1e-9) << "Problem " << i;
            EXPECT_LT(iterations(i), config.max_iter);
        }
    }
}

TEST_F(RootFindingTest, NewtonDeviceExecution) {
    // Test that Newton's method can execute on device in parallel
    const int num_problems = 100;

    Kokkos::View<double*, HostMemSpace> roots("roots", num_problems);
    Kokkos::View<size_t*, HostMemSpace> iterations("iterations", num_problems);
    Kokkos::View<int*, HostMemSpace> converged("converged", num_problems);

    RootFindingConfig config;
    config.tolerance = 1e-10;

    // Solve x^2 - 2 = 0 in parallel with different initial guesses
    Kokkos::parallel_for("newton_device_test", num_problems,
        KOKKOS_LAMBDA(const int i) {
            // Each problem uses slightly different initial guess
            double x0 = 1.0 + 0.01 * i;

            auto result = newton_find_root_device(test_func_sqrt2, test_func_sqrt2_deriv,
                                                  x0, 0.5, 2.5, config);

            if (result.has_value()) {
                roots(i) = result->root;
                iterations(i) = result->iterations;
                converged(i) = 1;
            } else {
                roots(i) = 0.0;
                iterations(i) = 0;
                converged(i) = 0;
            }
        });

    Kokkos::fence();

    // Verify all problems converged to sqrt(2)
    for (int i = 0; i < num_problems; ++i) {
        EXPECT_EQ(converged(i), 1) << "Problem " << i << " failed to converge";
        if (converged(i)) {
            EXPECT_NEAR(roots(i), std::sqrt(2.0), 1e-9) << "Problem " << i;
            EXPECT_LT(iterations(i), config.max_iter);
        }
    }
}

// ============================================================================
// Comparison Tests
// ============================================================================

TEST_F(RootFindingTest, BrentVsNewtonConvergence) {
    // Compare Brent and Newton on the same problem
    auto f = [](double x) { return x * x - 2.0; };
    auto df = [](double x) { return 2.0 * x; };

    RootFindingConfig config;
    config.brent_tol_abs = 1e-10;
    config.tolerance = 1e-10;

    auto brent_result = brent_find_root(f, 1.0, 2.0, config);
    auto newton_result = newton_find_root(f, df, 1.5, 0.0, 3.0, config);

    ASSERT_TRUE(brent_result.has_value());
    ASSERT_TRUE(newton_result.has_value());

    // Both should find the same root
    EXPECT_NEAR(brent_result->root, newton_result->root, 1e-9);

    // Newton should converge in fewer iterations (quadratic vs superlinear)
    EXPECT_LT(newton_result->iterations, brent_result->iterations);
}

TEST_F(RootFindingTest, BrentMoreRobustThanNewton) {
    // Function where Newton might struggle but Brent is robust
    // f(x) = x^3 - 2x + 2 has a root around x = -1.77
    auto f = [](double x) { return x * x * x - 2.0 * x + 2.0; };
    auto df = [](double x) { return 3.0 * x * x - 2.0; };

    RootFindingConfig config;
    config.brent_tol_abs = 1e-10;
    config.tolerance = 1e-10;

    // Brent with wide bracket should always work
    auto brent_result = brent_find_root(f, -3.0, 0.0, config);
    ASSERT_TRUE(brent_result.has_value());

    // Newton with poor initial guess might fail or take many iterations
    auto newton_result = newton_find_root(f, df, 0.5, -3.0, 0.0, config);

    // Both should find roots in the valid range if they converge
    if (newton_result.has_value()) {
        EXPECT_GE(newton_result->root, -3.0);
        EXPECT_LE(newton_result->root, 0.0);
    }
}

}  // namespace mango::kokkos::test
