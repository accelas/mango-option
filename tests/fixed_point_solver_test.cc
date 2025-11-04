#include "src/cpp/fixed_point_solver.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(FixedPointSolverTest, SimpleConvergence) {
    // Solve: x = cos(x) with x0 = 1.0
    // Known solution: x ≈ 0.7390851332151607

    double x = 1.0;
    size_t iterations = 0;

    auto iterate = [](double x) { return std::cos(x); };

    bool converged = mango::fixed_point_solve(
        x, iterate, 100, 1e-6, 0.7, iterations
    );

    EXPECT_TRUE(converged);
    EXPECT_NEAR(x, 0.7390851332151607, 1e-6);
    EXPECT_LT(iterations, 50);  // Should converge quickly
}

TEST(FixedPointSolverTest, UnderRelaxation) {
    // Test that under-relaxation parameter affects convergence
    double x1 = 1.0, x2 = 1.0;
    size_t iter1 = 0, iter2 = 0;

    auto iterate = [](double x) { return std::cos(x); };

    // Without relaxation (omega = 1.0)
    mango::fixed_point_solve(x1, iterate, 100, 1e-6, 1.0, iter1);

    // With relaxation (omega = 0.7)
    mango::fixed_point_solve(x2, iterate, 100, 1e-6, 0.7, iter2);

    // Both should converge to same value
    EXPECT_NEAR(x1, x2, 1e-6);
}

TEST(FixedPointSolverTest, FailToConverge) {
    // Diverging iteration: x = 2*x (diverges unless x=0)
    double x = 1.0;
    size_t iterations = 0;

    auto iterate = [](double x) { return 2.0 * x; };

    bool converged = mango::fixed_point_solve(
        x, iterate, 10, 1e-6, 1.0, iterations
    );

    EXPECT_FALSE(converged);
    EXPECT_EQ(iterations, 10);  // Hit max iterations
}
