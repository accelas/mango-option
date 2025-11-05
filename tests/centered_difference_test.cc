#include "src/cpp/operators/centered_difference.hpp"
#include "src/cpp/operators/grid_spacing.hpp"
#include "src/cpp/grid.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

TEST(CenteredDifferenceTest, UniformFirstDerivative) {
    // Uniform grid [0, 1] with dx = 0.1
    std::vector<double> x(11);
    for (size_t i = 0; i < 11; ++i) {
        x[i] = i * 0.1;
    }
    auto grid = mango::GridView<double>(x);
    auto spacing = mango::operators::GridSpacing<double>(grid);
    auto stencil = mango::operators::CenteredDifference<double>(spacing);

    // Test function: f(x) = x^2, f'(x) = 2x
    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = x[i] * x[i];
    }

    // Test interior point i=5 (x=0.5)
    // f'(0.5) = 2*0.5 = 1.0
    // Numerical: (u[6] - u[4]) / (2*dx) = (0.36 - 0.16) / 0.2 = 1.0
    double du_dx = stencil.first_derivative(u, 5);
    EXPECT_NEAR(du_dx, 1.0, 1e-10);
}

TEST(CenteredDifferenceTest, UniformSecondDerivative) {
    std::vector<double> x(11);
    for (size_t i = 0; i < 11; ++i) {
        x[i] = i * 0.1;
    }
    auto grid = mango::GridView<double>(x);
    auto spacing = mango::operators::GridSpacing<double>(grid);
    auto stencil = mango::operators::CenteredDifference<double>(spacing);

    // Test function: f(x) = x^2, f''(x) = 2
    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = x[i] * x[i];
    }

    // Test interior point i=5
    // f''(0.5) = 2.0
    // Numerical: (u[6] - 2*u[5] + u[4]) / dx^2 = (0.36 - 2*0.25 + 0.16) / 0.01 = 2.0
    double d2u_dx2 = stencil.second_derivative(u, 5);
    EXPECT_NEAR(d2u_dx2, 2.0, 1e-10);
}

TEST(CenteredDifferenceTest, ApplyUniformFusedKernel) {
    // Uniform grid [0, 1] with dx = 0.1
    std::vector<double> x(11);
    for (size_t i = 0; i < 11; ++i) {
        x[i] = i * 0.1;
    }
    auto grid = mango::GridView<double>(x);
    auto spacing = mango::operators::GridSpacing<double>(grid);
    auto stencil = mango::operators::CenteredDifference<double>(spacing);

    // Test function: f(x) = x^2
    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = x[i] * x[i];
    }

    // Test evaluator: Lu = a*d2u + b*du (simple linear combination)
    std::vector<double> Lu(11, 0.0);
    double a = 0.5;  // Coefficient for second derivative
    double b = 2.0;  // Coefficient for first derivative

    auto eval = [&](double d2u, double du, double val) {
        return a * d2u + b * du;
    };

    stencil.apply_uniform(u, Lu, 1, 10, eval);

    // Verify interior point i=5
    // f'(0.5) = 1.0, f''(0.5) = 2.0
    // Lu[5] = 0.5 * 2.0 + 2.0 * 1.0 = 3.0
    EXPECT_NEAR(Lu[5], 3.0, 1e-10);
}

TEST(CenteredDifferenceTest, NonUniformFirstDerivative) {
    // Non-uniform grid
    std::vector<double> x = {0.0, 0.5, 1.0, 2.0, 4.0};
    auto grid = mango::GridView<double>(x);
    auto spacing = mango::operators::GridSpacing<double>(grid);
    auto stencil = mango::operators::CenteredDifference<double>(spacing);

    // Test function: f(x) = x^2, f'(x) = 2x
    std::vector<double> u(5);
    for (size_t i = 0; i < 5; ++i) {
        u[i] = x[i] * x[i];
    }

    // Test interior point i=2 (x=1.0)
    // f'(1.0) = 2.0
    // dx_left = 0.5, dx_right = 1.0
    // Numerical: (u[3] - u[1]) / (dx_left + dx_right) = (4.0 - 0.25) / 1.5 = 2.5
    // (Note: Second-order accuracy on non-uniform grids degrades for highly non-uniform spacing)
    double du_dx = stencil.first_derivative(u, 2);
    EXPECT_NEAR(du_dx, 2.5, 1e-10);  // Exact numerical result for this spacing
}

TEST(CenteredDifferenceTest, ApplyNonUniformFusedKernel) {
    // Non-uniform grid
    std::vector<double> x = {0.0, 0.5, 1.0, 2.0, 4.0};
    auto grid = mango::GridView<double>(x);
    auto spacing = mango::operators::GridSpacing<double>(grid);
    auto stencil = mango::operators::CenteredDifference<double>(spacing);

    // Test function: f(x) = x^2
    std::vector<double> u(5);
    for (size_t i = 0; i < 5; ++i) {
        u[i] = x[i] * x[i];
    }

    // Test evaluator: Lu = a*d2u + b*du
    std::vector<double> Lu(5, 0.0);
    double a = 0.5;
    double b = 2.0;

    auto eval = [&](double d2u, double du, double val) {
        return a * d2u + b * du;
    };

    stencil.apply_non_uniform(u, Lu, 1, 4, eval);

    // Verify interior point i=2 (x=1.0)
    // First derivative (numerical): 2.5
    // Second derivative for non-uniform grid needs calculation
    // dx_left = 0.5, dx_right = 1.0, dx_center = 0.75
    // forward_diff = (u[3] - u[2]) / 1.0 = (4.0 - 1.0) / 1.0 = 3.0
    // backward_diff = (u[2] - u[1]) / 0.5 = (1.0 - 0.25) / 0.5 = 1.5
    // d2u_dx2 = (3.0 - 1.5) / 0.75 = 2.0
    // Lu[2] = 0.5 * 2.0 + 2.0 * 2.5 = 1.0 + 5.0 = 6.0
    EXPECT_NEAR(Lu[2], 6.0, 1e-10);
}

TEST(CenteredDifferenceTest, ComputeAllFirstUniform) {
    // Uniform grid [0, 1] with dx = 0.1
    std::vector<double> x(11);
    for (size_t i = 0; i < 11; ++i) {
        x[i] = i * 0.1;
    }
    auto grid = mango::GridView<double>(x);
    auto spacing = mango::operators::GridSpacing<double>(grid);
    auto stencil = mango::operators::CenteredDifference<double>(spacing);

    // Test function: f(x) = x^2, f'(x) = 2x
    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = x[i] * x[i];
    }

    std::vector<double> du_dx(11, 0.0);
    stencil.compute_all_first(u, du_dx, 1, 10);

    // Verify multiple points
    for (size_t i = 1; i < 10; ++i) {
        double expected = 2.0 * x[i];  // f'(x) = 2x
        EXPECT_NEAR(du_dx[i], expected, 1e-10);
    }
}

TEST(CenteredDifferenceTest, ComputeAllSecondUniform) {
    // Uniform grid [0, 1] with dx = 0.1
    std::vector<double> x(11);
    for (size_t i = 0; i < 11; ++i) {
        x[i] = i * 0.1;
    }
    auto grid = mango::GridView<double>(x);
    auto spacing = mango::operators::GridSpacing<double>(grid);
    auto stencil = mango::operators::CenteredDifference<double>(spacing);

    // Test function: f(x) = x^2, f''(x) = 2
    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = x[i] * x[i];
    }

    std::vector<double> d2u_dx2(11, 0.0);
    stencil.compute_all_second(u, d2u_dx2, 1, 10);

    // Verify multiple points (should all be 2.0)
    for (size_t i = 1; i < 10; ++i) {
        EXPECT_NEAR(d2u_dx2[i], 2.0, 1e-10);
    }
}

TEST(CenteredDifferenceTest, ComputeAllFirstNonUniform) {
    // Non-uniform grid
    std::vector<double> x = {0.0, 0.5, 1.0, 2.0, 4.0};
    auto grid = mango::GridView<double>(x);
    auto spacing = mango::operators::GridSpacing<double>(grid);
    auto stencil = mango::operators::CenteredDifference<double>(spacing);

    // Test function: f(x) = x^2, f'(x) = 2x
    std::vector<double> u(5);
    for (size_t i = 0; i < 5; ++i) {
        u[i] = x[i] * x[i];
    }

    std::vector<double> du_dx(5, 0.0);
    stencil.compute_all_first(u, du_dx, 1, 4);

    // Verify i=2: should match previous test
    EXPECT_NEAR(du_dx[2], 2.5, 1e-10);
}

TEST(CenteredDifferenceTest, ComputeAllSecondNonUniform) {
    // Non-uniform grid
    std::vector<double> x = {0.0, 0.5, 1.0, 2.0, 4.0};
    auto grid = mango::GridView<double>(x);
    auto spacing = mango::operators::GridSpacing<double>(grid);
    auto stencil = mango::operators::CenteredDifference<double>(spacing);

    // Test function: f(x) = x^2
    std::vector<double> u(5);
    for (size_t i = 0; i < 5; ++i) {
        u[i] = x[i] * x[i];
    }

    std::vector<double> d2u_dx2(5, 0.0);
    stencil.compute_all_second(u, d2u_dx2, 1, 4);

    // Verify i=2: should match calculation from ApplyNonUniformFusedKernel
    EXPECT_NEAR(d2u_dx2[2], 2.0, 1e-10);
}
