#include "src/operators/centered_difference_simd.hpp"
#include "src/operators/grid_spacing.hpp"
#include "src/grid.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

TEST(CenteredDifferenceSIMDTest, UniformSecondDerivative) {
    // Create uniform grid
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 11);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::operators::GridSpacing<double> spacing(grid.view());
    mango::operators::CenteredDifferenceSIMD<double> stencil(spacing);

    // Test function: u(x) = x^2, d2u/dx2 = 2.0 everywhere
    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = grid.span()[i] * grid.span()[i];
    }

    std::vector<double> d2u_dx2(11, 0.0);

    // Compute second derivative (interior points only)
    stencil.compute_second_derivative_uniform(u, d2u_dx2, 1, 10);

    // Check interior points (should be 2.0)
    for (size_t i = 1; i < 10; ++i) {
        EXPECT_NEAR(d2u_dx2[i], 2.0, 1e-10);
    }
}

TEST(CenteredDifferenceSIMDTest, TiledComputation) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 10.0, 101);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::operators::GridSpacing<double> spacing(grid.view());
    mango::operators::CenteredDifferenceSIMD<double> stencil(spacing, 32);  // L1 tile size

    // u(x) = sin(x), d2u/dx2 = -sin(x)
    std::vector<double> u(101);
    for (size_t i = 0; i < 101; ++i) {
        u[i] = std::sin(grid.span()[i]);
    }

    std::vector<double> d2u_dx2(101, 0.0);

    stencil.compute_second_derivative_tiled(u, d2u_dx2, 1, 100);

    // Verify against analytical derivative
    // Note: Finite differences have truncation error O(dx^2)
    // With dx = 0.1, expect errors ~1e-3
    for (size_t i = 1; i < 100; ++i) {
        double expected = -std::sin(grid.span()[i]);
        EXPECT_NEAR(d2u_dx2[i], expected, 1e-3);
    }
}

TEST(CenteredDifferenceSIMDTest, FirstDerivative) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 11);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::operators::GridSpacing<double> spacing(grid.view());
    mango::operators::CenteredDifferenceSIMD<double> stencil(spacing);

    // u(x) = x^3, du/dx = 3x^2
    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = grid.span()[i] * grid.span()[i] * grid.span()[i];
    }

    std::vector<double> du_dx(11, 0.0);

    stencil.compute_first_derivative_uniform(u, du_dx, 1, 10);

    // Note: Finite differences have truncation error O(dx^2)
    // With dx = 0.1, expect errors ~1e-2
    // For polynomials with third-order term, there's an additional systematic error
    for (size_t i = 1; i < 10; ++i) {
        double x = grid.span()[i];
        double expected = 3.0 * x * x;
        EXPECT_NEAR(du_dx[i], expected, 0.015);
    }
}

TEST(CenteredDifferenceSIMDTest, PaddedArraySafety) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 10);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::operators::GridSpacing<double> spacing(grid.view());
    mango::operators::CenteredDifferenceSIMD<double> stencil(spacing);

    // Allocate with SIMD padding (10 → 16)
    std::vector<double> u(16, 0.0);
    std::vector<double> d2u_dx2(16, 0.0);

    // Fill logical portion
    for (size_t i = 0; i < 10; ++i) {
        u[i] = static_cast<double>(i);
    }

    // Compute should not crash on padded array
    stencil.compute_second_derivative_uniform(u, d2u_dx2, 1, 9);

    // Padding should remain zero
    EXPECT_DOUBLE_EQ(d2u_dx2[10], 0.0);
}

TEST(CenteredDifferenceSIMDTest, NonUniformSecondDerivative) {
    // Non-uniform grid (tanh-clustered)
    std::vector<double> x(11);
    x[0] = -1.0; x[1] = -0.8; x[2] = -0.5; x[3] = -0.2; x[4] = -0.05;
    x[5] = 0.0; x[6] = 0.05; x[7] = 0.2; x[8] = 0.5; x[9] = 0.8; x[10] = 1.0;

    auto grid = mango::GridView<double>(x);
    auto spacing = mango::operators::GridSpacing<double>(grid);
    auto stencil = mango::operators::CenteredDifferenceSIMD<double>(spacing);

    // Test function: f(x) = x^2, f''(x) = 2
    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) u[i] = x[i] * x[i];

    std::vector<double> d2u_dx2(11, 0.0);
    stencil.compute_second_derivative_non_uniform(u, d2u_dx2, 1, 10);

    // Should be close to 2.0 (with truncation error)
    for (size_t i = 1; i < 10; ++i) {
        EXPECT_NEAR(d2u_dx2[i], 2.0, 0.05) << "at index " << i;
    }
}
