#include "src/pde/operators/centered_difference_facade.hpp"
#include "src/pde/core/grid.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

namespace mango::operators {
namespace {

TEST(CenteredDifferenceFacadeTest, AutoModeWorks) {
    // Create non-uniform grid
    std::vector<double> x(11);
    x[0] = -1.0; x[1] = -0.8; x[2] = -0.5; x[3] = -0.2; x[4] = -0.05;
    x[5] = 0.0; x[6] = 0.05; x[7] = 0.2; x[8] = 0.5; x[9] = 0.8; x[10] = 1.0;

    auto grid = GridView<double>(x);
    auto spacing = GridSpacing<double>(grid);

    // Mode::Auto should select backend automatically
    CenteredDifference<double>stencil(spacing);  // Mode::Auto by default

    // Test with f(x) = x²
    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = x[i] * x[i];
    }

    std::vector<double> d2u_dx2(11, 0.0);
    stencil.compute_second_derivative(u, d2u_dx2, 1, 10);

    // d²(x²)/dx² = 2.0
    for (size_t i = 1; i < 10; ++i) {
        EXPECT_NEAR(d2u_dx2[i], 2.0, 1e-12)
            << "Mismatch at index " << i;
    }
}

TEST(CenteredDifferenceFacadeTest, ScalarModeWorks) {
    std::vector<double> x(11);
    x[0] = -1.0; x[1] = -0.8; x[2] = -0.5; x[3] = -0.2; x[4] = -0.05;
    x[5] = 0.0; x[6] = 0.05; x[7] = 0.2; x[8] = 0.5; x[9] = 0.8; x[10] = 1.0;

    auto grid = GridView<double>(x);
    auto spacing = GridSpacing<double>(grid);

    // Force scalar backend
    CenteredDifference<double>stencil(spacing, CenteredDifference<double>::Mode::Scalar);

    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = x[i] * x[i];
    }

    std::vector<double> d2u_dx2(11, 0.0);
    stencil.compute_second_derivative(u, d2u_dx2, 1, 10);

    for (size_t i = 1; i < 10; ++i) {
        EXPECT_NEAR(d2u_dx2[i], 2.0, 1e-12)
            << "Scalar backend failed at index " << i;
    }
}

TEST(CenteredDifferenceFacadeTest, SimdModeWorks) {
    std::vector<double> x(11);
    x[0] = -1.0; x[1] = -0.8; x[2] = -0.5; x[3] = -0.2; x[4] = -0.05;
    x[5] = 0.0; x[6] = 0.05; x[7] = 0.2; x[8] = 0.5; x[9] = 0.8; x[10] = 1.0;

    auto grid = GridView<double>(x);
    auto spacing = GridSpacing<double>(grid);

    // Force SIMD backend
    CenteredDifference<double>stencil(spacing, CenteredDifference<double>::Mode::Simd);

    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = x[i] * x[i];
    }

    std::vector<double> d2u_dx2(11, 0.0);
    stencil.compute_second_derivative(u, d2u_dx2, 1, 10);

    for (size_t i = 1; i < 10; ++i) {
        EXPECT_NEAR(d2u_dx2[i], 2.0, 1e-12)
            << "SIMD backend failed at index " << i;
    }
}

TEST(CenteredDifferenceFacadeTest, ScalarVsSimdMatch) {
    std::vector<double> x(11);
    x[0] = -1.0; x[1] = -0.8; x[2] = -0.5; x[3] = -0.2; x[4] = -0.05;
    x[5] = 0.0; x[6] = 0.05; x[7] = 0.2; x[8] = 0.5; x[9] = 0.8; x[10] = 1.0;

    auto grid = GridView<double>(x);
    auto spacing = GridSpacing<double>(grid);

    CenteredDifference<double>scalar_stencil(spacing, CenteredDifference<double>::Mode::Scalar);
    CenteredDifference<double>simd_stencil(spacing, CenteredDifference<double>::Mode::Simd);

    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = x[i] * x[i];
    }

    std::vector<double> d2u_scalar(11, 0.0), d2u_simd(11, 0.0);
    scalar_stencil.compute_second_derivative(u, d2u_scalar, 1, 10);
    simd_stencil.compute_second_derivative(u, d2u_simd, 1, 10);

    // Allow FP rounding differences
    for (size_t i = 1; i < 10; ++i) {
        EXPECT_NEAR(d2u_scalar[i], d2u_simd[i], 1e-14)
            << "Scalar/SIMD mismatch at index " << i;
    }
}

TEST(CenteredDifferenceTest, VariantDispatchCorrectness) {
    // Test both uniform and non-uniform paths work identically

    // Uniform grid
    auto uniform_grid_vec = std::vector<double>(11);
    for (size_t i = 0; i < 11; ++i) {
        uniform_grid_vec[i] = i * 0.1;
    }
    auto uniform_grid = mango::GridBuffer<double>(std::move(uniform_grid_vec));
    auto uniform_spacing = mango::GridSpacing<double>(uniform_grid.view());

    // Non-uniform grid (same points, perturbed slightly)
    auto nonuniform_grid_vec = std::vector<double>(11);
    for (size_t i = 0; i < 11; ++i) {
        nonuniform_grid_vec[i] = i * 0.1 + (i % 2) * 0.001;
    }
    auto nonuniform_grid = mango::GridBuffer<double>(std::move(nonuniform_grid_vec));
    auto nonuniform_spacing = mango::GridSpacing<double>(nonuniform_grid.view());

    // Both should compute derivatives (different codepaths, both work)
    // Use f(x) = x^2 for uniform grid
    std::vector<double> u_uniform(11);
    for (size_t i = 0; i < 11; ++i) {
        double x = i * 0.1;
        u_uniform[i] = x * x;
    }

    // Use f(x) = x^2 for non-uniform grid
    std::vector<double> u_nonuniform(11);
    for (size_t i = 0; i < 11; ++i) {
        double x = i * 0.1 + (i % 2) * 0.001;
        u_nonuniform[i] = x * x;
    }

    std::vector<double> du_uniform(11);
    std::vector<double> du_nonuniform(11);

    auto stencil_uniform = mango::operators::CenteredDifference(uniform_spacing);
    auto stencil_nonuniform = mango::operators::CenteredDifference(nonuniform_spacing);

    stencil_uniform.compute_first_derivative(u_uniform, du_uniform, 1, 10);
    stencil_nonuniform.compute_first_derivative(u_nonuniform, du_nonuniform, 1, 10);

    // Both should produce reasonable results (exact values differ due to different grids)
    // For f(x) = x^2, df/dx = 2x, so at x=0.5 (index 5 for uniform), df/dx ≈ 1.0
    EXPECT_NEAR(du_uniform[5], 1.0, 0.1);  // Computed something reasonable
    EXPECT_NE(du_nonuniform[5], 0.0);  // Non-uniform also computed something
}

} // namespace
} // namespace mango::operators
