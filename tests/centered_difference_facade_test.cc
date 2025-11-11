#include "src/operators/centered_difference_facade.hpp"
#include "src/operators/grid_spacing.hpp"
#include "src/core/grid.hpp"
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

} // namespace
} // namespace mango::operators
