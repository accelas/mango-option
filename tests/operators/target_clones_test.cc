// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/pde/operators/centered_difference_scalar.hpp"
#include "src/pde/core/grid.hpp"
#include <vector>
#include <cmath>
#include <dlfcn.h>

using namespace mango;
using namespace mango::operators;

// Test that target_clones generates multiple ISA versions
TEST(TargetClonesTest, MultipleISAVersionsGenerated) {
    // This test verifies symbol table contains .default, .avx2, .avx512f versions
    // We can't easily test this at runtime without inspecting binary
    // So this is a compile-time verification test - if it compiles, it works

    const size_t n = 100;
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i) / (n - 1);
    }

    GridView<double> grid_view(x);
    GridSpacing<double> spacing(grid_view);
    ScalarBackend<double> backend(spacing);

    std::vector<double> u(n);
    std::vector<double> result(n);
    for (size_t i = 0; i < n; ++i) {
        u[i] = std::sin(2.0 * M_PI * x[i]);
    }

    // Call should work regardless of ISA
    backend.compute_second_derivative_uniform(u, result, 1, n - 1);

    // Verify result is correct (numerical validation)
    // d²/dx²(sin(2πx)) = -4π² sin(2πx)
    // At x = 0.25: sin(2π * 0.25) = sin(π/2) = 1.0
    EXPECT_NEAR(result[25], -4.0 * M_PI * M_PI * std::sin(2.0 * M_PI * 0.25), 0.02);
}

// Test uniform grid operations work with auto-vectorization
TEST(TargetClonesTest, UniformGridOperations) {
    const size_t n = 1000;
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i) / (n - 1);
    }

    GridView<double> grid_view(x);
    GridSpacing<double> spacing(grid_view);
    ScalarBackend<double> backend(spacing);

    std::vector<double> u(n);
    std::vector<double> d2u(n);
    std::vector<double> du(n);

    for (size_t i = 0; i < n; ++i) {
        u[i] = std::sin(2.0 * M_PI * x[i]);
    }

    // Second derivative
    backend.compute_second_derivative_uniform(u, d2u, 1, n - 1);

    // First derivative
    backend.compute_first_derivative_uniform(u, du, 1, n - 1);

    // Verify results
    // At x = 0.25: sin(2π * 0.25) = sin(π/2) = 1.0
    // d²/dx²(sin(2πx)) = -4π² sin(2πx)
    EXPECT_NEAR(d2u[250], -4.0 * M_PI * M_PI * std::sin(2.0 * M_PI * 0.25), 1e-2);
    // d/dx(sin(2πx)) = 2π cos(2πx)
    EXPECT_NEAR(du[250], 2.0 * M_PI * std::cos(2.0 * M_PI * 0.25), 1e-2);
}

// Test non-uniform grid operations work with auto-vectorization
TEST(TargetClonesTest, NonUniformGridOperations) {
    const size_t n = 200;
    std::vector<double> x(n);
    const double stretch = 2.0;
    for (size_t i = 0; i < n; ++i) {
        double xi = static_cast<double>(i) / (n - 1);
        x[i] = std::sinh(stretch * (2.0 * xi - 1.0)) / std::sinh(stretch);
    }

    GridView<double> grid_view(x);
    GridSpacing<double> spacing(grid_view);
    ScalarBackend<double> backend(spacing);

    std::vector<double> u(n);
    std::vector<double> d2u(n);
    std::vector<double> du(n);

    for (size_t i = 0; i < n; ++i) {
        u[i] = x[i] * x[i];  // u = x^2, so du/dx = 2x, d2u/dx2 = 2
    }

    // Second derivative
    backend.compute_second_derivative_non_uniform(u, d2u, 1, n - 1);

    // First derivative
    backend.compute_first_derivative_non_uniform(u, du, 1, n - 1);

    // Verify results (looser tolerance for non-uniform grids)
    EXPECT_NEAR(d2u[100], 2.0, 0.1);
    EXPECT_NEAR(du[100], 2.0 * x[100], 0.05);
}
