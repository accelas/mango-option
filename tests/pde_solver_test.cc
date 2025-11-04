#include "src/cpp/pde_solver.hpp"
#include "src/cpp/spatial_operators.hpp"
#include "src/cpp/boundary_conditions.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

TEST(PDESolverTest, HeatEquationDirichletBC) {
    // Heat equation: du/dt = D·d²u/dx² with D = 0.1
    // Domain: x ∈ [0, 1], t ∈ [0, 0.1]
    // BC: u(0,t) = 0, u(1,t) = 0
    // IC: u(x,0) = sin(π·x)
    // Analytical: u(x,t) = sin(π·x)·exp(-D·π²·t)

    const double D = 0.1;
    const double pi = std::numbers::pi;

    // Create grid
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, 51).generate();

    // Time domain
    mango::TimeDomain time(0.0, 0.1, 0.001);  // 100 time steps

    // TR-BDF2 config
    mango::TRBDF2Config trbdf2;

    // Boundary conditions: u(0,t) = 0, u(1,t) = 0
    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    // Spatial operator: L(u) = D·d²u/dx²
    auto heat_op = [D](double, std::span<const double> x,
                       std::span<const double> u, std::span<double> Lu,
                       std::span<const double> dx) {
        const size_t n = x.size();
        Lu[0] = Lu[n-1] = 0.0;  // Boundaries handled separately

        for (size_t i = 1; i < n - 1; ++i) {
            double dx_left = dx[i-1];
            double dx_right = dx[i];
            double dx_avg = (dx_left + dx_right) / 2.0;

            // Second derivative: d²u/dx²
            double d2u = (u[i+1] - u[i]) / dx_right - (u[i] - u[i-1]) / dx_left;
            d2u /= dx_avg;

            Lu[i] = D * d2u;
        }
    };

    // Initial condition: u(x,0) = sin(π·x)
    auto ic = [pi](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(pi * x[i]);
        }
    };

    // Create solver
    mango::PDESolver solver(grid.span(), time, trbdf2, left_bc, right_bc, heat_op);

    // Initialize with IC
    solver.initialize(ic);

    // Solve
    bool success = solver.solve();
    EXPECT_TRUE(success);

    // Verify against analytical solution
    auto solution = solver.solution();
    double decay = std::exp(-D * pi * pi * 0.1);

    for (size_t i = 0; i < grid.size(); ++i) {
        double x = grid.span()[i];
        double expected = std::sin(pi * x) * decay;
        EXPECT_NEAR(solution[i], expected, 5e-4);  // 0.05% relative error
    }
}

TEST(PDESolverTest, CacheBlockingCorrectness) {
    // Verify that cache-blocked solver produces identical results to single-block
    // Grid with n = 200 (triggers cache blocking at default threshold)

    const double D = 0.05;
    const double pi = std::numbers::pi;

    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, 200).generate();
    mango::TimeDomain time(0.0, 0.05, 0.001);
    mango::TRBDF2Config trbdf2;

    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    auto heat_op = [D](double, std::span<const double> x,
                       std::span<const double> u, std::span<double> Lu,
                       std::span<const double> dx) {
        const size_t n = x.size();
        Lu[0] = Lu[n-1] = 0.0;

        for (size_t i = 1; i < n - 1; ++i) {
            double dx_left = dx[i-1];
            double dx_right = dx[i];
            double dx_avg = (dx_left + dx_right) / 2.0;
            double d2u = (u[i+1] - u[i]) / dx_right - (u[i] - u[i-1]) / dx_left;
            d2u /= dx_avg;
            Lu[i] = D * d2u;
        }
    };

    auto ic = [pi](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(pi * x[i]);
        }
    };

    // Solve with default cache blocking (should use multiple blocks for n=200)
    mango::PDESolver solver1(grid.span(), time, trbdf2, left_bc, right_bc, heat_op);
    solver1.initialize(ic);
    bool success1 = solver1.solve();
    EXPECT_TRUE(success1);

    // Solve with forced single block (set cache threshold very high)
    mango::TRBDF2Config trbdf2_single_block = trbdf2;
    trbdf2_single_block.cache_blocking_threshold = 10000;  // Force single block

    mango::PDESolver solver2(grid.span(), time, trbdf2_single_block,
                              left_bc, right_bc, heat_op);
    solver2.initialize(ic);
    bool success2 = solver2.solve();
    EXPECT_TRUE(success2);

    // Results should be identical (within floating-point precision)
    auto sol1 = solver1.solution();
    auto sol2 = solver2.solution();

    for (size_t i = 0; i < grid.size(); ++i) {
        EXPECT_NEAR(sol1[i], sol2[i], 1e-12);  // Machine precision
    }
}
