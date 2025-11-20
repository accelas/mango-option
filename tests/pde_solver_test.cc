#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/operators/operator_factory.hpp"
#include "src/pde/operators/laplacian_pde.hpp"
#include "src/pde/core/pde_workspace.hpp"
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
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, 51).value().generate();

    // Time domain
    mango::TimeDomain time(0.0, 0.1, 0.001);  // 100 time steps

    // TR-BDF2 config
    mango::TRBDF2Config trbdf2;

    // Root-finding config

    // Boundary conditions: u(0,t) = 0, u(1,t) = 0
    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    // Spatial operator: L(u) = D·d²u/dx²
    auto pde_heat_op = mango::operators::LaplacianPDE<double>(D);
    auto grid_view_heat_op = mango::GridView<double>(grid.span());
    auto heat_op = mango::operators::create_spatial_operator(std::move(pde_heat_op), grid_view_heat_op);

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
    auto status = solver.solve();
    ASSERT_TRUE(status.has_value()) << status.error().message;

    // Verify against analytical solution
    auto solution = solver.solution();
    double decay = std::exp(-D * pi * pi * 0.1);

    for (size_t i = 0; i < grid.size(); ++i) {
        double x = grid.span()[i];
        double expected = std::sin(pi * x) * decay;
        EXPECT_NEAR(solution[i], expected, 5e-4);  // 0.05% relative error
    }
}

TEST(PDESolverTest, NewtonConvergence) {
    // Test that Newton converges in < 20 iterations for simple heat equation
    // This verifies the quasi-Newton implementation is working

    const size_t n = 101;
    const double D = 0.1;
    const double pi = std::numbers::pi;

    // Create grid
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, n).value().generate();

    // Time domain
    mango::TimeDomain time(0.0, 0.1, 0.01);  // 10 steps

    // TR-BDF2 config
    mango::TRBDF2Config config;

    // TR-BDF2 config (includes Newton parameters)
    config.max_iter = 20;  // Newton should converge well within this

    // Boundary conditions: u(0,t) = 0, u(1,t) = 0
    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    // Spatial operator: L(u) = D·d²u/dx²
    auto pde_heat_op = mango::operators::LaplacianPDE<double>(D);
    auto grid_view_heat_op = mango::GridView<double>(grid.span());
    auto heat_op = mango::operators::create_spatial_operator(std::move(pde_heat_op), grid_view_heat_op);

    // Initial condition: u(x,0) = sin(π·x)
    auto ic = [pi](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(pi * x[i]);
        }
    };

    // Create solver
    mango::PDESolver solver(grid.span(), time, config, left_bc, right_bc, heat_op);

    // Initialize with IC
    solver.initialize(ic);

    // Solve - should converge (Newton is robust)
    auto status3 = solver.solve();
    ASSERT_TRUE(status3.has_value()) << status3.error().message;

    // Solution should decay exponentially: u(x,t) ≈ exp(-π²Dt)sin(πx)
    auto solution = solver.solution();
    double expected_decay = std::exp(-pi * pi * D * 0.1);

    // Check middle point
    size_t mid = n / 2;
    double expected = expected_decay * std::sin(pi * grid.span()[mid]);
    EXPECT_NEAR(solution[mid], expected, 0.01);  // 1% tolerance
}

TEST(PDESolverTest, UsesNewtonSolverForStages) {
    // Setup PDE solver with Newton integration
    const size_t n = 101;
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, n).value().generate();

    mango::TimeDomain time{0.0, 0.1, 0.01};
    mango::TRBDF2Config trbdf2_config;

    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    const double D = 1.0;
    auto pde_spatial_op = mango::operators::LaplacianPDE<double>(D);
    auto grid_view_spatial_op = mango::GridView<double>(grid.span());
    auto spatial_op = mango::operators::create_spatial_operator(std::move(pde_spatial_op), grid_view_spatial_op);

    mango::PDESolver solver(grid.span(), time, trbdf2_config,
                           left_bc, right_bc, spatial_op);

    // Initial condition: u(x, 0) = sin(πx)
    const double pi = std::numbers::pi;
    auto ic = [pi](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(pi * x[i]);
        }
    };
    solver.initialize(ic);

    auto status4 = solver.solve();
    ASSERT_TRUE(status4.has_value()) << status4.error().message;

    // Verify solution decayed (heat equation with zero BCs)
    auto solution = solver.solution();
    for (size_t i = 1; i < n - 1; ++i) {
        EXPECT_LT(std::abs(solution[i]), std::abs(std::sin(pi * grid.span()[i])));
    }
}

TEST(PDESolverTest, NewtonConvergenceReported) {
    // Test that Newton convergence failures propagate
    const size_t n = 51;
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, n).value().generate();

    mango::TimeDomain time{0.0, 1.0, 0.5};  // Large dt
    mango::TRBDF2Config trbdf2_config{.max_iter = 2, .tolerance = 1e-12};  // Hard to converge

    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    const double D = 1.0;
    auto pde_spatial_op = mango::operators::LaplacianPDE<double>(D);
    auto grid_view_spatial_op = mango::GridView<double>(grid.span());
    auto spatial_op = mango::operators::create_spatial_operator(std::move(pde_spatial_op), grid_view_spatial_op);

    mango::PDESolver solver(grid.span(), time, trbdf2_config,
                           left_bc, right_bc, spatial_op);

    const double pi = std::numbers::pi;
    auto ic = [pi](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(pi * x[i]);
        }
    };
    solver.initialize(ic);

    auto status5 = solver.solve();

    // With harsh convergence requirements, should fail
    EXPECT_FALSE(status5.has_value());
}

TEST(PDESolverTest, WorksWithNewOperatorInterface) {
    // Test that PDESolver works with new SpatialOperator interface
    // This test uses the new composed operator architecture
    // Heat equation: du/dt = D·d²u/dx² with D = 0.1
    // Domain: x ∈ [0, 1], t ∈ [0, 0.1]
    // BC: u(0,t) = 0, u(1,t) = 0
    // IC: u(x,0) = sin(π·x)
    // Analytical: u(x,t) = sin(π·x)·exp(-D·π²·t)

    const double D = 0.1;
    const double pi = std::numbers::pi;

    // Create grid
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, 51).value().generate();
    auto grid_view = mango::GridView<double>(grid.span());

    // Create new spatial operator using factory
    auto spatial_op = mango::operators::create_spatial_operator(
        mango::operators::LaplacianPDE<double>(D),
        grid_view
    );

    // Time domain
    mango::TimeDomain time(0.0, 0.1, 0.001);  // 100 time steps

    // TR-BDF2 config
    mango::TRBDF2Config trbdf2;

    // Root-finding config

    // Boundary conditions: u(0,t) = 0, u(1,t) = 0
    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    // Initial condition: u(x,0) = sin(π·x)
    auto ic = [pi](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(pi * x[i]);
        }
    };

    // Create solver with new operator
    mango::PDESolver solver(grid.span(), time, trbdf2,
                           left_bc, right_bc, spatial_op);

    // Initialize with IC
    solver.initialize(ic);

    // Solve
    auto status2 = solver.solve();
    ASSERT_TRUE(status2.has_value()) << status2.error().message;

    // Verify against analytical solution
    auto solution = solver.solution();
    double decay = std::exp(-D * pi * pi * 0.1);

    for (size_t i = 0; i < grid.size(); ++i) {
        double x = grid.span()[i];
        double expected = std::sin(pi * x) * decay;
        EXPECT_NEAR(solution[i], expected, 5e-4);  // 0.05% relative error
    }
}

TEST(PDESolverTest, PDEWorkspaceIntegration) {
    // Verify PDEWorkspace is drop-in replacement for WorkspaceStorage
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    // This test will pass once we switch to PDEWorkspace
    mango::PDEWorkspace workspace(101, grid.span(), std::pmr::get_default_resource());

    EXPECT_EQ(workspace.u_current().size(), 101);
    EXPECT_EQ(workspace.dx().size(), 100);
}
