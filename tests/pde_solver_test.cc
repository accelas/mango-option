#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/spatial_operators.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/option/snapshot.hpp"
#include "src/pde/operators/operator_factory.hpp"
#include "src/pde/operators/laplacian_pde.hpp"
#include "src/pde/memory/pde_workspace.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

// Mock collector for testing
class MockCollector : public mango::SnapshotCollector {
public:
    std::vector<size_t> collected_indices;

    void collect(const mango::Snapshot& snapshot) override {
        collected_indices.push_back(snapshot.user_index);
    }
};

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
    mango::LaplacianOperator heat_op(D);

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
    mango::LaplacianOperator heat_op(D);

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
    mango::LaplacianOperator spatial_op(D);

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
    mango::LaplacianOperator spatial_op(D);

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

TEST(PDESolverTest, SnapshotRegistration) {
    mango::LaplacianOperator op(0.1);
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, 11).value().generate();
    mango::TimeDomain time(0.0, 1.0, 0.1);  // 10 steps
    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    mango::PDESolver solver(grid.span(), time, mango::TRBDF2Config{},
                           left_bc, right_bc, op);

    // Register snapshots at step indices 2, 5, 9
    MockCollector collector;
    solver.register_snapshot(2, 10, &collector);  // step_idx=2, user_idx=10
    solver.register_snapshot(5, 20, &collector);  // step_idx=5, user_idx=20
    solver.register_snapshot(9, 30, &collector);  // step_idx=9, user_idx=30

    // Verify registration (solve not called yet)
    EXPECT_EQ(collector.collected_indices.size(), 0u);
}

TEST(PDESolverTest, SnapshotCollection) {
    // Heat equation
    mango::LaplacianOperator op(0.1);
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, 21).value().generate();
    mango::TimeDomain time(0.0, 1.0, 0.25);  // 4 steps: 0.25, 0.5, 0.75, 1.0
    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    mango::PDESolver solver(grid.span(), time, mango::TRBDF2Config{},
                           left_bc, right_bc, op);

    // Initial condition: Gaussian
    auto ic = [](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            double dx = x[i] - 0.5;
            u[i] = std::exp(-50.0 * dx * dx);
        }
    };
    solver.initialize(ic);

    // Register snapshots at steps 1 and 3
    // user_index will be passed to collector (use for tau_idx)
    MockCollector collector;
    solver.register_snapshot(1, 0, &collector);  // step 1, tau_idx=0
    solver.register_snapshot(3, 1, &collector);  // step 3, tau_idx=1

    // Solve
    auto status6 = solver.solve();
    ASSERT_TRUE(status6.has_value()) << status6.error().message;

    // Verify snapshots collected with correct user_indices
    ASSERT_EQ(collector.collected_indices.size(), 2u);
    EXPECT_EQ(collector.collected_indices[0], 0u);  // tau_idx=0
    EXPECT_EQ(collector.collected_indices[1], 1u);  // tau_idx=1
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
    mango::PDEWorkspace workspace(101, grid.span());

    EXPECT_EQ(workspace.u_current().size(), 101);
    EXPECT_EQ(workspace.dx().size(), 100);
}
