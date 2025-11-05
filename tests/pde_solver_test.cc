#include "src/cpp/pde_solver.hpp"
#include "src/cpp/spatial_operators.hpp"
#include "src/cpp/boundary_conditions.hpp"
#include "src/cpp/root_finding.hpp"
#include "src/cpp/snapshot.hpp"
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
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, 51).generate();

    // Time domain
    mango::TimeDomain time(0.0, 0.1, 0.001);  // 100 time steps

    // TR-BDF2 config (force single block for small grid)
    mango::TRBDF2Config trbdf2;
    trbdf2.cache_blocking_threshold = 10000;

    // Root-finding config
    mango::RootFindingConfig root_config;

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
    mango::PDESolver solver(grid.span(), time, trbdf2, root_config, left_bc, right_bc, heat_op);

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

TEST(PDESolverTest, NewtonConvergence) {
    // Test that Newton converges in < 20 iterations for simple heat equation
    // This verifies the quasi-Newton implementation is working

    const size_t n = 101;
    const double D = 0.1;
    const double pi = std::numbers::pi;

    // Create grid
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, n).generate();

    // Time domain
    mango::TimeDomain time(0.0, 0.1, 0.01);  // 10 steps

    // TR-BDF2 config (force single block)
    mango::TRBDF2Config config;
    config.cache_blocking_threshold = 10000;

    // Root-finding config
    mango::RootFindingConfig root_config;
    root_config.max_iter = 20;  // Newton should converge well within this

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
    mango::PDESolver solver(grid.span(), time, config, root_config, left_bc, right_bc, heat_op);

    // Initialize with IC
    solver.initialize(ic);

    // Solve - should converge (Newton is robust)
    bool converged = solver.solve();
    EXPECT_TRUE(converged);

    // Solution should decay exponentially: u(x,t) ≈ exp(-π²Dt)sin(πx)
    auto solution = solver.solution();
    double expected_decay = std::exp(-pi * pi * D * 0.1);

    // Check middle point
    size_t mid = n / 2;
    double expected = expected_decay * std::sin(pi * grid.span()[mid]);
    EXPECT_NEAR(solution[mid], expected, 0.01);  // 1% tolerance
}

TEST(PDESolverTest, CacheBlockingCorrectness) {
    // Compare single-block vs multi-block on same PDE
    // Should produce identical results

    // Heat equation: du/dt = D * d2u/dx2
    mango::LaplacianOperator op(0.1);

    // Grid n=101 (force different blocking strategies via config)
    std::vector<double> grid(101);
    for (size_t i = 0; i < grid.size(); ++i) {
        grid[i] = static_cast<double>(i) / 100.0;
    }

    mango::TimeDomain time{0.0, 0.1, 0.01};
    mango::RootFindingConfig root_config;

    // Dirichlet BCs: u(0)=0, u(1)=0
    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    // Solver 1: Force single block
    mango::TRBDF2Config config1;
    config1.cache_blocking_threshold = 10000;  // Above n=101, so n_blocks=1

    mango::PDESolver solver1(grid, time, config1, root_config, left_bc, right_bc, op);

    // Solver 2: Force multi-block
    mango::TRBDF2Config config2;
    config2.cache_blocking_threshold = 20;  // Below n=101, so n_blocks > 1

    mango::PDESolver solver2(grid, time, config2, root_config, left_bc, right_bc, op);

    // Same initial condition: Gaussian
    const double pi = std::numbers::pi;
    auto ic = [pi](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            double dx = x[i] - 0.5;
            u[i] = std::exp(-50.0 * dx * dx);
        }
    };

    solver1.initialize(ic);
    solver2.initialize(ic);

    bool conv1 = solver1.solve();
    bool conv2 = solver2.solve();

    ASSERT_TRUE(conv1);
    ASSERT_TRUE(conv2);

    // Solutions should match to machine precision
    auto sol1 = solver1.solution();
    auto sol2 = solver2.solution();

    for (size_t i = 0; i < sol1.size(); ++i) {
        EXPECT_NEAR(sol1[i], sol2[i], 1e-12) << "Mismatch at i=" << i;
    }
}

TEST(PDESolverTest, UsesNewtonSolverForStages) {
    // Setup PDE solver with Newton integration
    const size_t n = 101;
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, n).generate();

    mango::TimeDomain time{0.0, 0.1, 0.01};
    mango::TRBDF2Config trbdf2_config;
    trbdf2_config.cache_blocking_threshold = 10000;  // Force single block
    mango::RootFindingConfig root_config{.max_iter = 20, .tolerance = 1e-6};

    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    const double D = 1.0;
    mango::LaplacianOperator spatial_op(D);

    mango::PDESolver solver(grid.span(), time, trbdf2_config, root_config,
                           left_bc, right_bc, spatial_op);

    // Initial condition: u(x, 0) = sin(πx)
    const double pi = std::numbers::pi;
    auto ic = [pi](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(pi * x[i]);
        }
    };
    solver.initialize(ic);

    bool converged = solver.solve();

    EXPECT_TRUE(converged);

    // Verify solution decayed (heat equation with zero BCs)
    auto solution = solver.solution();
    for (size_t i = 1; i < n - 1; ++i) {
        EXPECT_LT(std::abs(solution[i]), std::abs(std::sin(pi * grid.span()[i])));
    }
}

TEST(PDESolverTest, NewtonConvergenceReported) {
    // Test that Newton convergence failures propagate
    const size_t n = 51;
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, n).generate();

    mango::TimeDomain time{0.0, 1.0, 0.5};  // Large dt
    mango::TRBDF2Config trbdf2_config;
    trbdf2_config.cache_blocking_threshold = 10000;  // Force single block
    mango::RootFindingConfig root_config{.max_iter = 2, .tolerance = 1e-12};  // Hard to converge

    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    const double D = 1.0;
    mango::LaplacianOperator spatial_op(D);

    mango::PDESolver solver(grid.span(), time, trbdf2_config, root_config,
                           left_bc, right_bc, spatial_op);

    const double pi = std::numbers::pi;
    auto ic = [pi](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(pi * x[i]);
        }
    };
    solver.initialize(ic);

    bool converged = solver.solve();

    // With harsh convergence requirements, should fail
    EXPECT_FALSE(converged);
}

TEST(PDESolverTest, SnapshotRegistration) {
    mango::LaplacianOperator op(0.1);
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, 11).generate();
    mango::TimeDomain time(0.0, 1.0, 0.1);  // 10 steps
    mango::RootFindingConfig root_config;
    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    mango::PDESolver solver(grid.span(), time, mango::TRBDF2Config{},
                           root_config, left_bc, right_bc, op);

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
    auto grid = mango::GridSpec<>::uniform(0.0, 1.0, 21).generate();
    mango::TimeDomain time(0.0, 1.0, 0.25);  // 4 steps: 0.25, 0.5, 0.75, 1.0
    mango::RootFindingConfig root_config;
    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    mango::PDESolver solver(grid.span(), time, mango::TRBDF2Config{},
                           root_config, left_bc, right_bc, op);

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
    bool converged = solver.solve();
    ASSERT_TRUE(converged);

    // Verify snapshots collected with correct user_indices
    ASSERT_EQ(collector.collected_indices.size(), 2u);
    EXPECT_EQ(collector.collected_indices[0], 0u);  // tau_idx=0
    EXPECT_EQ(collector.collected_indices[1], 1u);  // tau_idx=1
}
