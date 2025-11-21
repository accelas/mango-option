#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/operators/operator_factory.hpp"
#include "src/pde/operators/laplacian_pde.hpp"
#include "src/pde/core/pde_workspace.hpp"
#include "src/pde/core/grid.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
#include <memory_resource>

namespace {

// Test helper: Generic PDE solver for tests using CRTP
template<typename LeftBC, typename RightBC, typename SpatialOp>
class TestPDESolver : public mango::PDESolver<TestPDESolver<LeftBC, RightBC, SpatialOp>> {
public:
    TestPDESolver(std::shared_ptr<mango::Grid<double>> grid,
                  mango::PDEWorkspace workspace,
                  LeftBC left_bc,
                  RightBC right_bc,
                  SpatialOp spatial_op)
        : mango::PDESolver<TestPDESolver>(grid, workspace)
        , left_bc_(std::move(left_bc))
        , right_bc_(std::move(right_bc))
        , spatial_op_(std::move(spatial_op))
    {}

    // CRTP interface
    const LeftBC& left_boundary() const { return left_bc_; }
    const RightBC& right_boundary() const { return right_bc_; }
    const SpatialOp& spatial_operator() const { return spatial_op_; }

private:
    LeftBC left_bc_;
    RightBC right_bc_;
    SpatialOp spatial_op_;
};

} // anonymous namespace

TEST(PDESolverTest, HeatEquationDirichletBC) {
    // Heat equation: du/dt = D·d²u/dx² with D = 0.1
    // Domain: x ∈ [0, 1], t ∈ [0, 0.1]
    // BC: u(0,t) = 0, u(1,t) = 0
    // IC: u(x,0) = sin(π·x)
    // Analytical: u(x,t) = sin(π·x)·exp(-D·π²·t)

    const double D = 0.1;
    const double pi = std::numbers::pi;

    // Create grid specification
    auto grid_spec = mango::GridSpec<double>::uniform(0.0, 1.0, 51).value();

    // Create time domain (from_n_steps: NOT direct constructor with dt!)
    auto time = mango::TimeDomain::from_n_steps(0.0, 0.1, 100);  // 100 time steps

    // Create Grid with solution storage
    auto grid_result = mango::Grid<double>::create(grid_spec, time);
    ASSERT_TRUE(grid_result.has_value()) << grid_result.error();
    auto grid = grid_result.value();

    // Create workspace with PMR
    std::pmr::monotonic_buffer_resource pool;
    size_t buffer_size = mango::PDEWorkspace::required_size(grid->n_space());
    std::pmr::vector<double> pmr_buffer(buffer_size, 0.0, &pool);
    auto workspace_result = mango::PDEWorkspace::from_buffer_and_grid(
        std::span{pmr_buffer.data(), pmr_buffer.size()},
        grid->x(),
        grid->n_space()
    );
    ASSERT_TRUE(workspace_result.has_value()) << workspace_result.error();
    auto workspace = workspace_result.value();

    // Boundary conditions: u(0,t) = 0, u(1,t) = 0
    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    // Spatial operator: L(u) = D·d²u/dx²
    auto pde_heat_op = mango::operators::LaplacianPDE<double>(D);
    auto spacing = std::make_shared<mango::GridSpacing<double>>(grid->spacing());
    auto heat_op = mango::operators::create_spatial_operator(std::move(pde_heat_op), spacing);

    // Initial condition: u(x,0) = sin(π·x)
    auto ic = [pi](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(pi * x[i]);
        }
    };

    // Create solver
    auto solver = TestPDESolver(grid, workspace, left_bc, right_bc, heat_op);

    // Increase Newton iterations for tests without analytical Jacobian
    mango::TRBDF2Config config;
    config.max_iter = 100;
    solver.set_config(config);

    // Initialize with IC
    solver.initialize(ic);

    // Solve
    auto status = solver.solve();
    ASSERT_TRUE(status.has_value()) << status.error().message;

    // Verify against analytical solution at t = 0.1
    auto solution = solver.solution();
    auto x = grid->x();
    const double t_final = 0.1;
    const double decay_factor = std::exp(-D * pi * pi * t_final);

    for (size_t i = 0; i < solution.size(); ++i) {
        double analytical = std::sin(pi * x[i]) * decay_factor;
        EXPECT_NEAR(solution[i], analytical, 1e-4)
            << "Mismatch at x[" << i << "] = " << x[i];
    }
}

TEST(PDESolverTest, HeatEquationNeumannBC) {
    // Heat equation: du/dt = D·d²u/dx² with D = 0.1
    // Domain: x ∈ [0, 1], t ∈ [0, 0.1]
    // BC: du/dx(0,t) = 0, du/dx(1,t) = 0 (insulated boundaries)
    // IC: u(x,0) = cos(π·x)
    // Analytical: u(x,t) = cos(π·x)·exp(-D·π²·t)

    const double D = 0.1;
    const double pi = std::numbers::pi;

    // Create grid specification
    auto grid_spec = mango::GridSpec<double>::uniform(0.0, 1.0, 51).value();

    // Create time domain
    auto time = mango::TimeDomain::from_n_steps(0.0, 0.1, 100);

    // Create Grid
    auto grid_result = mango::Grid<double>::create(grid_spec, time);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value();

    // Create workspace
    std::pmr::monotonic_buffer_resource pool;
    size_t buffer_size = mango::PDEWorkspace::required_size(grid->n_space());
    std::pmr::vector<double> pmr_buffer(buffer_size, 0.0, &pool);
    auto workspace_result = mango::PDEWorkspace::from_buffer_and_grid(
        std::span{pmr_buffer.data(), pmr_buffer.size()},
        grid->x(),
        grid->n_space()
    );
    ASSERT_TRUE(workspace_result.has_value());
    auto workspace = workspace_result.value();

    // Neumann boundary conditions: du/dx = 0 at both ends
    auto left_bc = mango::NeumannBC([](double, double) { return 0.0; }, D);
    auto right_bc = mango::NeumannBC([](double, double) { return 0.0; }, D);

    // Spatial operator
    auto pde_heat_op = mango::operators::LaplacianPDE<double>(D);
    auto spacing = std::make_shared<mango::GridSpacing<double>>(grid->spacing());
    auto heat_op = mango::operators::create_spatial_operator(std::move(pde_heat_op), spacing);

    // Initial condition: u(x,0) = cos(π·x)
    auto ic = [pi](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::cos(pi * x[i]);
        }
    };

    // Create solver
    auto solver = TestPDESolver(grid, workspace, left_bc, right_bc, heat_op);

    // Increase Newton iterations for tests without analytical Jacobian
    mango::TRBDF2Config config;
    config.max_iter = 100;
    solver.set_config(config);

    // Initialize and solve
    solver.initialize(ic);
    auto status = solver.solve();
    ASSERT_TRUE(status.has_value()) << status.error().message;

    // Verify against analytical solution
    auto solution = solver.solution();
    auto x = grid->x();
    const double t_final = 0.1;
    const double decay_factor = std::exp(-D * pi * pi * t_final);

    for (size_t i = 0; i < solution.size(); ++i) {
        double analytical = std::cos(pi * x[i]) * decay_factor;
        EXPECT_NEAR(solution[i], analytical, 1e-4)
            << "Mismatch at x[" << i << "] = " << x[i];
    }
}

TEST(PDESolverTest, SteadyStateConvergence) {
    // Test convergence to steady state: du/dt = D·d²u/dx²
    // With Dirichlet BC: u(0,t) = 0, u(1,t) = 1
    // Steady state: u(x) = x (linear profile)

    const double D = 0.1;

    auto grid_spec = mango::GridSpec<double>::uniform(0.0, 1.0, 51).value();
    auto time = mango::TimeDomain::from_n_steps(0.0, 1.0, 1000);  // Long time for steady state

    auto grid_result = mango::Grid<double>::create(grid_spec, time);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value();

    std::pmr::monotonic_buffer_resource pool;
    size_t buffer_size = mango::PDEWorkspace::required_size(grid->n_space());
    std::pmr::vector<double> pmr_buffer(buffer_size, 0.0, &pool);
    auto workspace_result = mango::PDEWorkspace::from_buffer_and_grid(
        std::span{pmr_buffer.data(), pmr_buffer.size()},
        grid->x(),
        grid->n_space()
    );
    ASSERT_TRUE(workspace_result.has_value());
    auto workspace = workspace_result.value();

    // Dirichlet BCs: u(0,t) = 0, u(1,t) = 1
    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 1.0; });

    auto pde_op = mango::operators::LaplacianPDE<double>(D);
    auto spacing = std::make_shared<mango::GridSpacing<double>>(grid->spacing());
    auto spatial_op = mango::operators::create_spatial_operator(std::move(pde_op), spacing);

    // Initial condition: u(x,0) = 0 (away from steady state)
    auto ic = [](std::span<const double>, std::span<double> u) {
        std::fill(u.begin(), u.end(), 0.0);
    };

    auto solver = TestPDESolver(grid, workspace, left_bc, right_bc, spatial_op);

    // Increase Newton iterations for tests without analytical Jacobian
    mango::TRBDF2Config config;
    config.max_iter = 100;
    solver.set_config(config);

    solver.initialize(ic);
    auto status = solver.solve();
    ASSERT_TRUE(status.has_value());

    // Verify convergence to linear profile u(x) = x
    auto solution = solver.solution();
    auto x = grid->x();

    for (size_t i = 0; i < solution.size(); ++i) {
        EXPECT_NEAR(solution[i], x[i], 1e-3)
            << "Steady state mismatch at x[" << i << "] = " << x[i];
    }
}
