#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/operators/laplacian_pde.hpp"
#include "src/pde/operators/operator_factory.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/core/pde_workspace.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
#include <memory_resource>

namespace {

// Simple diffusion solver for testing
template<typename LeftBC, typename RightBC, typename SpatialOp>
class DiffusionSolver : public mango::PDESolver<DiffusionSolver<LeftBC, RightBC, SpatialOp>> {
public:
    DiffusionSolver(std::shared_ptr<mango::Grid<double>> grid,
                   mango::PDEWorkspace workspace,
                   LeftBC left_bc,
                   RightBC right_bc,
                   SpatialOp spatial_op)
        : mango::PDESolver<DiffusionSolver>(grid, workspace)
        , grid_(grid)
        , left_bc_(std::move(left_bc))
        , right_bc_(std::move(right_bc))
        , spatial_op_(std::move(spatial_op))
    {}

    const LeftBC& left_boundary() const { return left_bc_; }
    const RightBC& right_boundary() const { return right_bc_; }
    const SpatialOp& spatial_operator() const { return spatial_op_; }

private:
    std::shared_ptr<mango::Grid<double>> grid_;
    LeftBC left_bc_;
    RightBC right_bc_;
    SpatialOp spatial_op_;
};

} // anonymous namespace

TEST(PDESolverSnapshotTest, RecordsInitialCondition) {
    auto grid_spec = mango::GridSpec<double>::uniform(0.0, 1.0, 11).value();
    auto time_domain = mango::TimeDomain::from_n_steps(0.0, 0.1, 10);
    std::vector<double> snapshot_times = {0.0};  // Just initial condition

    auto grid = mango::Grid<double>::create(grid_spec, time_domain, snapshot_times).value();

    std::pmr::monotonic_buffer_resource pool;

    // Create workspace with grid initialization
    size_t buffer_size = mango::PDEWorkspace::required_size(grid->n_space());
    std::pmr::vector<double> pmr_buffer(buffer_size, 0.0, &pool);
    auto workspace_result = mango::PDEWorkspace::from_buffer_and_grid(
        std::span{pmr_buffer.data(), pmr_buffer.size()},
        grid->x(),
        grid->n_space()
    );
    ASSERT_TRUE(workspace_result.has_value());
    auto workspace = workspace_result.value();

    // Boundary conditions
    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });

    // Spatial operator
    auto pde = mango::operators::LaplacianPDE<double>(0.1);
    auto spacing = std::make_shared<mango::GridSpacing<double>>(grid->spacing());
    auto spatial_op = mango::operators::create_spatial_operator(std::move(pde), spacing, workspace);

    DiffusionSolver solver(grid, workspace, left_bc, right_bc, spatial_op);

    // Set initial condition: u = sin(pi*x)
    const double pi = std::numbers::pi;
    solver.initialize([pi](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(pi * x[i]);
        }
    });

    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    // Verify initial condition was recorded
    EXPECT_TRUE(grid->has_snapshots());
    auto snap0 = grid->at(0);
    EXPECT_NEAR(snap0[5], std::sin(pi * 0.5), 1e-10);  // Middle point
}

TEST(PDESolverSnapshotTest, RecordsMultipleSnapshots) {
    auto grid_spec = mango::GridSpec<double>::uniform(0.0, 1.0, 11).value();
    auto time_domain = mango::TimeDomain::from_n_steps(0.0, 1.0, 10);  // dt=0.1
    std::vector<double> snapshot_times = {0.0, 0.5, 1.0};

    auto grid = mango::Grid<double>::create(grid_spec, time_domain, snapshot_times).value();

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

    auto left_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto right_bc = mango::DirichletBC([](double, double) { return 0.0; });
    auto pde = mango::operators::LaplacianPDE<double>(0.1);
    auto spacing = std::make_shared<mango::GridSpacing<double>>(grid->spacing());
    auto spatial_op = mango::operators::create_spatial_operator(std::move(pde), spacing, workspace);

    DiffusionSolver solver(grid, workspace, left_bc, right_bc, spatial_op);

    const double pi = std::numbers::pi;
    solver.initialize([pi](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(pi * x[i]);
        }
    });

    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    // Should have 3 snapshots
    EXPECT_EQ(grid->num_snapshots(), 3);

    // Each snapshot should have 11 points
    EXPECT_EQ(grid->at(0).size(), 11);
    EXPECT_EQ(grid->at(1).size(), 11);
    EXPECT_EQ(grid->at(2).size(), 11);

    // Solution should decay over time (diffusion)
    double initial_peak = grid->at(0)[5];
    double mid_peak = grid->at(1)[5];
    double final_peak = grid->at(2)[5];

    EXPECT_GT(initial_peak, mid_peak);
    EXPECT_GT(mid_peak, final_peak);
}
