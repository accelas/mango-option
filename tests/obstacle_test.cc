#include <gtest/gtest.h>
#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/operators/laplacian_pde.hpp"
#include "src/pde/operators/operator_factory.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/core/pde_workspace.hpp"
#include <cmath>
#include <algorithm>
#include <memory_resource>

namespace mango {
namespace {

// Test helper: PDE solver with obstacle
template<typename LeftBC, typename RightBC, typename SpatialOp, typename ObstacleFunc>
class TestPDESolverWithObstacle : public mango::PDESolver<TestPDESolverWithObstacle<LeftBC, RightBC, SpatialOp, ObstacleFunc>> {
public:
    TestPDESolverWithObstacle(std::shared_ptr<Grid<double>> grid,
                              PDEWorkspace workspace,
                              LeftBC left_bc,
                              RightBC right_bc,
                              SpatialOp spatial_op,
                              ObstacleFunc obstacle_func)
        : mango::PDESolver<TestPDESolverWithObstacle>(grid, workspace)
        , left_bc_(std::move(left_bc))
        , right_bc_(std::move(right_bc))
        , spatial_op_(std::move(spatial_op))
        , obstacle_func_(std::move(obstacle_func))
    {}

    // CRTP interface
    const LeftBC& left_boundary() const { return left_bc_; }
    const RightBC& right_boundary() const { return right_bc_; }
    const SpatialOp& spatial_operator() const { return spatial_op_; }

    // Obstacle method for CRTP
    void obstacle(double t, std::span<const double> x, std::span<double> psi) const {
        obstacle_func_(t, x, psi);
    }

private:
    LeftBC left_bc_;
    RightBC right_bc_;
    SpatialOp spatial_op_;
    ObstacleFunc obstacle_func_;
};

TEST(ObstacleTest, ProjectionDuringNewtonIteration) {
    // Create grid specification
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 51).value();
    auto time = TimeDomain::from_n_steps(0.0, 1.0, 100);

    // Create Grid
    auto grid_result = Grid<double>::create(grid_spec, time);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value();

    // Create workspace
    std::pmr::monotonic_buffer_resource pool;
    size_t buffer_size = PDEWorkspace::required_size(grid->n_space());
    std::pmr::vector<double> pmr_buffer(buffer_size, 0.0, &pool);
    auto workspace_result = PDEWorkspace::from_buffer_and_grid(
        std::span{pmr_buffer.data(), pmr_buffer.size()},
        grid->x(),
        grid->n_space()
    );
    ASSERT_TRUE(workspace_result.has_value());
    auto workspace = workspace_result.value();

    // Zero spatial operator (no PDE evolution) - isolates obstacle enforcement
    auto pde = operators::LaplacianPDE<double>(0.0);
    auto spacing = std::make_shared<GridSpacing<double>>(grid->spacing());
    auto spatial_op = operators::create_spatial_operator(std::move(pde), spacing);

    // BCs compatible with obstacle (both = 0.5)
    DirichletBC left_bc{[](double, double) { return 0.5; }};
    DirichletBC right_bc{[](double, double) { return 0.5; }};

    // Define obstacle: ψ(x,t) = 0.5 everywhere
    auto obstacle_func = [](double, std::span<const double>, std::span<double> psi) {
        std::fill(psi.begin(), psi.end(), 0.5);
    };

    auto solver = TestPDESolverWithObstacle(
        grid, workspace,
        left_bc, right_bc, spatial_op, obstacle_func);

    // Initial condition: u = 0.3 everywhere (violates obstacle u ≥ ψ = 0.5)
    solver.initialize([](auto, auto u) {
        std::fill(u.begin(), u.end(), 0.3);
    });

    // After initialization, obstacle should be enforced: u ≥ 0.5
    auto initial_solution = solver.solution();
    for (size_t i = 0; i < initial_solution.size(); ++i) {
        EXPECT_GE(initial_solution[i], 0.5)
            << "Initial obstacle violated at i=" << i;
    }

    // Solve (zero PDE, so solution should remain constant at obstacle value)
    auto status = solver.solve();
    ASSERT_TRUE(status.has_value()) << status.error();

    // Verify final solution still respects obstacle
    auto final_solution = solver.solution();
    for (size_t i = 0; i < final_solution.size(); ++i) {
        EXPECT_NEAR(final_solution[i], 0.5, 1e-10)
            << "Final solution incorrect at i=" << i;
    }
}

TEST(ObstacleTest, DiffusionWithLowerBound) {
    // Test diffusion with lower obstacle bound
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 51).value();
    auto time = TimeDomain::from_n_steps(0.0, 0.1, 100);

    auto grid_result = Grid<double>::create(grid_spec, time);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value();

    std::pmr::monotonic_buffer_resource pool;
    size_t buffer_size = PDEWorkspace::required_size(grid->n_space());
    std::pmr::vector<double> pmr_buffer(buffer_size, 0.0, &pool);
    auto workspace_result = PDEWorkspace::from_buffer_and_grid(
        std::span{pmr_buffer.data(), pmr_buffer.size()},
        grid->x(),
        grid->n_space()
    );
    ASSERT_TRUE(workspace_result.has_value());
    auto workspace = workspace_result.value();

    // Diffusion operator
    auto pde = operators::LaplacianPDE<double>(0.1);
    auto spacing = std::make_shared<GridSpacing<double>>(grid->spacing());
    auto spatial_op = operators::create_spatial_operator(std::move(pde), spacing);

    // Dirichlet BCs: u(0,t) = 0.2, u(1,t) = 0.2 (compatible with obstacle)
    DirichletBC left_bc{[](double, double) { return 0.2; }};
    DirichletBC right_bc{[](double, double) { return 0.2; }};

    // Obstacle: u ≥ 0.2 everywhere
    auto obstacle_func = [](double, std::span<const double>, std::span<double> psi) {
        std::fill(psi.begin(), psi.end(), 0.2);
    };

    auto solver = TestPDESolverWithObstacle(
        grid, workspace,
        left_bc, right_bc, spatial_op, obstacle_func);

    // Initial condition: u = 0.5 everywhere (above obstacle)
    solver.initialize([](auto, auto u) {
        std::fill(u.begin(), u.end(), 0.5);
    });

    auto status = solver.solve();
    ASSERT_TRUE(status.has_value()) << status.error();

    // Final solution should respect obstacle: u ≥ 0.2 everywhere
    auto final_solution = solver.solution();
    for (size_t i = 0; i < final_solution.size(); ++i) {
        EXPECT_GE(final_solution[i], 0.2 - 1e-10)
            << "Obstacle violated at i=" << i;
    }
}

} // namespace
} // namespace mango
