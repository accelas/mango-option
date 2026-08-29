// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/pde/internal/pde_solver.hpp"
#include "mango/pde/internal/pde_workspace.hpp"
#include "mango/pde/core/grid.hpp"
#include "mango/pde/core/boundary_conditions.hpp"
#include "mango/pde/operators/laplacian_pde.hpp"
#include "mango/pde/internal/spatial_operator.hpp"
#include "mango/pde/internal/operator_factory.hpp"
#include "mango/pde/core/grid.hpp"
#include "mango/pde/core/time_domain.hpp"
#include <cmath>
#include <algorithm>
#include <memory_resource>

namespace mango {
namespace {

// Test helper: Generic PDE solver for tests
template<typename LeftBC, typename RightBC, typename SpatialOp>
class TestPDESolver : public mango::PDESolver<TestPDESolver<LeftBC, RightBC, SpatialOp>> {
public:
    TestPDESolver(std::shared_ptr<Grid<double>> grid,
                  PDEWorkspace workspace,
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

// Helper function to create test solver with deduced types
template<typename LeftBC, typename RightBC, typename SpatialOp>
auto make_test_solver(std::shared_ptr<Grid<double>> grid,
                      PDEWorkspace workspace,
                      LeftBC left_bc,
                      RightBC right_bc,
                      SpatialOp spatial_op) {
    return TestPDESolver<LeftBC, RightBC, SpatialOp>(
        grid, workspace, std::move(left_bc), std::move(right_bc), std::move(spatial_op));
}

TEST(TemporalEventTest, EventAppliedAfterStep) {
    // Create grid spec
    auto grid_spec = GridSpec<double>::uniform(-1.0, 1.0, 51);
    ASSERT_TRUE(grid_spec.has_value());

    // Create TimeDomain
    TimeDomain time = TimeDomain::from_n_steps(0.0, 1.0, 10);

    // Create Grid
    auto grid_with_sol = Grid<double>::create(grid_spec.value(), time);
    ASSERT_TRUE(grid_with_sol.has_value());

    // Create PMR workspace
    std::pmr::synchronized_pool_resource pool;
    size_t n = grid_spec->n_points();
    size_t buffer_size = PDEWorkspace::required_size(n);
    std::pmr::vector<double> pmr_buffer(buffer_size, 0.0, &pool);

    auto workspace_spans = PDEWorkspace::from_buffer_and_grid(
        std::span{pmr_buffer.data(), pmr_buffer.size()},
        grid_with_sol.value()->x(),
        n
    );
    ASSERT_TRUE(workspace_spans.has_value());
    auto workspace = workspace_spans.value();

    // Zero spatial operator (no PDE evolution) - use LaplacianPDE with D=0
    auto pde = operators::LaplacianPDE<double>(0.0);
    auto spacing = std::make_shared<GridSpacing<double>>(grid_with_sol.value()->spacing());
    auto spatial_op = operators::create_spatial_operator(std::move(pde), spacing, workspace);

    DirichletBC left_bc{[](double t, double x) { return 0.0; }};
    DirichletBC right_bc{[](double t, double x) { return 0.0; }};

    auto solver = make_test_solver(grid_with_sol.value(), workspace,
                                   left_bc, right_bc, spatial_op);

    // Initial condition: u = 1 everywhere
    solver.initialize([](auto x, auto u) {
        std::fill(u.begin(), u.end(), 1.0);
    });

    // Add event at t=0.5 that doubles all values
    bool event_fired = false;
    solver.add_temporal_event(0.5, [&](double t, auto x, auto u) {
        event_fired = true;
        for (size_t i = 0; i < u.size(); ++i) {
            u[i] *= 2.0;
        }
    });

    auto status = solver.solve();
    ASSERT_TRUE(status.has_value()) << status.error();

    // Verify event was applied
    EXPECT_TRUE(event_fired);

    auto solution = solver.solution();
    // After event at t=0.5, all values should be 2.0
    EXPECT_NEAR(solution[25], 2.0, 1e-10);
}

// Regression: a reused solver must replay temporal events
// Bug: next_event_idx_ was never reset, so a second initialize()+solve()
//      on the same instance silently skipped all events (dividends dropped).
TEST(TemporalEventTest, ReusedSolverReplaysEvents) {
    // Create grid spec
    auto grid_spec = GridSpec<double>::uniform(-1.0, 1.0, 51);
    ASSERT_TRUE(grid_spec.has_value());

    // Create TimeDomain
    TimeDomain time = TimeDomain::from_n_steps(0.0, 1.0, 10);

    // Create Grid
    auto grid_with_sol = Grid<double>::create(grid_spec.value(), time);
    ASSERT_TRUE(grid_with_sol.has_value());

    // Create PMR workspace
    std::pmr::synchronized_pool_resource pool;
    size_t n = grid_spec->n_points();
    size_t buffer_size = PDEWorkspace::required_size(n);
    std::pmr::vector<double> pmr_buffer(buffer_size, 0.0, &pool);

    auto workspace_spans = PDEWorkspace::from_buffer_and_grid(
        std::span{pmr_buffer.data(), pmr_buffer.size()},
        grid_with_sol.value()->x(),
        n
    );
    ASSERT_TRUE(workspace_spans.has_value());
    auto workspace = workspace_spans.value();

    // Zero spatial operator (no PDE evolution) - use LaplacianPDE with D=0
    auto pde = operators::LaplacianPDE<double>(0.0);
    auto spacing = std::make_shared<GridSpacing<double>>(grid_with_sol.value()->spacing());
    auto spatial_op = operators::create_spatial_operator(std::move(pde), spacing, workspace);

    DirichletBC left_bc{[](double t, double x) { return 0.0; }};
    DirichletBC right_bc{[](double t, double x) { return 0.0; }};

    auto solver = make_test_solver(grid_with_sol.value(), workspace,
                                   left_bc, right_bc, spatial_op);

    int event_calls = 0;
    // register one mid-interval event that bumps the solution and counts calls
    solver.add_temporal_event(0.5, [&](double t, auto x, auto u) {
        ++event_calls;
        for (size_t i = 0; i < u.size(); ++i) {
            u[i] += 1.0;
        }
    });

    auto ic = [](std::span<const double> /*x*/, std::span<double> u) {
        std::fill(u.begin(), u.end(), 0.0);
    };

    solver.initialize(ic);
    ASSERT_TRUE(solver.solve().has_value());
    auto solver_solution = solver.solution();
    std::vector<double> first(solver_solution.begin(), solver_solution.end());
    EXPECT_EQ(event_calls, 1);

    solver.initialize(ic);
    ASSERT_TRUE(solver.solve().has_value());
    solver_solution = solver.solution();
    std::vector<double> second(solver_solution.begin(), solver_solution.end());

    EXPECT_EQ(event_calls, 2);  // fired on BOTH runs
    for (size_t i = 0; i < first.size(); ++i) {
        EXPECT_DOUBLE_EQ(second[i], first[i]);
    }
}

}  // namespace
}  // namespace mango
