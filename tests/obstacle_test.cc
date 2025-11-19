#include <gtest/gtest.h>
#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/operators/laplacian_pde.hpp"
#include "src/pde/operators/operator_factory.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"
#include <cmath>
#include <algorithm>

namespace mango {
namespace {

// Test helper: Generic PDE solver for tests
template<typename LeftBC, typename RightBC, typename SpatialOp>
class TestPDESolver : public mango::PDESolver<TestPDESolver<LeftBC, RightBC, SpatialOp>> {
public:
    TestPDESolver(std::span<const double> grid,
                  const mango::TimeDomain& time,
                  LeftBC left_bc,
                  RightBC right_bc,
                  SpatialOp spatial_op,
                  std::optional<mango::ObstacleCallback> obstacle = std::nullopt)
        : mango::PDESolver<TestPDESolver>(
              grid, time, obstacle, nullptr, {})
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
auto make_test_solver(std::span<const double> grid,
                      const mango::TimeDomain& time,
                      LeftBC left_bc,
                      RightBC right_bc,
                      SpatialOp spatial_op,
                      std::optional<mango::ObstacleCallback> obstacle = std::nullopt) {
    return TestPDESolver<LeftBC, RightBC, SpatialOp>(
        grid, time, std::move(left_bc), std::move(right_bc), std::move(spatial_op), obstacle);
}

TEST(ObstacleTest, ProjectionDuringNewtonIteration) {
    // Create grid
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 51);
    auto grid = grid_spec.value().generate();

    TimeDomain time(0.0, 1.0, 0.01);

    // Zero spatial operator (no PDE evolution) - isolates obstacle enforcement
    auto pde = operators::LaplacianPDE<double>(0.0);
    auto grid_view = GridView<double>(grid.span());
    auto spatial_op = operators::create_spatial_operator(std::move(pde), grid_view);

    // BCs compatible with obstacle (both = 0.5)
    DirichletBC left_bc{[](double t, double x) { return 0.5; }};
    DirichletBC right_bc{[](double t, double x) { return 0.5; }};

    // Define obstacle: ψ(x,t) = 0.5 everywhere
    auto obstacle = [](double t, std::span<const double> x, std::span<double> psi) {
        std::fill(psi.begin(), psi.end(), 0.5);
    };

    auto solver = make_test_solver(grid.span(), time,
                     left_bc, right_bc, spatial_op, obstacle);

    // Initial condition: u = 0.3 everywhere (violates obstacle u ≥ ψ = 0.5)
    solver.initialize([](auto x, auto u) {
        std::fill(u.begin(), u.end(), 0.3);
    });

    auto status = solver.solve();
    ASSERT_TRUE(status.has_value()) << status.error().message;

    // Verify solution ≥ obstacle everywhere
    auto solution = solver.solution();
    for (size_t i = 0; i < solution.size(); ++i) {
        EXPECT_GE(solution[i], 0.5 - 1e-10)
            << "Solution violates obstacle at index " << i
            << ": u[" << i << "] = " << solution[i];
    }

    // Since spatial operator is zero and BCs are 0.5, obstacle projection
    // should enforce u = max(u, ψ) = max(0.3, 0.5) = 0.5 everywhere
    for (size_t i = 0; i < solution.size(); ++i) {
        EXPECT_NEAR(solution[i], 0.5, 1e-6)
            << "All points should equal obstacle value";
    }
}

TEST(ObstacleTest, TimeVaryingObstacle) {
    // Create grid
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 51);
    auto grid = grid_spec.value().generate();

    TimeDomain time(0.0, 1.0, 0.01);

    // Zero spatial operator
    auto pde = operators::LaplacianPDE<double>(0.0);
    auto grid_view = GridView<double>(grid.span());
    auto spatial_op = operators::create_spatial_operator(std::move(pde), grid_view);

    DirichletBC left_bc{[](double t, double x) { return 1.0; }};
    DirichletBC right_bc{[](double t, double x) { return 1.0; }};

    // Time-varying obstacle: ψ(x,t) = t (grows from 0 to 1)
    auto obstacle = [](double t, std::span<const double> x, std::span<double> psi) {
        std::fill(psi.begin(), psi.end(), t);
    };

    auto solver = make_test_solver(grid.span(), time,
                     left_bc, right_bc, spatial_op, obstacle);

    // Initial condition: u = 0.5 everywhere
    solver.initialize([](auto x, auto u) {
        std::fill(u.begin(), u.end(), 0.5);
    });

    auto status2 = solver.solve();
    ASSERT_TRUE(status2.has_value()) << status2.error().message;

    // At t=1.0, obstacle is ψ=1.0, so solution should be ≥ 1.0 everywhere
    auto solution = solver.solution();
    for (size_t i = 0; i < solution.size(); ++i) {
        EXPECT_GE(solution[i], 1.0 - 1e-10)
            << "Final solution should satisfy obstacle ψ(t=1) = 1.0";
    }
}

TEST(ObstacleTest, NoObstacleOptional) {
    // Create grid
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 51);
    auto grid = grid_spec.value().generate();

    TimeDomain time(0.0, 0.1, 0.01);

    // Zero spatial operator
    auto pde = operators::LaplacianPDE<double>(0.0);
    auto grid_view = GridView<double>(grid.span());
    auto spatial_op = operators::create_spatial_operator(std::move(pde), grid_view);

    DirichletBC left_bc{[](double t, double x) { return 0.0; }};
    DirichletBC right_bc{[](double t, double x) { return 0.0; }};

    // No obstacle provided (std::nullopt)
    auto solver = make_test_solver(grid.span(), time,
                     left_bc, right_bc, spatial_op);

    // Initial condition: u = 0.5 everywhere
    solver.initialize([](auto x, auto u) {
        std::fill(u.begin(), u.end(), 0.5);
    });

    auto status3 = solver.solve();
    ASSERT_TRUE(status3.has_value()) << status3.error().message;

    // Without obstacle and zero spatial operator, solution should remain ~0.5
    // (BCs are zero, so boundaries will be zero)
    auto solution = solver.solution();
    EXPECT_NEAR(solution[0], 0.0, 1e-6) << "Left BC should be enforced";
    EXPECT_NEAR(solution[solution.size() - 1], 0.0, 1e-6) << "Right BC should be enforced";
}

}  // namespace
}  // namespace mango
