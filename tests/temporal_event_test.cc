#include <gtest/gtest.h>
#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/operators/laplacian_pde.hpp"
#include "src/pde/operators/spatial_operator.hpp"
#include "src/pde/operators/operator_factory.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"
#include "src/pde/core/trbdf2_config.hpp"
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
                  const mango::TRBDF2Config& config,
                  LeftBC left_bc,
                  RightBC right_bc,
                  SpatialOp spatial_op)
        : mango::PDESolver<TestPDESolver>(
              grid, time, config, std::nullopt, nullptr, {})
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
                      const mango::TRBDF2Config& config,
                      LeftBC left_bc,
                      RightBC right_bc,
                      SpatialOp spatial_op) {
    return TestPDESolver<LeftBC, RightBC, SpatialOp>(
        grid, time, config, std::move(left_bc), std::move(right_bc), std::move(spatial_op));
}

TEST(TemporalEventTest, EventAppliedAfterStep) {
    // Create grid
    auto grid_spec = GridSpec<double>::uniform(-1.0, 1.0, 51);
    ASSERT_TRUE(grid_spec.has_value());
    auto grid = grid_spec->generate();

    TimeDomain time(0.0, 1.0, 0.1);

    // Zero spatial operator (no PDE evolution) - use LaplacianPDE with D=0
    auto pde = operators::LaplacianPDE<double>(0.0);
    auto grid_view = GridView<double>(grid.span());
    auto spatial_op = operators::create_spatial_operator(std::move(pde), grid_view);

    DirichletBC left_bc{[](double t, double x) { return 0.0; }};
    DirichletBC right_bc{[](double t, double x) { return 0.0; }};

    TRBDF2Config trbdf2_config{};

    auto solver = make_test_solver(grid.span(), time, trbdf2_config,
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
    ASSERT_TRUE(status.has_value()) << status.error().message;

    // Verify event was applied
    EXPECT_TRUE(event_fired);

    auto solution = solver.solution();
    // After event at t=0.5, all values should be 2.0
    EXPECT_NEAR(solution[25], 2.0, 1e-10);
}

}  // namespace
}  // namespace mango
