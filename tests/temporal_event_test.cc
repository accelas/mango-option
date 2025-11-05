#include <gtest/gtest.h>
#include "src/cpp/pde_solver.hpp"
#include "src/cpp/boundary_conditions.hpp"
#include "src/cpp/spatial_operators.hpp"
#include "src/cpp/grid.hpp"
#include "src/cpp/time_domain.hpp"
#include "src/cpp/trbdf2_config.hpp"
#include "src/cpp/root_finding.hpp"
#include <cmath>
#include <algorithm>

namespace mango {
namespace {

TEST(TemporalEventTest, EventAppliedAfterStep) {
    // Create grid
    auto grid_spec = GridSpec<double>::uniform(-1.0, 1.0, 51);
    auto grid = grid_spec.generate();

    TimeDomain time(0.0, 1.0, 0.1);

    // Zero spatial operator (no PDE evolution) - use LaplacianOperator with D=0
    LaplacianOperator spatial_op(0.0);

    DirichletBC left_bc{[](double t, double x) { return 0.0; }};
    DirichletBC right_bc{[](double t, double x) { return 0.0; }};

    TRBDF2Config trbdf2_config{};
    RootFindingConfig root_config{};

    PDESolver solver(grid.span(), time, trbdf2_config, root_config,
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

    ASSERT_TRUE(solver.solve());

    // Verify event was applied
    EXPECT_TRUE(event_fired);

    auto solution = solver.solution();
    // After event at t=0.5, all values should be 2.0
    EXPECT_NEAR(solution[25], 2.0, 1e-10);
}

}  // namespace
}  // namespace mango
