#include <gtest/gtest.h>
#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/core/spatial_operators.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"
#include "src/pde/core/trbdf2_config.hpp"
#include "src/pde/core/root_finding.hpp"
#include <cmath>
#include <algorithm>

namespace mango {
namespace {

TEST(ObstacleTest, ProjectionDuringNewtonIteration) {
    // Create grid
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 51);
    auto grid = grid_spec.value().generate();

    TimeDomain time(0.0, 1.0, 0.01);

    // Zero spatial operator (no PDE evolution) - isolates obstacle enforcement
    LaplacianOperator spatial_op(0.0);

    // BCs compatible with obstacle (both = 0.5)
    DirichletBC left_bc{[](double t, double x) { return 0.5; }};
    DirichletBC right_bc{[](double t, double x) { return 0.5; }};

    TRBDF2Config trbdf2_config{};
    RootFindingConfig root_config{};

    // Define obstacle: ψ(x,t) = 0.5 everywhere
    auto obstacle = [](double t, std::span<const double> x, std::span<double> psi) {
        std::fill(psi.begin(), psi.end(), 0.5);
    };

    PDESolver solver(grid.span(), time, trbdf2_config, root_config,
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
    LaplacianOperator spatial_op(0.0);

    DirichletBC left_bc{[](double t, double x) { return 1.0; }};
    DirichletBC right_bc{[](double t, double x) { return 1.0; }};

    TRBDF2Config trbdf2_config{};
    RootFindingConfig root_config{};

    // Time-varying obstacle: ψ(x,t) = t (grows from 0 to 1)
    auto obstacle = [](double t, std::span<const double> x, std::span<double> psi) {
        std::fill(psi.begin(), psi.end(), t);
    };

    PDESolver solver(grid.span(), time, trbdf2_config, root_config,
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
    LaplacianOperator spatial_op(0.0);

    DirichletBC left_bc{[](double t, double x) { return 0.0; }};
    DirichletBC right_bc{[](double t, double x) { return 0.0; }};

    TRBDF2Config trbdf2_config{};
    RootFindingConfig root_config{};

    // No obstacle provided (std::nullopt)
    PDESolver solver(grid.span(), time, trbdf2_config, root_config,
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
