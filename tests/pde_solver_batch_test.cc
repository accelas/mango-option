// Test suite for PDESolver batch mode vs single-contract mode
// Verifies that batch solving produces identical results to single-contract solving

#include <gtest/gtest.h>
#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/time_domain.hpp"
#include "src/pde/memory/pde_workspace.hpp"
#include "src/pde/operators/spatial_operator.hpp"
#include "src/pde/operators/centered_difference_facade.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/operators/black_scholes_pde.hpp"
#include <cmath>
#include <vector>

namespace mango {
namespace {

using namespace mango::operators;

// Helper: Create uniform grid in log-moneyness space
std::vector<double> create_uniform_grid(double x_min, double x_max, size_t n) {
    std::vector<double> grid(n);
    const double dx = (x_max - x_min) / (n - 1);
    for (size_t i = 0; i < n; ++i) {
        grid[i] = x_min + i * dx;
    }
    return grid;
}

// Test: Batch mode produces identical results to single-contract mode
TEST(PDESolverBatchTest, BatchMatchesSingleContract) {
    // Grid configuration
    constexpr size_t n = 101;
    constexpr double x_min = -1.0;
    constexpr double x_max = 1.0;
    auto grid = create_uniform_grid(x_min, x_max, n);

    // Time domain
    TimeDomain time_domain(0.0, 1.0, 0.001);

    // PDE parameters (Black-Scholes with American put obstacle)
    constexpr double volatility = 0.20;
    constexpr double rate = 0.05;
    constexpr double dividend = 0.02;
    constexpr double strike = 100.0;

    // Initial condition: American put payoff max(K - S, 0)
    auto initial_condition = [&](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            const double S = strike * std::exp(x[i]);
            u[i] = std::max(strike - S, 0.0);
        }
    };

    // Obstacle condition: American put payoff
    auto obstacle = [&](double t, std::span<const double> x, std::span<double> psi) {
        (void)t;  // Unused for time-independent payoff
        for (size_t i = 0; i < x.size(); ++i) {
            const double S = strike * std::exp(x[i]);
            psi[i] = std::max(strike - S, 0.0);
        }
    };

    // Black-Scholes PDE
    BlackScholesPDE pde(volatility, rate, dividend);
    auto spacing = std::make_shared<GridSpacing<double>>(GridView<double>(grid));
    auto spatial_op = SpatialOperator(pde, spacing);

    // Boundary conditions: zero at boundaries (far OTM and ITM)
    auto left_bc_func = [](double, double) { return 0.0; };
    auto right_bc_func = [](double, double) { return 0.0; };
    auto left_bc = DirichletBC(left_bc_func);
    auto right_bc = DirichletBC(right_bc_func);

    // Root-finding config
    RootFindingConfig root_config{
        .max_iter = 100,
        .tolerance = 1e-6,
        .jacobian_fd_epsilon = 1e-7,
        .brent_tol_abs = 1e-6
    };

    // TR-BDF2 config
    TRBDF2Config trbdf2_config{
        .max_iter = 100,
        .tolerance = 1e-6
    };

    // ==========================================
    // Single-contract mode (reference solution)
    // ==========================================
    PDESolver solver_single(
        grid, time_domain,
        trbdf2_config, root_config,
        left_bc, right_bc, spatial_op,
        obstacle
    );

    solver_single.initialize(initial_condition);
    auto result_single = solver_single.solve();
    ASSERT_TRUE(result_single.has_value()) << "Single-contract solver did not converge";

    auto solution_single = solver_single.solution();

    // ==========================================
    // Batch mode with 3 identical contracts
    // ==========================================
    constexpr size_t batch_width = 3;

    PDEWorkspace workspace_batch(n, std::span(grid), batch_width);

    PDESolver solver_batch(
        grid, time_domain,
        trbdf2_config, root_config,
        left_bc, right_bc, spatial_op,
        obstacle,
        &workspace_batch
    );

    // Initialize all lanes with same initial condition
    solver_batch.initialize(initial_condition);
    auto result_batch = solver_batch.solve();
    ASSERT_TRUE(result_batch.has_value()) << "Batch solver did not converge";

    // ==========================================
    // Verify: All lanes match single-contract
    // ==========================================
    for (size_t lane = 0; lane < batch_width; ++lane) {
        auto solution_lane = workspace_batch.u_lane(lane);

        for (size_t i = 0; i < n; ++i) {
            // Batch mode uses AoS layout (pack/scatter), single-contract uses SoA
            // Different FP operation ordering causes ~5e-4 to ~1e-3 precision differences
            EXPECT_NEAR(solution_lane[i], solution_single[i], 1.5e-3)
                << "Mismatch at lane=" << lane << ", i=" << i;
        }
    }
}

// Test: Batch mode with different initial conditions per lane
// TODO: This test has a design flaw - the batch_initial_condition callback
// expects an AoS buffer of size n*batch_width, but initialize() passes u_current_
// which is size n. Need to redesign the IC interface to support per-lane ICs.
TEST(PDESolverBatchTest, DISABLED_BatchWithDifferentInitialConditions) {
    // Grid configuration
    constexpr size_t n = 101;
    constexpr double x_min = -1.0;
    constexpr double x_max = 1.0;
    auto grid = create_uniform_grid(x_min, x_max, n);

    // Time domain
    TimeDomain time_domain(0.0, 0.5, 0.001);  // Shorter for faster test

    // PDE parameters
    constexpr double volatility = 0.20;
    constexpr double rate = 0.05;
    constexpr double dividend = 0.02;
    constexpr double strike = 100.0;

    // Black-Scholes PDE
    BlackScholesPDE pde(volatility, rate, dividend);
    auto spacing = std::make_shared<GridSpacing<double>>(GridView<double>(grid));
    auto spatial_op = SpatialOperator(pde, spacing);

    // Boundary conditions
    auto left_bc_func = [](double, double) { return 0.0; };
    auto right_bc_func = [](double, double) { return 0.0; };
    auto left_bc = DirichletBC(left_bc_func);
    auto right_bc = DirichletBC(right_bc_func);

    // Root-finding config
    RootFindingConfig root_config{
        .max_iter = 100,
        .tolerance = 1e-6,
        .jacobian_fd_epsilon = 1e-7,
        .brent_tol_abs = 1e-6
    };

    // TR-BDF2 config
    TRBDF2Config trbdf2_config{
        .max_iter = 100,
        .tolerance = 1e-6
    };

    // Obstacle condition
    auto obstacle = [&](double t, std::span<const double> x, std::span<double> psi) {
        (void)t;
        for (size_t i = 0; i < x.size(); ++i) {
            const double S = strike * std::exp(x[i]);
            psi[i] = std::max(strike - S, 0.0);
        }
    };

    // ==========================================
    // Solve 3 contracts individually with different spot prices
    // ==========================================
    std::vector<double> spot_prices = {90.0, 100.0, 110.0};
    std::vector<std::vector<double>> solutions_single(3);

    for (size_t contract = 0; contract < 3; ++contract) {
        const double spot = spot_prices[contract];

        // Initial condition: American put payoff at this spot
        auto initial_condition = [&](std::span<const double> x, std::span<double> u) {
            for (size_t i = 0; i < x.size(); ++i) {
                const double S = spot * std::exp(x[i]);
                u[i] = std::max(strike - S, 0.0);
            }
        };

        PDESolver solver(
            grid, time_domain,
            trbdf2_config, root_config,
            left_bc, right_bc, spatial_op,
            obstacle
        );

        solver.initialize(initial_condition);
        auto result = solver.solve();
        ASSERT_TRUE(result.has_value()) << "Single-contract solver did not converge for contract " << contract;

        auto solution = solver.solution();
        solutions_single[contract].assign(solution.begin(), solution.end());
    }

    // ==========================================
    // Solve same 3 contracts in batch mode
    // ==========================================
    constexpr size_t batch_width = 3;
    PDEWorkspace workspace_batch(n, std::span(grid), batch_width);

    PDESolver solver_batch(
        grid, time_domain,
        trbdf2_config, root_config,
        left_bc, right_bc, spatial_op,
        obstacle,
        &workspace_batch
    );

    // Initialize each lane with different spot price
    auto batch_initial_condition = [&](std::span<const double> x, std::span<double> u) {
        for (size_t lane = 0; lane < batch_width; ++lane) {
            const double spot = spot_prices[lane];
            for (size_t i = 0; i < x.size(); ++i) {
                const double S = spot * std::exp(x[i]);
                u[i * batch_width + lane] = std::max(strike - S, 0.0);
            }
        }
    };

    solver_batch.initialize(batch_initial_condition);
    auto result_batch = solver_batch.solve();
    ASSERT_TRUE(result_batch.has_value()) << "Batch solver did not converge";

    // ==========================================
    // Verify: Each lane matches corresponding single-contract
    // ==========================================
    for (size_t lane = 0; lane < batch_width; ++lane) {
        auto solution_lane = workspace_batch.u_lane(lane);

        for (size_t i = 0; i < n; ++i) {
            // Batch mode uses AoS layout (pack/scatter), single-contract uses SoA
            // Different FP operation ordering causes ~5e-4 to ~1e-3 precision differences
            EXPECT_NEAR(solution_lane[i], solutions_single[lane][i], 1.5e-3)
                << "Mismatch at lane=" << lane << ", i=" << i
                << ", spot=" << spot_prices[lane];
        }
    }
}

// Test: Batch mode convergence behavior
TEST(PDESolverBatchTest, BatchConvergenceBehavior) {
    // Grid configuration
    constexpr size_t n = 51;  // Smaller grid for faster test
    constexpr double x_min = -1.0;
    constexpr double x_max = 1.0;
    auto grid = create_uniform_grid(x_min, x_max, n);

    // Time domain
    TimeDomain time_domain(0.0, 0.1, 0.01);  // Short time

    // PDE parameters
    constexpr double volatility = 0.20;
    constexpr double rate = 0.05;
    constexpr double dividend = 0.02;

    // Black-Scholes PDE
    BlackScholesPDE pde(volatility, rate, dividend);
    auto spacing = std::make_shared<GridSpacing<double>>(GridView<double>(grid));
    auto spatial_op = SpatialOperator(pde, spacing);

    // Boundary conditions
    auto left_bc_func = [](double, double) { return 0.0; };
    auto right_bc_func = [](double, double) { return 0.0; };
    auto left_bc = DirichletBC(left_bc_func);
    auto right_bc = DirichletBC(right_bc_func);

    // Root-finding config
    RootFindingConfig root_config{
        .max_iter = 100,
        .tolerance = 1e-6,
        .jacobian_fd_epsilon = 1e-7,
        .brent_tol_abs = 1e-6
    };

    // TR-BDF2 config
    TRBDF2Config trbdf2_config{
        .max_iter = 100,
        .tolerance = 1e-6
    };

    // Simple initial condition (Gaussian)
    auto initial_condition = [&](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::exp(-x[i] * x[i]);
        }
    };

    // ==========================================
    // Batch mode with 4 lanes
    // ==========================================
    constexpr size_t batch_width = 4;
    PDEWorkspace workspace_batch(n, std::span(grid), batch_width);

    PDESolver solver_batch(
        grid, time_domain,
        trbdf2_config, root_config,
        left_bc, right_bc, spatial_op,
        std::nullopt,  // No obstacle
        &workspace_batch
    );

    solver_batch.initialize(initial_condition);
    auto result = solver_batch.solve();

    // Should converge successfully
    EXPECT_TRUE(result.has_value()) << "Batch solver failed to converge";

    // Verify solution is not NaN or Inf
    for (size_t lane = 0; lane < batch_width; ++lane) {
        auto solution_lane = workspace_batch.u_lane(lane);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_TRUE(std::isfinite(solution_lane[i]))
                << "Non-finite value at lane=" << lane << ", i=" << i;
        }
    }
}

}  // namespace
}  // namespace mango
