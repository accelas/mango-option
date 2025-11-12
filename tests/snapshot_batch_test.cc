// Test suite for batch snapshot collection
// Verifies that per-lane snapshot collectors receive correct data

#include <gtest/gtest.h>
#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/time_domain.hpp"
#include "src/pde/memory/pde_workspace.hpp"
#include "src/pde/operators/spatial_operator.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/operators/black_scholes_pde.hpp"
#include "src/option/snapshot.hpp"
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

// Test collector that stores snapshots for verification
class TestSnapshotCollector : public SnapshotCollector {
public:
    void collect(const Snapshot& snapshot) override {
        snapshots.push_back(snapshot);
        // Store a copy of solution data
        solution_data.emplace_back(snapshot.solution.begin(), snapshot.solution.end());
        // Update last pointer to point to our copy
        if (!snapshots.empty()) {
            snapshots.back().solution = std::span{solution_data.back()};
        }
    }

    std::vector<Snapshot> snapshots;
    std::vector<std::vector<double>> solution_data;
};

// Test: Batch snapshot collection produces per-lane snapshots
TEST(SnapshotBatchTest, PerLaneSnapshotCollection) {
    // Grid configuration
    constexpr size_t n = 101;
    constexpr double x_min = -1.0;
    constexpr double x_max = 1.0;
    auto grid = create_uniform_grid(x_min, x_max, n);

    // Time domain (short run for quick test)
    TimeDomain time_domain(0.0, 0.1, 0.01);

    // PDE parameters (same for all lanes for now - testing snapshot unpacking)
    // TODO: Once per-lane PDEs are properly supported, test with different params
    constexpr double volatility = 0.20;
    constexpr double rate = 0.05;
    constexpr double dividend = 0.02;
    constexpr double strike = 100.0;

    // Initial condition: American put payoff
    auto initial_condition = [&](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            const double S = strike * std::exp(x[i]);
            u[i] = std::max(strike - S, 0.0);
        }
    };

    // Obstacle condition: American put payoff
    auto obstacle = [&](double t, std::span<const double> x, std::span<double> psi) {
        (void)t;
        for (size_t i = 0; i < x.size(); ++i) {
            const double S = strike * std::exp(x[i]);
            psi[i] = std::max(strike - S, 0.0);
        }
    };

    // ==========================================
    // Batch mode with 2 lanes (same PDE for now)
    // ==========================================
    constexpr size_t batch_width = 2;

    // Create single PDE (will be reused for all lanes)
    BlackScholesPDE pde(volatility, rate, dividend);

    auto spacing = std::make_shared<GridSpacing<double>>(GridView<double>(grid));
    auto spatial_op = SpatialOperator(pde, spacing);

    // Boundary conditions
    auto left_bc = DirichletBC([](double, double) { return 0.0; });
    auto right_bc = DirichletBC([](double, double) { return 0.0; });

    // Root-finding config
    RootFindingConfig root_config{
        .max_iter = 100,
        .tolerance = 1e-6
    };

    // TR-BDF2 config
    TRBDF2Config trbdf2_config{
        .max_iter = 100,
        .tolerance = 1e-6
    };

    // Create batch workspace
    PDEWorkspace workspace_batch(n, std::span(grid), batch_width);

    // Create solver
    PDESolver solver(
        grid, time_domain,
        trbdf2_config, root_config,
        left_bc, right_bc, spatial_op,
        obstacle,
        &workspace_batch
    );

    // Create per-lane collectors
    TestSnapshotCollector collector0;
    TestSnapshotCollector collector1;

    // Register batch snapshot at step 5 (halfway through)
    std::vector<SnapshotCollector*> collectors = {&collector0, &collector1};
    solver.register_snapshot_batch(5, 42, std::move(collectors));

    // Initialize and solve
    solver.initialize(initial_condition);
    auto result = solver.solve();
    ASSERT_TRUE(result.has_value()) << "Batch solver did not converge";

    // ==========================================
    // Verify: Each collector received exactly one snapshot
    // ==========================================
    ASSERT_EQ(collector0.snapshots.size(), 1u) << "Lane 0 collector should receive 1 snapshot";
    ASSERT_EQ(collector1.snapshots.size(), 1u) << "Lane 1 collector should receive 1 snapshot";

    // ==========================================
    // Verify: Snapshot metadata is correct
    // ==========================================
    const auto& snap0 = collector0.snapshots[0];
    const auto& snap1 = collector1.snapshots[0];

    EXPECT_EQ(snap0.user_index, 42u);
    EXPECT_EQ(snap1.user_index, 42u);

    EXPECT_GT(snap0.time, 0.0) << "Snapshot time should be > 0";
    EXPECT_GT(snap1.time, 0.0) << "Snapshot time should be > 0";

    EXPECT_EQ(snap0.spatial_grid.size(), n);
    EXPECT_EQ(snap1.spatial_grid.size(), n);

    EXPECT_EQ(snap0.solution.size(), n);
    EXPECT_EQ(snap1.solution.size(), n);

    // ==========================================
    // Verify: Solutions are identical (same PDE params)
    // ==========================================
    // Both lanes use identical PDEs, so solutions should match

    double max_diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs(snap0.solution[i] - snap1.solution[i]);
        max_diff = std::max(max_diff, diff);
    }

    // Allow small numerical differences due to FP rounding in batch operations
    EXPECT_LT(max_diff, 1e-10) << "Solutions should match (identical PDEs)";

    // ==========================================
    // Verify: Solution is sensible (put option bounds)
    // ==========================================
    for (size_t i = 0; i < n; ++i) {
        const double S = strike * std::exp(grid[i]);
        const double intrinsic = std::max(strike - S, 0.0);

        // American put value should be >= intrinsic (except at boundaries where BC overrides)
        // At far OTM (i=0) and ITM (i=n-1), boundary conditions apply
        if (i > 0 && i < n - 1) {
            EXPECT_GE(snap0.solution[i], intrinsic - 1e-6)
                << "Lane 0 solution violates lower bound at i=" << i;
            EXPECT_GE(snap1.solution[i], intrinsic - 1e-6)
                << "Lane 1 solution violates lower bound at i=" << i;
        }

        // American put value should be <= strike (always)
        EXPECT_LE(snap0.solution[i], strike + 1e-10)
            << "Lane 0 solution violates upper bound at i=" << i;
        EXPECT_LE(snap1.solution[i], strike + 1e-10)
            << "Lane 1 solution violates upper bound at i=" << i;
    }
}

// Test: Multiple snapshots per lane
TEST(SnapshotBatchTest, MultipleSnapshotsPerLane) {
    // Grid configuration
    constexpr size_t n = 51;  // Smaller grid for faster test
    constexpr double x_min = -1.0;
    constexpr double x_max = 1.0;
    auto grid = create_uniform_grid(x_min, x_max, n);

    // Time domain
    TimeDomain time_domain(0.0, 0.1, 0.01);  // 10 steps

    // PDE parameters
    constexpr double volatility = 0.20;
    constexpr double rate = 0.05;
    constexpr double dividend = 0.02;
    constexpr double strike = 100.0;

    // Initial condition: American put payoff
    auto initial_condition = [&](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            const double S = strike * std::exp(x[i]);
            u[i] = std::max(strike - S, 0.0);
        }
    };

    // Obstacle condition
    auto obstacle = [&](double t, std::span<const double> x, std::span<double> psi) {
        (void)t;
        for (size_t i = 0; i < x.size(); ++i) {
            const double S = strike * std::exp(x[i]);
            psi[i] = std::max(strike - S, 0.0);
        }
    };

    // Batch mode with 2 lanes
    constexpr size_t batch_width = 2;

    // Create single PDE (will be reused for all lanes)
    BlackScholesPDE pde(volatility, rate, dividend);

    auto spacing = std::make_shared<GridSpacing<double>>(GridView<double>(grid));
    auto spatial_op = SpatialOperator(pde, spacing);

    auto left_bc = DirichletBC([](double, double) { return 0.0; });
    auto right_bc = DirichletBC([](double, double) { return 0.0; });

    RootFindingConfig root_config{.max_iter = 100, .tolerance = 1e-6};
    TRBDF2Config trbdf2_config{.max_iter = 100, .tolerance = 1e-6};

    PDEWorkspace workspace_batch(n, std::span(grid), batch_width);

    PDESolver solver(
        grid, time_domain,
        trbdf2_config, root_config,
        left_bc, right_bc, spatial_op,
        obstacle,
        &workspace_batch
    );

    // Create per-lane collectors
    TestSnapshotCollector collector0;
    TestSnapshotCollector collector1;

    // Register 3 snapshots at different times
    solver.register_snapshot_batch(2, 0, {&collector0, &collector1});  // t ≈ 0.03
    solver.register_snapshot_batch(5, 1, {&collector0, &collector1});  // t ≈ 0.06
    solver.register_snapshot_batch(8, 2, {&collector0, &collector1});  // t ≈ 0.09

    // Initialize and solve
    solver.initialize(initial_condition);
    auto result = solver.solve();
    ASSERT_TRUE(result.has_value()) << "Batch solver did not converge";

    // Verify: Each collector received 3 snapshots
    EXPECT_EQ(collector0.snapshots.size(), 3u);
    EXPECT_EQ(collector1.snapshots.size(), 3u);

    // Verify: Snapshots are ordered by time
    for (size_t i = 1; i < collector0.snapshots.size(); ++i) {
        EXPECT_GT(collector0.snapshots[i].time, collector0.snapshots[i-1].time)
            << "Lane 0 snapshots not ordered by time";
        EXPECT_GT(collector1.snapshots[i].time, collector1.snapshots[i-1].time)
            << "Lane 1 snapshots not ordered by time";
    }

    // Verify: User indices match
    EXPECT_EQ(collector0.snapshots[0].user_index, 0u);
    EXPECT_EQ(collector0.snapshots[1].user_index, 1u);
    EXPECT_EQ(collector0.snapshots[2].user_index, 2u);
}

}  // namespace
}  // namespace mango
