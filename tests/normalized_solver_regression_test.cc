// SPDX-License-Identifier: MIT
/**
 * @file normalized_solver_regression_test.cc
 * @brief Regression tests for normalized chain solver integration
 *
 * Tests cover critical issues identified during code review:
 * - Issue #1: Price table snapshot registration
 * - Issue #2: SetupCallback disables normalized path
 */

#include "src/option/american_option_batch.hpp"
#include <gtest/gtest.h>
#include <mutex>
#include <vector>

using namespace mango;

// Test that SetupCallback disables the normalized path
// Regression: Normalized solver was invoking callback for index 0 of temporary
// normalized option, not for original option indices
TEST(NormalizedSolverRegressionTest, SetupCallbackDisablesNormalizedPath) {
    // Create batch of options that would be eligible for normalized solver
    std::vector<PricingParams> batch(5);
    for (size_t i = 0; i < 5; ++i) {
        batch[i] = PricingParams(OptionSpec{.spot = 100.0 + i * 10.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20);
    }

    // Track callback invocations
    std::vector<size_t> callback_indices;
    std::mutex callback_mutex;

    // Setup callback that records which indices are invoked
    auto setup_callback = [&](size_t idx, [[maybe_unused]] AmericanOptionSolver& solver) {
        std::lock_guard<std::mutex> lock(callback_mutex);
        callback_indices.push_back(idx);
    };

    BatchAmericanOptionSolver solver;
    auto result = solver.solve_batch(batch, false, setup_callback);

    // Verify all solves succeeded
    ASSERT_EQ(result.results.size(), 5);
    EXPECT_EQ(result.failed_count, 0);

    // Critical assertion: Callback should be invoked for ALL original option indices
    // If normalized path was used, callback would only be invoked for index 0
    EXPECT_EQ(callback_indices.size(), 5) << "SetupCallback should be invoked for all 5 options";

    std::sort(callback_indices.begin(), callback_indices.end());
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(callback_indices[i], i) << "Callback should be invoked for index " << i;
    }
}

// Test that price table builder setup callback registers snapshots
// Regression: PriceTable4DBuilder wasn't registering snapshots, causing divide-by-zero
TEST(NormalizedSolverRegressionTest, PriceTableSnapshotRegistration) {
    // This test verifies that the fix (adding SetupCallback to register snapshots) is in place.
    // We test the mechanism by checking that a solver with set_snapshot_times() actually
    // registers snapshots, which is what the price table builder's callback does.

    // The actual price table end-to-end test is in price_table_workspace_test.cc
    // This test just verifies the snapshot registration mechanism works.

    std::vector<double> maturities = {0.25, 0.5, 1.0};
    bool callback_invoked = false;
    std::vector<double> registered_times;

    // Simulate what PriceTable4DBuilder's callback does
    auto snapshot_callback = [&]([[maybe_unused]] size_t idx, AmericanOptionSolver& solver) {
        callback_invoked = true;
        solver.set_snapshot_times(std::span{maturities});
    };

    // Create a simple batch
    std::vector<PricingParams> batch(1);
    batch[0] = PricingParams(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20);

    BatchAmericanOptionSolver solver;
    auto result = solver.solve_batch(batch, false, snapshot_callback);

    // Verify callback was invoked
    EXPECT_TRUE(callback_invoked) << "Setup callback should have been invoked";

    // Verify solve succeeded
    ASSERT_EQ(result.results.size(), 1);
    ASSERT_TRUE(result.results[0].has_value()) << "Solve should succeed with snapshots";

    // Verify snapshots were registered
    const auto& grid = result.results[0].value().grid();
    EXPECT_EQ(grid->num_snapshots(), 3) << "Should have registered 3 snapshot times";
}

// Test that set_snapshot_times() works correctly
TEST(NormalizedSolverRegressionTest, SetSnapshotTimesMethod) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20);

    // Create workspace
    auto grid_spec_result = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0);
    ASSERT_TRUE(grid_spec_result.has_value());
    auto grid_spec = grid_spec_result.value();

    size_t n_points = grid_spec.n_points();
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n_points), std::pmr::get_default_resource());
    auto workspace_result = PDEWorkspace::from_buffer(buffer, n_points);
    ASSERT_TRUE(workspace_result.has_value());
    auto workspace = workspace_result.value();

    // Create solver with custom grid to avoid auto-estimation
    TimeDomain time_domain = TimeDomain::from_n_steps(0.0, params.maturity, 100);
    auto solver = AmericanOptionSolver::create(params, workspace,
        PDEGridConfig{grid_spec, time_domain.n_steps(), {}}).value();

    // Set snapshot times using the new method
    std::vector<double> snapshot_times = {0.25, 0.5, 0.75, 1.0};
    solver.set_snapshot_times(std::span{snapshot_times});

    // Solve
    auto result = solver.solve();
    if (!result.has_value()) {
        FAIL() << "Solve should succeed with registered snapshots: " << result.error();
    }
    ASSERT_TRUE(result.has_value());

    // Verify snapshots were registered
    const auto& grid = result.value().grid();
    EXPECT_EQ(grid->num_snapshots(), 4) << "Should have 4 snapshots registered";

    auto registered_times = grid->snapshot_times();
    ASSERT_EQ(registered_times.size(), 4);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(registered_times[i], snapshot_times[i]);
    }
}

// Test that normalized solver is NOT used when callback is provided
TEST(NormalizedSolverRegressionTest, CallbackForcesRegularBatch) {
    // Create batch that would be eligible for normalized solver
    std::vector<PricingParams> batch(3);
    for (size_t i = 0; i < 3; ++i) {
        batch[i] = PricingParams(OptionSpec{.spot = 100.0 + i * 10.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20);
    }

    // Enable normalized solver
    BatchAmericanOptionSolver solver;
    solver.set_use_normalized(true);

    // Without callback, normalized path should be used (if eligible)
    auto result_without_callback = solver.solve_batch(batch, false);
    EXPECT_EQ(result_without_callback.failed_count, 0);

    // With callback, regular batch path should be used
    size_t callback_count = 0;
    auto setup_callback = [&callback_count]([[maybe_unused]] size_t idx,
                                            [[maybe_unused]] AmericanOptionSolver& solver) {
        ++callback_count;
    };

    auto result_with_callback = solver.solve_batch(batch, false, setup_callback);
    EXPECT_EQ(result_with_callback.failed_count, 0);

    // Critical assertion: Callback should be invoked for each option
    EXPECT_EQ(callback_count, 3) << "SetupCallback should force regular batch path";
}

// Test that normalized solver WORKS with snapshot_times (via dedicated API)
TEST(NormalizedSolverRegressionTest, NormalizedPathWorksWithSnapshots) {
    // Create batch eligible for normalized solver
    std::vector<PricingParams> batch(5);
    for (size_t i = 0; i < 5; ++i) {
        batch[i] = PricingParams(OptionSpec{.spot = 100.0 + i * 10.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20);
    }

    // Configure snapshots using dedicated API (preserves normalized path)
    std::vector<double> snapshot_times = {0.25, 0.5, 0.75, 1.0};
    BatchAmericanOptionSolver solver;
    solver.set_use_normalized(true);
    solver.set_snapshot_times(std::span{snapshot_times});

    // Solve using shared grid (enables normalized path)
    auto result = solver.solve_batch(batch, true);

    // Verify all solves succeeded
    ASSERT_EQ(result.results.size(), 5);
    EXPECT_EQ(result.failed_count, 0);

    // Verify snapshots were registered for all results
    for (size_t i = 0; i < 5; ++i) {
        ASSERT_TRUE(result.results[i].has_value()) << "Option " << i << " should succeed";
        const auto& opt_result = result.results[i].value();
        auto grid = opt_result.grid();
        EXPECT_EQ(grid->num_snapshots(), 4) << "Option " << i << " should have 4 snapshots";
    }
}
