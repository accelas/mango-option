// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/table/price_table_builder.hpp"
#include "src/pde/core/time_domain.hpp"

namespace mango {
namespace {

// Test the normalized chain solver with custom_grid
// This tests the ACTUAL code path used by price table builder
TEST(PriceTableBuilderCustomGridAdvancedTest, NormalizedChainWithCustomGrid) {
    // Setup to trigger normalized chain solver
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .grid_estimator = GridSpec<double>::uniform(-3.0, 3.0, 101).value(),
        .n_time = 100,
        .dividend_yield = 0.02
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.9, 1.0, 1.1, 1.2};
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};
    axes.grids[2] = {0.20, 0.25};
    axes.grids[3] = {0.05, 0.06};

    // Generate batch with normalized parameters
    auto batch = builder.make_batch_for_testing(axes);
    ASSERT_EQ(batch.size(), 4);  // 2 vols × 2 rates

    std::cout << "=== Testing normalized chain solver ===" << std::endl;

    // Verify eligibility for normalized chain
    // All options have same maturity (1.0), no discrete dividends, spot==strike
    bool all_same_maturity = true;
    for (const auto& params : batch) {
        if (std::abs(params.maturity - batch[0].maturity) > 1e-10) {
            all_same_maturity = false;
        }
    }
    EXPECT_TRUE(all_same_maturity);

    // Create solver and check normalized eligibility
    BatchAmericanOptionSolver solver;
    GridAccuracyParams accuracy;
    accuracy.min_spatial_points = std::min(config.grid_estimator.n_points(), size_t(100));
    accuracy.max_spatial_points = std::max(config.grid_estimator.n_points(), size_t(1200));
    accuracy.max_time_steps = config.n_time;
    solver.set_grid_accuracy(accuracy);
    solver.set_snapshot_times(axes.grids[1]);

    // Test WITHOUT custom_grid (uses normalized chain if eligible)
    std::cout << "\nTest 1: use_shared_grid=true, no custom_grid (normalized path)" << std::endl;
    auto result1 = solver.solve_batch(batch, true, nullptr, std::nullopt);
    std::cout << "  Failed count: " << result1.failed_count << std::endl;
    EXPECT_EQ(result1.failed_count, 0);

    // Test WITH custom_grid (what does this do?)
    std::cout << "\nTest 2: use_shared_grid=true, WITH custom_grid" << std::endl;
    GridSpec<double> user_grid = config.grid_estimator;
    auto time_domain = TimeDomain::from_n_steps(0.0, axes.grids[1].back(), config.n_time);
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid =
        std::make_pair(user_grid, time_domain);

    auto result2 = solver.solve_batch(batch, true, nullptr, custom_grid);
    std::cout << "  Failed count: " << result2.failed_count << std::endl;
    EXPECT_EQ(result2.failed_count, 0);

    // Test 3: Force regular batch path (use_shared_grid=false)
    std::cout << "\nTest 3: use_shared_grid=false, WITH custom_grid (regular batch)" << std::endl;
    auto result3 = solver.solve_batch(batch, false, nullptr, custom_grid);
    std::cout << "  Failed count: " << result3.failed_count << std::endl;
    EXPECT_EQ(result3.failed_count, 0);

    std::cout << "\n*** All paths work correctly with custom_grid ***" << std::endl;
}

// Test edge case: moneyness == 1.0 (spot == strike in log-space: log(1.0) = 0.0)
TEST(PriceTableBuilderCustomGridAdvancedTest, EdgeCaseLogMoneyness) {
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .grid_estimator = GridSpec<double>::uniform(-3.0, 3.0, 101).value(),
        .n_time = 100,
        .dividend_yield = 0.02
    };

    PriceTableBuilder<4> builder(config);

    // Include moneyness = 1.0 (ATM point, log(1.0) = 0.0)
    PriceTableAxes<4> axes;
    axes.grids[0] = {0.95, 1.0, 1.05, 1.10};  // Includes exact ATM
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};
    axes.grids[2] = {0.20};
    axes.grids[3] = {0.05};

    // Generate batch (normalized: spot=strike=100)
    auto batch = builder.make_batch_for_testing(axes);
    ASSERT_EQ(batch.size(), 1);

    // Verify normalized parameters
    EXPECT_DOUBLE_EQ(batch[0].spot, 100.0);
    EXPECT_DOUBLE_EQ(batch[0].strike, 100.0);

    std::cout << "=== Testing edge case: moneyness grid includes 1.0 ===" << std::endl;
    std::cout << "Normalized batch: spot=" << batch[0].spot
              << ", strike=" << batch[0].strike << std::endl;

    // Create custom grid
    GridSpec<double> user_grid = config.grid_estimator;
    auto time_domain = TimeDomain::from_n_steps(0.0, axes.grids[1].back(), config.n_time);
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid =
        std::make_pair(user_grid, time_domain);

    // Test with custom_grid
    BatchAmericanOptionSolver solver;
    GridAccuracyParams accuracy;
    accuracy.min_spatial_points = 101;
    accuracy.max_spatial_points = 101;
    accuracy.max_time_steps = config.n_time;
    solver.set_grid_accuracy(accuracy);
    solver.set_snapshot_times(axes.grids[1]);

    auto result = solver.solve_batch(batch, true, nullptr, custom_grid);

    std::cout << "Failed count: " << result.failed_count << std::endl;
    EXPECT_EQ(result.failed_count, 0);

    if (result.results[0].has_value()) {
        const auto& opt_result = result.results[0].value();
        auto grid = opt_result.grid();

        std::cout << "Grid spatial points: " << grid->n_space() << std::endl;
        std::cout << "Grid x_min: " << grid->x()[0] << std::endl;
        std::cout << "Grid x_max: " << grid->x()[grid->n_space() - 1] << std::endl;

        // Check that log(1.0) = 0.0 is within the grid
        double log_m = std::log(1.0);  // = 0.0
        EXPECT_GE(log_m, grid->x()[0]);
        EXPECT_LE(log_m, grid->x()[grid->n_space() - 1]);

        std::cout << "log(1.0) = " << log_m << " is within grid bounds" << std::endl;
    }

    std::cout << "\n*** Edge case works correctly ***" << std::endl;
}

// Test the exact scenario from the plan: build() with custom_grid modification
TEST(PriceTableBuilderCustomGridAdvancedTest, SimulatePlanModification) {
    // This simulates what would happen if we modified solve_batch() in
    // price_table_builder.cpp to use custom_grid as the plan specified

    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .grid_estimator = GridSpec<double>::uniform(-3.0, 3.0, 101).value(),
        .n_time = 100,
        .dividend_yield = 0.02
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1};
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};
    axes.grids[2] = {0.15, 0.20, 0.25, 0.30};
    axes.grids[3] = {0.02, 0.04, 0.06, 0.08};

    std::cout << "=== Simulating plan modification ===" << std::endl;
    std::cout << "Grid size: " << axes.grids[0].size() << "×"
              << axes.grids[1].size() << "×"
              << axes.grids[2].size() << "×"
              << axes.grids[3].size() << " = "
              << (axes.grids[0].size() * axes.grids[1].size() *
                  axes.grids[2].size() * axes.grids[3].size())
              << " points" << std::endl;
    std::cout << "Expected PDE solves: " << (axes.grids[2].size() * axes.grids[3].size())
              << " (Nσ × Nr)" << std::endl;

    // Generate batch as plan does
    auto batch = builder.make_batch_for_testing(axes);
    std::cout << "Batch size: " << batch.size() << std::endl;

    // Create custom_grid as plan specifies
    GridSpec<double> user_grid = config.grid_estimator;
    auto time_domain = TimeDomain::from_n_steps(0.0, axes.grids[1].back(), config.n_time);
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid =
        std::make_pair(user_grid, time_domain);

    std::cout << "\nCustom grid specification:" << std::endl;
    std::cout << "  Spatial: [" << user_grid.x_min() << ", " << user_grid.x_max()
              << "], n=" << user_grid.n_points() << std::endl;
    std::cout << "  Time: [" << time_domain.t_start() << ", " << time_domain.t_end()
              << "], n_steps=" << time_domain.n_steps() << std::endl;

    // Simulate the solve_batch call with custom_grid
    BatchAmericanOptionSolver solver;
    GridAccuracyParams accuracy;
    accuracy.min_spatial_points = std::min(config.grid_estimator.n_points(), size_t(100));
    accuracy.max_spatial_points = std::max(config.grid_estimator.n_points(), size_t(1200));
    accuracy.max_time_steps = config.n_time;
    if (config.grid_estimator.type() == GridSpec<double>::Type::SinhSpaced) {
        accuracy.alpha = config.grid_estimator.concentration();
    }
    solver.set_grid_accuracy(accuracy);
    solver.set_snapshot_times(axes.grids[1]);

    auto result = solver.solve_batch(batch, true, nullptr, custom_grid);

    std::cout << "\nResults:" << std::endl;
    std::cout << "  Total: " << result.results.size() << std::endl;
    std::cout << "  Failed: " << result.failed_count << std::endl;
    std::cout << "  Success: " << (result.results.size() - result.failed_count) << std::endl;

    for (size_t i = 0; i < result.results.size(); ++i) {
        if (!result.results[i].has_value()) {
            const auto& error = result.results[i].error();
            std::cout << "  FAILURE at index " << i << ": code="
                      << static_cast<int>(error.code) << std::endl;
        }
    }

    EXPECT_EQ(result.failed_count, 0);

    std::cout << "\n*** Plan modification works correctly - no failures ***" << std::endl;
}

} // namespace
} // namespace mango
