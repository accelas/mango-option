// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/price_table_builder.hpp"
#include "tests/price_table_builder_test_access.hpp"
#include "mango/pde/core/time_domain.hpp"
#include <cmath>

namespace mango {
namespace {

using Access = testing::PriceTableBuilderAccess<4>;

// Test the normalized chain solver with custom_grid
// This tests the ACTUAL code path used by price table builder
TEST(PriceTableBuilderCustomGridAdvancedTest, NormalizedChainWithCustomGrid) {
    // Setup to trigger normalized chain solver
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .pde_grid = PDEGridConfig{GridSpec<double>::uniform(-3.0, 3.0, 101).value(), 100},
        .dividends = {.dividend_yield = 0.02}
    };

    PriceTableBuilder builder(config);

    PriceTableAxes axes;
    axes.grids[0] = {std::log(0.9), std::log(1.0), std::log(1.1), std::log(1.2)};
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};
    axes.grids[2] = {0.20, 0.25};
    axes.grids[3] = {0.05, 0.06};

    // Generate batch with normalized parameters
    auto batch = Access::make_batch(builder, axes);
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
    const auto& adv_grid1 = std::get<PDEGridConfig>(config.pde_grid);
    GridAccuracyParams accuracy;
    accuracy.min_spatial_points = std::min(adv_grid1.grid_spec.n_points(), size_t(100));
    accuracy.max_spatial_points = std::max(adv_grid1.grid_spec.n_points(), size_t(1200));
    accuracy.max_time_steps = adv_grid1.n_time;
    solver.set_grid_accuracy(accuracy);
    solver.set_snapshot_times(axes.grids[1]);

    // Test WITHOUT custom_grid (uses normalized chain if eligible)
    std::cout << "\nTest 1: use_shared_grid=true, no custom_grid (normalized path)" << std::endl;
    auto result1 = solver.solve_batch(batch, true, nullptr, std::nullopt);
    std::cout << "  Failed count: " << result1.failed_count << std::endl;
    EXPECT_EQ(result1.failed_count, 0);

    // Test WITH custom_grid (what does this do?)
    std::cout << "\nTest 2: use_shared_grid=true, WITH custom_grid" << std::endl;
    GridSpec<double> user_grid = adv_grid1.grid_spec;
    auto time_domain = TimeDomain::from_n_steps(0.0, axes.grids[1].back(), adv_grid1.n_time);
    std::optional<PDEGridSpec> custom_grid =
        PDEGridConfig{user_grid, time_domain.n_steps(), {}};

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
        .pde_grid = PDEGridConfig{GridSpec<double>::uniform(-3.0, 3.0, 101).value(), 100},
        .dividends = {.dividend_yield = 0.02}
    };

    PriceTableBuilder builder(config);

    // Include moneyness = 1.0 (ATM point, log(1.0) = 0.0)
    PriceTableAxes axes;
    axes.grids[0] = {std::log(0.95), std::log(1.0), std::log(1.05), std::log(1.10)};  // Includes exact ATM (log(1.0)=0.0)
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};
    axes.grids[2] = {0.20};
    axes.grids[3] = {0.05};

    // Generate batch (normalized: spot=strike=100)
    auto batch = Access::make_batch(builder, axes);
    ASSERT_EQ(batch.size(), 1);

    // Verify normalized parameters
    EXPECT_DOUBLE_EQ(batch[0].spot, 100.0);
    EXPECT_DOUBLE_EQ(batch[0].strike, 100.0);

    std::cout << "=== Testing edge case: moneyness grid includes 1.0 ===" << std::endl;
    std::cout << "Normalized batch: spot=" << batch[0].spot
              << ", strike=" << batch[0].strike << std::endl;

    // Create custom grid
    const auto& edge_grid = std::get<PDEGridConfig>(config.pde_grid);
    GridSpec<double> user_grid = edge_grid.grid_spec;
    auto time_domain = TimeDomain::from_n_steps(0.0, axes.grids[1].back(), edge_grid.n_time);
    std::optional<PDEGridSpec> custom_grid =
        PDEGridConfig{user_grid, time_domain.n_steps(), {}};

    // Test with custom_grid
    BatchAmericanOptionSolver solver;
    GridAccuracyParams accuracy;
    accuracy.min_spatial_points = 101;
    accuracy.max_spatial_points = 101;
    accuracy.max_time_steps = edge_grid.n_time;
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
        .pde_grid = PDEGridConfig{GridSpec<double>::uniform(-3.0, 3.0, 101).value(), 100},
        .dividends = {.dividend_yield = 0.02}
    };

    PriceTableBuilder builder(config);

    PriceTableAxes axes;
    axes.grids[0] = {std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1)};
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
    auto batch = Access::make_batch(builder, axes);
    std::cout << "Batch size: " << batch.size() << std::endl;

    // Create custom_grid as plan specifies
    const auto& sim_grid = std::get<PDEGridConfig>(config.pde_grid);
    GridSpec<double> user_grid = sim_grid.grid_spec;
    auto time_domain = TimeDomain::from_n_steps(0.0, axes.grids[1].back(), sim_grid.n_time);
    std::optional<PDEGridSpec> custom_grid =
        PDEGridConfig{user_grid, time_domain.n_steps(), {}};

    std::cout << "\nCustom grid specification:" << std::endl;
    std::cout << "  Spatial: [" << user_grid.x_min() << ", " << user_grid.x_max()
              << "], n=" << user_grid.n_points() << std::endl;
    std::cout << "  Time: [" << time_domain.t_start() << ", " << time_domain.t_end()
              << "], n_steps=" << time_domain.n_steps() << std::endl;

    // Simulate the solve_batch call with custom_grid
    BatchAmericanOptionSolver solver;
    GridAccuracyParams accuracy;
    accuracy.min_spatial_points = std::min(sim_grid.grid_spec.n_points(), size_t(100));
    accuracy.max_spatial_points = std::max(sim_grid.grid_spec.n_points(), size_t(1200));
    accuracy.max_time_steps = sim_grid.n_time;
    if (sim_grid.grid_spec.type() == GridSpec<double>::Type::SinhSpaced) {
        accuracy.alpha = sim_grid.grid_spec.concentration();
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
