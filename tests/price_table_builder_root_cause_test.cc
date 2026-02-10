// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "tests/price_table_builder_test_access.hpp"
#include "mango/pde/core/time_domain.hpp"
#include <cmath>

namespace mango {
namespace {

using Access = testing::PriceTableBuilderAccess<4>;

// Test to identify the ROOT CAUSE: grid width exceeds normalized chain solver limits
TEST(PriceTableBuilderRootCauseTest, GridWidthExceedsLimit) {
    // The normalized chain solver has MAX_WIDTH = 5.8
    // The custom grid has width = 6.0 (from -3 to 3)
    // This causes the normalized chain solver to be ineligible

    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .pde_grid = PDEGridConfig{GridSpec<double>::uniform(-3.0, 3.0, 21).value(), 100},  // width = 6.0
        .dividends = {.dividend_yield = 0.02}
    };

    const auto& explicit_grid = std::get<PDEGridConfig>(config.pde_grid);
    std::cout << "=== ROOT CAUSE ANALYSIS ===" << std::endl;
    std::cout << "Custom grid width: " << (explicit_grid.grid_spec.x_max() - explicit_grid.grid_spec.x_min()) << std::endl;
    std::cout << "Normalized chain MAX_WIDTH: 5.8" << std::endl;
    std::cout << "Result: Custom grid width EXCEEDS limit" << std::endl;

    // Test 1: Use a narrower grid that fits within MAX_WIDTH
    std::cout << "\n=== Test 1: Narrower grid (width=5.0 < 5.8) ===" << std::endl;
    PriceTableConfig config_narrow{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .pde_grid = PDEGridConfig{GridSpec<double>::uniform(-2.5, 2.5, 21).value(), 100},  // width = 5.0
        .dividends = {.dividend_yield = 0.02}
    };

    PriceTableBuilder builder_narrow(config_narrow);
    PriceTableAxes axes;
    axes.grids[0] = {std::log(0.9), std::log(1.0)};
    axes.grids[1] = {0.1, 0.5, 1.0};
    axes.grids[2] = {0.20};
    axes.grids[3] = {0.05};

    auto batch = Access::make_batch(builder_narrow, axes);
    const auto& narrow_explicit = std::get<PDEGridConfig>(config_narrow.pde_grid);
    GridSpec<double> narrow_grid = narrow_explicit.grid_spec;
    auto time_domain = TimeDomain::from_n_steps(0.0, axes.grids[1].back(), narrow_explicit.n_time);
    std::optional<PDEGridSpec> custom_grid_narrow =
        PDEGridConfig{narrow_grid, time_domain.n_steps(), {}};

    BatchAmericanOptionSolver solver_narrow;
    GridAccuracyParams accuracy;
    accuracy.min_spatial_points = 21;
    accuracy.max_spatial_points = 1200;
    accuracy.max_time_steps = narrow_explicit.n_time;
    solver_narrow.set_grid_accuracy(accuracy);
    solver_narrow.set_snapshot_times(axes.grids[1]);

    auto result_narrow = solver_narrow.solve_batch(batch, true, nullptr, custom_grid_narrow);
    std::cout << "Grid width: " << (narrow_grid.x_max() - narrow_grid.x_min()) << std::endl;
    std::cout << "Failed count: " << result_narrow.failed_count << std::endl;
    std::cout << "Result: " << (result_narrow.failed_count == 0 ? "SUCCESS" : "FAILED") << std::endl;

    // Test 2: Use the original wide grid
    std::cout << "\n=== Test 2: Wide grid (width=6.0 > 5.8) ===" << std::endl;
    PriceTableBuilder builder_wide(config);
    auto batch2 = Access::make_batch(builder_wide, axes);
    GridSpec<double> wide_grid = explicit_grid.grid_spec;
    auto time_domain2 = TimeDomain::from_n_steps(0.0, axes.grids[1].back(), explicit_grid.n_time);
    std::optional<PDEGridSpec> custom_grid_wide =
        PDEGridConfig{wide_grid, time_domain2.n_steps(), {}};

    BatchAmericanOptionSolver solver_wide;
    solver_wide.set_grid_accuracy(accuracy);
    solver_wide.set_snapshot_times(axes.grids[1]);

    auto result_wide = solver_wide.solve_batch(batch2, true, nullptr, custom_grid_wide);
    std::cout << "Grid width: " << (wide_grid.x_max() - wide_grid.x_min()) << std::endl;
    std::cout << "Failed count: " << result_wide.failed_count << std::endl;
    std::cout << "Result: " << (result_wide.failed_count == 0 ? "SUCCESS" : "FAILED") << std::endl;

    if (result_wide.failed_count > 0) {
        std::cout << "Error code: " << static_cast<int>(result_wide.results[0].error().code) << std::endl;
    }

    // Test 3: Try with use_normalized=false to force regular batch path
    std::cout << "\n=== Test 3: Wide grid with normalized=false (regular batch) ===" << std::endl;
    BatchAmericanOptionSolver solver_regular;
    solver_regular.set_grid_accuracy(accuracy);
    solver_regular.set_snapshot_times(axes.grids[1]);
    solver_regular.set_use_normalized(false);  // Disable normalized path

    auto result_regular = solver_regular.solve_batch(batch2, true, nullptr, custom_grid_wide);
    std::cout << "Failed count: " << result_regular.failed_count << std::endl;
    std::cout << "Result: " << (result_regular.failed_count == 0 ? "SUCCESS" : "FAILED") << std::endl;

    std::cout << "\n=== CONCLUSION ===" << std::endl;
    std::cout << "The issue is that custom_grid with width > 5.8 makes" << std::endl;
    std::cout << "the batch INELIGIBLE for normalized chain solver." << std::endl;
    std::cout << "When it falls back to regular batch, snapshots may not work correctly." << std::endl;
}

// Test: Check dx constraint
TEST(PriceTableBuilderRootCauseTest, GridSpacingCheck) {
    std::cout << "=== Grid spacing (dx) analysis ===" << std::endl;

    // Custom grid: 21 points over [-3, 3]
    double x_min = -3.0;
    double x_max = 3.0;
    size_t n_points = 21;
    double dx = (x_max - x_min) / (n_points - 1);

    std::cout << "Custom grid: [" << x_min << ", " << x_max << "], n=" << n_points << std::endl;
    std::cout << "dx = " << dx << std::endl;
    std::cout << "MAX_DX = 0.05" << std::endl;
    std::cout << "Passes dx check? " << (dx <= 0.05 ? "YES" : "NO") << std::endl;

    // This is another constraint that might fail
    if (dx > 0.05) {
        std::cout << "\nThis grid ALSO fails the dx constraint!" << std::endl;
        std::cout << "The grid is too coarse (dx=" << dx << " > 0.05)" << std::endl;
    }
}

} // namespace
} // namespace mango
