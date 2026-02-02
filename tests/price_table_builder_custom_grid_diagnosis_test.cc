// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/table/price_table_builder.hpp"
#include "tests/price_table_builder_test_access.hpp"
#include "src/pde/core/time_domain.hpp"

namespace mango {
namespace {

using Access = testing::PriceTableBuilderAccess<4>;

// Reproduce the exact failing test scenario
TEST(PriceTableBuilderCustomGridDiagnosisTest, ReproduceFailure) {
    // Same config as failing test
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .pde_grid = PDEGridConfig{GridSpec<double>::uniform(-3.0, 3.0, 21).value(), 100},
        .dividends = {.dividend_yield = 0.02}
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.9, 1.0};
    axes.grids[1] = {0.1, 0.5, 1.0};  // 3 maturity points
    axes.grids[2] = {0.20};           // 1 vol
    axes.grids[3] = {0.05};           // 1 rate

    std::cout << "=== Diagnosing failure ===" << std::endl;

    // Generate batch
    auto batch_params = Access::make_batch(builder, axes);
    std::cout << "Batch size: " << batch_params.size() << std::endl;
    ASSERT_EQ(batch_params.size(), 1);

    // Print batch parameters
    const auto& params = batch_params[0];
    std::cout << "Batch params:" << std::endl;
    std::cout << "  spot: " << params.spot << std::endl;
    std::cout << "  strike: " << params.strike << std::endl;
    std::cout << "  maturity: " << params.maturity << std::endl;
    std::cout << "  volatility: " << params.volatility << std::endl;
    std::cout << "  rate: ";
    if (std::holds_alternative<double>(params.rate)) {
        std::cout << std::get<double>(params.rate);
    } else {
        std::cout << "<YieldCurve>";
    }
    std::cout << std::endl;
    std::cout << "  dividend_yield: " << params.dividend_yield << std::endl;

    // Check grid estimator
    const auto& diag_grid = std::get<PDEGridConfig>(config.pde_grid);
    std::cout << "\nGrid estimator from config:" << std::endl;
    std::cout << "  x_min: " << diag_grid.grid_spec.x_min() << std::endl;
    std::cout << "  x_max: " << diag_grid.grid_spec.x_max() << std::endl;
    std::cout << "  n_points: " << diag_grid.grid_spec.n_points() << std::endl;

    // Check what grid would be estimated for this option
    GridAccuracyParams accuracy;
    accuracy.min_spatial_points = std::min(diag_grid.grid_spec.n_points(), size_t(100));
    accuracy.max_spatial_points = std::max(diag_grid.grid_spec.n_points(), size_t(1200));
    accuracy.max_time_steps = diag_grid.n_time;

    auto [auto_grid, auto_time] = estimate_grid_for_option(params, accuracy);
    std::cout << "\nAuto-estimated grid for these params:" << std::endl;
    std::cout << "  x_min: " << auto_grid.x_min() << std::endl;
    std::cout << "  x_max: " << auto_grid.x_max() << std::endl;
    std::cout << "  n_points: " << auto_grid.n_points() << std::endl;
    std::cout << "  time: [" << auto_time.t_start() << ", " << auto_time.t_end() << "]" << std::endl;

    // Now check what custom_grid would be
    GridSpec<double> user_grid = diag_grid.grid_spec;
    auto time_domain = TimeDomain::from_n_steps(0.0, axes.grids[1].back(), diag_grid.n_time);
    std::cout << "\nCustom grid (as plan specifies):" << std::endl;
    std::cout << "  x_min: " << user_grid.x_min() << std::endl;
    std::cout << "  x_max: " << user_grid.x_max() << std::endl;
    std::cout << "  n_points: " << user_grid.n_points() << std::endl;
    std::cout << "  time: [" << time_domain.t_start() << ", " << time_domain.t_end() << "]" << std::endl;

    // Check if log(spot/strike) = log(1.0) = 0.0 is in grid
    double log_moneyness = std::log(params.spot / params.strike);
    std::cout << "\nLog-moneyness check:" << std::endl;
    std::cout << "  log(spot/strike) = " << log_moneyness << std::endl;
    std::cout << "  Is in auto grid? " << (log_moneyness >= auto_grid.x_min() && log_moneyness <= auto_grid.x_max() ? "YES" : "NO") << std::endl;
    std::cout << "  Is in custom grid? " << (log_moneyness >= user_grid.x_min() && log_moneyness <= user_grid.x_max() ? "YES" : "NO") << std::endl;

    // Try solving with auto (no custom_grid)
    std::cout << "\n=== Test 1: WITHOUT custom_grid (auto estimation) ===" << std::endl;
    BatchAmericanOptionSolver solver1;
    solver1.set_grid_accuracy(accuracy);
    solver1.set_snapshot_times(axes.grids[1]);
    auto result1 = solver1.solve_batch(batch_params, true, nullptr, std::nullopt);
    std::cout << "Failed count: " << result1.failed_count << std::endl;
    if (result1.results[0].has_value()) {
        std::cout << "SUCCESS" << std::endl;
        auto grid = result1.results[0]->grid();
        std::cout << "  Grid: [" << grid->x()[0] << ", " << grid->x()[grid->n_space()-1] << "]" << std::endl;
        std::cout << "  n_space: " << grid->n_space() << std::endl;
        std::cout << "  snapshots: " << grid->num_snapshots() << std::endl;
    } else {
        std::cout << "FAILED: code=" << static_cast<int>(result1.results[0].error().code) << std::endl;
    }

    // Try solving with custom_grid
    std::cout << "\n=== Test 2: WITH custom_grid ===" << std::endl;
    std::optional<PDEGridSpec> custom_grid =
        PDEGridConfig{user_grid, time_domain.n_steps(), {}};
    BatchAmericanOptionSolver solver2;
    solver2.set_grid_accuracy(accuracy);
    solver2.set_snapshot_times(axes.grids[1]);
    auto result2 = solver2.solve_batch(batch_params, true, nullptr, custom_grid);
    std::cout << "Failed count: " << result2.failed_count << std::endl;
    if (result2.results[0].has_value()) {
        std::cout << "SUCCESS" << std::endl;
        auto grid = result2.results[0]->grid();
        std::cout << "  Grid: [" << grid->x()[0] << ", " << grid->x()[grid->n_space()-1] << "]" << std::endl;
        std::cout << "  n_space: " << grid->n_space() << std::endl;
        std::cout << "  snapshots: " << grid->num_snapshots() << std::endl;
    } else {
        std::cout << "FAILED: code=" << static_cast<int>(result2.results[0].error().code) << std::endl;
    }

    // Check snapshot times registration
    std::cout << "\n=== Snapshot times analysis ===" << std::endl;
    std::cout << "Requested snapshot times: ";
    for (double t : axes.grids[1]) {
        std::cout << t << " ";
    }
    std::cout << std::endl;
}

} // namespace
} // namespace mango
