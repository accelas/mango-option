#include <gtest/gtest.h>
#include "src/option/price_table_builder.hpp"
#include "src/pde/core/time_domain.hpp"

namespace mango {
namespace {

// Test to investigate the claimed failure when using custom_grid
// The claim is: "custom_grid causes all PDE solves to fail when options have spot==strike (normalized case)"
TEST(PriceTableBuilderCustomGridTest, CustomGridWithNormalizedCase) {
    // Setup: minimal 4D grid with small spatial grid for fast test
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .grid_estimator = GridSpec<double>::uniform(-3.0, 3.0, 101).value(),
        .n_time = 100,
        .dividend_yield = 0.02
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.9, 1.0, 1.1, 1.2};     // moneyness: 4 points
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};   // maturity: 4 points
    axes.grids[2] = {0.20, 0.25};             // volatility: 2 points
    axes.grids[3] = {0.05, 0.06};             // rate: 2 points

    // Generate batch (should be 2×2=4 entries with spot==strike==K_ref)
    auto batch = builder.make_batch_for_testing(axes);
    ASSERT_EQ(batch.size(), 4);  // 2 vols × 2 rates

    // Verify all batch entries are normalized (spot == strike == K_ref)
    for (const auto& params : batch) {
        EXPECT_DOUBLE_EQ(params.spot, 100.0);
        EXPECT_DOUBLE_EQ(params.strike, 100.0);
    }

    // Now test solve_batch WITHOUT custom_grid (baseline - should work)
    auto batch_result_baseline = builder.solve_batch_for_testing(batch, axes);

    std::cout << "=== Baseline (no custom_grid) ===" << std::endl;
    std::cout << "Total results: " << batch_result_baseline.results.size() << std::endl;
    std::cout << "Failed count: " << batch_result_baseline.failed_count << std::endl;

    for (size_t i = 0; i < batch_result_baseline.results.size(); ++i) {
        if (batch_result_baseline.results[i].has_value()) {
            std::cout << "  Result " << i << ": SUCCESS" << std::endl;
        } else {
            const auto& error = batch_result_baseline.results[i].error();
            std::cout << "  Result " << i << ": FAILED - code="
                      << static_cast<int>(error.code) << std::endl;
        }
    }

    // Baseline should succeed
    EXPECT_EQ(batch_result_baseline.failed_count, 0);

    // Now test the CLAIMED FAILURE: using custom_grid with normalized case
    // Build custom_grid as the plan specified
    std::cout << "\n=== Testing custom_grid (claimed to fail) ===" << std::endl;

    // Create custom grid/time domain as plan specified
    GridSpec<double> user_grid = config.grid_estimator;
    auto time_domain = TimeDomain::from_n_steps(0.0, axes.grids[1].back(), config.n_time);
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid =
        std::make_pair(user_grid, time_domain);

    // Create a BatchAmericanOptionSolver to test with custom_grid
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

    // Call solve_batch with custom_grid parameter
    auto batch_result_custom = solver.solve_batch(batch, true, nullptr, custom_grid);

    std::cout << "Total results: " << batch_result_custom.results.size() << std::endl;
    std::cout << "Failed count: " << batch_result_custom.failed_count << std::endl;

    for (size_t i = 0; i < batch_result_custom.results.size(); ++i) {
        if (batch_result_custom.results[i].has_value()) {
            std::cout << "  Result " << i << ": SUCCESS" << std::endl;
        } else {
            const auto& error = batch_result_custom.results[i].error();
            std::cout << "  Result " << i << ": FAILED - code="
                      << static_cast<int>(error.code)
                      << " (iterations=" << error.iterations << ")" << std::endl;
        }
    }

    // Check if the claim is true: does custom_grid cause failures?
    if (batch_result_custom.failed_count > 0) {
        std::cout << "\n*** CLAIM VERIFIED: custom_grid causes "
                  << batch_result_custom.failed_count << " failures ***" << std::endl;
    } else {
        std::cout << "\n*** CLAIM REFUTED: custom_grid works fine ***" << std::endl;
    }

    // For investigation: print details about the grid
    std::cout << "\nGrid details:" << std::endl;
    std::cout << "  x_min: " << user_grid.x_min() << std::endl;
    std::cout << "  x_max: " << user_grid.x_max() << std::endl;
    std::cout << "  n_points: " << user_grid.n_points() << std::endl;
    std::cout << "  time domain: [" << time_domain.t_start() << ", " << time_domain.t_end() << "]" << std::endl;
    std::cout << "  n_steps: " << time_domain.n_steps() << std::endl;
}

// Additional test: verify the exact spot==strike condition
TEST(PriceTableBuilderCustomGridTest, VerifyNormalizedBatchConditions) {
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .grid_estimator = GridSpec<double>::uniform(-3.0, 3.0, 101).value(),
        .n_time = 100
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.9, 1.0, 1.1, 1.2};
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};
    axes.grids[2] = {0.20};
    axes.grids[3] = {0.05};

    auto batch = builder.make_batch_for_testing(axes);

    ASSERT_EQ(batch.size(), 1);

    // Verify the normalized condition: spot == strike
    const auto& params = batch[0];
    EXPECT_DOUBLE_EQ(params.spot, params.strike);
    EXPECT_DOUBLE_EQ(params.spot, config.K_ref);

    // This is the "normalized case" mentioned in the claim
    double moneyness = params.spot / params.strike;
    EXPECT_DOUBLE_EQ(moneyness, 1.0);

    std::cout << "Normalized batch parameters:" << std::endl;
    std::cout << "  spot: " << params.spot << std::endl;
    std::cout << "  strike: " << params.strike << std::endl;
    std::cout << "  moneyness: " << moneyness << std::endl;
    std::cout << "  maturity: " << params.maturity << std::endl;
    std::cout << "  volatility: " << params.volatility << std::endl;
    std::cout << "  rate: " << params.rate << std::endl;
}

} // namespace
} // namespace mango
