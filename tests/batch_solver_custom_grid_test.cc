// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/american_option_batch.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"

TEST(BatchSolverCustomGridTest, AcceptsCustomGrid) {
    // Test without custom grid first (baseline)
    std::vector<mango::PricingParams> batch;
    batch.push_back(mango::PricingParams(
        mango::OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02, .option_type = mango::OptionType::PUT}, 0.20));

    mango::BatchAmericanOptionSolver solver;
    solver.set_use_normalized(false);

    // First, verify it works without custom_grid
    auto result_baseline = solver.solve_batch(batch, true);
    EXPECT_EQ(result_baseline.failed_count, 0);
    EXPECT_EQ(result_baseline.results.size(), 1);
    if (result_baseline.all_succeeded()) {
        // Now try with custom grid - use wider bounds than auto-estimation would choose
        auto grid_spec = mango::GridSpec<double>::uniform(-4.0, 4.0, 101).value();
        auto time_domain = mango::TimeDomain::from_n_steps(0.0, 1.0, 200);

        std::optional<mango::PDEGridSpec> custom_grid =
        mango::ExplicitPDEGrid{grid_spec, time_domain.n_steps(), {}};

        auto result = solver.solve_batch(batch, true, nullptr, custom_grid);

        EXPECT_EQ(result.results.size(), 1);
        ASSERT_TRUE(result.all_succeeded()) << "With custom_grid failed with error code: "
                   << static_cast<int>(result.results[0].error().code)
                   << ", iterations: " << result.results[0].error().iterations
                   << ", residual: " << result.results[0].error().residual;

        EXPECT_GT(result.results[0]->value(), 0.0) << "Price should be positive for ATM put";
    } else {
        FAIL() << "Baseline test failed - can't proceed";
    }
}

TEST(BatchSolverCustomGridTest, NulloptUsesAutoEstimation) {
    std::vector<mango::PricingParams> batch;
    batch.push_back(mango::PricingParams(
        mango::OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02, .option_type = mango::OptionType::PUT}, 0.20));

    mango::BatchAmericanOptionSolver solver;
    // Pass nullopt explicitly - should use auto-estimation
    auto result = solver.solve_batch(batch, true, nullptr, std::nullopt);

    EXPECT_TRUE(result.all_succeeded());
}
