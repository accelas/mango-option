// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/american_option_batch.hpp"

using namespace mango;

TEST(BatchAmericanOptionSolver, NormalizedEligibility) {
    // Test eligible batch: varying strikes with same maturity
    std::vector<PricingParams> eligible_params;
    double spot = 100.0;
    std::vector<double> strikes = {90, 95, 100, 105, 110};

    for (double K : strikes) {
        eligible_params.push_back(PricingParams(OptionSpec{.spot = spot, .strike = K, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20));
    }

    BatchAmericanOptionSolver solver;
    auto result = solver.solve_batch(eligible_params, /*use_shared_grid=*/true);

    // Should use normalized path: 1 PDE solve for 5 options
    EXPECT_EQ(result.failed_count, 0);
    EXPECT_EQ(result.results.size(), 5);

    // All results should have converged
    for (const auto& r : result.results) {
        ASSERT_TRUE(r.has_value());
        EXPECT_TRUE(r->converged);
        EXPECT_GT(r->value(), 0.0);
    }
}

TEST(BatchAmericanOptionSolver, NormalizedIneligibleDividends) {
    // Test ineligible batch (discrete dividends)
    std::vector<PricingParams> ineligible_params;
    double spot = 100.0;

    for (int i = 0; i < 5; ++i) {
        ineligible_params.push_back(PricingParams(OptionSpec{.spot = spot, .strike = 90.0 + i * 5.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20, {{0.5, 2.0}}));
    }

    BatchAmericanOptionSolver solver;
    auto result = solver.solve_batch(ineligible_params, /*use_shared_grid=*/true);

    // Should fall back to regular path
    EXPECT_EQ(result.failed_count, 0);
    EXPECT_EQ(result.results.size(), 5);
}

TEST(BatchAmericanOptionSolver, DisableNormalizedOptimization) {
    // Test forcing regular path
    std::vector<PricingParams> params;
    double spot = 100.0;

    for (int i = 0; i < 5; ++i) {
        params.push_back(PricingParams(OptionSpec{.spot = spot, .strike = 90.0 + i * 5.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20));
    }

    BatchAmericanOptionSolver solver;
    solver.set_use_normalized(false);  // Force regular path

    auto result = solver.solve_batch(params, /*use_shared_grid=*/true);
    EXPECT_EQ(result.failed_count, 0);
    EXPECT_EQ(result.results.size(), 5);
}

// ===========================================================================
// Regression tests for bugs found during code review
// ===========================================================================

// Regression: Batch solver must pass grid config to AmericanOptionSolver
// Bug: Issue 272 - solver_grid_config was not passed to AmericanOptionSolver,
//      causing solver to re-estimate grid with different size than workspace
//      allocation. This resulted in 100% PDE failure rate for production configs.
TEST(AmericanOptionBatch, RegressionIssue272_WorkspaceGridSizeConsistency) {
    // Create a batch that uses shared grid with varying strikes
    std::vector<PricingParams> params;
    for (double K : {85.0, 92.5, 100.0, 107.5, 115.0}) {
        params.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = K, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20));
    }

    BatchAmericanOptionSolver solver;

    // Solve with shared grid - this was failing before the fix with
    // SolverErrorCode::InvalidConfiguration due to workspace/grid size mismatch
    auto results = solver.solve_batch(params, /*use_shared_grid=*/true);

    // All solves should succeed (not fail with InvalidConfiguration)
    EXPECT_EQ(results.failed_count, 0)
        << "Workspace/grid size mismatch causes failures";

    for (size_t i = 0; i < results.results.size(); ++i) {
        ASSERT_TRUE(results.results[i].has_value())
            << "Option " << i << " failed with error code "
            << static_cast<int>(results.results[i].error().code);
    }
}

// Regression: Per-option grid path must also track solver_grid_config
// Bug: Issue 272 - the fix must cover all code paths including per-option grids
TEST(AmericanOptionBatch, RegressionIssue272_PerOptionGridConsistency) {
    std::vector<PricingParams> params;
    for (double K : {90.0, 100.0, 110.0}) {
        params.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = K, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20));
    }

    BatchAmericanOptionSolver solver;

    // Solve WITHOUT shared grid (per-option grids)
    auto results = solver.solve_batch(params, /*use_shared_grid=*/false);

    EXPECT_EQ(results.failed_count, 0);
    for (const auto& result : results.results) {
        ASSERT_TRUE(result.has_value());
    }
}
