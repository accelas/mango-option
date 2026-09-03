// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/american_option_batch.hpp"

#include <cmath>
#include <utility>
#include <variant>
#include <vector>

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

TEST(BatchAmericanOptionSolver, NormalizedMixedMaturities) {
    // Mixed maturities should be grouped into separate PDE groups,
    // each solved with a single normalized PDE
    std::vector<PricingParams> params;
    double spot = 100.0;

    // Group 1: maturity = 0.5, varying strikes
    for (double K : {90.0, 100.0, 110.0}) {
        params.push_back(PricingParams(OptionSpec{.spot = spot, .strike = K, .maturity = 0.5, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20));
    }
    // Group 2: maturity = 1.0, varying strikes
    for (double K : {90.0, 100.0, 110.0}) {
        params.push_back(PricingParams(OptionSpec{.spot = spot, .strike = K, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20));
    }

    BatchAmericanOptionSolver solver;
    auto result = solver.solve_batch(params, /*use_shared_grid=*/true);

    EXPECT_EQ(result.failed_count, 0);
    EXPECT_EQ(result.results.size(), 6);

    for (size_t i = 0; i < result.results.size(); ++i) {
        ASSERT_TRUE(result.results[i].has_value())
            << "Option " << i << " failed";
        EXPECT_GT(result.results[i]->value(), 0.0);
    }

    // Cross-check: normalized results should match individual solves
    for (size_t i = 0; i < params.size(); ++i) {
        auto individual = solve_american_option(params[i]);
        ASSERT_TRUE(individual.has_value());
        double normalized_price = result.results[i]->value();
        double individual_price = individual->value();
        EXPECT_NEAR(normalized_price, individual_price, 0.05)
            << "Option " << i << " (K=" << params[i].strike
            << ", T=" << params[i].maturity << "): normalized="
            << normalized_price << " vs individual=" << individual_price;
    }
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

// ===========================================================================
// Log-moneyness coverage on GridAccuracyParams (#480 rework)
// ===========================================================================

namespace {
PricingParams atm_put(double sigma, double T, std::vector<Dividend> divs = {}) {
    PricingParams p(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = T,
        .rate = 0.05, .dividend_yield = 0.0, .option_type = OptionType::PUT}, sigma);
    p.discrete_dividends = std::move(divs);
    return p;
}
}  // namespace

// D14: eligibility is judged with coverage cleared.  Reach 3.0 makes the
// coverage-widened first-contract grid 7.2 wide (> MAX_WIDTH = 5.8) while its
// base grid (half-width 1.0, margin 1.0) is eligible, so the batch must STILL
// route through the normalized chain: each group solves on its own grid,
// covering with its own clearance, and the two grids differ.  Without D14
// the batch would fall to the shared regular path and both results would
// share one grid.
// Note that the per-group grids solved on here are therefore WIDER than
// MAX_WIDTH = 5.8 by design of D14; #487 (eligibility judging the grid
// actually solved on) may renegotiate that.
TEST(BatchAmericanOptionSolver, CoverageDoesNotChangeRoutingAndEveryGroupCovers) {
    std::vector<PricingParams> batch = {atm_put(0.20, 1.0), atm_put(0.40, 1.0)};
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-3.0, 3.0};
    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(acc);
    auto result = solver.solve_batch(batch, /*use_shared_grid=*/true);
    ASSERT_EQ(result.failed_count, 0u);
    auto x0 = result.results[0]->grid()->x();
    auto x1 = result.results[1]->grid()->x();
    EXPECT_LE(x0.front(), -(3.0 + 3.0 * 0.20) + 1e-9);
    EXPECT_LE(x1.front(), -(3.0 + 3.0 * 0.40) + 1e-9);
    EXPECT_NE(x0.front(), x1.front()) << "per-group grids expected (normalized routing)";
}

// Shared regular route (dividends make the batch ineligible): one grid for
// all, its edge set by the LARGEST sigma*sqrt(T).
TEST(BatchAmericanOptionSolver, SharedRegularRouteCoversWithSigmaMax) {
    const std::vector<Dividend> divs = {Dividend{.calendar_time = 0.5, .amount = 1.0}};
    std::vector<PricingParams> batch = {atm_put(0.20, 1.0, divs), atm_put(0.40, 1.0, divs)};
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-1.5, 1.5};
    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(acc);
    auto result = solver.solve_batch(batch, /*use_shared_grid=*/true);
    ASSERT_EQ(result.failed_count, 0u);
    auto x0 = result.results[0]->grid()->x();
    auto x1 = result.results[1]->grid()->x();
    EXPECT_DOUBLE_EQ(x0.front(), x1.front());
    EXPECT_LE(x0.front(), -(1.5 + 3.0 * 0.40) + 1e-9);
    EXPECT_GE(x0.back(),   (1.5 + 3.0 * 0.40) - 1e-9);
}

// Per-contract route: a probe 1.3 beyond the strike with s = 0.2*sqrt(0.02).
TEST(BatchAmericanOptionSolver, PerContractRouteHonoursCoverage) {
    std::vector<PricingParams> batch = {atm_put(0.20, 0.02)};
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-1.3, -1.3};
    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(acc);
    auto result = solver.solve_batch(batch, /*use_shared_grid=*/false);
    ASSERT_TRUE(result.results[0].has_value());
    EXPECT_LE(result.results[0]->grid()->x().front(),
              -(1.3 + 3.0 * 0.20 * std::sqrt(0.02)) + 1e-9);
}

// D15: an accuracy spec passed as custom_grid is estimated the same way the
// solver's own accuracy is -- over the batch on the shared path (edge from
// sigma_max, not from params[0]) ...
TEST(BatchAmericanOptionSolver, AccuracyCustomGridIsEstimatedOverTheBatch) {
    const std::vector<Dividend> divs = {Dividend{.calendar_time = 0.5, .amount = 1.0}};
    std::vector<PricingParams> batch = {atm_put(0.20, 1.0, divs), atm_put(0.40, 1.0, divs)};
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-1.5, 1.5};
    BatchAmericanOptionSolver solver;
    auto result = solver.solve_batch(batch, /*use_shared_grid=*/true, nullptr, PDEGridSpec{acc});
    ASSERT_EQ(result.failed_count, 0u);
    auto x0 = result.results[0]->grid()->x();
    auto x1 = result.results[1]->grid()->x();
    EXPECT_DOUBLE_EQ(x0.front(), x1.front());
    EXPECT_DOUBLE_EQ(x0.back(),  x1.back());
    EXPECT_LE(x0.front(), -(1.5 + 3.0 * 0.40) + 1e-9);
}

// ... and per contract otherwise: two sigmas -> two different grids (before
// the repair both contracts reused params[0]'s grid), and the coverage set
// on the custom accuracy spec is honoured by each (range [-2.5, 2.5] lies
// beyond both base domains, +-1.0 and +-2.0).
TEST(BatchAmericanOptionSolver, AccuracyCustomGridIsEstimatedPerContractWithCoverage) {
    std::vector<PricingParams> batch = {atm_put(0.20, 1.0), atm_put(0.40, 1.0)};
    GridAccuracyParams acc;
    acc.log_moneyness_coverage = LogMoneynessRange{-2.5, 2.5};
    BatchAmericanOptionSolver solver;
    auto result = solver.solve_batch(batch, /*use_shared_grid=*/false, nullptr,
                                     PDEGridSpec{acc});
    ASSERT_EQ(result.failed_count, 0u);
    auto x0 = result.results[0]->grid()->x();
    auto x1 = result.results[1]->grid()->x();
    EXPECT_LE(x0.front(), -(2.5 + 3.0 * 0.20) + 1e-9);
    EXPECT_LE(x1.front(), -(2.5 + 3.0 * 0.40) + 1e-9);
    EXPECT_NE(x0.front(), x1.front());
}
