// SPDX-License-Identifier: MIT
#include "mango/option/american_option.hpp"
#include "mango/option/american_option_batch.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <memory_resource>
#include <vector>

namespace mango {
namespace {

class AmericanOptionPricingTest : public ::testing::Test {
protected:
    void SetUp() override {
        pool_ = std::make_unique<std::pmr::synchronized_pool_resource>();
    }

    [[nodiscard]] AmericanOptionResult Solve(const PricingParams& params) const {
        // Use convenience function that creates appropriately-sized workspace
        auto result = solve_american_option(params);
        if (!result) {
            const auto& error = result.error();
            ADD_FAILURE() << "Solver failed: " << error
                          << " (code=" << static_cast<int>(error.code)
                          << ", iterations=" << error.iterations << ")";
            // Cannot return empty AmericanOptionResult (not default constructible)
            // Throw to abort test
            throw std::runtime_error("Solver failed");
        }
        return std::move(result.value());
    }

    std::unique_ptr<std::pmr::synchronized_pool_resource> pool_;
};

TEST_F(AmericanOptionPricingTest, SolverWithPMRWorkspace) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 110.0, .maturity = 1.0,
            .rate = 0.03, .option_type = OptionType::PUT}, 0.25);

    // Use convenience function that automatically sizes the workspace
    auto result = solve_american_option(params);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->converged);
}

TEST_F(AmericanOptionPricingTest, PutValueRespectsIntrinsicBound) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 110.0, .maturity = 1.0,
            .rate = 0.03, .option_type = OptionType::PUT}, 0.25);

    AmericanOptionResult result = Solve(params);
    ASSERT_TRUE(result.converged);

    double intrinsic = std::max(params.strike - params.spot, 0.0);
    EXPECT_GE(result.value_at(params.spot), intrinsic - 1e-6);
    EXPECT_LT(result.value_at(params.spot), params.strike);
}

TEST_F(AmericanOptionPricingTest, CallValueIncreasesWithVolatility) {
    const double spot = 100.0;
    const double strike = 100.0;
    const double maturity = 1.0;
    const double rate = 0.01;
    std::vector<double> volatilities = {0.15, 0.25, 0.4};
    double previous_value = 0.0;
    double previous_vol = 0.0;
    for (size_t i = 0; i < volatilities.size(); ++i) {
        double vol = volatilities[i];
        PricingParams params(
            OptionSpec{.spot = spot, .strike = strike, .maturity = maturity,
                .rate = rate, .option_type = OptionType::CALL}, vol);
        AmericanOptionResult result = Solve(params);
        ASSERT_TRUE(result.converged);

        if (i > 0) {
            EXPECT_GT(result.value_at(params.spot), previous_value)
                << "Value did not increase when volatility went from "
                << previous_vol << " to " << vol;
        }
        previous_value = result.value_at(params.spot);
        previous_vol = vol;
    }
}

TEST_F(AmericanOptionPricingTest, PutValueIncreasesWithMaturity) {
    std::vector<double> maturities = {0.25, 0.5, 1.0, 2.0};
    double previous_value = 0.0;
    for (double maturity : maturities) {
        PricingParams params(
            OptionSpec{.spot = 100.0, .strike = 95.0, .maturity = maturity,
                .rate = 0.02, .option_type = OptionType::PUT}, 0.2);

        AmericanOptionResult result = Solve(params);
        ASSERT_TRUE(result.converged);

        if (previous_value > 0.0) {
            EXPECT_GE(result.value_at(params.spot), previous_value - 5e-3);
        }
        previous_value = result.value_at(params.spot);
    }
}

TEST_F(AmericanOptionPricingTest, DividendsReduceCallValue) {
    PricingParams no_dividends(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.02, .option_type = OptionType::CALL}, 0.3);

    PricingParams with_dividends(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.02, .option_type = OptionType::CALL}, 0.3,
        {{.calendar_time = 0.5, .amount = 3.0}});

    AmericanOptionResult result_no_div = Solve(no_dividends);
    AmericanOptionResult result_with_div = Solve(with_dividends);

    ASSERT_TRUE(result_no_div.converged);
    ASSERT_TRUE(result_with_div.converged);

    EXPECT_GT(result_no_div.value_at(no_dividends.spot), result_with_div.value_at(with_dividends.spot));
}

TEST_F(AmericanOptionPricingTest, BatchSolverMatchesSingleSolver) {
    std::vector<PricingParams> params;
    params.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.75, .rate = 0.01, .option_type = OptionType::CALL}, 0.25));
    params.push_back(PricingParams(OptionSpec{.spot = 120.0, .strike = 100.0, .maturity = 1.5, .rate = 0.02, .option_type = OptionType::PUT}, 0.2));
    params.push_back(PricingParams(OptionSpec{.spot = 90.0, .strike = 95.0, .maturity = 0.5, .rate = -0.01, .dividend_yield = 0.01, .option_type = OptionType::PUT}, 0.35));

    // Use automatic grid determination for batch solver
    auto batch_result = BatchAmericanOptionSolver().solve_batch(params);
    ASSERT_EQ(batch_result.results.size(), params.size());
    EXPECT_EQ(batch_result.failed_count, 0u);

    // Compare with single option automatic grid solver
    for (size_t i = 0; i < params.size(); ++i) {
        ASSERT_TRUE(batch_result.results[i].has_value()) << "Batch solve failed for index " << i;

        auto single_result = solve_american_option(params[i]);
        ASSERT_TRUE(single_result.has_value()) << "Single solve failed for index " << i;
        ASSERT_TRUE(single_result->converged);

        const double batch_value = batch_result.results[i]->value_at(params[i].spot);
        const double single_value = single_result->value_at(params[i].spot);
        EXPECT_NEAR(single_value, batch_value, 1e-3) << "Mismatch at index " << i;
    }
}

TEST_F(AmericanOptionPricingTest, PutImmediateExerciseAtBoundary) {
    // Deep ITM put test - verifies active set method locks nodes to payoff
    // Fixed by implementing proper complementarity enforcement in Newton solver
    PricingParams params(
        OptionSpec{.spot = 0.25, .strike = 100.0, .maturity = 0.75,
            .rate = 0.05, .option_type = OptionType::PUT}, 0.2);

    // Use convenience function - it will automatically size the grid appropriately
    auto result_exp = solve_american_option(params);
    ASSERT_TRUE(result_exp.has_value()) << result_exp.error();
    AmericanOptionResult result = std::move(result_exp.value());
    ASSERT_TRUE(result.converged);

    const double intrinsic = params.strike - params.spot;
    EXPECT_NEAR(result.value_at(params.spot), intrinsic, 1e-3)
        << "Left boundary should equal immediate exercise for deep ITM put (error < 0.001)";
}

TEST_F(AmericanOptionPricingTest, ATMOptionsRetainTimeValue) {
    // Regression test for Issue #196 IV solver failure
    // Verifies that ATM options develop time value and don't lock to payoff=0
    // This guards against the known limitation of the 50% time window guard
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .option_type = OptionType::PUT}, 0.25);

    AmericanOptionResult result = Solve(params);
    ASSERT_TRUE(result.converged);

    // ATM put should have significant time value (not lock to payoff=0)
    // With σ=0.25, T=1.0, r=0.05, ATM American put should be worth ~$8
    const double intrinsic = std::max(params.strike - params.spot, 0.0);  // 0 for ATM
    EXPECT_GT(result.value_at(params.spot), intrinsic + 7.0)
        << "ATM put must develop time value, not lock to payoff=0";
    EXPECT_LT(result.value_at(params.spot), 12.0)
        << "ATM put price seems unreasonably high";
}

TEST_F(AmericanOptionPricingTest, PricingWithYieldCurve) {
    // Integration test for yield curve support
    // Upward sloping curve: 5% for first 6 months, 6% for second 6 months
    std::vector<TenorPoint> points = {
        {0.0, 0.0},
        {0.5, -0.025},   // 5% for first 6 months (integral: 0.05 * 0.5 = 0.025)
        {1.0, -0.055}    // 6% for second 6 months (integral: 0.025 + 0.06 * 0.5 = 0.055)
    };
    auto curve = YieldCurve::from_points(points).value();

    // Create params with yield curve
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = curve, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20);

    // Solve with yield curve
    auto result = solve_american_option(params);
    ASSERT_TRUE(result.has_value()) << "Solver failed with yield curve";
    ASSERT_TRUE(result->converged);

    // Price should be positive and reasonable
    double price = result->value_at(params.spot);
    EXPECT_GT(price, 0.0);
    EXPECT_LT(price, params.strike);  // Put can't exceed strike

    // Compare with flat rate at average (5.5%)
    PricingParams flat_params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.055, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20);

    auto flat_result = solve_american_option(flat_params);
    ASSERT_TRUE(flat_result.has_value()) << "Solver failed with flat rate";
    ASSERT_TRUE(flat_result->converged);

    // Prices should be close (within 2% for similar average rate)
    // Note: Some difference expected due to convexity effects with sloping curve
    double flat_price = flat_result->value_at(flat_params.spot);
    EXPECT_NEAR(price, flat_price, flat_price * 0.02)
        << "Yield curve price differs significantly from flat rate average";
}

TEST_F(AmericanOptionPricingTest, DiscreteDividendPutPriceHigherThanNoDividend) {
    // A discrete dividend increases put value (spot drops)
    PricingParams no_div(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20);
    PricingParams with_div(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20,
                                  {{.calendar_time = 0.5, .amount = 3.0}});

    auto result_no_div = solve_american_option(no_div);
    auto result_with_div = solve_american_option(with_div);

    ASSERT_TRUE(result_no_div.has_value());
    ASSERT_TRUE(result_with_div.has_value());

    EXPECT_GT(result_with_div->value_at(with_div.spot), result_no_div->value_at(no_div.spot))
        << "Put with discrete dividend should be worth more than without";
}

TEST_F(AmericanOptionPricingTest, DiscreteDividendCallPriceLowerThanNoDividend) {
    PricingParams no_div(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::CALL}, 0.20);
    PricingParams with_div(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::CALL}, 0.20,
                                  {{.calendar_time = 0.5, .amount = 3.0}});

    auto result_no_div = solve_american_option(no_div);
    auto result_with_div = solve_american_option(with_div);

    ASSERT_TRUE(result_no_div.has_value());
    ASSERT_TRUE(result_with_div.has_value());

    EXPECT_LT(result_with_div->value_at(with_div.spot), result_no_div->value_at(no_div.spot))
        << "Call with discrete dividend should be worth less than without";
}

TEST_F(AmericanOptionPricingTest, DiscreteDividendCallLargeDividend) {
    PricingParams params(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::CALL}, 0.30,
                                {{.calendar_time = 0.5, .amount = 50.0}});

    auto result = solve_american_option(params);
    ASSERT_TRUE(result.has_value());
    EXPECT_GE(result->value_at(params.spot), 0.0);
    EXPECT_TRUE(std::isfinite(result->value_at(params.spot)));
}

TEST_F(AmericanOptionPricingTest, DiscreteDividendMultiple) {
    PricingParams one_div(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20,
                                 {{.calendar_time = 0.5, .amount = 2.0}});
    PricingParams two_div(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20,
                                 {{.calendar_time = 0.3, .amount = 2.0}, {.calendar_time = 0.7, .amount = 2.0}});

    auto result_one = solve_american_option(one_div);
    auto result_two = solve_american_option(two_div);

    ASSERT_TRUE(result_one.has_value());
    ASSERT_TRUE(result_two.has_value());

    EXPECT_GT(result_two->value_at(two_div.spot), result_one->value_at(one_div.spot))
        << "Two dividends should increase put value more than one";
}

TEST_F(AmericanOptionPricingTest, RegularBatchWithDiscreteDividends) {
    // Batch of options with discrete dividends — uses regular path (not normalized)
    std::vector<PricingParams> batch;
    batch.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20,
                       std::vector<mango::Dividend>{{.calendar_time = 0.5, .amount = 3.0}}));
    batch.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = 110.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20,
                       std::vector<mango::Dividend>{{.calendar_time = 0.5, .amount = 3.0}}));

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(batch);

    EXPECT_EQ(results.failed_count, 0u);
    for (size_t i = 0; i < batch.size(); ++i) {
        ASSERT_TRUE(results.results[i].has_value()) << "Batch solve failed for index " << i;
        EXPECT_GT(results.results[i]->value_at(batch[i].spot), 0.0);
    }
}

TEST_F(AmericanOptionPricingTest, NormalizedChainFallsBackWithDividends) {
    // When requesting shared grid with dividends, should fall back to regular batch
    // (normalized chain rejects discrete dividends)
    std::vector<PricingParams> batch;
    batch.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = 90.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20,
                       std::vector<mango::Dividend>{{.calendar_time = 0.5, .amount = 3.0}}));
    batch.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20,
                       std::vector<mango::Dividend>{{.calendar_time = 0.5, .amount = 3.0}}));
    batch.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = 110.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20,
                       std::vector<mango::Dividend>{{.calendar_time = 0.5, .amount = 3.0}}));

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(batch, /*use_shared_grid=*/true);

    // Should still succeed via regular batch fallback
    EXPECT_EQ(results.failed_count, 0u);
    for (size_t i = 0; i < batch.size(); ++i) {
        ASSERT_TRUE(results.results[i].has_value()) << "Batch solve failed for index " << i;
        EXPECT_GT(results.results[i]->value_at(batch[i].spot), 0.0);
    }
}

// ===========================================================================
// Regression: create() must validate workspace/grid at construction time
// Bug: Mismatch between workspace size and grid was only caught at solve()
// ===========================================================================

TEST(AmericanOptionTest, CreateRejectsMismatchedWorkspace) {
    // #306: Mismatch should fail at create(), not at solve()
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.05, .option_type = OptionType::PUT},
        0.20);

    // Create a workspace that is deliberately too small (10 points)
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(10));
    auto ws = PDEWorkspace::from_buffer(buffer, 10);
    ASSERT_TRUE(ws.has_value());

    // create() should now fail because auto-estimated grid needs ~100+ points
    auto result = AmericanOptionSolver::create(params, ws.value());
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ValidationErrorCode::InvalidGridSize);
}

// ===========================================================================
// solve_american_option accessible from primary header
// ===========================================================================

TEST(AmericanOptionTest, SolveAutoFromPrimaryHeader) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT},
        0.20);
    auto result = solve_american_option(params);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value_at(100.0), 6.35, 0.5);
}

// European PDE (projection disabled) should produce value <= American
TEST(AmericanOptionTest, ProjectionDisabledProducesEuropeanValue) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.05, .dividend_yield = 0.02,
                   .option_type = OptionType::PUT},
        0.20);

    // American solve (projection enabled, default)
    auto am = solve_american_option(params);
    ASSERT_TRUE(am.has_value());
    double am_price = am->value_at(100.0);

    // European solve (projection disabled)
    auto [grid_spec, time_domain] = estimate_pde_grid(params);
    size_t n = grid_spec.n_points();
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n),
                                     std::pmr::get_default_resource());
    auto ws = PDEWorkspace::from_buffer(buffer, n).value();
    auto solver = AmericanOptionSolver::create(params, ws).value();
    solver.set_projection_enabled(false);
    auto eu = solver.solve();
    ASSERT_TRUE(eu.has_value());
    double eu_price = eu->value_at(100.0);

    // European <= American (early exercise adds value)
    EXPECT_LE(eu_price, am_price + 1e-10);
    // European should still be positive
    EXPECT_GT(eu_price, 0.0);
    // Difference (EEP) should be small but positive for ATM put
    EXPECT_GT(am_price - eu_price, 0.0);
}

}  // namespace
}  // namespace mango
