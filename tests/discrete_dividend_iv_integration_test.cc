// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/iv_solver_factory.hpp"
#include "src/option/american_option.hpp"
#include "src/option/american_option_batch.hpp"

using namespace mango;

// ===========================================================================
// End-to-end integration test: discrete dividend IV round-trip
//
// Pipeline: make_iv_solver (factory) -> segmented build -> IV solve
// Approach: Price an American option at known vol via FDM, then recover
//           that vol using the interpolated IV solver.
// ===========================================================================

class DiscreteDividendIVIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        IVSolverConfig config{
            .option_type = OptionType::PUT,
            .spot = 100.0,
            .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}},
            .moneyness_grid = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
            .maturity = 1.0,
            .vol_grid = {0.10, 0.15, 0.20, 0.25, 0.30, 0.40},
            .rate_grid = {0.02, 0.03, 0.05, 0.07},
            // Use default K_ref config (9 log-spaced points)
        };
        auto result = make_iv_solver(config);
        ASSERT_TRUE(result.has_value()) << "Failed to build solver";
        solver_ = std::make_unique<IVSolver>(std::move(*result));
    }

    std::unique_ptr<IVSolver> solver_;
};

TEST_F(DiscreteDividendIVIntegrationTest, ATMPutIVRoundTrip) {
    // Price an ATM put at known vol=0.20 using FDM
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.8,
            .rate = 0.05, .option_type = OptionType::PUT},
        0.20, {{.calendar_time = 0.5, .amount = 2.0}});

    auto price_result = solve_american_option_auto(params);
    ASSERT_TRUE(price_result.has_value());
    double market_price = price_result->value();
    EXPECT_GT(market_price, 0.0) << "FDM price should be positive";

    // Now solve for IV using the interpolated solver
    IVQuery query;
    query.spot = 100.0;
    query.strike = 100.0;
    query.maturity = 0.8;
    query.rate = RateSpec{0.05};
    query.dividend_yield = 0.0;
    query.option_type = OptionType::PUT;
    query.market_price = market_price;

    auto iv_result = solver_->solve(query);
    ASSERT_TRUE(iv_result.has_value())
        << "IV solve must succeed; error code: "
        << (iv_result.has_value() ? 0 : static_cast<int>(iv_result.error().code));
    {
        EXPECT_NEAR(iv_result->implied_vol, 0.20, 0.02)
            << "Recovered IV should be close to true vol=0.20, got "
            << iv_result->implied_vol << " for market_price=" << market_price;
    }
}

TEST_F(DiscreteDividendIVIntegrationTest, OTMPutIVRoundTrip) {
    // OTM put: strike=90, vol=0.25
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 90.0, .maturity = 0.8,
            .rate = 0.05, .option_type = OptionType::PUT},
        0.25, {{.calendar_time = 0.5, .amount = 2.0}});

    auto price_result = solve_american_option_auto(params);
    ASSERT_TRUE(price_result.has_value());
    EXPECT_GT(price_result->value(), 0.0);

    IVQuery query;
    query.spot = 100.0;
    query.strike = 90.0;
    query.maturity = 0.8;
    query.rate = RateSpec{0.05};
    query.dividend_yield = 0.0;
    query.option_type = OptionType::PUT;
    query.market_price = price_result->value();

    auto iv_result = solver_->solve(query);
    ASSERT_TRUE(iv_result.has_value())
        << "IV solve must succeed; error code: "
        << static_cast<int>(iv_result.error().code);
    EXPECT_NEAR(iv_result->implied_vol, 0.25, 0.02)
        << "Recovered IV should be close to true vol=0.25, got "
        << iv_result->implied_vol;
}

TEST_F(DiscreteDividendIVIntegrationTest, ITMPutIVRoundTrip) {
    // ITM put: strike=110, vol=0.20 at various maturities
    // Regression: spot adjustment must NOT apply to RawPrice (chained) segments
    for (double tau : {1.0, 0.8, 0.6}) {
        SCOPED_TRACE("maturity=" + std::to_string(tau));

        std::vector<Dividend> divs;
        if (tau > 0.5) divs = {{.calendar_time = 0.5, .amount = 2.0}};

        PricingParams params(
            OptionSpec{.spot = 100.0, .strike = 110.0, .maturity = tau,
                .rate = 0.05, .option_type = OptionType::PUT}, 0.20, divs);

        auto price_result = solve_american_option_auto(params);
        ASSERT_TRUE(price_result.has_value());
        EXPECT_GT(price_result->value(), 0.0);

        IVQuery query;
        query.spot = 100.0;
        query.strike = 110.0;
        query.maturity = tau;
        query.rate = RateSpec{0.05};
        query.dividend_yield = 0.0;
        query.option_type = OptionType::PUT;
        query.market_price = price_result->value();

        auto iv_result = solver_->solve(query);
        ASSERT_TRUE(iv_result.has_value())
            << "IV solve must succeed; error code: "
            << static_cast<int>(iv_result.error().code);
        EXPECT_NEAR(iv_result->implied_vol, 0.20, 0.02)
            << "Recovered IV should be close to true vol=0.20, got "
            << iv_result->implied_vol;
    }
}

TEST_F(DiscreteDividendIVIntegrationTest, NearExpiryIV) {
    // Near expiry: maturity=0.3 (before the dividend at t=0.5)
    // The FDM solver validates that discrete dividends fall within maturity,
    // so we price WITHOUT discrete dividends for the FDM reference price.
    // The interpolated solver (built with dividends) should still produce
    // reasonable results for maturities before the first dividend.
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.3,
            .rate = 0.05, .option_type = OptionType::PUT}, 0.20);

    auto price_result = solve_american_option_auto(params);
    ASSERT_TRUE(price_result.has_value());
    EXPECT_GT(price_result->value(), 0.0);

    IVQuery query;
    query.spot = 100.0;
    query.strike = 100.0;
    query.maturity = 0.3;
    query.rate = RateSpec{0.05};
    query.dividend_yield = 0.0;
    query.option_type = OptionType::PUT;
    query.market_price = price_result->value();

    auto iv_result = solver_->solve(query);
    ASSERT_TRUE(iv_result.has_value())
        << "IV solve must succeed; error code: "
        << static_cast<int>(iv_result.error().code);
    // For maturities before the dividend, the interpolated surface
    // (which knows about dividends) should still recover a reasonable IV.
    EXPECT_GT(iv_result->implied_vol, 0.05);
    EXPECT_LT(iv_result->implied_vol, 1.0);
}

TEST_F(DiscreteDividendIVIntegrationTest, HighVolRoundTrip) {
    // High vol: vol=0.35
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.8,
            .rate = 0.03, .option_type = OptionType::PUT},
        0.35, {{.calendar_time = 0.5, .amount = 2.0}});

    auto price_result = solve_american_option_auto(params);
    ASSERT_TRUE(price_result.has_value());
    EXPECT_GT(price_result->value(), 0.0);

    IVQuery query;
    query.spot = 100.0;
    query.strike = 100.0;
    query.maturity = 0.8;
    query.rate = RateSpec{0.03};
    query.dividend_yield = 0.0;
    query.option_type = OptionType::PUT;
    query.market_price = price_result->value();

    auto iv_result = solver_->solve(query);
    ASSERT_TRUE(iv_result.has_value())
        << "IV solve must succeed; error code: "
        << static_cast<int>(iv_result.error().code);
    EXPECT_NEAR(iv_result->implied_vol, 0.35, 0.02)
        << "High-vol IV should be close to true vol=0.35, got "
        << iv_result->implied_vol;
}

// ===========================================================================
// Regression: Verify that the factory builds and works without dividends
// ===========================================================================

TEST(DiscreteDividendIVRegressionTest, NoDividendMatchesExisting) {
    IVSolverConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .dividend_yield = 0.02,
        .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},
        .maturity_grid = {0.1, 0.25, 0.5, 1.0},
        .vol_grid = {0.10, 0.15, 0.20, 0.30, 0.40},
        .rate_grid = {0.02, 0.03, 0.05, 0.07},
    };
    auto solver = make_iv_solver(config);
    ASSERT_TRUE(solver.has_value()) << "No-dividend factory should succeed";

    // Price an option with continuous dividends via FDM for round-trip test
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.5,
            .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20);

    auto price_result = solve_american_option_auto(params);
    ASSERT_TRUE(price_result.has_value());

    IVQuery query;
    query.spot = 100.0;
    query.strike = 100.0;
    query.maturity = 0.5;
    query.rate = RateSpec{0.05};
    query.dividend_yield = 0.02;
    query.option_type = OptionType::PUT;
    query.market_price = price_result->value();

    auto result = solver->solve(query);
    ASSERT_TRUE(result.has_value())
        << "IV solve must succeed; error code: "
        << static_cast<int>(result.error().code);
    EXPECT_GT(result->implied_vol, 0.0);
    EXPECT_LT(result->implied_vol, 3.0);
    EXPECT_NEAR(result->implied_vol, 0.20, 0.02)
        << "Continuous dividend round-trip should recover vol=0.20, got "
        << result->implied_vol;
}
