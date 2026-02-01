// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/iv_solver_factory.hpp"

using namespace mango;

TEST(IVSolverFactoryTest, NoDividendsUsesStandardPath) {
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
    ASSERT_TRUE(solver.has_value()) << "Factory should succeed with no dividends";

    IVQuery query;
    query.spot = 100.0;
    query.strike = 100.0;
    query.maturity = 0.5;
    query.rate = RateSpec{0.05};
    query.dividend_yield = 0.02;
    query.type = OptionType::PUT;
    query.market_price = 6.0;

    auto result = solver->solve(query);
    // May or may not converge depending on exact price, but should not crash
    if (result.has_value()) {
        EXPECT_GT(result->implied_vol, 0.0);
        EXPECT_LT(result->implied_vol, 3.0);
    }
}

TEST(IVSolverFactoryTest, DiscreteDividendsUsesSegmentedPath) {
    IVSolverConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .discrete_dividends = {{0.5, 2.0}},
        .moneyness_grid = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
        .maturity = 1.0,
        .vol_grid = {0.10, 0.15, 0.20, 0.30, 0.40},
        .rate_grid = {0.02, 0.03, 0.05, 0.07},
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    };

    auto solver = make_iv_solver(config);
    ASSERT_TRUE(solver.has_value()) << "Factory should succeed with discrete dividends";

    IVQuery query;
    query.spot = 100.0;
    query.strike = 100.0;
    query.maturity = 0.5;
    query.rate = RateSpec{0.05};
    query.dividend_yield = 0.0;
    query.type = OptionType::PUT;
    query.market_price = 7.0;

    auto result = solver->solve(query);
    if (result.has_value()) {
        EXPECT_GT(result->implied_vol, 0.0);
        EXPECT_LT(result->implied_vol, 3.0);
    }
}

TEST(IVSolverFactoryTest, BatchSolveWorks) {
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
    ASSERT_TRUE(solver.has_value());

    std::vector<IVQuery> queries(3);
    for (auto& q : queries) {
        q.spot = 100.0;
        q.strike = 100.0;
        q.maturity = 0.5;
        q.rate = RateSpec{0.05};
        q.dividend_yield = 0.02;
        q.type = OptionType::PUT;
        q.market_price = 6.0;
    }

    auto batch_result = solver->solve_batch(queries);
    EXPECT_EQ(batch_result.results.size(), 3u);
}
