// SPDX-License-Identifier: MIT
/**
 * @file discrete_dividend_event_test.cc
 * @brief Tests for discrete dividend handling in AmericanOptionSolver
 *
 * Verifies that discrete dividends are correctly applied during PDE solve
 * by testing observable effects on option prices.
 */
#include <gtest/gtest.h>
#include "mango/option/american_option.hpp"
#include <cmath>

using namespace mango;

namespace {

// Helper: solve an American option with the given params
std::expected<AmericanOptionResult, SolverError> solve(const PricingParams& params) {
    return solve_american_option(params);
}

PricingParams make_put(double spot, double strike, double maturity,
                       double vol, double rate, double div_yield,
                       std::vector<Dividend> dividends = {}) {
    PricingParams p(
        OptionSpec{.spot = spot, .strike = strike, .maturity = maturity,
                   .rate = rate, .dividend_yield = div_yield,
                   .option_type = OptionType::PUT},
        vol);
    p.discrete_dividends = std::move(dividends);
    return p;
}

PricingParams make_call(double spot, double strike, double maturity,
                        double vol, double rate, double div_yield,
                        std::vector<Dividend> dividends = {}) {
    PricingParams p(
        OptionSpec{.spot = spot, .strike = strike, .maturity = maturity,
                   .rate = rate, .dividend_yield = div_yield,
                   .option_type = OptionType::CALL},
        vol);
    p.discrete_dividends = std::move(dividends);
    return p;
}

}  // namespace

TEST(DiscreteDividendTest, EventSnapshotsPreserveBothSidesWithoutChangingSolve) {
    for (auto type : {OptionType::PUT, OptionType::CALL}) {
        for (bool startup : {false, true}) {
            SCOPED_TRACE(static_cast<int>(type));
            SCOPED_TRACE(startup);
            auto params = make_put(100, 100, 1.0, 0.2, 0.05, 0.0,
                                  {{0.75, 2.0}, {0.5, 2.0}});
            params.option_type = type;
            PDEGridConfig grid{
                .grid_spec = GridSpec<double>::uniform(-1.0, 1.0, 81).value(),
                .n_time = 4,
            };
            // First event lands on the startup endpoint; second uses TR-BDF2.
            const std::vector<double> times{0.25, 0.5};
            auto ordinary = AmericanOptionSolver::create(params, PDEGridSpec{grid}, times);
            auto sided = AmericanOptionSolver::create(params, PDEGridSpec{grid}, times);
            ASSERT_TRUE(ordinary.has_value());
            ASSERT_TRUE(sided.has_value());
            TRBDF2Config method;
            method.rannacher_startup = startup;
            ordinary->set_trbdf2_config(method);
            sided->set_trbdf2_config(method);
            sided->set_before_event_snapshot_times(times);
            auto a = ordinary->solve();
            auto b = sided->solve();
            ASSERT_TRUE(a.has_value());
            ASSERT_TRUE(b.has_value());
            EXPECT_DOUBLE_EQ(a->value(), b->value());
            EXPECT_TRUE(a->grid()->before_event_snapshot_times().empty());
            ASSERT_EQ(b->grid()->before_event_snapshot_times().size(), times.size());
            // Independent solve of the post-dividend calendar regime. It has
            // no remaining dividends and ends just before the backward jump.
            auto remaining = params;
            remaining.maturity = times.front();
            remaining.discrete_dividends.clear();
            auto short_grid = grid;
            short_grid.n_time = 1;
            auto post_solver = AmericanOptionSolver::create(remaining, PDEGridSpec{short_grid});
            ASSERT_TRUE(post_solver.has_value());
            post_solver->set_trbdf2_config(method);
            auto post = post_solver->solve();
            ASSERT_TRUE(post.has_value());
            const auto first_side = b->grid()->at_before_events(0);
            ASSERT_EQ(first_side.size(), post->grid()->solution().size());
            for (size_t i = 0; i < first_side.size(); ++i) {
                EXPECT_NEAR(first_side[i], post->grid()->solution()[i], 1e-12);
            }
            for (size_t i = 0; i < a->grid()->solution().size(); ++i) {
                EXPECT_DOUBLE_EQ(a->grid()->solution()[i], b->grid()->solution()[i]);
                EXPECT_DOUBLE_EQ(a->grid()->solution_prev()[i], b->grid()->solution_prev()[i]);
            }
            for (size_t j = 0; j < times.size(); ++j) {
                auto before = b->grid()->at_before_events(j);
                auto after = b->at_time(j);
                auto x = b->grid()->x();
                ASSERT_EQ(before.size(), x.size());
                EXPECT_DOUBLE_EQ(b->grid()->before_event_snapshot_times()[j], times[j]);
                CubicSpline<double> continuation;
                ASSERT_FALSE(continuation.build(x, before).has_value());
                double max_jump = 0.0;
                for (size_t i = 0; i < x.size(); ++i) {
                    EXPECT_DOUBLE_EQ(a->at_time(j)[i], after[i]);
                    max_jump = std::max(max_jump, std::abs(after[i] - before[i]));
                    const double shifted_x = std::log(std::exp(x[i]) - 0.02);
                    if (i == 0 || i + 1 == x.size() || shifted_x < x.front()) continue;
                    const double exercise = intrinsic_value(100.0 * std::exp(x[i]), 100.0, type) / 100.0;
                    EXPECT_NEAR(after[i], std::max(exercise, continuation.eval(shifted_x)), 1e-13);
                }
                EXPECT_GT(max_jump, 1e-3);
            }
        }
    }
}

TEST(DiscreteDividendTest, PutValueIncreasesWithDividend) {
    // A discrete dividend lowers the effective spot, increasing put value
    auto no_div = solve(make_put(100, 100, 1.0, 0.20, 0.05, 0.0));
    auto with_div = solve(make_put(100, 100, 1.0, 0.20, 0.05, 0.0,
        {Dividend{.calendar_time = 0.25, .amount = 5.0}}));

    ASSERT_TRUE(no_div.has_value());
    ASSERT_TRUE(with_div.has_value());
    EXPECT_GT(with_div->value(), no_div->value())
        << "Put value should increase when a discrete dividend is present";
}

TEST(DiscreteDividendTest, CallValueDecreasesWithDividend) {
    // A discrete dividend lowers the effective spot, decreasing call value
    auto no_div = solve(make_call(100, 100, 1.0, 0.20, 0.05, 0.0));
    auto with_div = solve(make_call(100, 100, 1.0, 0.20, 0.05, 0.0,
        {Dividend{.calendar_time = 0.25, .amount = 5.0}}));

    ASSERT_TRUE(no_div.has_value());
    ASSERT_TRUE(with_div.has_value());
    EXPECT_LT(with_div->value(), no_div->value())
        << "Call value should decrease when a discrete dividend is present";
}

TEST(DiscreteDividendTest, ZeroDividendMatchesNoDividend) {
    auto no_div = solve(make_put(100, 100, 1.0, 0.20, 0.05, 0.0));
    auto zero_div = solve(make_put(100, 100, 1.0, 0.20, 0.05, 0.0,
        {Dividend{.calendar_time = 0.25, .amount = 0.0}}));

    ASSERT_TRUE(no_div.has_value());
    ASSERT_TRUE(zero_div.has_value());
    EXPECT_NEAR(zero_div->value(), no_div->value(), 1e-10)
        << "Zero dividend should produce same price as no dividend";
}

TEST(DiscreteDividendTest, LargerDividendIncreasesDeepITMPut) {
    // Deep ITM put with large dividend should push price closer to intrinsic
    auto small_div = solve(make_put(80, 100, 1.0, 0.20, 0.05, 0.0,
        {Dividend{.calendar_time = 0.5, .amount = 2.0}}));
    auto large_div = solve(make_put(80, 100, 1.0, 0.20, 0.05, 0.0,
        {Dividend{.calendar_time = 0.5, .amount = 10.0}}));

    ASSERT_TRUE(small_div.has_value());
    ASSERT_TRUE(large_div.has_value());
    EXPECT_GT(large_div->value(), small_div->value())
        << "Larger dividend should increase put value further";
}
