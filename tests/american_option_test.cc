// SPDX-License-Identifier: MIT
#include "mango/option/american_option.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/european_option.hpp"
#include "mango/option/detail/call_boundary_envelope.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

namespace mango {
namespace {

class AmericanOptionPricingTest : public ::testing::Test {
protected:
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

// ===========================================================================
// Regression tests for bugs found during code review
// ===========================================================================

// Regression: Solver results must not depend on prior solves on the same
// thread (issue #433).
// Bug: build_jacobian_boundaries never wrote jacobian.lower()[n-2] for a
// right Dirichlet BC, so the Thomas solve consumed stale bytes left in the
// reused thread-local workspace arena by a previous solve with a different
// grid size. A fresh thread's arena is zero-initialized (masking the bug),
// so the same solve gave different prices depending on thread history.
TEST(AmericanOptionTest, PriceIndependentOfThreadWorkspaceHistory) {
    // A CALL on a narrow grid makes the bug visible: the right Dirichlet
    // value g = e^x - e^{-r tau} is O(1) there (for a put it is ~0, so the
    // stale sub-diagonal multiplies near-zero values and the error hides),
    // and psi(x_max) = e^0.6 - 1 < 0.95 keeps deep-ITM locking from
    // rewriting the rows adjacent to the boundary.
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.05, .dividend_yield = 0.0,
                   .option_type = OptionType::CALL},
        0.20);

    auto solve_price = [&params](size_t n_space) {
        PDEGridConfig grid{GridSpec<double>::uniform(-0.6, 0.6, n_space).value(), 200};
        auto solver = AmericanOptionSolver::create(params, PDEGridSpec{grid});
        EXPECT_TRUE(solver.has_value());
        auto result = solver->solve();
        EXPECT_TRUE(result.has_value());
        return result->value_at(params.spot);
    };

    // Reference price from a fresh thread: zero-initialized TLS arena.
    double clean_price = 0.0;
    std::thread([&] { clean_price = solve_price(201); }).join();

    // Same solve on another fresh thread, after a different-sized solve
    // dirtied that thread's arena.
    double dirty_price = 0.0;
    std::thread([&] {
        (void)solve_price(301);
        dirty_price = solve_price(201);
    }).join();

    EXPECT_EQ(clean_price, dirty_price)
        << "identical solve returned a different price depending on "
           "thread-local workspace history (stale jacobian.lower()[n-2])";
}

// Regression: no-dividend American call must equal the European price and
// never fall below intrinsic (issue #432).
// Bug: the deep-ITM lock used an absolute threshold psi > 0.95, derived for
// the bounded put payoff (psi <= 1  =>  x < -3). The call payoff e^x - 1 is
// unbounded and crosses 0.95 at S > 1.95K, so moderately ITM calls had
// continuation-valued nodes permanently ratcheted to intrinsic — the price
// came out below intrinsic (arbitrage violation).
TEST(AmericanOptionTest, NoDividendCallEqualsEuropean) {
    for (double spot : {150.0, 200.0, 300.0}) {
        PricingParams params(
            OptionSpec{.spot = spot, .strike = 100.0, .maturity = 1.0,
                       .rate = 0.05, .dividend_yield = 0.0,
                       .option_type = OptionType::CALL},
            0.20);

        auto result = solve_american_option(params);
        ASSERT_TRUE(result.has_value()) << "spot=" << spot;

        const double american = result->value_at(spot);
        const double european = EuropeanOptionResult(params).value();
        const double intrinsic = spot - params.strike;

        EXPECT_GE(american, intrinsic - 1e-6) << "spot=" << spot;
        EXPECT_NEAR(american, european, 0.05) << "spot=" << spot;
    }
}

// Regression: with r=0 early exercise of a put is never optimal, so the
// American put equals the European put — even deep ITM (issue #432).
// Bug: the deep-ITM lock clamped every node with psi > 0.95 to intrinsic
// regardless of whether holding the payoff loses value (L(psi) <= 0), so a
// zero-rate deep-ITM put lost its continuation value.
TEST(AmericanOptionTest, ZeroRateDeepITMPutEqualsEuropean) {
    PricingParams params(
        OptionSpec{.spot = 4.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.0, .dividend_yield = 0.04,
                   .option_type = OptionType::PUT},
        0.20);

    auto result = solve_american_option(params);
    ASSERT_TRUE(result.has_value());

    const double american = result->value_at(params.spot);
    const double european = EuropeanOptionResult(params).value();

    EXPECT_GE(american, european - 0.05);
    EXPECT_NEAR(american, european, 0.05);
}

// Regression/spec test for the public complementarity report (issue #439).
// An ATM put solve is an M-matrix regime end-to-end, so a clean solve must
// report zero KKT violations. This is the strongest single assertion that
// the new put sweep is exact; if it fails, the sweep is still wrong and the
// assertion must not be loosened.
TEST(AmericanOptionTest, ComplementarityReportCleanForATMPut) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.05, .dividend_yield = 0.02,
                   .option_type = OptionType::PUT},
        0.20);

    auto solver = AmericanOptionSolver::create(params);
    ASSERT_TRUE(solver.has_value());

    auto result = solver->solve();
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(solver->complementarity_report().violation_count, 0u)
        << "max_violation=" << solver->complementarity_report().max_violation
        << " worst_kind=" << solver->complementarity_report().worst_kind;
}

// ===========================================================================
// Regression tests: call right-boundary stopping envelope (#439 item 2 / B5)
// ===========================================================================

// Regression: custom time grids must include dividend taus (#439 batch, B5)
// Bug: process_temporal_events fires an event only at a completed grid step,
// so a custom grid whose mandatory_times omit the ex-date applies the
// dividend jump to a state that has already evolved past the true ex-date
// by up to one step -- an O(dt) phase error, largest on coarse grids. Both
// prices below use the IDENTICAL spatial grid (from estimate_pde_grid) so
// the comparison isolates the time-grid merge; n_time=27 (dt≈0.037) puts
// the tau=0.5 dividend roughly half a step off a uniform 27-step grid.
// Measured on this branch: pre-fix (mandatory_times not merged) delta =
// 5.53e-3 (FAILS a 5e-3 bound); post-fix (this change) delta = 2.87e-3
// (residual is ordinary n_time-vs-auto-grid time-discretization difference,
// not a phase bug -- both grids now land exactly on the dividend tau).
TEST(AmericanOptionTest, CustomGridOmittingDividendDateStillAligns) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.05, .option_type = OptionType::PUT},
        0.20);
    params.discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 3.0}};

    // (a) Auto grid: estimate_pde_grid already merges dividend taus.
    auto auto_result = solve_american_option(params);
    ASSERT_TRUE(auto_result.has_value());
    const double price_auto = auto_result->value_at(params.spot);

    // (b) Same spatial grid as (a), but routed through the PDEGridConfig
    // (custom-grid) path with an EMPTY mandatory_times list, at a
    // deliberately non-divisor n_time so the dividend tau cannot land on
    // the grid by coincidence.
    auto grid_pair = estimate_pde_grid(params);
    PDEGridConfig custom_cfg{grid_pair.first, 27, {}};

    auto solver = AmericanOptionSolver::create(params, PDEGridSpec{custom_cfg});
    ASSERT_TRUE(solver.has_value());
    auto custom_result = solver->solve();
    ASSERT_TRUE(custom_result.has_value());
    const double price_custom = custom_result->value_at(params.spot);

    EXPECT_NEAR(price_custom, price_auto, 5e-3)
        << "price_auto=" << price_auto << " price_custom=" << price_custom
        << " delta=" << std::abs(price_custom - price_auto);
}

// Regression: dividend-free call right BC unchanged (#439 item 2 guard)
// Bug guard: replacing the naive `e^x - forward_discount(t)` right BC with
// the general stopping-value envelope must be a no-op when there are no
// discrete dividends and q=0 -- pinned to the value measured on this branch
// immediately before the envelope wiring landed (bazel run of a throwaway
// solve at the same params gave 10.447090628631905 both before and after).
TEST(AmericanOptionTest, NoDivCallPriceUnchangedByEnvelopeBC) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.05, .dividend_yield = 0.0,
                   .option_type = OptionType::CALL},
        0.20);

    auto result = solve_american_option(params);
    ASSERT_TRUE(result.has_value());

    constexpr double kPinnedPrice = 10.447090628631905;
    EXPECT_NEAR(result->value_at(params.spot), kPinnedPrice, 1e-12);
}

// Regression: dividend-paying call right BC no longer pinned high (#439
// item 2). Direct envelope check plus a full solve.
// Bug: the old right BC `e^x - forward_discount(t)` ignored discrete
// dividends entirely, overstating the boundary value (and hence any query
// point close enough to the boundary to feel it) by the discounted dividend
// once the ex-date is in the option's remaining life.
TEST(AmericanOptionTest, DiscreteDivCallRightBoundaryEnvelope) {
    using mango::detail::CallBoundaryEnvelope;
    constexpr size_t kTimeBased = std::numeric_limits<size_t>::max();

    const double K = 100.0;
    const double x_max = std::log(2.0);   // deep ITM, S = 2K
    const double maturity = 1.0;
    const double r = 0.05;
    const double D_over_K = 1.5 / K;
    const double calendar_div = 0.25;
    const double tau_d = maturity - calendar_div;  // 0.75

    CallBoundaryEnvelope no_div{
        .x_max = x_max, .dividend_yield = 0.0, .maturity = maturity,
        .rate = RateSpec{r}, .dividends = {}};
    CallBoundaryEnvelope with_div{
        .x_max = x_max, .dividend_yield = 0.0, .maturity = maturity,
        .rate = RateSpec{r},
        .dividends = {Dividend{.calendar_time = calendar_div, .amount = D_over_K}}};

    // Just before the ex-date crosses into the remaining set (tau < tau_d,
    // strict-< phase rule of B1): the dividend has not yet entered the
    // option's remaining life as of this evaluation, so the envelope must
    // match the no-dividend case exactly.
    const double tau_before = tau_d - 0.01;
    EXPECT_NEAR(with_div.value(tau_before, kTimeBased),
                no_div.value(tau_before, kTimeBased), 1e-12);

    // Just after the ex-date (tau > tau_d): the dividend is now in the
    // remaining set, and the deep-ITM "hold to expiry" candidate (s=0) must
    // be reduced by exactly the dividend's forward-discounted value.
    // At this deep-ITM point with r>0 and q=0 the hold-to-expiry candidate
    // dominates both "stop now" (=intrinsic) and "stop at expiry" for both
    // envelopes, so the closed-form difference is exact, not approximate.
    const double tau_after = tau_d + 0.01;
    const double expected_drop = D_over_K * std::exp(-r * (tau_after - tau_d));
    EXPECT_NEAR(with_div.value(tau_after, kTimeBased),
                no_div.value(tau_after, kTimeBased) - expected_drop, 1e-12);

    // Full solve sanity: the discrete-dividend call must price below its
    // no-dividend counterpart (the boundary and interior both now see the
    // dividend), stay at or above intrinsic, and remain finite/converged.
    // Accuracy against QuantLib for this exact scenario shape (single
    // discrete dividend, ATM call) is covered by
    // DiscreteDividendAccuracyTest.CallSingleDividendVsQuantLib
    // (tests/discrete_dividend_accuracy_test.cc), which this change must
    // keep passing at rel_err < 1% -- verified separately as part of this
    // task's required test run rather than duplicated here (avoids adding
    // a QuantLib dependency to this target).
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = K, .maturity = maturity,
                   .rate = r, .dividend_yield = 0.0,
                   .option_type = OptionType::CALL},
        0.20);
    auto no_div_result = solve_american_option(params);
    params.discrete_dividends = {Dividend{.calendar_time = calendar_div, .amount = 1.5}};
    auto with_div_result = solve_american_option(params);

    ASSERT_TRUE(no_div_result.has_value());
    ASSERT_TRUE(with_div_result.has_value());
    EXPECT_TRUE(with_div_result->converged);
    EXPECT_LT(with_div_result->value_at(params.spot), no_div_result->value_at(params.spot));
    EXPECT_GE(with_div_result->value_at(params.spot),
              std::max(params.spot - params.strike, 0.0) - 1e-6);
}

}  // namespace
}  // namespace mango
