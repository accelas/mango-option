// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/american_option.hpp"
#include "src/option/american_option_batch.hpp"
#include <cmath>
#include <memory_resource>
#include <ql/quantlib.hpp>

using namespace mango;
namespace ql = QuantLib;

// ============================================================================
// QuantLib reference: American option with discrete cash dividends
// ============================================================================

namespace {

double price_american_discrete_div_quantlib(
    double spot, double strike, double maturity,
    double volatility, double rate,
    const std::vector<Dividend>& dividends,
    bool is_call,
    size_t grid_steps = 401,
    size_t time_steps = 4000)
{
    ql::Date today = ql::Date::todaysDate();
    ql::Settings::instance().evaluationDate() = today;

    auto option_type = is_call ? ql::Option::Call : ql::Option::Put;
    ql::Date maturity_date = today + ql::Period(
        static_cast<int>(maturity * 365), ql::Days);

    auto exercise = ql::ext::make_shared<ql::AmericanExercise>(today, maturity_date);
    auto payoff = ql::ext::make_shared<ql::PlainVanillaPayoff>(option_type, strike);

    ql::VanillaOption option(payoff, exercise);

    // Convert dividend schedule to QuantLib DividendSchedule
    ql::DividendSchedule div_schedule;
    for (const auto& [t_cal, amount] : dividends) {
        if (t_cal > 0.0 && t_cal < maturity) {
            ql::Date div_date = today + ql::Period(
                static_cast<int>(t_cal * 365), ql::Days);
            div_schedule.push_back(
                ql::ext::make_shared<ql::FixedDividend>(amount, div_date));
        }
    }

    // Market data (zero continuous dividend yield — discrete only)
    ql::Handle<ql::Quote> spot_h(ql::ext::make_shared<ql::SimpleQuote>(spot));
    ql::Handle<ql::YieldTermStructure> rate_ts(
        ql::ext::make_shared<ql::FlatForward>(today, rate, ql::Actual365Fixed()));
    ql::Handle<ql::YieldTermStructure> div_ts(
        ql::ext::make_shared<ql::FlatForward>(today, 0.0, ql::Actual365Fixed()));
    ql::Handle<ql::BlackVolTermStructure> vol_ts(
        ql::ext::make_shared<ql::BlackConstantVol>(
            today, ql::NullCalendar(), volatility, ql::Actual365Fixed()));

    auto process = ql::ext::make_shared<ql::BlackScholesMertonProcess>(
        spot_h, div_ts, rate_ts, vol_ts);

    // FD engine with discrete dividend schedule
    option.setPricingEngine(
        ql::ext::make_shared<ql::FdBlackScholesVanillaEngine>(
            process, std::move(div_schedule), time_steps, grid_steps));

    return option.NPV();
}

// Helper: solve with mango at a given accuracy profile
double solve_mango(const PricingParams& params,
                   const GridAccuracyParams& accuracy = GridAccuracyParams{}) {
    auto [grid_spec, time_domain] = estimate_pde_grid(params, accuracy);
    size_t n = grid_spec.n_points();
    std::pmr::vector<double> buffer(
        PDEWorkspace::required_size(n), std::pmr::get_default_resource());
    auto ws = PDEWorkspace::from_buffer(buffer, n).value();
    auto solver = AmericanOptionSolver::create(params, ws,
                                PDEGridConfig{grid_spec, time_domain.n_steps(), {}}).value();
    auto result = solver.solve();
    return result->value();
}

}  // namespace

// ============================================================================
// QuantLib comparison tests
// ============================================================================

TEST(DiscreteDividendAccuracyTest, PutSingleDividendVsQuantLib) {
    // ATM put, S=100, K=100, T=1, sigma=0.20, r=0.05
    // Single $3 dividend at t=0.5
    PricingParams params(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20,
                         {{.calendar_time = 0.5, .amount = 3.0}});

    double mango_price = solve_mango(params, make_grid_accuracy(GridAccuracyProfile::Medium));
    double ql_price = price_american_discrete_div_quantlib(
        100.0, 100.0, 1.0, 0.20, 0.05, {{0.5, 3.0}}, false);

    double rel_err = std::abs(mango_price - ql_price) / ql_price;
    EXPECT_LT(rel_err, 0.01)
        << "mango=" << mango_price << " ql=" << ql_price
        << " rel_err=" << rel_err * 100 << "%";
}

TEST(DiscreteDividendAccuracyTest, CallSingleDividendVsQuantLib) {
    // ATM call, S=100, K=100, T=1, sigma=0.25, r=0.05
    // Single $4 dividend at t=0.3
    PricingParams params(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::CALL}, 0.25,
                         {{.calendar_time = 0.3, .amount = 4.0}});

    double mango_price = solve_mango(params, make_grid_accuracy(GridAccuracyProfile::Medium));
    double ql_price = price_american_discrete_div_quantlib(
        100.0, 100.0, 1.0, 0.25, 0.05, {{0.3, 4.0}}, true);

    double rel_err = std::abs(mango_price - ql_price) / ql_price;
    EXPECT_LT(rel_err, 0.01)
        << "mango=" << mango_price << " ql=" << ql_price
        << " rel_err=" << rel_err * 100 << "%";
}

TEST(DiscreteDividendAccuracyTest, MultipleDividendsVsQuantLib) {
    // ITM put, two dividends
    PricingParams params(OptionSpec{.spot = 95.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20,
                         {{.calendar_time = 0.25, .amount = 2.0}, {.calendar_time = 0.75, .amount = 2.0}});

    double mango_price = solve_mango(params, make_grid_accuracy(GridAccuracyProfile::Medium));
    double ql_price = price_american_discrete_div_quantlib(
        95.0, 100.0, 1.0, 0.20, 0.05, {{0.25, 2.0}, {0.75, 2.0}}, false);

    double rel_err = std::abs(mango_price - ql_price) / ql_price;
    EXPECT_LT(rel_err, 0.01)
        << "mango=" << mango_price << " ql=" << ql_price
        << " rel_err=" << rel_err * 100 << "%";
}

TEST(DiscreteDividendAccuracyTest, LargeDividendVsQuantLib) {
    // Large dividend: $15 on S=100 (15% of spot)
    PricingParams params(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.30,
                         {{.calendar_time = 0.5, .amount = 15.0}});

    double mango_price = solve_mango(params, make_grid_accuracy(GridAccuracyProfile::Medium));
    double ql_price = price_american_discrete_div_quantlib(
        100.0, 100.0, 1.0, 0.30, 0.05, {{0.5, 15.0}}, false);

    // Larger dividend → larger model differences, relax to 2%
    double rel_err = std::abs(mango_price - ql_price) / ql_price;
    EXPECT_LT(rel_err, 0.02)
        << "mango=" << mango_price << " ql=" << ql_price
        << " rel_err=" << rel_err * 100 << "%";
}

TEST(DiscreteDividendAccuracyTest, DividendNearExpiryVsQuantLib) {
    // Dividend close to expiry (t=0.9, T=1.0)
    PricingParams params(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20,
                         {{.calendar_time = 0.9, .amount = 2.0}});

    double mango_price = solve_mango(params, make_grid_accuracy(GridAccuracyProfile::Medium));
    double ql_price = price_american_discrete_div_quantlib(
        100.0, 100.0, 1.0, 0.20, 0.05, {{0.9, 2.0}}, false);

    double rel_err = std::abs(mango_price - ql_price) / ql_price;
    EXPECT_LT(rel_err, 0.01)
        << "mango=" << mango_price << " ql=" << ql_price
        << " rel_err=" << rel_err * 100 << "%";
}

// ============================================================================
// Grid convergence: prices should converge as resolution increases
// ============================================================================

TEST(DiscreteDividendAccuracyTest, GridConvergence) {
    PricingParams params(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20,
                         {{.calendar_time = 0.5, .amount = 3.0}});

    double p_low  = solve_mango(params, make_grid_accuracy(GridAccuracyProfile::Low));
    double p_med  = solve_mango(params, make_grid_accuracy(GridAccuracyProfile::Medium));
    double p_high = solve_mango(params, make_grid_accuracy(GridAccuracyProfile::High));

    // Successive differences should shrink
    double diff_1 = std::abs(p_med - p_low);
    double diff_2 = std::abs(p_high - p_med);

    EXPECT_LT(diff_2, diff_1)
        << "Grid convergence: |high-med|=" << diff_2
        << " should be < |med-low|=" << diff_1;

    // High and medium should agree to within 0.1%
    double rel = std::abs(p_high - p_med) / p_high;
    EXPECT_LT(rel, 0.001)
        << "High and medium grids should agree to <0.1%: rel=" << rel * 100 << "%";
}

// ============================================================================
// Structural tests (no QuantLib dependency)
// ============================================================================

TEST(DiscreteDividendAccuracyTest, EventAlignsWithMandatoryTimePoint) {
    PricingParams params(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20,
                         {{.calendar_time = 0.3, .amount = 2.0}});

    auto [grid_spec, td] = estimate_pde_grid(params);
    auto pts = td.time_points();

    double tau_div = 0.7;  // tau = T - t_cal = 1.0 - 0.3
    bool found = false;
    for (double p : pts) {
        if (std::abs(p - tau_div) < 1e-14) { found = true; break; }
    }
    EXPECT_TRUE(found) << "Time grid must land exactly on dividend tau=" << tau_div;
}

TEST(DiscreteDividendAccuracyTest, DividendAtBoundariesIgnored) {
    PricingParams params(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20,
                         {{.calendar_time = 0.0, .amount = 5.0}, {.calendar_time = 1.0, .amount = 5.0}});

    auto result = solve_american_option(params);
    ASSERT_TRUE(result.has_value());

    PricingParams no_div(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20);
    auto result_no_div = solve_american_option(no_div);
    ASSERT_TRUE(result_no_div.has_value());

    EXPECT_NEAR(result->value(), result_no_div->value(), 1e-10)
        << "Boundary dividends should be ignored";
}

TEST(DiscreteDividendAccuracyTest, SharedGridBatchIncludesDividendTimePoints) {
    std::vector<PricingParams> batch;
    batch.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20,
                       std::vector<Dividend>{{.calendar_time = 0.4, .amount = 3.0}}));
    batch.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = 110.0, .maturity = 1.0, .rate = 0.05, .option_type = OptionType::PUT}, 0.20,
                       std::vector<Dividend>{{.calendar_time = 0.4, .amount = 3.0}}));

    auto [grid_spec, td] = estimate_batch_pde_grid(batch);
    auto pts = td.time_points();

    double tau_div = 0.6;  // tau = T - t_cal = 1.0 - 0.4
    bool found = false;
    for (double p : pts) {
        if (std::abs(p - tau_div) < 1e-14) { found = true; break; }
    }
    EXPECT_TRUE(found) << "Shared grid time domain must land on dividend tau=" << tau_div;
}
