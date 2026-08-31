// SPDX-License-Identifier: MIT
// Regression: put sweep orientation (#439).
//
// The implicit-stage projected solve used the same Brennan-Schwartz sweep
// direction for both puts (left-active obstacle) and calls (right-active
// obstacle). Only one direction is exact for a given active-set shape, so
// puts picked up a systematic error from sweeping the wrong way. Task 2 on
// this branch fixed puts to use the mirrored sweep.
//
// This test pins mango-option's American pricing accuracy against a
// high-resolution QuantLib finite-difference reference (8000 time steps,
// 801 space steps -- 4x the framework default) for a mix of put and call
// scenarios. The put thresholds are tight enough that the old (unmirrored)
// sweep fails them -- e.g. ATM put |err| was 6.9e-3 under the old sweep,
// against a 5.5e-3 threshold measured on the mirrored sweep at 4.4e-3.
// Call thresholds were never affected by the sweep bug; they pin current
// accuracy so future regressions in the call path are caught too.
//
// Bug: solve_implicit_stage_projected used one fixed sweep direction
// (exact for right-active/call-like obstacles) for both option types,
// instead of selecting the sweep exact for the option's active-set shape.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>

#include <ql/quantlib.hpp>

#include "mango/option/american_option.hpp"

namespace ql = QuantLib;
using namespace mango;

namespace {

// High-resolution QuantLib American reference price. 8000x801 grid, 4x the
// framework default (see quantlib_validation_framework.hpp), so the
// reference itself is accurate well below the thresholds pinned here.
double ql_american(double spot, double strike, double maturity, double vol,
                    double rate, double q, bool is_call) {
    ql::Date today = ql::Date::todaysDate();
    ql::Settings::instance().evaluationDate() = today;
    ql::Option::Type type = is_call ? ql::Option::Call : ql::Option::Put;
    // int(maturity * 365) truncates to a whole number of days, so a
    // maturity that isn't an exact multiple of 1/365 gets a QuantLib
    // reference for a slightly different T than `maturity` itself. The
    // T=0.25 row ("put OTM T.25") is the one case in kRows this bites:
    // int(0.25*365) = 91 days = 91/365 = 0.24932 (not 0.25), so that row's
    // reference is priced at T=0.24932 -- the resulting cross-convention
    // mismatch (~3e-4 in T) is already folded into that row's measured
    // abs_err and threshold, not a separate error source to account for.
    ql::Date mat = today + ql::Period(int(maturity * 365), ql::Days);
    auto exercise = ql::ext::make_shared<ql::AmericanExercise>(today, mat);
    auto payoff = ql::ext::make_shared<ql::PlainVanillaPayoff>(type, strike);
    ql::VanillaOption opt(payoff, exercise);
    ql::Handle<ql::Quote> s(ql::ext::make_shared<ql::SimpleQuote>(spot));
    ql::Handle<ql::YieldTermStructure> r_ts(
        ql::ext::make_shared<ql::FlatForward>(today, rate,
                                               ql::Actual365Fixed()));
    ql::Handle<ql::YieldTermStructure> q_ts(
        ql::ext::make_shared<ql::FlatForward>(today, q,
                                               ql::Actual365Fixed()));
    ql::Handle<ql::BlackVolTermStructure> v_ts(
        ql::ext::make_shared<ql::BlackConstantVol>(today, ql::NullCalendar(),
                                                     vol, ql::Actual365Fixed()));
    auto proc = ql::ext::make_shared<ql::BlackScholesMertonProcess>(s, q_ts,
                                                                     r_ts, v_ts);
    opt.setPricingEngine(ql::ext::make_shared<ql::FdBlackScholesVanillaEngine>(
        proc, 8000, 801));
    return opt.NPV();
}

double mango_american(double spot, double strike, double maturity, double vol,
                       double rate, double q, bool is_call) {
    PricingParams params(
        OptionSpec{.spot = spot, .strike = strike, .maturity = maturity,
                   .rate = rate, .dividend_yield = q,
                   .option_type = is_call ? OptionType::CALL : OptionType::PUT},
        vol);
    auto solver = AmericanOptionSolver::create(params).value();
    auto result = solver.solve();
    EXPECT_TRUE(result.has_value());
    return result->value_at(spot);
}

struct Row {
    const char* name;
    double S, K, T, vol, r, q;
    bool call;
    double max_abs_err;
};

// Scenarios and thresholds from the task spec: absolute error vs the
// 8000x801 QuantLib reference. Put thresholds tighten on measured
// mirrored-sweep errors (spike table) by roughly a 1.25x margin.
const Row kRows[] = {
    {"put ATM",        100, 100, 1.0, .20, .05, .00, false, 5.5e-3},
    {"put ITM S90",     90, 100, 1.0, .20, .05, .00, false, 3.0e-3},
    {"put ITM S80",     80, 100, 1.0, .20, .05, .00, false, 1.0e-3},
    {"put deep S70",    70, 100, 1.0, .20, .05, .00, false, 5.0e-4},
    {"put nearFB r8",   85, 100, 1.0, .20, .08, .00, false, 6.5e-3},
    {"put OTM T.25",   110, 100, 0.25, .30, .05, .00, false, 5.0e-3},
    {"put T2 v25",      90, 100, 2.0, .25, .05, .00, false, 3.0e-3},
    {"put q2",         100, 100, 1.0, .20, .05, .02, false, 6.0e-3},
    {"call ATM q8",    100, 100, 1.0, .20, .05, .08, true,  7.5e-3},
    {"call S120 q8",   120, 100, 1.0, .20, .05, .08, true,  7.5e-3},
    {"call S130 q8",   130, 100, 1.0, .20, .05, .08, true,  7.5e-3},
    {"call S150 q8",   150, 100, 1.0, .20, .05, .08, true,  7.5e-3},
    {"call S200 q8",   200, 100, 1.0, .20, .05, .08, true,  7.5e-3},
    {"call S300 q8",   300, 100, 1.0, .20, .05, .08, true,  7.5e-3},
    {"call S120 q4r2", 120, 100, 2.0, .25, .02, .04, true,  7.5e-3},
};

}  // namespace

TEST(QuantLibSweepRegression, PricingAccuracyAcrossPutsAndCalls) {
    printf("%-16s %12s %12s %12s %10s %10s  %s\n", "scenario", "quantlib",
           "mango", "abs_err", "threshold", "margin", "result");
    bool any_failed = false;
    for (const auto& row : kRows) {
        SCOPED_TRACE(row.name);
        double ref = ql_american(row.S, row.K, row.T, row.vol, row.r, row.q,
                                  row.call);
        double mango = mango_american(row.S, row.K, row.T, row.vol, row.r,
                                       row.q, row.call);
        double abs_err = std::abs(mango - ref);
        bool pass = abs_err <= row.max_abs_err;
        any_failed |= !pass;
        printf("%-16s %12.6f %12.6f %12.3e %10.3e %10.3e  %s\n", row.name, ref,
               mango, abs_err, row.max_abs_err, row.max_abs_err - abs_err,
               pass ? "PASS" : "FAIL");
        EXPECT_LE(abs_err, row.max_abs_err)
            << row.name << ": |mango(" << mango << ") - quantlib(" << ref
            << ")| = " << abs_err << " exceeds threshold "
            << row.max_abs_err;
    }
    EXPECT_FALSE(any_failed);
}
