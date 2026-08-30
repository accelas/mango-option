// SPDX-License-Identifier: MIT
#include "mango/option/detail/call_boundary_envelope.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

using mango::Dividend;
using mango::RateSpec;
using mango::YieldCurve;
using mango::detail::CallBoundaryEnvelope;

constexpr size_t kTimeBased = std::numeric_limits<size_t>::max();

// Discount factor between backward times a and b: DF(a,b) = D_cal(T-b)/D_cal(T-a).
double DF_calendar(const RateSpec& rate, double T, double a, double b) {
    auto discount_cal = [&](double t) -> double {
        return std::visit(
            [t](const auto& r) -> double {
                using R = std::decay_t<decltype(r)>;
                if constexpr (std::is_same_v<R, double>) {
                    return std::exp(-r * t);
                } else {
                    return r.discount(t);
                }
            },
            rate);
    };
    return discount_cal(T - b) / discount_cal(T - a);
}

// Phase-aware remaining-dividend resolution, mirroring the envelope's own rule.
std::vector<Dividend> remaining_at(const std::vector<Dividend>& divs, double T, double tau,
                                    size_t n_applied) {
    std::vector<Dividend> out;
    if (n_applied == kTimeBased) {
        for (const auto& d : divs) {
            if (T - d.calendar_time < tau) out.push_back(d);
        }
    } else {
        size_t m = divs.size();
        size_t n = std::min(n_applied, m);
        for (size_t i = m - n; i < m; ++i) out.push_back(divs[i]);
    }
    return out;
}

// Brute-force oracle: dense scan of the stopping value f(s) over s in [0, tau].
// Independent of the envelope's segment-walk implementation -- agreement is
// the proof the candidate enumeration is complete.
double dense_scan_max(const CallBoundaryEnvelope& env, double tau, size_t n_applied) {
    auto remaining = remaining_at(env.dividends, env.maturity, tau, n_applied);
    double A = std::exp(env.x_max);
    double best = A - 1.0;  // intrinsic floor
    constexpr int N = 20000;
    for (int i = 0; i <= N; ++i) {
        double s = tau * static_cast<double>(i) / N;
        double sum = 0.0;
        for (const auto& d : remaining) {
            double tau_i = env.maturity - d.calendar_time;
            if (tau_i > s) {
                sum += d.amount * DF_calendar(env.rate, env.maturity, tau, tau_i) *
                       std::exp(-env.dividend_yield * (tau_i - s));
            }
        }
        double f = A * std::exp(-env.dividend_yield * (tau - s)) - sum -
                   DF_calendar(env.rate, env.maturity, tau, s);
        best = std::max(best, f);
    }
    return best;
}

}  // namespace

TEST(Envelope, NoDivZeroQReducesToForwardDiscount) {
    double T = 1.0;
    CallBoundaryEnvelope env{0.6, 0.0, T, RateSpec{0.05}, {}};

    for (double tau : {0.1, 0.4, 0.75, 1.0}) {
        double expected = std::max(std::exp(env.x_max) - 1.0,
                                    std::exp(env.x_max) - std::exp(-0.05 * tau));
        EXPECT_NEAR(env.value(tau, kTimeBased), expected, 1e-12) << "tau=" << tau;
    }
}

TEST(Envelope, TauZeroIsIntrinsic) {
    CallBoundaryEnvelope env{0.8, 0.02, 1.0, RateSpec{0.05}, {}};
    double expected = std::exp(0.8) - 1.0;
    EXPECT_NEAR(env.value(0.0, 0), expected, 1e-15);
}

TEST(Envelope, StrictPhasePredicate) {
    double T = 1.0;
    double tau_j = 0.4;
    double calendar_j = T - tau_j;
    Dividend div{calendar_j, 0.02};

    CallBoundaryEnvelope with_div{0.5, 0.0, T, RateSpec{0.05}, {div}};
    CallBoundaryEnvelope without_div{0.5, 0.0, T, RateSpec{0.05}, {}};

    const double eps = 1e-6;
    // Just below tau_j: dividend not yet in remaining life -> excluded.
    EXPECT_NEAR(with_div.value(tau_j - eps, kTimeBased),
                without_div.value(tau_j - eps, kTimeBased), 1e-9);
    // Just above tau_j: dividend now in remaining life -> included, so the
    // two envelopes must disagree materially.
    double included_diff = std::abs(with_div.value(tau_j + eps, kTimeBased) -
                                     without_div.value(tau_j + eps, kTimeBased));
    EXPECT_GT(included_diff, 1e-6);
}

TEST(Envelope, EpochCounterOverridesTime) {
    double T = 1.0;
    double tau_j = 0.4;
    double calendar_j = T - tau_j;
    Dividend div{calendar_j, 0.02};

    CallBoundaryEnvelope with_div{0.5, 0.0, T, RateSpec{0.05}, {div}};

    // At tau == tau_j exactly: n=0 must exclude the dividend (matches the
    // time-based strict predicate, which also excludes at equality).
    double excluded = with_div.value(tau_j, 0);
    double time_based_at_tau_j = with_div.value(tau_j, kTimeBased);
    EXPECT_NEAR(excluded, time_based_at_tau_j, 1e-9);

    // n=1 must include it, materially changing the result.
    double included = with_div.value(tau_j, 1);
    EXPECT_GT(std::abs(included - excluded), 1e-6);
}

TEST(Envelope, IntermediateExDateDominates) {
    double T = 1.0;
    std::vector<Dividend> divs = {
        {0.3, 0.03},
        {0.7, 0.02},
    };
    CallBoundaryEnvelope env{0.7, 0.0, T, RateSpec{0.06}, divs};
    // tau == T: chosen (along with the dividend dates) so both ex-dates
    // land exactly on the dense scan's grid -- the true optimum sits right
    // at the later ex-date (a kink), and an off-grid sample would
    // understate it by far more than the 1e-9 tolerance below.
    double tau = 1.0;

    double got = env.value(tau, kTimeBased);
    double want = dense_scan_max(env, tau, kTimeBased);
    EXPECT_NEAR(got, want, 1e-9);
}

// Flat rate, no dividends: f(s) = A*e^{-q(tau-s)} - e^{-r(tau-s)}. With
// r > q > 0 this is B*e^{qs} - C*e^{rs}, which has a genuine interior
// maximum when B, C > 0. Parameters below (T=1.5, x=0.5, q=3%, r=5%,
// tau=1.0) put that interior optimum at s* ~= 0.4587, strictly dominating
// both endpoints: f(s*) ~= 0.648866 > f(0) ~= 0.648765 > f(tau) ~= 0.648721.
// (An earlier version of this test used tau=1.6 with r < q effectively
// reversed, for which s* fell outside (0, tau) and the envelope degenerated
// to the endpoint candidate -- verified below via explicit margins so that
// regression can't reoccur silently.)
TEST(Envelope, FlatRateInteriorStationaryPoint) {
    double T = 1.5;
    CallBoundaryEnvelope env{0.5, 0.03, T, RateSpec{0.05}, {}};
    double tau = 1.0;

    double got = env.value(tau, kTimeBased);
    double want = dense_scan_max(env, tau, kTimeBased);
    EXPECT_NEAR(got, want, 1e-9);

    // The interior stationary point must strictly dominate both endpoints
    // -- otherwise this test would silently degenerate to an endpoint-only
    // check and never exercise the closed-form stationary-point branch.
    // f(s) is evaluated directly at the two candidate stopping times s=0
    // (hold to expiry) and s=tau (exercise now); NOTE this is deliberately
    // NOT `env.value(0, ...)` / `env.value(tau, ...)` -- those re-evaluate
    // the whole envelope at a *different* outer solver time, which is a
    // different quantity (e.g. value(0, ...) is always intrinsic by
    // definition, not f(0) within *this* tau=1.0 evaluation).
    constexpr double kMargin = 5e-5;
    double f0 = std::exp(env.x_max) * std::exp(-env.dividend_yield * tau) -
                DF_calendar(env.rate, env.maturity, tau, 0.0);
    double f_tau = std::exp(env.x_max) - 1.0;  // DF(tau,tau)=1, sum term vanishes at s=tau
    EXPECT_GT(got - f0, kMargin);
    EXPECT_GT(got - f_tau, kMargin);
}

// Two-segment yield curve: forward rate 5% for calendar [0, 0.6] (the
// "near-now" backward-time segment s in (0.4, 1.0]) and 2% for calendar
// [0.6, 1.0] (the "near-expiry" segment s in [0, 0.4)). The winning
// candidate is the near-now segment's own closed-form stationary point at
// s* ~= 0.4587 -- strictly inside (0.4, 1.0), away from both the knot
// breakpoint at s=0.4 and the endpoints -- which reproduces the isolated
// flat-5%-rate result from FlatRateInteriorStationaryPoint exactly, because
// DF(tau, s) for s in that segment never crosses the knot. The near-expiry
// segment's 2% rate only affects f(0) (which must discount across the
// knot), not the winning candidate: this is exactly the two-region
// consistency that a mismapped knot (s = tenor instead of s = T - tenor)
// would break, since the segment split would move (or the rate
// attribution per segment would swap) and the closed form would then
// disagree with the dense scan.
TEST(Envelope, YieldCurveKnotsRespected) {
    double t_knot = 0.6;   // tenor (calendar time) of the curve knot
    double rho1 = 0.05;    // forward rate for calendar [0, t_knot]
    double rho2 = 0.02;    // forward rate for calendar [t_knot, T]
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.0},
        {t_knot, -rho1 * t_knot},
        {1.0, -rho1 * t_knot - rho2 * (1.0 - t_knot)},
    };
    auto curve = YieldCurve::from_points(points).value();

    double T = 1.0;
    CallBoundaryEnvelope env{0.5, 0.03, T, RateSpec{curve}, {}};
    double tau = 1.0;

    double got = env.value(tau, kTimeBased);
    double want = dense_scan_max(env, tau, kTimeBased);
    EXPECT_NEAR(got, want, 1e-9);

    // Dominance: the interior candidate beats both endpoints by a comfortable
    // margin (measured: ~0.0116 over f(0), ~0.000145 over f(tau)=intrinsic).
    // See FlatRateInteriorStationaryPoint for why f(0)/f(tau) are evaluated
    // directly rather than via env.value(0, ...) / env.value(tau, ...).
    constexpr double kMargin = 5e-5;
    double f0 = std::exp(env.x_max) * std::exp(-env.dividend_yield * tau) -
                DF_calendar(env.rate, env.maturity, tau, 0.0);
    double f_tau = std::exp(env.x_max) - 1.0;
    EXPECT_GT(got - f0, kMargin);
    EXPECT_GT(got - f_tau, kMargin);
}

TEST(Envelope, CombinedContinuousAndDiscrete) {
    double T = 1.0;
    std::vector<Dividend> divs = {{0.5, 0.025}};
    CallBoundaryEnvelope env{0.4, 0.02, T, RateSpec{0.05}, divs};
    // tau == T so the ex-date kink lands exactly on the dense scan's grid
    // (see IntermediateExDateDominates for why an off-grid sample fails).
    double tau = 1.0;

    double got = env.value(tau, kTimeBased);
    double want = dense_scan_max(env, tau, kTimeBased);
    EXPECT_NEAR(got, want, 1e-9);
}
