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

TEST(Envelope, FlatRateInteriorStationaryPoint) {
    double T = 2.0;
    CallBoundaryEnvelope env{0.15, 0.01, T, RateSpec{0.08}, {}};
    double tau = 1.6;

    double got = env.value(tau, kTimeBased);
    double want = dense_scan_max(env, tau, kTimeBased);
    EXPECT_NEAR(got, want, 1e-9);
}

TEST(Envelope, YieldCurveKnotsRespected) {
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.0},
        {0.25, -0.25 * 0.03},
        {0.5, -0.25 * 0.03 - 0.25 * 0.09},
        {1.0, -0.25 * 0.03 - 0.25 * 0.09 - 0.5 * 0.05},
    };
    auto curve = YieldCurve::from_points(points).value();

    double T = 1.0;
    CallBoundaryEnvelope env{0.2, 0.015, T, RateSpec{curve}, {}};
    double tau = 0.9;

    double got = env.value(tau, kTimeBased);
    double want = dense_scan_max(env, tau, kTimeBased);
    EXPECT_NEAR(got, want, 1e-9);
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
