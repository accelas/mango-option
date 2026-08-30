// SPDX-License-Identifier: MIT
/**
 * @file call_boundary_envelope.hpp
 * @brief Deep-ITM American call right-boundary stopping-value envelope
 *
 * Evaluates the exact optimal-stopping value used as the right boundary
 * condition for the American call PDE in log-moneyness space, at a fixed
 * `x = x_max` deep in the money. See docs/plans/
 * 2026-08-30-boundary-correctness-439-455-design.md section B for the
 * derivation and time/phase conventions.
 *
 * All times below are solver (backward) time: 0 at expiry, `maturity` at
 * the valuation date. Calendar time is `maturity - backward_time`.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <variant>
#include <vector>

#include "mango/option/option_spec.hpp"
#include "mango/option/yield_curve.hpp"

namespace mango::detail {

/// Stopping-value envelope for the American call's right boundary.
///
/// `dividends` must be calendar-ascending (the output of
/// `filter_and_merge_dividends`), and each `amount` must already be
/// normalized by strike (D_i/K) -- Task 6 (RightBCFunction) performs that
/// division when constructing this struct.
struct CallBoundaryEnvelope {
    double x_max;              ///< log-moneyness at the right boundary
    double dividend_yield;     ///< continuous dividend yield q
    double maturity;           ///< time to maturity T
    RateSpec rate;             ///< constant rate or yield curve
    std::vector<Dividend> dividends;  ///< calendar-ascending, amount = D_i/K

    /// Value at solver (backward) time `tau`, with `n_events_applied`
    /// dividends already crossed by temporal events.
    ///
    /// Phase rule (binding, see design doc B1/B3): the last
    /// `n_events_applied` entries of `dividends` (calendar-ascending, so
    /// these are the latest-calendar / first-crossed dividends as the
    /// solver steps backward time forward from expiry) are the ones
    /// still counted as "remaining" life for this evaluation -- this is
    /// the phase-aware override used once the temporal event for a
    /// dividend has fired at tau == tau_j. Pass `SIZE_MAX` to derive the
    /// remaining set purely from time: `tau_i < tau` strictly (used by
    /// tests and by any caller without a phase counter).
    double value(double tau, size_t n_events_applied) const {
        const double A = std::exp(x_max);
        const double intrinsic = A - 1.0;

        // --- discount-factor helpers over the configured RateSpec -------
        // discount_cal(t): D_cal(t), calendar time t from the valuation date.
        auto discount_cal = [this](double t) -> double {
            return std::visit(
                [t](const auto& r) -> double {
                    using T = std::decay_t<decltype(r)>;
                    if constexpr (std::is_same_v<T, double>) {
                        return std::exp(-r * t);
                    } else {
                        return r.discount(t);
                    }
                },
                rate);
        };
        // DF(a, b): discount factor between backward times a and b,
        // DF(a,b) = D_cal(T-b) / D_cal(T-a).
        auto DF = [this, &discount_cal](double a, double b) -> double {
            return discount_cal(maturity - b) / discount_cal(maturity - a);
        };
        // Local flat-forward rate over backward-time segment [lo, hi]
        // (evaluated at the segment's calendar midpoint; safe because
        // every curve knot in range is already a segment breakpoint).
        auto local_forward_rate = [this](double lo, double hi) -> double {
            double t_cal = maturity - 0.5 * (lo + hi);
            return std::visit(
                [t_cal](const auto& r) -> double {
                    using T = std::decay_t<decltype(r)>;
                    if constexpr (std::is_same_v<T, double>) {
                        return r;
                    } else {
                        return r.rate(t_cal);
                    }
                },
                rate);
        };

        // --- remaining dividend set (phase source of truth) -------------
        std::vector<size_t> remaining;
        if (n_events_applied == std::numeric_limits<size_t>::max()) {
            for (size_t i = 0; i < dividends.size(); ++i) {
                double tau_i = maturity - dividends[i].calendar_time;
                if (tau_i < tau) remaining.push_back(i);
            }
        } else {
            size_t m = dividends.size();
            size_t n = std::min(n_events_applied, m);
            for (size_t i = m - n; i < m; ++i) remaining.push_back(i);
        }

        auto tau_of = [this](size_t i) { return maturity - dividends[i].calendar_time; };

        // f(s): stopping value at backward time s in [0, tau].
        auto f = [&](double s) -> double {
            double sum = 0.0;
            for (size_t i : remaining) {
                double tau_i = tau_of(i);
                if (tau_i > s) {
                    sum += dividends[i].amount * DF(tau, tau_i) *
                           std::exp(-dividend_yield * (tau_i - s));
                }
            }
            return A * std::exp(-dividend_yield * (tau - s)) - sum - DF(tau, s);
        };

        // --- candidate breakpoints: 0, tau, remaining ex-dates, curve knots
        std::vector<double> breakpoints = {0.0, tau};
        for (size_t i : remaining) {
            double tau_i = tau_of(i);
            if (tau_i > 0.0 && tau_i < tau) breakpoints.push_back(tau_i);
        }
        if (std::holds_alternative<YieldCurve>(rate)) {
            for (const auto& p : std::get<YieldCurve>(rate).points()) {
                double s_knot = maturity - p.tenor;
                if (s_knot > 0.0 && s_knot < tau) breakpoints.push_back(s_knot);
            }
        }
        std::sort(breakpoints.begin(), breakpoints.end());
        breakpoints.erase(
            std::unique(breakpoints.begin(), breakpoints.end(),
                        [](double a, double b) { return std::abs(a - b) < 1e-12; }),
            breakpoints.end());

        double best = f(breakpoints.front());
        for (double s : breakpoints) best = std::max(best, f(s));

        // --- per-segment closed-form interior stationary point ---------
        const double q = dividend_yield;
        for (size_t k = 0; k + 1 < breakpoints.size(); ++k) {
            double lo = breakpoints[k];
            double hi = breakpoints[k + 1];
            if (hi - lo < 1e-14) continue;

            // Active dividends throughout the open segment: tau_i >= hi
            // (breakpoints guarantee no remaining tau_i lies strictly
            // inside (lo, hi)).
            double B = A * std::exp(-q * tau);
            for (size_t i : remaining) {
                double tau_i = tau_of(i);
                if (tau_i >= hi - 1e-12) {
                    B -= dividends[i].amount * DF(tau, tau_i) * std::exp(-q * tau_i);
                }
            }

            double r_f = local_forward_rate(lo, hi);
            double C = DF(tau, hi) * std::exp(-r_f * hi);

            if (q > 0.0 && r_f > 0.0 && B > 0.0 && C > 0.0 &&
                std::abs(r_f - q) > 1e-14) {
                double s_star = std::log((q * B) / (r_f * C)) / (r_f - q);
                if (s_star > lo + 1e-14 && s_star < hi - 1e-14) {
                    best = std::max(best, f(s_star));
                }
            }
        }

        return std::max(best, intrinsic);
    }
};

}  // namespace mango::detail
