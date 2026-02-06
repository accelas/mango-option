// SPDX-License-Identifier: MIT
#pragma once
#include "mango/option/option_spec.hpp"  // Dividend
#include <cmath>
#include <limits>
#include <vector>

namespace mango::bench {

inline constexpr double kSpot = 100.0;
inline constexpr double kRate = 0.05;
inline constexpr double kDivYield = 0.02;

// Quarterly $0.50 dividends scaled to maturity
inline std::vector<Dividend> make_div_schedule(double maturity) {
    return {
        Dividend{.calendar_time = maturity * 0.25, .amount = 0.50},
        Dividend{.calendar_time = maturity * 0.50, .amount = 0.50},
        Dividend{.calendar_time = maturity * 0.75, .amount = 0.50},
    };
}

// Generic Brent solver for IV recovery.
// Returns vol on success, NaN on failure.
template <typename PriceFn>
double brent_solve_iv(PriceFn&& price_fn, double target_price,
                      double a = 0.01, double b = 3.0) {
    double fa = price_fn(a) - target_price;
    double fb = price_fn(b) - target_price;

    if (!std::isfinite(fa) || !std::isfinite(fb) || fa * fb > 0)
        return std::numeric_limits<double>::quiet_NaN();

    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c = a, fc = fa;
    bool mflag = true;
    double d = 0.0;
    constexpr double tol = 1e-6;
    constexpr size_t max_iter = 100;

    for (size_t iter = 0; iter < max_iter; ++iter) {
        if (std::abs(fb) < tol || std::abs(b - a) < tol) {
            return b;
        }

        double s;
        if (fa != fc && fb != fc) {
            s = a * fb * fc / ((fa - fb) * (fa - fc))
              + b * fa * fc / ((fb - fa) * (fb - fc))
              + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            s = b - fb * (b - a) / (fb - fa);
        }

        double bisect = (3.0 * a + b) / 4.0;
        bool cond1 = !((s > bisect && s < b) || (s < bisect && s > b));
        bool cond2 = mflag && std::abs(s - b) >= std::abs(b - c) / 2.0;
        bool cond3 = !mflag && std::abs(s - b) >= std::abs(c - d) / 2.0;
        bool cond4 = mflag && std::abs(b - c) < tol;
        bool cond5 = !mflag && std::abs(c - d) < tol;

        if (cond1 || cond2 || cond3 || cond4 || cond5) {
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }

        double fs = price_fn(s) - target_price;
        if (!std::isfinite(fs))
            return std::numeric_limits<double>::quiet_NaN();

        d = c; c = b; fc = fb;
        if (fa * fs < 0.0) { b = s; fb = fs; }
        else { a = s; fa = fs; }
        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b); std::swap(fa, fb);
        }
    }
    return b;
}

}  // namespace mango::bench
