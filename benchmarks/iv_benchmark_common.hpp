// SPDX-License-Identifier: MIT
#pragma once
#include "mango/option/option_spec.hpp"  // Dividend
#include "mango/math/root_finding.hpp"
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

// Fixed quarterly $0.50 calendar, filtered to the option's life. Used by
// interp_iv_safety's dividends path (#462): unlike make_div_schedule it does
// not scale with maturity, so a 7-day option carries no dividend and a
// 1-year option carries three. See the retune spec for why the scaled
// schedule refuses on the segmented B-spline path at short maturities (#501).
inline std::vector<Dividend> quarterly_div_schedule(double maturity) {
    std::vector<Dividend> out;
    for (double t = 0.25; t < maturity; t += 0.25)
        out.push_back(Dividend{.calendar_time = t, .amount = 0.50});
    return out;
}

// Brent solver for IV recovery using the library's find_root.
// Returns vol on success, NaN on failure.
template <typename PriceFn>
double brent_solve_iv(PriceFn&& price_fn, double target_price,
                      double a = 0.01, double b = 3.0) {
    auto objective = [&](double vol) { return price_fn(vol) - target_price; };
    RootFindingConfig config{.max_iter = 100, .brent_tol_abs = 1e-6};
    auto result = find_root(objective, a, b, config);
    if (result.has_value()) {
        return result->root;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace mango::bench
