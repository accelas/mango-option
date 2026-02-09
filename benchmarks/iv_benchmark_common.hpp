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

// Full Brent result with convergence diagnostics.
struct BrentIVResult {
    double iv = std::numeric_limits<double>::quiet_NaN();
    bool converged = false;
    size_t iterations = 0;
    double residual = std::numeric_limits<double>::quiet_NaN();  // |price(σ) - target|
    bool at_boundary = false;  // σ within margin of bracket bound
};

template <typename PriceFn>
BrentIVResult brent_solve_iv_full(PriceFn&& price_fn, double target_price,
                                   double a = 0.01, double b = 3.0,
                                   double boundary_margin = 0.005) {
    auto objective = [&](double vol) { return price_fn(vol) - target_price; };
    RootFindingConfig config{.max_iter = 100, .brent_tol_abs = 1e-6};
    auto result = find_root(objective, a, b, config);

    BrentIVResult r;
    if (result.has_value()) {
        r.iv = result->root;
        r.converged = true;
        r.iterations = result->iterations;
        r.residual = result->final_error;
        r.at_boundary = (result->root <= a + boundary_margin) ||
                        (result->root >= b - boundary_margin);
    } else {
        r.converged = false;
        r.iterations = result.error().iterations;
        r.residual = result.error().final_error;
        if (result.error().last_value)
            r.iv = *result.error().last_value;
    }
    return r;
}

}  // namespace mango::bench
