// SPDX-License-Identifier: MIT
#pragma once

#include "src/pde/core/pde_solver.hpp"
#include "src/math/cubic_spline_solver.hpp"
#include <cmath>
#include <span>

namespace mango {

namespace detail {

/// Core dividend shift logic shared by put and call variants.
///
/// Rebuilds a pre-allocated cubic spline with the current solution, then for
/// each grid point evaluates u(x') where x' = ln(exp(x) - D/K).  Points
/// that shift below the grid domain or where S - D <= 0 are set to the
/// caller-supplied intrinsic fallback value.
///
/// @param spline Pre-built spline (grid already set via build(); coefficients
///               are refreshed each invocation via rebuild_same_grid()).
///               Must outlive the returned callback.
inline TemporalEventCallback make_dividend_event_impl(
    double dividend_amount, double strike, double intrinsic_fallback,
    CubicSpline<double>* spline)
{
    const double d = dividend_amount / strike;  // normalized dividend
    const double fallback = intrinsic_fallback;

    return [d, fallback, spline](double /*t*/, std::span<const double> x, std::span<double> u) {
        if (d <= 0.0) return;  // no-op for zero dividend

        const size_t n = x.size();

        // Rebuild spline coefficients with current solution (zero-alloc)
        auto err = spline->rebuild_same_grid(std::span<const double>(u.data(), u.size()));
        if (err.has_value()) return;  // spline rebuild failed — leave u unchanged

        // Apply dividend shift: x' = ln(exp(x) - d)
        const double x_lo = x[0];      // spline domain lower bound
        const double x_hi = x[n - 1];  // spline domain upper bound

        for (size_t i = 0; i < n; ++i) {
            double S_over_K = std::exp(x[i]);
            double S_adj_over_K = S_over_K - d;

            if (S_adj_over_K > 1e-10) {
                double x_shifted = std::log(S_adj_over_K);
                // Guard against extrapolation outside the spline domain
                if (x_shifted < x_lo) {
                    u[i] = fallback;
                } else {
                    u[i] = spline->eval(std::min(x_shifted, x_hi));
                }
            } else {
                // Spot drops to zero or below
                u[i] = fallback;
            }
        }
    };
}

}  // namespace detail

/// Create a put dividend event callback.
///
/// When S - D <= 0, the put is deep ITM with normalized payoff = 1.0.
///
/// @param dividend_amount Cash dividend amount D (in dollars)
/// @param strike Reference strike K (normalization base)
/// @param spline Pre-built spline; must outlive the returned callback
inline TemporalEventCallback make_put_dividend_event(
    double dividend_amount, double strike, CubicSpline<double>* spline)
{
    return detail::make_dividend_event_impl(dividend_amount, strike, 1.0, spline);
}

/// Create a call dividend event callback.
///
/// When S - D <= 0, the call is worthless with payoff = 0.0.
///
/// @param dividend_amount Cash dividend amount D (in dollars)
/// @param strike Reference strike K (normalization base)
/// @param spline Pre-built spline; must outlive the returned callback
inline TemporalEventCallback make_call_dividend_event(
    double dividend_amount, double strike, CubicSpline<double>* spline)
{
    return detail::make_dividend_event_impl(dividend_amount, strike, 0.0, spline);
}

}  // namespace mango
