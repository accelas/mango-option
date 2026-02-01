// SPDX-License-Identifier: MIT
#pragma once

#include "src/pde/core/pde_solver.hpp"
#include "src/math/cubic_spline_solver.hpp"
#include "src/option/option_spec.hpp"
#include <cmath>
#include <span>
#include <vector>

namespace mango {

/// Create a temporal event callback for a discrete cash dividend.
///
/// At the dividend date, the spot drops from S to S - D. In log-moneyness
/// coordinates x = ln(S/K), this shifts the solution: for each grid point
/// x[i], the new value is u(x') where x' = ln(exp(x[i]) - D/K).
///
/// Uses cubic spline interpolation for the shifted evaluation points.
///
/// @param dividend_amount Cash dividend amount D (in dollars)
/// @param strike Reference strike K (normalization base)
/// @param option_type PUT or CALL (determines fallback when S - D <= 0)
/// @return TemporalEventCallback suitable for PDESolver::add_temporal_event()
inline TemporalEventCallback make_dividend_event(
    double dividend_amount, double strike, OptionType option_type)
{
    const double d = dividend_amount / strike;  // normalized dividend
    const bool is_put = (option_type == OptionType::PUT);

    return [d, is_put](double /*t*/, std::span<const double> x, std::span<double> u) {
        if (d <= 0.0) return;  // no-op for zero dividend

        const size_t n = x.size();

        // Build cubic spline of current solution
        CubicSpline<double> spline;
        auto err = spline.build(x, std::span<const double>(u.data(), u.size()));
        if (err.has_value()) return;  // spline build failed — leave u unchanged

        // Apply dividend shift: x' = ln(exp(x) - d)
        for (size_t i = 0; i < n; ++i) {
            double S_over_K = std::exp(x[i]);
            double S_adj_over_K = S_over_K - d;

            if (S_adj_over_K > 1e-10) {
                double x_shifted = std::log(S_adj_over_K);
                u[i] = spline.eval(x_shifted);
            } else {
                // Spot drops to zero or below: use option-type-aware intrinsic
                if (is_put) {
                    u[i] = 1.0;  // Put: deep ITM, normalized payoff = 1.0
                } else {
                    u[i] = 0.0;  // Call: worthless when spot <= 0
                }
            }
        }
    };
}

}  // namespace mango
