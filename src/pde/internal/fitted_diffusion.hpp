// SPDX-License-Identifier: MIT
/**
 * @file fitted_diffusion.hpp
 * @brief Il'in exponentially-fitted diffusion coefficient (issue #472)
 *
 * Replaces the raw diffusion coefficient a = sigma^2/2 with
 * a_f = a * rho * coth(rho), rho = |b| * h_binding / (2a), where
 * h_binding is the neighbor spacing whose off-diagonal the drift can
 * flip (dx_right for b > 0, dx_left for b < 0). Guarantees
 * a_f >= max(a, |b| * h_binding / 2) exactly in floating point, which
 * makes both off-diagonals of the discrete spatial operator
 * non-negative for every sigma/h/b combination — the Z-matrix half of
 * the Brennan-Schwartz one-pass exactness requirement.
 *
 * Contract (binding; see the #472 design spec):
 *  - a < 0 is outside the contract (debug assert).
 *  - a == 0, b != 0 returns the representable convection limit a_f = z.
 *  - z == 0 (b == 0, or a degenerate zero-width binding cell) returns
 *    exactly a (bit-exact pure-diffusion behavior).
 *  - Callers sample a and b once per assembly/apply invocation and pass
 *    them in; this function is pure.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>

namespace mango::operators::detail {

/// Fitted diffusion for one interior cell.
struct FittedDiffusion {
    double a_f;  ///< fitted diffusion coefficient; >= max(a, z) exactly
    double z;    ///< binding half-cell drift mass |b| * h_binding / 2
};

inline FittedDiffusion fitted_diffusion(double a, double b,
                                        double dx_left, double dx_right) {
    assert(a >= 0.0 && "negative diffusion is outside the fitting contract");
    const double h_binding = (b > 0.0) ? dx_right : dx_left;
    const double z = 0.5 * std::abs(b) * h_binding;
    if (z == 0.0) {
        return {a, 0.0};  // b == 0: pure diffusion, bit-exact passthrough
    }
    if (a == 0.0) {
        return {z, z};  // sigma^2/2 underflowed: exact convection limit
    }
    const double rho = z / a;
    double a_f;
    if (rho < 1e-4) {
        // rho*coth(rho) = 1 + rho^2/3 - rho^4/45 + ...; truncation error
        // <= rho^4/45 < 1e-17 relative at the cutoff.
        a_f = a * (1.0 + rho * rho / 3.0);
    } else {
        // a * rho * coth(rho) == z / tanh(rho); tanh saturates to 1 for
        // large rho, so a_f -> z with no overflow.
        a_f = z / std::tanh(rho);
    }
    // Exact-in-FP sign invariant: a_f - z >= 0 and a_f >= a.
    a_f = std::max({a_f, a, z});
    return {a_f, z};
}

}  // namespace mango::operators::detail
