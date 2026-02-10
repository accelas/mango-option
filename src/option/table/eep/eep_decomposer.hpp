// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/european_option.hpp"
#include "mango/option/option_spec.hpp"
#include <algorithm>
#include <cmath>

namespace mango {

/// Compute EEP value from an American price.
/// EEP = American - European, with debiased softplus floor for non-negativity.
///
/// Inline so it can be inlined into both B-spline tensor loops and
/// Chebyshev CGL node loops without function-call overhead.
inline double compute_eep(double american_price, double spot, double strike,
                          double tau, double sigma, double rate,
                          double dividend_yield, OptionType option_type) {
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = dividend_yield,
            .option_type = option_type}, sigma).solve();

    double eep_raw = eu.has_value() ? american_price - eu->value() : 0.0;

    constexpr double kSharpness = 100.0;
    if (kSharpness * eep_raw > 500.0) {
        return eep_raw;
    }
    double softplus = std::log1p(std::exp(kSharpness * eep_raw)) / kSharpness;
    double bias = std::log(2.0) / kSharpness;
    return std::max(0.0, softplus - bias);
}

}  // namespace mango
