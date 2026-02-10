// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/european_option.hpp"
#include "mango/option/option_spec.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>

namespace mango {

/// Debiased softplus floor for EEP non-negativity.
///
/// Smoothly clamps eep_raw = (American - European) to non-negative values.
/// Zero-bias correction ensures eep_floor(0) == 0 exactly.
///
/// Use directly when the European price comes from a non-analytical source
/// (e.g. numerical PDE). For analytical Black-Scholes, use
/// analytical_eep_decompose().
inline double eep_floor(double eep_raw) {
    constexpr double kSharpness = 100.0;
    if (kSharpness * eep_raw > 500.0) {
        return eep_raw;
    }
    double softplus = std::log1p(std::exp(kSharpness * eep_raw)) / kSharpness;
    double bias = std::log(2.0) / kSharpness;
    return std::max(0.0, softplus - bias);
}

/// Per-point analytical EEP: European via Black-Scholes + softplus floor.
///
/// Use for call sites that don't fit the accessor pattern (e.g. cache-based
/// extraction with per-slice spline lookup).
inline double compute_eep(double american_price, double spot, double strike,
                          double tau, double sigma, double rate,
                          double dividend_yield, OptionType option_type) {
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = dividend_yield,
            .option_type = option_type}, sigma).solve();

    double eep_raw = eu.has_value() ? american_price - eu->value() : 0.0;
    return eep_floor(eep_raw);
}

/// Accessor concept for EEP decomposition.
///
/// Backends implement this to adapt their storage (mdspan tensor, flat vector
/// + spline eval, etc.) into a uniform point-wise interface that the EEP
/// module iterates.
template <typename A>
concept EEPAccessor = requires(const A a, size_t i) {
    { a.size() } -> std::convertible_to<size_t>;
    { a.american_price(i) } -> std::convertible_to<double>;
    { a.spot(i) } -> std::convertible_to<double>;
    { a.strike() } -> std::convertible_to<double>;
    { a.tau(i) } -> std::convertible_to<double>;
    { a.sigma(i) } -> std::convertible_to<double>;
    { a.rate(i) } -> std::convertible_to<double>;
} && requires(A a, size_t i, double v) {
    a.set_value(i, v);
};

/// Apply analytical EEP decomposition over any accessor.
///
/// Iterates all points, computes European price via Black-Scholes,
/// subtracts from American, and applies the softplus floor.
template <EEPAccessor A>
void analytical_eep_decompose(A&& accessor,
                              OptionType option_type,
                              double dividend_yield) {
    const size_t n = accessor.size();
    const double strike = accessor.strike();

    for (size_t i = 0; i < n; ++i) {
        double am = accessor.american_price(i);
        double spot = accessor.spot(i);
        double tau = accessor.tau(i);
        double sigma = accessor.sigma(i);
        double rate = accessor.rate(i);

        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
                .rate = rate, .dividend_yield = dividend_yield,
                .option_type = option_type}, sigma).solve();

        double eep_raw = eu.has_value() ? am - eu->value() : 0.0;
        accessor.set_value(i, eep_floor(eep_raw));
    }
}

}  // namespace mango
