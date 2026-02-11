// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/surface_concepts.hpp"

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

// ===========================================================================
// Generic strategy-based API
// ===========================================================================

/// Decompose American prices to EEP using any EEP strategy.
/// The strategy's european_price() is the single source of truth.
template <EEPAccessor A, EEPStrategy EEP>
void eep_decompose(A&& accessor, const EEP& eep) {
    const size_t n = accessor.size();
    const double strike = accessor.strike();
    for (size_t i = 0; i < n; ++i) {
        double am = accessor.american_price(i);
        double eu = eep.european_price(accessor.spot(i), strike,
                        accessor.tau(i), accessor.sigma(i), accessor.rate(i));
        accessor.set_value(i, eep_floor(am - eu));
    }
}

/// Per-point EEP using any EEP strategy.
template <EEPStrategy EEP>
double compute_eep(double american_price, double spot, double strike,
                   double tau, double sigma, double rate, const EEP& eep) {
    return eep_floor(american_price
                     - eep.european_price(spot, strike, tau, sigma, rate));
}

// ===========================================================================
// Backward-compatible wrappers (delegate to generic API)
// ===========================================================================

/// Convenience wrapper: construct AnalyticalEEP internally.
template <EEPAccessor A>
void analytical_eep_decompose(A&& accessor,
                              OptionType option_type,
                              double dividend_yield) {
    AnalyticalEEP eep(option_type, dividend_yield);
    eep_decompose(std::forward<A>(accessor), eep);
}

/// Per-point analytical EEP: European via Black-Scholes + softplus floor.
///
/// Use for call sites that don't fit the accessor pattern (e.g. cache-based
/// extraction with per-slice spline lookup).
inline double compute_eep(double american_price, double spot, double strike,
                          double tau, double sigma, double rate,
                          double dividend_yield, OptionType option_type) {
    AnalyticalEEP eep(option_type, dividend_yield);
    return compute_eep(american_price, spot, strike, tau, sigma, rate, eep);
}

}  // namespace mango
