// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"
#include <cmath>

namespace mango {

/// European put price in dimensionless coordinates, normalized by K.
///
/// V_put / K = N(-d2) * exp(-κτ') - exp(x) * N(-d1)
///
/// where d1 = (x + (κ+1)τ') / √(2τ'), d2 = d1 - √(2τ').
///
/// @param x Log-moneyness ln(S/K)
/// @param tau_prime Dimensionless time σ²τ/2
/// @param kappa Dimensionless rate 2r/σ²
/// @return Normalized European put price V/K
[[nodiscard]] inline double
dimensionless_european_put(double x, double tau_prime, double kappa) noexcept {
    if (tau_prime <= 0.0) {
        return std::max(1.0 - std::exp(x), 0.0);
    }
    const double sqrt_2tp = std::sqrt(2.0 * tau_prime);
    const double d1 = (x + (kappa + 1.0) * tau_prime) / sqrt_2tp;
    const double d2 = d1 - sqrt_2tp;
    const double Nd1 = 0.5 * std::erfc(d1 * M_SQRT1_2);   // N(-d1)
    const double Nd2 = 0.5 * std::erfc(d2 * M_SQRT1_2);   // N(-d2)
    return Nd2 * std::exp(-kappa * tau_prime) - std::exp(x) * Nd1;
}

/// European call price in dimensionless coordinates, normalized by K.
///
/// V_call / K = exp(x) * N(d1) - N(d2) * exp(-κτ')
[[nodiscard]] inline double
dimensionless_european_call(double x, double tau_prime, double kappa) noexcept {
    if (tau_prime <= 0.0) {
        return std::max(std::exp(x) - 1.0, 0.0);
    }
    const double sqrt_2tp = std::sqrt(2.0 * tau_prime);
    const double d1 = (x + (kappa + 1.0) * tau_prime) / sqrt_2tp;
    const double d2 = d1 - sqrt_2tp;
    const double Nd1 = 0.5 * std::erfc(-d1 * M_SQRT1_2);  // N(d1)
    const double Nd2 = 0.5 * std::erfc(-d2 * M_SQRT1_2);  // N(d2)
    return std::exp(x) * Nd1 - Nd2 * std::exp(-kappa * tau_prime);
}

/// Dispatch to put or call based on option type.
[[nodiscard]] inline double
dimensionless_european(double x, double tau_prime, double kappa,
                       OptionType type) noexcept {
    return type == OptionType::PUT
        ? dimensionless_european_put(x, tau_prime, kappa)
        : dimensionless_european_call(x, tau_prime, kappa);
}

}  // namespace mango
