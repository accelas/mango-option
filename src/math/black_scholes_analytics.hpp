// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>

namespace mango {

/// Standard normal PDF: φ(x) = exp(-x²/2) / sqrt(2π)
inline double norm_pdf(double x) {
    static constexpr double kInvSqrt2Pi = 0.3989422804014327;  // 1/sqrt(2π)
    return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}

/// Standard normal CDF: Φ(x) using Abramowitz & Stegun approximation
inline double norm_cdf(double x) {
    // Use erfc for numerical stability
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

/// Black-Scholes d1 term
/// d1 = [ln(S/K) + (r - q + σ²/2)τ] / (σ√τ)
inline double bs_d1(double spot, double strike, double tau, double sigma, double rate,
                    double dividend_yield = 0.0) {
    double sigma_sqrt_tau = sigma * std::sqrt(tau);
    return (std::log(spot / strike) + (rate - dividend_yield + 0.5 * sigma * sigma) * tau) /
           sigma_sqrt_tau;
}

/// Black-Scholes Vega: ∂V/∂σ = S · e^(-qτ) · √τ · φ(d1)
/// Same for puts and calls
///
/// @param spot Current underlying price
/// @param strike Strike price
/// @param tau Time to expiry in years
/// @param sigma Volatility
/// @param rate Risk-free rate
/// @param dividend_yield Continuous dividend yield (default = 0.0)
/// @return Vega (price change per unit volatility change)
inline double bs_vega(double spot, double strike, double tau, double sigma, double rate,
                      double dividend_yield = 0.0) {
    if (tau <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }
    double d1 = bs_d1(spot, strike, tau, sigma, rate, dividend_yield);
    return spot * std::exp(-dividend_yield * tau) * std::sqrt(tau) * norm_pdf(d1);
}

}  // namespace mango
