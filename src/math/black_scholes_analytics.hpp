// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cmath>

#include "mango/option/option_spec.hpp"

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

/// Black-Scholes option price
///
/// @param spot Current underlying price
/// @param strike Strike price
/// @param tau Time to expiry in years
/// @param sigma Volatility
/// @param rate Risk-free rate
/// @param dividend_yield Continuous dividend yield
/// @param option_type PUT or CALL
/// @return European option price
inline double bs_price(double spot, double strike, double tau, double sigma, double rate,
                       double dividend_yield, OptionType option_type) {
    // Edge cases: zero maturity or zero vol -> intrinsic value
    if (tau <= 0.0 || sigma <= 0.0) {
        if (tau <= 0.0) {
            return intrinsic_value(spot, strike, option_type);
        }
        // Zero vol, positive maturity: discounted intrinsic
        double S_fwd = spot * std::exp(-dividend_yield * tau);
        double K_disc = strike * std::exp(-rate * tau);
        return intrinsic_value(S_fwd, K_disc, option_type);
    }

    double d1 = bs_d1(spot, strike, tau, sigma, rate, dividend_yield);
    double d2 = d1 - sigma * std::sqrt(tau);
    double exp_qt = std::exp(-dividend_yield * tau);
    double exp_rt = std::exp(-rate * tau);

    if (option_type == OptionType::PUT) {
        return strike * exp_rt * norm_cdf(-d2) - spot * exp_qt * norm_cdf(-d1);
    } else {
        return spot * exp_qt * norm_cdf(d1) - strike * exp_rt * norm_cdf(d2);
    }
}

}  // namespace mango
