// SPDX-License-Identifier: MIT
/**
 * @file european_option.hpp
 * @brief European option pricing with closed-form Black-Scholes formulas
 *
 * Provides EuropeanOptionResult (satisfies OptionResult and OptionResultWithVega)
 * and EuropeanOptionSolver for analytical European option pricing with full Greeks.
 */

#pragma once

#include "mango/option/option_spec.hpp"
#include "mango/option/option_concepts.hpp"
#include <cmath>
#include <expected>

namespace mango {

// ===========================================================================
// Black-Scholes analytical helpers
// ===========================================================================

/// Standard normal PDF: phi(x) = exp(-x^2/2) / sqrt(2pi)
inline double norm_pdf(double x) {
    static constexpr double kInvSqrt2Pi = 0.3989422804014327;  // 1/sqrt(2pi)
    return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}

/// Standard normal CDF using erfc for numerical stability
inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

/// Black-Scholes d1 = [ln(S/K) + (r - q + sigma^2/2)tau] / (sigma*sqrt(tau))
inline double bs_d1(double spot, double strike, double tau, double sigma, double rate,
                    double dividend_yield = 0.0) {
    double sigma_sqrt_tau = sigma * std::sqrt(tau);
    return (std::log(spot / strike) + (rate - dividend_yield + 0.5 * sigma * sigma) * tau) /
           sigma_sqrt_tau;
}

/// Black-Scholes Vega: S * e^(-q*tau) * sqrt(tau) * phi(d1)
inline double bs_vega(double spot, double strike, double tau, double sigma, double rate,
                      double dividend_yield = 0.0) {
    if (tau <= 0.0 || sigma <= 0.0) return 0.0;
    double d1 = bs_d1(spot, strike, tau, sigma, rate, dividend_yield);
    return spot * std::exp(-dividend_yield * tau) * std::sqrt(tau) * norm_pdf(d1);
}

/// Black-Scholes European option price
inline double bs_price(double spot, double strike, double tau, double sigma, double rate,
                       double dividend_yield, OptionType option_type) {
    if (tau <= 0.0 || sigma <= 0.0) {
        if (tau <= 0.0) return intrinsic_value(spot, strike, option_type);
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

// ===========================================================================
// European option result and solver
// ===========================================================================

/**
 * @brief European option pricing result with closed-form Greeks
 *
 * Stores all pricing parameters and computes price/Greeks analytically.
 * Satisfies both OptionResult and OptionResultWithVega concepts.
 *
 * Thread-safety: All methods are const and thread-safe.
 */
class EuropeanOptionResult {
public:
    EuropeanOptionResult(const PricingParams& params);

    /// Option value at current spot
    double value() const;

    /// Option value at arbitrary spot price
    double value_at(double S) const;

    /// Delta: dV/dS
    double delta() const;

    /// Gamma: d²V/dS²
    double gamma() const;

    /// Vega: dV/dσ
    double vega() const;

    /// Theta: dV/dt (time decay, typically negative)
    double theta() const;

    /// Rho: dV/dr
    double rho() const;

    // Parameter accessors
    double spot() const { return params_.spot; }
    double strike() const { return params_.strike; }
    double maturity() const { return params_.maturity; }
    double volatility() const { return params_.volatility; }
    OptionType option_type() const { return params_.option_type; }

private:
    /// Compute d1, d2 for given spot price
    std::pair<double, double> compute_d1_d2(double S) const;

    /// Compute price for given spot (used by value() and value_at())
    double compute_price(double S) const;

    /// Compute delta for given spot
    double compute_delta(double S) const;

    PricingParams params_;
    double rate_;  ///< Flat rate extracted from RateSpec
};

/**
 * @brief European option solver using closed-form Black-Scholes
 *
 * Lightweight solver that delegates to analytical formulas.
 * No PDE discretization needed.
 */
class EuropeanOptionSolver {
public:
    /// Construct solver from pricing parameters (no validation)
    explicit EuropeanOptionSolver(const PricingParams& params);

    /// Construct from option spec + volatility (convenience)
    EuropeanOptionSolver(const OptionSpec& spec, double sigma);

    /// Factory with validation via validate_pricing_params()
    static std::expected<EuropeanOptionSolver, ValidationError>
    create(const PricingParams& params) noexcept;

    /// Compute European option price and Greeks (always succeeds)
    std::expected<EuropeanOptionResult, SolverError> solve() const;

private:
    PricingParams params_;
};

static_assert(OptionResultWithVega<EuropeanOptionResult>);
static_assert(OptionSolver<EuropeanOptionSolver>);

}  // namespace mango
