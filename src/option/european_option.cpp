// SPDX-License-Identifier: MIT
#include "mango/option/european_option.hpp"
#include "mango/math/black_scholes_analytics.hpp"
#include <algorithm>
#include <cmath>

namespace mango {

// ===========================================================================
// EuropeanOptionResult
// ===========================================================================

EuropeanOptionResult::EuropeanOptionResult(const PricingParams& params)
    : params_(params)
    , rate_(get_zero_rate(params.rate, params.maturity))
{}

std::pair<double, double> EuropeanOptionResult::compute_d1_d2(double S) const {
    double tau = params_.maturity;
    double sigma = params_.volatility;
    double d1 = bs_d1(S, params_.strike, tau, sigma, rate_, params_.dividend_yield);
    double d2 = d1 - sigma * std::sqrt(tau);
    return {d1, d2};
}

double EuropeanOptionResult::compute_price(double S) const {
    double tau = params_.maturity;
    double sigma = params_.volatility;
    double K = params_.strike;
    double r = rate_;
    double q = params_.dividend_yield;

    // Edge cases: zero maturity or zero vol -> intrinsic/discounted intrinsic
    if (tau <= 0.0 || sigma <= 0.0) {
        if (tau <= 0.0) {
            // At expiry: intrinsic value
            if (params_.option_type == OptionType::PUT) {
                return std::max(K - S, 0.0);
            } else {
                return std::max(S - K, 0.0);
            }
        }
        // Zero vol, positive maturity: discounted intrinsic
        double S_fwd = S * std::exp(-q * tau);
        double K_disc = K * std::exp(-r * tau);
        if (params_.option_type == OptionType::PUT) {
            return std::max(K_disc - S_fwd, 0.0);
        } else {
            return std::max(S_fwd - K_disc, 0.0);
        }
    }

    auto [d1, d2] = compute_d1_d2(S);
    double exp_qt = std::exp(-q * tau);
    double exp_rt = std::exp(-r * tau);

    if (params_.option_type == OptionType::PUT) {
        return K * exp_rt * norm_cdf(-d2) - S * exp_qt * norm_cdf(-d1);
    } else {
        return S * exp_qt * norm_cdf(d1) - K * exp_rt * norm_cdf(d2);
    }
}

double EuropeanOptionResult::compute_delta(double S) const {
    double tau = params_.maturity;
    double sigma = params_.volatility;
    double q = params_.dividend_yield;

    if (tau <= 0.0 || sigma <= 0.0) {
        // Edge case: discounted delta for ITM, 0 for OTM
        double exp_qt = std::exp(-q * tau);
        if (params_.option_type == OptionType::PUT) {
            return (S < params_.strike) ? -exp_qt : 0.0;
        } else {
            return (S > params_.strike) ? exp_qt : 0.0;
        }
    }

    auto [d1, d2] = compute_d1_d2(S);
    double exp_qt = std::exp(-q * tau);

    if (params_.option_type == OptionType::PUT) {
        return -exp_qt * norm_cdf(-d1);
    } else {
        return exp_qt * norm_cdf(d1);
    }
}

double EuropeanOptionResult::value() const {
    return compute_price(params_.spot);
}

double EuropeanOptionResult::value_at(double S) const {
    return compute_price(S);
}

double EuropeanOptionResult::delta() const {
    return compute_delta(params_.spot);
}

double EuropeanOptionResult::gamma() const {
    double tau = params_.maturity;
    double sigma = params_.volatility;
    double S = params_.spot;
    double q = params_.dividend_yield;

    if (tau <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }

    auto [d1, d2] = compute_d1_d2(S);
    double exp_qt = std::exp(-q * tau);
    return exp_qt * norm_pdf(d1) / (S * sigma * std::sqrt(tau));
}

double EuropeanOptionResult::vega() const {
    return bs_vega(params_.spot, params_.strike, params_.maturity,
                   params_.volatility, rate_, params_.dividend_yield);
}

double EuropeanOptionResult::theta() const {
    double tau = params_.maturity;
    double sigma = params_.volatility;
    double S = params_.spot;
    double K = params_.strike;
    double r = rate_;
    double q = params_.dividend_yield;

    if (tau <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }

    auto [d1, d2] = compute_d1_d2(S);
    double sqrt_tau = std::sqrt(tau);
    double exp_qt = std::exp(-q * tau);
    double exp_rt = std::exp(-r * tau);

    // Common term: -S·e^(-qτ)·φ(d1)·σ/(2√τ)
    double common = -S * exp_qt * norm_pdf(d1) * sigma / (2.0 * sqrt_tau);

    if (params_.option_type == OptionType::PUT) {
        return common + r * K * exp_rt * norm_cdf(-d2) - q * S * exp_qt * norm_cdf(-d1);
    } else {
        return common - r * K * exp_rt * norm_cdf(d2) + q * S * exp_qt * norm_cdf(d1);
    }
}

double EuropeanOptionResult::rho() const {
    double tau = params_.maturity;
    double sigma = params_.volatility;
    double K = params_.strike;
    double r = rate_;

    if (tau <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }

    auto [d1, d2] = compute_d1_d2(params_.spot);
    double exp_rt = std::exp(-r * tau);

    if (params_.option_type == OptionType::PUT) {
        return -K * tau * exp_rt * norm_cdf(-d2);
    } else {
        return K * tau * exp_rt * norm_cdf(d2);
    }
}

// ===========================================================================
// EuropeanOptionSolver
// ===========================================================================

EuropeanOptionSolver::EuropeanOptionSolver(const PricingParams& params)
    : params_(params)
{}

EuropeanOptionSolver::EuropeanOptionSolver(const OptionSpec& spec, double sigma)
    : params_(spec, sigma)
{}

std::expected<EuropeanOptionSolver, ValidationError>
EuropeanOptionSolver::create(const PricingParams& params) noexcept {
    auto validation = validate_pricing_params(params);
    if (!validation.has_value()) {
        return std::unexpected(validation.error());
    }
    return EuropeanOptionSolver(params);
}

std::expected<EuropeanOptionResult, SolverError> EuropeanOptionSolver::solve() const {
    return EuropeanOptionResult(params_);
}

}  // namespace mango
