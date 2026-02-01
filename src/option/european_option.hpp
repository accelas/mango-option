// SPDX-License-Identifier: MIT
/**
 * @file european_option.hpp
 * @brief European option pricing with closed-form Black-Scholes formulas
 *
 * Provides EuropeanOptionResult (satisfies OptionResult and OptionResultWithVega)
 * and EuropeanOptionSolver for analytical European option pricing with full Greeks.
 */

#pragma once

#include "src/option/option_spec.hpp"
#include "src/option/option_concepts.hpp"
#include <expected>

namespace mango {

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
    OptionType option_type() const { return params_.type; }

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
