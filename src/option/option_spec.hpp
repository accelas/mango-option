// SPDX-License-Identifier: MIT
/**
 * @file option_spec.hpp
 * @brief Unified option specification for IV solvers
 */

#pragma once

#include <expected>
#include "src/support/error_types.hpp"
#include "src/math/yield_curve.hpp"
#include <string>
#include <vector>
#include <utility>
#include <variant>
#include <functional>

namespace mango {

/// Rate specification: constant or yield curve
///
/// Full yield curve support is available in the FDM solver (AmericanOptionSolver,
/// IVSolverFDM) where the time-varying rate flows through the PDE discretization.
///
/// Limitation: Interpolation-based solvers (IVSolverInterpolated, PriceTableSurface)
/// use a scalar rate axis. When a YieldCurve is provided, it is collapsed to a
/// zero rate: -ln(D(T))/T. This provides a reasonable flat-rate approximation
/// but does not capture the full term structure dynamics.
using RateSpec = std::variant<double, YieldCurve>;

/// Check if RateSpec contains a yield curve (vs constant rate)
inline bool is_yield_curve(const RateSpec& spec) {
    return std::holds_alternative<YieldCurve>(spec);
}

/// Helper to extract rate function from RateSpec for PDE solver
///
/// The PDE solver uses time-to-expiry τ (0 at expiry, T at valuation date).
/// Yield curves use calendar time s from valuation date (0 = today, T = expiry).
/// Conversion: s = T - τ (calendar time = maturity - time_to_expiry)
///
/// Returns a callable that takes time-to-expiry τ and returns the instantaneous
/// forward rate at calendar time s = T - τ.
///
/// @param spec Rate specification (constant or yield curve)
/// @param maturity Total time to maturity T
inline std::function<double(double)> make_rate_fn(const RateSpec& spec, double maturity) {
    return std::visit([maturity](const auto& arg) -> std::function<double(double)> {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, double>) {
            return [r = arg](double) { return r; };
        } else {
            // Convert time-to-expiry τ to calendar time s = T - τ
            return [curve = arg, maturity](double tau) {
                double s = std::max(0.0, maturity - tau);
                return curve.rate(s);
            };
        }
    }, spec);
}

/// Helper to extract forward discount function from RateSpec for boundary conditions
///
/// The boundary condition needs the forward discount factor from current calendar
/// time s to expiry T: D(T)/D(s). For constant rate, this is exp(-r*τ) where τ = T - s.
///
/// Returns a callable that takes time-to-expiry τ and returns the forward discount
/// factor from calendar time s = T - τ to expiry T.
///
/// @param spec Rate specification (constant or yield curve)
/// @param maturity Total time to maturity T
inline std::function<double(double)> make_forward_discount_fn(const RateSpec& spec, double maturity) {
    return std::visit([maturity](const auto& arg) -> std::function<double(double)> {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, double>) {
            // For constant rate: forward discount = exp(-r*τ)
            return [r = arg](double tau) { return std::exp(-r * tau); };
        } else {
            // Forward discount from s to T: D(T)/D(s) where s = T - τ
            return [curve = arg, maturity](double tau) {
                double s = std::max(0.0, maturity - tau);
                double D_T = curve.discount(maturity);
                double D_s = curve.discount(s);
                return D_T / D_s;
            };
        }
    }, spec);
}

/// Helper to extract zero rate at a specific maturity from RateSpec
///
/// Returns the continuously compounded rate such that exp(-zero_rate*T) = D(T).
/// For constant rate, returns the constant.
/// For YieldCurve, returns -ln(D(T))/T.
inline double get_zero_rate(const RateSpec& spec, double maturity) {
    return std::visit([maturity](const auto& arg) -> double {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, double>) {
            return arg;
        } else {
            return arg.zero_rate(maturity);
        }
    }, spec);
}

/**
 * Option type enumeration.
 */
enum class OptionType {
    CALL,
    PUT
};

/// Discrete dividend event
///
/// Represents a known future dividend payment at a specific calendar time.
/// Calendar time is measured in years from the valuation date.
struct Dividend {
    double calendar_time = 0.0;  ///< Years from valuation date
    double amount = 0.0;         ///< Dollar amount
};

/**
 * @brief Complete specification of an option contract
 *
 * This is a POD struct that can be trivially copied and used in
 * batch processing contexts. All parameters are in consistent units:
 * - Prices in dollars
 * - Time in years
 * - Rates as decimals (e.g., 0.05 for 5%)
 *
 * Note: This struct does NOT include volatility, as it's used for
 * IV solving (where volatility is the unknown). For pricing with
 * known volatility, see PricingParams.
 */
struct OptionSpec {
    double spot = 0.0;             ///< Current spot price (S)
    double strike = 0.0;           ///< Strike price (K)
    double maturity = 0.0;         ///< Time to maturity in years (T)
    RateSpec rate = 0.0;           ///< Risk-free rate (constant or yield curve)
    double dividend_yield = 0.0;   ///< Continuous dividend yield (annualized, decimal)
    OptionType option_type = OptionType::CALL; ///< CALL or PUT (default CALL)
};

/**
 * @brief Validate option specification parameters
 *
 * Checks for:
 * - Positive prices (spot, strike)
 * - Positive maturity
 * - Finite and reasonable rate/dividend values
 *
 * @param spec Option specification to validate
 * @return void on success, ValidationError on failure
 */
std::expected<void, ValidationError> validate_option_spec(const OptionSpec& spec);

/**
 * @brief IV solver query: option spec + observed market price
 *
 * This struct contains everything needed to solve for implied volatility:
 * the option contract specification and the observed market price.
 *
 * Inherits from OptionSpec to provide direct access to spot, strike,
 * maturity, rate, dividend_yield, and type fields.
 */
struct IVQuery : OptionSpec {
    double market_price = 0.0;    ///< Observed market price to match

    IVQuery() = default;

    IVQuery(const OptionSpec& spec, double market_price_)
        : OptionSpec(spec), market_price(market_price_) {}
};

/**
 * @brief Validate IV query (option spec + market price)
 *
 * Performs comprehensive validation:
 * - Option spec validation (via validate_option_spec)
 * - Market price: finite, positive
 * - Arbitrage checks: price <= upper bound, price >= intrinsic value
 *
 * @param query IV query to validate
 * @return void on success, ValidationError on failure
 */
std::expected<void, ValidationError> validate_iv_query(const IVQuery& query);

/**
 * @brief Complete pricing parameters including volatility
 *
 * This struct contains all parameters needed for option pricing:
 * the option contract specification plus volatility and optional
 * discrete dividends.
 *
 * Inherits from OptionSpec to provide direct access to spot, strike,
 * maturity, rate, dividend_yield, and type fields.
 *
 * All parameters are in consistent units:
 * - Prices in dollars
 * - Time in years
 * - Rates and volatility as decimals (e.g., 0.05 for 5%)
 */
struct PricingParams : OptionSpec {
    double volatility = 0.0;  ///< Volatility (fraction, annualized)

    /// Discrete dividend schedule: (time, amount) pairs
    /// Time is in years from now, amount is in dollars
    /// Can be used simultaneously with dividend_yield
    std::vector<Dividend> discrete_dividends;

    PricingParams() = default;

    PricingParams(const OptionSpec& spec,
                  double volatility_,
                  std::vector<Dividend> discrete_dividends_ = {})
        : OptionSpec(spec)
        , volatility(volatility_)
        , discrete_dividends(std::move(discrete_dividends_))
    {}


};

/**
 * @brief Validate pricing parameters
 *
 * Checks for:
 * - Positive prices (spot, strike)
 * - Positive maturity
 * - Positive volatility
 * - Finite and reasonable rate/dividend values
 * - Valid discrete dividend schedule
 *
 * @param params Pricing parameters to validate
 * @return void on success, ValidationError on failure
 */
std::expected<void, ValidationError> validate_pricing_params(const PricingParams& params);

} // namespace mango
