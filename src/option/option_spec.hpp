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
#include <initializer_list>
#include <variant>
#include <functional>

namespace mango {

/// Rate specification: constant or yield curve
using RateSpec = std::variant<double, YieldCurve>;

/// Helper to extract rate function from RateSpec
///
/// Returns a callable that takes time t and returns the rate at that time.
/// For constant rate, returns the constant regardless of t.
/// For YieldCurve, delegates to curve.rate(t).
inline std::function<double(double)> make_rate_fn(const RateSpec& spec) {
    return std::visit([](const auto& arg) -> std::function<double(double)> {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, double>) {
            return [r = arg](double) { return r; };
        } else {
            // Capture by value to ensure curve lifetime
            return [curve = arg](double t) { return curve.rate(t); };
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
 * known volatility, see AmericanOptionParams.
 */
struct OptionSpec {
    double spot = 0.0;             ///< Current spot price (S)
    double strike = 0.0;           ///< Strike price (K)
    double maturity = 0.0;         ///< Time to maturity in years (T)
    RateSpec rate = 0.0;           ///< Risk-free rate (constant or yield curve)
    double dividend_yield = 0.0;   ///< Continuous dividend yield (annualized, decimal)
    OptionType type = OptionType::CALL; ///< CALL or PUT (default CALL)
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

    IVQuery(double spot_,
            double strike_,
            double maturity_,
            double rate_,
            double dividend_yield_,
            OptionType type_,
            double market_price_)
        : market_price(market_price_)
    {
        spot = spot_;
        strike = strike_;
        maturity = maturity_;
        rate = rate_;
        dividend_yield = dividend_yield_;
        type = type_;
    }
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
 * @brief Option solver grid configuration
 *
 * Specifies the computational grid and option parameters for
 * solving the option pricing PDE using finite difference methods.
 */
struct OptionSolverGrid {
    OptionType option_type;    ///< Call or Put
    double x_min;              ///< Minimum log-moneyness
    double x_max;              ///< Maximum log-moneyness
    size_t n_space;            ///< Number of spatial grid points
    size_t n_time;             ///< Number of time steps
    double dividend_yield;     ///< Continuous dividend yield
};

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
    std::vector<std::pair<double, double>> discrete_dividends;

    PricingParams() = default;

    PricingParams(const OptionSpec& spec,
                  double volatility_,
                  std::vector<std::pair<double, double>> discrete_dividends_ = {})
        : OptionSpec(spec)
        , volatility(volatility_)
        , discrete_dividends(std::move(discrete_dividends_))
    {}

    PricingParams(double spot_,
                  double strike_,
                  double maturity_,
                  double rate_,
                  double dividend_yield_,
                  OptionType type_,
                  double volatility_,
                  std::vector<std::pair<double, double>> discrete_dividends_ = {})
        : volatility(volatility_)
        , discrete_dividends(std::move(discrete_dividends_))
    {
        spot = spot_;
        strike = strike_;
        maturity = maturity_;
        rate = rate_;
        dividend_yield = dividend_yield_;
        type = type_;
    }

    PricingParams(double spot_,
                  double strike_,
                  double maturity_,
                  double rate_,
                  double dividend_yield_,
                  OptionType type_,
                  double volatility_,
                  std::initializer_list<std::pair<double, double>> discrete_dividends_)
        : PricingParams(spot_, strike_, maturity_, rate_, dividend_yield_, type_, volatility_,
                        std::vector<std::pair<double, double>>(discrete_dividends_))
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
