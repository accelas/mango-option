/**
 * @file option_spec.hpp
 * @brief Unified option specification for IV solvers
 */

#pragma once

#include <expected>
#include "src/support/error_types.hpp"
#include <string>

namespace mango {

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
    double spot;             ///< Current spot price (S)
    double strike;           ///< Strike price (K)
    double maturity;         ///< Time to maturity in years (T)
    double rate;             ///< Risk-free rate (annualized, decimal)
    double dividend_yield = 0.0;  ///< Continuous dividend yield (annualized, decimal)
    OptionType type;         ///< CALL or PUT
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
 * @return void on success, error message on failure
 */
std::expected<void, std::string> validate_option_spec(const OptionSpec& spec);

/**
 * @brief IV solver query: option spec + observed market price
 *
 * This struct contains everything needed to solve for implied volatility:
 * the option contract specification and the observed market price.
 */
struct IVQuery {
    OptionSpec option;      ///< Option contract specification
    double market_price;    ///< Observed market price to match
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
 * @return void on success, error message on failure
 */
std::expected<void, std::string> validate_iv_query(const IVQuery& query);

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
 * All parameters are in consistent units:
 * - Prices in dollars
 * - Time in years
 * - Rates and volatility as decimals (e.g., 0.05 for 5%)
 */
struct PricingParams {
    double strike;                      ///< Strike price (dollars)
    double spot;                        ///< Current spot price (dollars)
    double maturity;                    ///< Time to maturity (years)
    double volatility;                  ///< Volatility (fraction, annualized)
    double rate;                        ///< Risk-free rate (fraction, annualized)
    double continuous_dividend_yield;   ///< Continuous dividend yield (fraction, annualized)
    OptionType option_type;             ///< Call or Put

    /// Discrete dividend schedule: (time, amount) pairs
    /// Time is in years from now, amount is in dollars
    /// Can be used simultaneously with continuous_dividend_yield
    std::vector<std::pair<double, double>> discrete_dividends;
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
 * @return void on success, error message on failure
 */
std::expected<void, std::string> validate_pricing_params(const PricingParams& params);

} // namespace mango
