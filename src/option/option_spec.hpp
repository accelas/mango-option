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

} // namespace mango
