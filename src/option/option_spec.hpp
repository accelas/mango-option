/**
 * @file option_spec.hpp
 * @brief Unified option specification for IV solvers
 */

#pragma once

#include "src/option/american_option.hpp"  // For OptionType enum
#include "src/support/expected.hpp"
#include <string>

namespace mango {

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
expected<void, std::string> validate_option_spec(const OptionSpec& spec);

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
expected<void, std::string> validate_iv_query(const IVQuery& query);

} // namespace mango
