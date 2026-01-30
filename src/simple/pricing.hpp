// SPDX-License-Identifier: MIT
#pragma once

#include "src/option/option_spec.hpp"
#include <expected>
#include <string>

namespace mango::simple {

using mango::OptionType;

/// Price an American option using finite difference methods
///
/// Convenience wrapper around AmericanOptionSolver with automatic grid estimation.
///
/// @param spot Current underlying price
/// @param strike Strike price
/// @param maturity Time to expiry in years
/// @param volatility Annualized volatility (decimal, e.g. 0.20 for 20%)
/// @param rate Risk-free rate (decimal)
/// @param dividend_yield Continuous dividend yield (default 0.0)
/// @param type Option type (default PUT)
/// @return Option price or error string
std::expected<double, std::string> price(
    double spot, double strike, double maturity,
    double volatility, double rate,
    double dividend_yield = 0.0,
    OptionType type = OptionType::PUT);

/// Compute implied volatility for an American option using FDM
///
/// Convenience wrapper around IVSolverFDM with default configuration.
///
/// @param spot Current underlying price
/// @param strike Strike price
/// @param maturity Time to expiry in years
/// @param market_price Observed market price
/// @param rate Risk-free rate (decimal)
/// @param dividend_yield Continuous dividend yield (default 0.0)
/// @param type Option type (default PUT)
/// @return Implied volatility or error string
std::expected<double, std::string> implied_vol(
    double spot, double strike, double maturity,
    double market_price, double rate,
    double dividend_yield = 0.0,
    OptionType type = OptionType::PUT);

}  // namespace mango::simple
