// SPDX-License-Identifier: MIT
#pragma once

#include "src/option/option_spec.hpp"
#include <expected>
#include <string>
#include <vector>

namespace mango::simple {

using mango::OptionType;
using mango::PricingParams;
using mango::IVQuery;

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

/// Result for a single option in a batch
struct BatchPriceResult {
    std::vector<std::expected<double, std::string>> prices;
    size_t failed_count = 0;
};

/// Result for a single IV query in a batch
struct BatchIVResult {
    std::vector<std::expected<double, std::string>> vols;
    size_t failed_count = 0;
};

/// Price a batch of American options in parallel
///
/// Uses BatchAmericanOptionSolver with automatic routing to the
/// normalized chain solver when eligible (same maturity, no discrete
/// dividends). This provides up to 19,000x speedup for option chains.
///
/// @param params Vector of pricing parameters
/// @return BatchPriceResult with per-option prices and failure count
BatchPriceResult price_batch(const std::vector<PricingParams>& params);

/// Compute implied volatility for a batch of options in parallel
///
/// Uses IVSolverFDM with OpenMP parallelization. Each query is solved
/// independently using Brent's method.
///
/// @param queries Vector of IV queries
/// @return BatchIVResult with per-query IVs and failure count
BatchIVResult implied_vol_batch(const std::vector<IVQuery>& queries);

}  // namespace mango::simple
