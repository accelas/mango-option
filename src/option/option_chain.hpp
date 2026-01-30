// SPDX-License-Identifier: MIT
#pragma once

#include <vector>
#include <string>

namespace mango {

/// Market option chain data (from exchanges)
///
/// Represents raw option chain data as typically received from market data
/// feeds or exchanges. Can contain duplicate strikes/maturities (e.g., multiple
/// options with same parameters but different bid/ask spreads).
///
/// Extracted from PriceTable4DBuilder for reusability.
struct OptionChain {
    std::string ticker;                  ///< Underlying ticker symbol
    double spot = 0.0;                   ///< Current underlying price
    std::vector<double> strikes;         ///< Strike prices (may have duplicates)
    std::vector<double> maturities;      ///< Times to expiration in years (may have duplicates)
    std::vector<double> implied_vols;    ///< Market implied volatilities (for grid)
    std::vector<double> rates;           ///< Risk-free rates (may have duplicates)
    double dividend_yield = 0.0;         ///< Continuous dividend yield
};

} // namespace mango
