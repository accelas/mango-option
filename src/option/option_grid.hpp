// SPDX-License-Identifier: MIT
#pragma once

#include <vector>
#include <string>

namespace mango {

/// Option parameter grid for price table construction
///
/// Represents a grid of option parameters used to build price tables.
/// Can contain duplicate strikes/maturities which are deduplicated during
/// grid construction.
///
/// Extracted from PriceTable4DBuilder for reusability.
struct OptionGrid {
    std::string ticker;                  ///< Underlying ticker symbol
    double spot = 0.0;                   ///< Current underlying price
    std::vector<double> strikes;         ///< Strike prices (may have duplicates)
    std::vector<double> maturities;      ///< Times to expiration in years (may have duplicates)
    std::vector<double> implied_vols;    ///< Market implied volatilities (for grid)
    std::vector<double> rates;           ///< Risk-free rates (may have duplicates)
    double dividend_yield = 0.0;         ///< Continuous dividend yield
};

} // namespace mango
