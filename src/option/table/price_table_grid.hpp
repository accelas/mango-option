// SPDX-License-Identifier: MIT
/**
 * @file price_table_grid.hpp
 * @brief Grid specification for price table construction
 */

#pragma once

#include <span>

namespace mango {

/**
 * @brief Price table grid specification
 *
 * Represents a 4D grid of option parameters to be solved.
 * Output will be prices_4d[i, j, k, l] for (moneyness[i], maturity[j], volatility[k], rate[l]).
 */
struct PriceTableGrid {
    std::span<const double> moneyness;   ///< Moneyness grid (M/K_ref)
    std::span<const double> maturity;    ///< Time to maturity grid (years)
    std::span<const double> volatility;  ///< Volatility grid
    std::span<const double> rate;        ///< Interest rate grid
    double K_ref;                         ///< Reference strike price

    /// Get total number of grid points
    size_t size() const {
        return moneyness.size() * maturity.size() * volatility.size() * rate.size();
    }
};

} // namespace mango
