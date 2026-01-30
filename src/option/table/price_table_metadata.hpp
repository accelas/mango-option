// SPDX-License-Identifier: MIT
#pragma once

#include <vector>
#include <utility>

namespace mango {

/// Metadata for price table surface
///
/// Stores reference strike, dividend information, moneyness bounds,
/// and discrete dividend schedule.
///
/// The moneyness bounds (m_min, m_max) store the original user-specified
/// moneyness range. Internally, the price table stores log-moneyness for
/// better B-spline interpolation, but the user-facing API remains in moneyness.
struct PriceTableMetadata {
    double K_ref = 0.0;                                     ///< Reference strike price
    double dividend_yield = 0.0;                            ///< Continuous dividend yield
    double m_min = 0.0;                                     ///< Minimum moneyness (S/K)
    double m_max = 0.0;                                     ///< Maximum moneyness (S/K)
    std::vector<std::pair<double, double>> discrete_dividends;  ///< (time, amount) pairs
};

} // namespace mango
