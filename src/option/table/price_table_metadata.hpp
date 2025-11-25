#pragma once

#include <vector>
#include <utility>

namespace mango {

/// Metadata for price table surface
///
/// Stores reference strike, dividend information, and discrete dividend schedule.
struct PriceTableMetadata {
    double K_ref = 0.0;                                     ///< Reference strike price
    double dividend_yield = 0.0;                            ///< Continuous dividend yield
    std::vector<std::pair<double, double>> discrete_dividends;  ///< (time, amount) pairs
};

} // namespace mango
