// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <vector>
#include "src/option/option_spec.hpp"

namespace mango {

/// What the surface tensor contains
enum class SurfaceContent : uint8_t {
    RawPrice = 0,              ///< Raw American option prices
    EarlyExercisePremium = 1   ///< P_Am - P_Eu (requires reconstruction)
};

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
    SurfaceContent content = SurfaceContent::EarlyExercisePremium;  ///< What tensor stores
    std::vector<Dividend> discrete_dividends;  ///< Discrete dividend schedule
};

} // namespace mango
