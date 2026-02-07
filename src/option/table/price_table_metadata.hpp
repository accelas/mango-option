// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <vector>
#include "mango/option/option_spec.hpp"

namespace mango {

/// What the surface tensor contains
enum class SurfaceContent : uint8_t {
    RawPrice = 0,              ///< Raw American option prices
    EarlyExercisePremium = 1   ///< P_Am - P_Eu (requires reconstruction)
};

/// Metadata for price table surface
///
/// Stores reference strike, dividend information, log-moneyness bounds,
/// and discrete dividend schedule.
struct PriceTableMetadata {
    double K_ref = 0.0;                                     ///< Reference strike price
    DividendSpec dividends;                                  ///< Continuous yield + discrete schedule
    double m_min = 0.0;                                     ///< Minimum log-moneyness ln(S/K)
    double m_max = 0.0;                                     ///< Maximum log-moneyness ln(S/K)
    SurfaceContent content = SurfaceContent::EarlyExercisePremium;  ///< What tensor stores
};

} // namespace mango
