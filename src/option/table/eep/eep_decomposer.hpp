// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cmath>

namespace mango {

/// Debiased softplus floor for EEP non-negativity.
///
/// Smoothly clamps eep_raw = (American - European) to non-negative values.
/// Zero-bias correction ensures eep_floor(0) == 0 exactly.
///
/// This is the shared primitive for EEP decomposition — callers compute
/// (American - European) using whatever European source they have
/// (analytical Black-Scholes, numerical PDE, etc).
inline double eep_floor(double eep_raw) {
    constexpr double kSharpness = 100.0;
    if (kSharpness * eep_raw > 500.0) {
        return eep_raw;
    }
    double softplus = std::log1p(std::exp(kSharpness * eep_raw)) / kSharpness;
    double bias = std::log(2.0) / kSharpness;
    return std::max(0.0, softplus - bias);
}

}  // namespace mango
