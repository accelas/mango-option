// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/grid_spec_types.hpp"  // PDEGridSpec, PDEGridConfig, GridAccuracyParams
#include "mango/option/option_spec.hpp"      // OptionType
#include "mango/option/table/price_table_metadata.hpp"  // For SurfaceContent
#include <utility>
#include <optional>
#include <string>

namespace mango {

/// Configuration for price table pre-computation
struct PriceTableConfig {
    OptionType option_type = OptionType::PUT;  ///< Option type (call/put)
    double K_ref = 100.0;                      ///< Reference strike price for normalization
    PDEGridSpec pde_grid = GridAccuracyParams{};  ///< PDE grid: explicit or auto-estimated
    DividendSpec dividends;                    ///< Continuous yield + discrete schedule
    double max_failure_rate = 0.0;             ///< Maximum tolerable failure rate: 0.0 = strict, 0.1 = allow 10%
    SurfaceContent surface_content = SurfaceContent::EarlyExercisePremium;  ///< Output mode
    bool allow_tau_zero = false;               ///< Allow τ=0 in maturity grid (requires custom IC)
};

/// Validate PriceTableConfig fields
/// @param config Configuration to validate
/// @return Error message if invalid, nullopt if valid
inline std::optional<std::string> validate_config(const PriceTableConfig& config) {
    if (config.max_failure_rate < 0.0 || config.max_failure_rate > 1.0) {
        return "max_failure_rate must be in [0.0, 1.0], got " +
               std::to_string(config.max_failure_rate);
    }
    return std::nullopt;
}

} // namespace mango
