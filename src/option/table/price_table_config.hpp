// SPDX-License-Identifier: MIT
#pragma once

#include "src/option/american_option.hpp"  // For OptionType
#include "src/pde/core/grid.hpp"  // For GridSpec
#include <variant>
#include <vector>
#include <utility>
#include <optional>
#include <string>

namespace mango {

/// Explicit PDE grid: caller-specified spatial grid and time steps
struct ExplicitPDEGrid {
    GridSpec<double> grid_spec = GridSpec<double>::uniform(-3.0, 3.0, 101).value();
    size_t n_time = 1000;
};

/// PDE grid specification: either explicit grid or auto-estimated from accuracy params
using PDEGridSpec = std::variant<ExplicitPDEGrid, GridAccuracyParams>;

/// Configuration for price table pre-computation
struct PriceTableConfig {
    OptionType option_type = OptionType::PUT;  ///< Option type (call/put)
    double K_ref = 100.0;                      ///< Reference strike price for normalization
    PDEGridSpec pde_grid;                      ///< PDE grid: explicit or auto-estimated
    double dividend_yield = 0.0;               ///< Continuous dividend yield
    std::vector<std::pair<double, double>> discrete_dividends;  ///< (time, amount) schedule
    double max_failure_rate = 0.0;             ///< Maximum tolerable failure rate: 0.0 = strict, 0.1 = allow 10%
    bool store_eep = true;                     ///< Store early exercise premium instead of raw prices
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
