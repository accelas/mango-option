#pragma once

#include "src/option/american_option.hpp"  // For OptionType
#include "src/pde/core/grid.hpp"  // For GridSpec
#include <vector>
#include <utility>
#include <optional>

namespace mango {

/// Configuration for price table pre-computation
struct PriceTableConfig {
    OptionType option_type = OptionType::PUT;  ///< Option type (call/put)
    double K_ref = 100.0;                      ///< Reference strike price for normalization
    GridSpec<double> grid_estimator = GridSpec<double>::uniform(-3.0, 3.0, 101).value();  ///< Grid for PDE solves
    size_t n_time = 1000;                      ///< Time steps for TR-BDF2
    double dividend_yield = 0.0;               ///< Continuous dividend yield
    std::vector<std::pair<double, double>> discrete_dividends;  ///< (time, amount) schedule
};

} // namespace mango
