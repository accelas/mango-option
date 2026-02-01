// SPDX-License-Identifier: MIT
#pragma once

#include <vector>
#include <expected>
#include "src/option/table/segmented_multi_kref_surface.hpp"
#include "src/option/table/segmented_price_table_builder.hpp"
#include "src/option/option_spec.hpp"
#include "src/support/error_types.hpp"

namespace mango {

struct MultiKRefConfig {
    std::vector<double> K_refs;   // explicit list; if empty, use auto selection
    int K_ref_count = 11;         // used when K_refs is empty
    double K_ref_span = 0.3;      // +/-span around spot for auto mode (log-spaced)
};

class SegmentedMultiKRefBuilder {
public:
    struct Config {
        double spot;
        OptionType option_type;
        double dividend_yield = 0.0;
        std::vector<std::pair<double, double>> dividends;  // (calendar_time, amount)
        std::vector<double> moneyness_grid;
        double maturity;
        std::vector<double> vol_grid;
        std::vector<double> rate_grid;
        MultiKRefConfig kref_config;
    };

    static std::expected<SegmentedMultiKRefSurface, ValidationError> build(const Config& config);
};

}  // namespace mango
