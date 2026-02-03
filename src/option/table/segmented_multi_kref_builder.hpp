// SPDX-License-Identifier: MIT
#pragma once

#include <vector>
#include <expected>
#include "mango/option/table/segmented_multi_kref_surface.hpp"
#include "mango/option/table/segmented_price_table_builder.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/support/error_types.hpp"

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
        DividendSpec dividends;  ///< Continuous yield + discrete schedule
        std::vector<double> moneyness_grid;
        double maturity;
        std::vector<double> vol_grid;
        std::vector<double> rate_grid;
        MultiKRefConfig kref_config;
        int tau_points_per_segment = 5;
        bool skip_moneyness_expansion = false;
    };

    static std::expected<SegmentedMultiKRefSurface, ValidationError> build(const Config& config);
};

}  // namespace mango
