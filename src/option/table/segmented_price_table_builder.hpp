// SPDX-License-Identifier: MIT
#pragma once

#include <vector>
#include <expected>
#include "src/option/table/segmented_price_surface.hpp"
#include "src/option/option_spec.hpp"
#include "src/support/error_types.hpp"

namespace mango {

/// Orchestrates backward-chained construction of a SegmentedPriceSurface for a
/// single K_ref.  Splits maturity at discrete dividend dates, builds the last
/// segment normally (EEP, payoff IC), and chains earlier segments backward
/// using RawPrice mode with initial conditions sourced from the previous
/// segment's surface.
class SegmentedPriceTableBuilder {
public:
    struct Config {
        double K_ref;
        OptionType option_type;
        DividendSpec dividends;  ///< Continuous yield + discrete schedule
        std::vector<double> moneyness_grid;
        double maturity;  // T in years
        std::vector<double> vol_grid;
        std::vector<double> rate_grid;
        /// Minimum tau points per segment (actual count may be higher)
        int tau_points_per_segment = 5;
        /// If true, skip internal moneyness expansion (caller pre-expanded).
        bool skip_moneyness_expansion = false;
    };

    /// Build a SegmentedPriceSurface from the given configuration.
    ///
    /// Algorithm:
    ///   1. Filter dividends outside (0, T), sort, compute segment boundaries in τ.
    ///   2. Expand moneyness grid downward to accommodate spot adjustment.
    ///   3. Build last segment (closest to expiry) with EEP and payoff IC.
    ///   4. Build earlier segments backward with RawPrice and chained IC.
    ///   5. Assemble into SegmentedPriceSurface.
    static std::expected<SegmentedPriceSurface, ValidationError> build(const Config& config);
};

}  // namespace mango
