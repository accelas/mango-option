// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/spliced_surface.hpp"
#include "mango/support/error_types.hpp"
#include <expected>
#include <vector>

namespace mango {

// ===========================================================================
// Segmented surface builder
// ===========================================================================

struct SegmentConfig {
    AmericanPriceSurface surface;
    double tau_start;
    double tau_end;
};

struct SegmentedConfig {
    std::vector<SegmentConfig> segments;
    std::vector<Dividend> dividends;
    double K_ref;
    double T;  // expiry in calendar time
};

[[nodiscard]] std::expected<SegmentedSurface<>, PriceTableError>
build_segmented_surface(SegmentedConfig config);

// ===========================================================================
// Multi-K_ref surface builder
// ===========================================================================

struct MultiKRefEntry {
    double K_ref;
    SegmentedSurface<> surface;
};

struct StrikeEntry {
    double strike;
    SegmentedSurface<> surface;
};

[[nodiscard]] std::expected<MultiKRefSurface<>, PriceTableError>
build_multi_kref_surface(std::vector<MultiKRefEntry> entries);

[[nodiscard]] std::expected<StrikeSurface<>, PriceTableError>
build_strike_surface(std::vector<StrikeEntry> entries,
                     bool use_nearest = true);

}  // namespace mango
