// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/spliced_surface.hpp"
#include "mango/support/error_types.hpp"
#include <expected>
#include <memory>
#include <vector>

namespace mango {

// ===========================================================================
// Per-maturity surface builder
// ===========================================================================

struct PerMaturityConfig {
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces;
    std::vector<double> tau_grid;
    double K_ref;
    OptionType option_type;
    double dividend_yield;
};

[[nodiscard]] std::expected<PerMaturitySurface, PriceTableError>
build_per_maturity_surface(PerMaturityConfig config);

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

[[nodiscard]] std::expected<MultiKRefSurface<>, PriceTableError>
build_multi_kref_surface(std::vector<MultiKRefEntry> entries);

}  // namespace mango
