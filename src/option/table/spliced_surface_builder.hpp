// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/standard_surface.hpp"
#include "mango/support/error_types.hpp"
#include <expected>
#include <memory>
#include <vector>

namespace mango {

// ===========================================================================
// Segmented surface builder
// ===========================================================================

struct SegmentConfig {
    std::shared_ptr<const PriceTableSurface> surface;
    double tau_start;
    double tau_end;
};

struct SegmentedConfig {
    std::vector<SegmentConfig> segments;
    double K_ref;
};

[[nodiscard]] std::expected<SegmentedPriceSurface, PriceTableError>
build_segmented_surface(SegmentedConfig config);

// ===========================================================================
// Multi-K_ref surface builder
// ===========================================================================

struct MultiKRefEntry {
    double K_ref;
    SegmentedPriceSurface surface;
};

[[nodiscard]] std::expected<MultiKRefPriceSurface, PriceTableError>
build_multi_kref_surface(std::vector<MultiKRefEntry> entries);

}  // namespace mango
