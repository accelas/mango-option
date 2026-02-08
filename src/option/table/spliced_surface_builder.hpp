// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/standard_surface.hpp"
#include "mango/support/error_types.hpp"
#include <expected>
#include <memory>
#include <vector>

namespace mango {

template <size_t N> class PriceTableSurface;

// ===========================================================================
// Segmented surface builder
// ===========================================================================

struct SegmentConfig {
    std::shared_ptr<const PriceTableSurface<4>> surface;
    double tau_start;
    double tau_end;
};

struct SegmentedConfig {
    std::vector<SegmentConfig> segments;
    double K_ref;
};

[[nodiscard]] std::expected<SegmentedSurfacePI, PriceTableError>
build_segmented_surface(SegmentedConfig config);

// ===========================================================================
// Multi-K_ref surface builder
// ===========================================================================

struct MultiKRefEntry {
    double K_ref;
    SegmentedSurfacePI surface;
};

[[nodiscard]] std::expected<MultiKRefSurfacePI, PriceTableError>
build_multi_kref_surface(std::vector<MultiKRefEntry> entries);

}  // namespace mango
