// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/spliced_surface.hpp"
#include "mango/option/table/standard_surface.hpp"
#include "mango/support/error_types.hpp"
#include <expected>
#include <memory>
#include <vector>

namespace mango {

template <size_t N> class PriceTableSurface;

// ===========================================================================
// Standard surface builder
// ===========================================================================

/// Build a StandardSurfaceWrapper from a pre-built EEP price table surface.
///
/// Wraps the surface in EEPPriceTableInner (query-time EEP reconstruction)
/// and SplicedSurfaceWrapper (PriceSurface-compatible bounds).
///
/// @param surface EEP price table surface (from PriceTableBuilder)
/// @param option_type PUT or CALL
/// @param dividend_yield Continuous dividend yield
/// @return StandardSurfaceWrapper or error
[[nodiscard]] std::expected<StandardSurfaceWrapper, PriceTableError>
build_standard_surface(std::shared_ptr<const PriceTableSurface<4>> surface,
                       OptionType option_type,
                       double dividend_yield);

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
    double K_ref;
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
