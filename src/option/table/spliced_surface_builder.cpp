// SPDX-License-Identifier: MIT

#include "mango/option/table/spliced_surface_builder.hpp"
#include "mango/option/table/price_table_surface.hpp"
#include <algorithm>

namespace mango {

// ===========================================================================
// Standard surface builder
// ===========================================================================

std::expected<StandardSurfaceWrapper, PriceTableError>
build_standard_surface(std::shared_ptr<const PriceTableSurface<4>> surface,
                       OptionType option_type,
                       double dividend_yield) {
    if (!surface) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig, 0, 0});
    }

    const auto& meta = surface->metadata();
    const auto& axes = surface->axes();
    double K_ref = meta.K_ref;

    // Create EEP-reconstructing inner adapter
    EEPPriceTableInner inner(surface, option_type, K_ref, dividend_yield);

    // Wrap in StandardSurface (1 slice, identity transform)
    StandardSurface std_surface(
        {std::move(inner)}, SingleBracket{}, IdentityTransform{}, WeightedSum{});

    // Extract bounds from surface metadata/axes
    StandardSurfaceWrapper::Bounds bounds{
        .m_min = meta.m_min, .m_max = meta.m_max,
        .tau_min = axes.grids[1].front(), .tau_max = axes.grids[1].back(),
        .sigma_min = axes.grids[2].front(), .sigma_max = axes.grids[2].back(),
        .rate_min = axes.grids[3].front(), .rate_max = axes.grids[3].back(),
    };

    return StandardSurfaceWrapper(
        std::move(std_surface), bounds, option_type, dividend_yield);
}

// ===========================================================================
// Segmented surface builder
// ===========================================================================

std::expected<SegmentedSurfacePI, PriceTableError>
build_segmented_surface(SegmentedConfig config) {
    // Validate: non-empty segments
    if (config.segments.empty()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig, 0, 0});
    }

    // Extract tau bounds and build slices
    std::vector<double> tau_start;
    std::vector<double> tau_end;
    std::vector<double> tau_min;
    std::vector<double> tau_max;
    std::vector<PriceTableInner> slices;

    tau_start.reserve(config.segments.size());
    tau_end.reserve(config.segments.size());
    tau_min.reserve(config.segments.size());
    tau_max.reserve(config.segments.size());
    slices.reserve(config.segments.size());

    for (auto& seg : config.segments) {
        tau_start.push_back(seg.tau_start);
        tau_end.push_back(seg.tau_end);
        tau_min.push_back(seg.surface->axes().grids[1].front());
        tau_max.push_back(seg.surface->axes().grids[1].back());
        slices.emplace_back(seg.surface);
    }

    // Construct SegmentLookup
    SegmentLookup lookup(tau_start, tau_end);

    // Construct SegmentedTransform
    SegmentedTransform xform{
        .tau_start = std::move(tau_start),
        .tau_min = std::move(tau_min),
        .tau_max = std::move(tau_max),
        .K_ref = config.K_ref,
    };

    WeightedSum combiner;

    return SegmentedSurfacePI(
        std::move(slices),
        std::move(lookup),
        std::move(xform),
        std::move(combiner));
}

// ===========================================================================
// Multi-K_ref surface builder
// ===========================================================================

std::expected<MultiKRefSurfacePI, PriceTableError>
build_multi_kref_surface(std::vector<MultiKRefEntry> entries) {
    // Validate: non-empty
    if (entries.empty()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig, 0, 0});
    }

    // Sort entries by K_ref
    std::sort(entries.begin(), entries.end(),
              [](const MultiKRefEntry& a, const MultiKRefEntry& b) {
                  return a.K_ref < b.K_ref;
              });

    // Extract k_refs vector and slices
    std::vector<double> k_refs;
    std::vector<SegmentedSurfacePI> slices;
    k_refs.reserve(entries.size());
    slices.reserve(entries.size());

    for (auto& entry : entries) {
        k_refs.push_back(entry.K_ref);
        slices.push_back(std::move(entry.surface));
    }

    // Construct KRefBracket
    KRefBracket bracket(k_refs);

    // Construct KRefTransform
    KRefTransform xform{.k_refs = std::move(k_refs)};

    WeightedSum combiner;

    return MultiKRefSurfacePI(
        std::move(slices),
        std::move(bracket),
        std::move(xform),
        std::move(combiner));
}

}  // namespace mango
