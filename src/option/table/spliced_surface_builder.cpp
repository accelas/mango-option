// SPDX-License-Identifier: MIT

#include "mango/option/table/spliced_surface_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include <algorithm>

namespace mango {

// ===========================================================================
// Segmented surface builder
// ===========================================================================

std::expected<SegmentedPriceSurface, PriceTableError>
build_segmented_surface(SegmentedConfig config) {
    if (config.segments.empty()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig, 0, 0});
    }

    std::vector<double> tau_start;
    std::vector<double> tau_end;
    std::vector<double> tau_min;
    std::vector<double> tau_max;
    std::vector<SegmentedLeaf> leaves;

    tau_start.reserve(config.segments.size());
    tau_end.reserve(config.segments.size());
    tau_min.reserve(config.segments.size());
    tau_max.reserve(config.segments.size());
    leaves.reserve(config.segments.size());

    for (auto& seg : config.segments) {
        tau_start.push_back(seg.tau_start);
        tau_end.push_back(seg.tau_end);
        tau_min.push_back(seg.surface->axes().grids[1].front());
        tau_max.push_back(seg.surface->axes().grids[1].back());

        SharedBSplineInterp<4> interp(seg.surface);
        StandardTransform4D xform;
        IdentityEEP eep;
        leaves.emplace_back(std::move(interp), xform, eep, config.K_ref);
    }

    TauSegmentSplit split(
        std::move(tau_start), std::move(tau_end),
        std::move(tau_min), std::move(tau_max),
        config.K_ref);

    return SegmentedPriceSurface(std::move(leaves), std::move(split));
}

// ===========================================================================
// Multi-K_ref surface builder
// ===========================================================================

std::expected<MultiKRefInner, PriceTableError>
build_multi_kref_surface(std::vector<MultiKRefEntry> entries) {
    if (entries.empty()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig, 0, 0});
    }

    // Sort entries by K_ref
    std::sort(entries.begin(), entries.end(),
              [](const MultiKRefEntry& a, const MultiKRefEntry& b) {
                  return a.K_ref < b.K_ref;
              });

    std::vector<double> k_refs;
    std::vector<SegmentedPriceSurface> slices;
    k_refs.reserve(entries.size());
    slices.reserve(entries.size());

    for (auto& entry : entries) {
        k_refs.push_back(entry.K_ref);
        slices.push_back(std::move(entry.surface));
    }

    MultiKRefSplit split(std::move(k_refs));

    return MultiKRefInner(std::move(slices), std::move(split));
}

}  // namespace mango
