// SPDX-License-Identifier: MIT

#include "mango/option/table/serialization/from_data.hpp"
#include "mango/option/table/serialization/reconstruct.hpp"

// Surface type headers (define the type aliases)
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/bspline/bspline_3d_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_3d_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"

namespace mango {

// Surface type string constants — must match surface_type_string<> in to_data.cpp.
namespace {
constexpr const char* kBSpline4D            = "bspline_4d";
constexpr const char* kBSpline4DSegmented   = "bspline_4d_segmented";
constexpr const char* kChebyshev4D          = "chebyshev_4d";
constexpr const char* kChebyshev4DRaw       = "chebyshev_4d_raw";
constexpr const char* kChebyshev4DSegmented = "chebyshev_4d_segmented";
constexpr const char* kBSpline3D            = "bspline_3d";
constexpr const char* kChebyshev3D          = "chebyshev_3d";
}  // anonymous namespace

// ============================================================================
// BSplineLeaf (bspline_4d): standard 4D B-spline with EEP
// ============================================================================

template <>
std::expected<PriceTable<BSplineLeaf>, PriceTableError>
from_data<BSplineLeaf>(const PriceTableData& data) {
    if (data.surface_type != kBSpline4D) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig});
    }
    if (data.segments.size() != 1) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig, 0, data.segments.size()});
    }

    const auto& seg = data.segments[0];
    auto leaf = reconstruct_bspline_leaf<4, StandardTransform4D>(seg);
    if (!leaf) return std::unexpected(leaf.error());

    auto eep_leaf = reconstruct_eep(std::move(*leaf),
                                    data.option_type, data.dividend_yield);
    auto bounds = bounds_from_4d_segment(seg);

    return PriceTable<BSplineLeaf>(
        std::move(eep_leaf), bounds, data.option_type, data.dividend_yield);
}

// ============================================================================
// BSplineMultiKRefInner (bspline_4d_segmented): segmented B-spline
// ============================================================================

template <>
std::expected<PriceTable<BSplineMultiKRefInner>, PriceTableError>
from_data<BSplineMultiKRefInner>(const PriceTableData& data) {
    if (data.surface_type != kBSpline4DSegmented) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig});
    }
    if (data.segments.empty()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig, 0, 0});
    }

    auto groups = group_segments_by_kref(data.segments);
    std::vector<double> k_refs;
    std::vector<BSplineSegmentedSurface> kref_surfaces;

    for (const auto& [k_ref, group] : groups) {
        k_refs.push_back(k_ref);
        auto tau_split = reconstruct_bspline_tau_split<4, StandardTransform4D>(
            group, k_ref);
        if (!tau_split) return std::unexpected(tau_split.error());
        kref_surfaces.push_back(std::move(*tau_split));
    }

    MultiKRefSplit multi_split(std::move(k_refs));
    BSplineMultiKRefInner inner(std::move(kref_surfaces), std::move(multi_split));
    auto bounds = bounds_from_segments(data.segments);

    return PriceTable<BSplineMultiKRefInner>(
        std::move(inner), bounds, data.option_type, data.dividend_yield);
}

// ============================================================================
// ChebyshevLeaf (chebyshev_4d): Tucker Chebyshev with EEP
// ============================================================================

// Tucker Chebyshev surfaces serialize to raw values (Tucker expansion).
// Reconstruction as Tucker would require re-decomposition which is not
// supported. Use from_data<ChebyshevRawLeaf> instead, which accepts both
// "chebyshev_4d" and "chebyshev_4d_raw" surface_type strings.
template <>
std::expected<PriceTable<ChebyshevLeaf>, PriceTableError>
from_data<ChebyshevLeaf>(const PriceTableData& data) {
    (void)data;
    return std::unexpected(PriceTableError{
        PriceTableErrorCode::InvalidConfig});
}

// ============================================================================
// ChebyshevRawLeaf (chebyshev_4d_raw): Raw Chebyshev with EEP
// ============================================================================

template <>
std::expected<PriceTable<ChebyshevRawLeaf>, PriceTableError>
from_data<ChebyshevRawLeaf>(const PriceTableData& data) {
    // Accept both "chebyshev_4d" and "chebyshev_4d_raw" since Tucker
    // surfaces serialize to raw values.
    if (data.surface_type != kChebyshev4DRaw &&
        data.surface_type != kChebyshev4D) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig});
    }
    if (data.segments.size() != 1) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig, 0, data.segments.size()});
    }

    const auto& seg = data.segments[0];
    auto leaf = reconstruct_chebyshev_leaf<4, StandardTransform4D>(seg);
    if (!leaf) return std::unexpected(leaf.error());

    auto eep_leaf = reconstruct_eep(std::move(*leaf),
                                    data.option_type, data.dividend_yield);
    auto bounds = bounds_from_4d_segment(seg);

    return PriceTable<ChebyshevRawLeaf>(
        std::move(eep_leaf), bounds, data.option_type, data.dividend_yield);
}

// ============================================================================
// ChebyshevMultiKRefInner (chebyshev_4d_segmented): segmented Chebyshev
// ============================================================================

template <>
std::expected<PriceTable<ChebyshevMultiKRefInner>, PriceTableError>
from_data<ChebyshevMultiKRefInner>(const PriceTableData& data) {
    if (data.surface_type != kChebyshev4DSegmented) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig});
    }
    if (data.segments.empty()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig, 0, 0});
    }

    auto groups = group_segments_by_kref(data.segments);
    std::vector<double> k_refs;
    std::vector<ChebyshevTauSegmented> kref_surfaces;

    for (const auto& [k_ref, group] : groups) {
        k_refs.push_back(k_ref);
        auto tau_split = reconstruct_chebyshev_tau_split<4, StandardTransform4D>(
            group, k_ref);
        if (!tau_split) return std::unexpected(tau_split.error());
        kref_surfaces.push_back(std::move(*tau_split));
    }

    MultiKRefSplit multi_split(std::move(k_refs));
    ChebyshevMultiKRefInner inner(std::move(kref_surfaces), std::move(multi_split));
    auto bounds = bounds_from_segments(data.segments);

    return PriceTable<ChebyshevMultiKRefInner>(
        std::move(inner), bounds, data.option_type, data.dividend_yield);
}

// ============================================================================
// BSpline3DLeaf (bspline_3d): 3D dimensionless B-spline with EEP
// ============================================================================

template <>
std::expected<PriceTable<BSpline3DLeaf>, PriceTableError>
from_data<BSpline3DLeaf>(const PriceTableData& data) {
    if (data.surface_type != kBSpline3D) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig});
    }
    if (data.segments.size() != 1) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig, 0, data.segments.size()});
    }

    const auto& seg = data.segments[0];
    auto leaf = reconstruct_bspline_leaf<3, DimensionlessTransform3D>(seg);
    if (!leaf) return std::unexpected(leaf.error());

    auto eep_leaf = reconstruct_eep(std::move(*leaf),
                                    data.option_type, data.dividend_yield);
    auto bounds = bounds_from_3d_segment(seg, data.maturity);

    return PriceTable<BSpline3DLeaf>(
        std::move(eep_leaf), bounds, data.option_type, data.dividend_yield);
}

// ============================================================================
// Chebyshev3DLeaf (chebyshev_3d): 3D dimensionless Chebyshev with EEP
// ============================================================================

// The Chebyshev3DLeaf uses TuckerTensor<3>, but Tucker surfaces serialize
// to raw values. Reconstruction as Tucker would require re-decomposition.
// Same limitation as the 4D Tucker case.
template <>
std::expected<PriceTable<Chebyshev3DLeaf>, PriceTableError>
from_data<Chebyshev3DLeaf>(const PriceTableData& data) {
    (void)data;
    return std::unexpected(PriceTableError{
        PriceTableErrorCode::InvalidConfig});
}

}  // namespace mango
