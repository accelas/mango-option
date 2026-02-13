// SPDX-License-Identifier: MIT

#include "mango/option/table/serialization/to_data.hpp"
#include "mango/option/table/serialization/extract_segments.hpp"

// Surface type headers (define the type aliases)
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/bspline/bspline_3d_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_3d_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"

namespace mango {

// ============================================================================
// surface_type_string specializations
// ============================================================================

template <> constexpr const char* surface_type_string<BSplineLeaf>() {
    return "bspline_4d";
}

template <> constexpr const char* surface_type_string<BSplineMultiKRefInner>() {
    return "bspline_4d_segmented";
}

template <> constexpr const char* surface_type_string<ChebyshevLeaf>() {
    return "chebyshev_4d";
}

template <> constexpr const char* surface_type_string<ChebyshevRawLeaf>() {
    return "chebyshev_4d_raw";
}

template <> constexpr const char* surface_type_string<ChebyshevMultiKRefInner>() {
    return "chebyshev_4d_segmented";
}

template <> constexpr const char* surface_type_string<BSpline3DLeaf>() {
    return "bspline_3d";
}

template <> constexpr const char* surface_type_string<Chebyshev3DLeaf>() {
    return "chebyshev_3d";
}

template <> constexpr const char* surface_type_string<Chebyshev3DRawLeaf>() {
    return "chebyshev_3d_raw";
}

// ============================================================================
// to_data implementation
// ============================================================================

template <typename Inner>
PriceTableData to_data(const PriceTable<Inner>& table) {
    PriceTableData data;
    data.surface_type = surface_type_string<Inner>();
    data.option_type = table.option_type();
    data.dividend_yield = table.dividend_yield();
    data.maturity = table.tau_max();

    extract_segments(table.inner(), data.segments,
                     /*K_ref_hint=*/0.0,
                     /*tau_start=*/0.0, /*tau_end=*/table.tau_max(),
                     /*tau_min=*/table.tau_min(), /*tau_max=*/table.tau_max());
    return data;
}

// ============================================================================
// Explicit instantiations for all 7 Inner types
// ============================================================================

template PriceTableData to_data<BSplineLeaf>(const PriceTable<BSplineLeaf>&);
template PriceTableData to_data<BSplineMultiKRefInner>(const PriceTable<BSplineMultiKRefInner>&);
template PriceTableData to_data<ChebyshevLeaf>(const PriceTable<ChebyshevLeaf>&);
template PriceTableData to_data<ChebyshevRawLeaf>(const PriceTable<ChebyshevRawLeaf>&);
template PriceTableData to_data<ChebyshevMultiKRefInner>(const PriceTable<ChebyshevMultiKRefInner>&);
template PriceTableData to_data<BSpline3DLeaf>(const PriceTable<BSpline3DLeaf>&);
template PriceTableData to_data<Chebyshev3DLeaf>(const PriceTable<Chebyshev3DLeaf>&);
template PriceTableData to_data<Chebyshev3DRawLeaf>(const PriceTable<Chebyshev3DRawLeaf>&);

}  // namespace mango
