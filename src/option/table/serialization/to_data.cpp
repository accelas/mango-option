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
    return surface_types::kBSpline4D;
}

template <> constexpr const char* surface_type_string<BSplineMultiKRefInner>() {
    return surface_types::kBSpline4DSegmented;
}

template <> constexpr const char* surface_type_string<ChebyshevLeaf>() {
    return surface_types::kChebyshev4DRaw;
}

template <> constexpr const char* surface_type_string<ChebyshevMultiKRefInner>() {
    return surface_types::kChebyshev4DSegmented;
}

template <> constexpr const char* surface_type_string<BSpline3DLeaf>() {
    return surface_types::kBSpline3D;
}

template <> constexpr const char* surface_type_string<Chebyshev3DLeaf>() {
    return surface_types::kChebyshev3DRaw;
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

    data.bounds_m_min = table.m_min();
    data.bounds_m_max = table.m_max();
    data.bounds_tau_min = table.tau_min();
    data.bounds_tau_max = table.tau_max();
    data.bounds_sigma_min = table.sigma_min();
    data.bounds_sigma_max = table.sigma_max();
    data.bounds_rate_min = table.rate_min();
    data.bounds_rate_max = table.rate_max();

    extract_segments(table.inner(), data.segments,
                     /*K_ref_hint=*/0.0,
                     /*tau_start=*/0.0, /*tau_end=*/table.tau_max(),
                     /*tau_min=*/table.tau_min(), /*tau_max=*/table.tau_max());
    return data;
}

// ============================================================================
// Explicit instantiations for all 8 Inner types
// ============================================================================

template PriceTableData to_data<BSplineLeaf>(const PriceTable<BSplineLeaf>&);
template PriceTableData to_data<BSplineMultiKRefInner>(const PriceTable<BSplineMultiKRefInner>&);
template PriceTableData to_data<ChebyshevLeaf>(const PriceTable<ChebyshevLeaf>&);
template PriceTableData to_data<ChebyshevMultiKRefInner>(const PriceTable<ChebyshevMultiKRefInner>&);
template PriceTableData to_data<BSpline3DLeaf>(const PriceTable<BSpline3DLeaf>&);
template PriceTableData to_data<Chebyshev3DLeaf>(const PriceTable<Chebyshev3DLeaf>&);

}  // namespace mango
