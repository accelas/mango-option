// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/bspline/bspline_interpolant.hpp"
#include "mango/option/table/bounded_surface.hpp"
#include "mango/option/table/eep_surface_adapter.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/eep/identity_eep.hpp"
#include "mango/option/table/split_surface.hpp"
#include "mango/option/table/splits/tau_segment.hpp"
#include "mango/option/table/splits/multi_kref.hpp"
#include "mango/option/table/transforms/standard_4d.hpp"
#include <expected>
#include <memory>
#include <string>

namespace mango {

// ===========================================================================
// New type aliases — concept-based layered architecture
// ===========================================================================

/// Leaf adapter for standard (EEP) surfaces
using StandardLeaf = EEPSurfaceAdapter<SharedBSplineInterp<4>,
                                        StandardTransform4D, AnalyticalEEP>;

/// Standard surface (satisfies PriceSurface concept)
using StandardSurface = BoundedSurface<StandardLeaf>;

/// Leaf adapter for segmented surfaces (no EEP decomposition)
using SegmentedLeaf = EEPSurfaceAdapter<SharedBSplineInterp<4>,
                                         StandardTransform4D, IdentityEEP>;

/// Tau-segmented surface
using SegmentedPriceSurface = SplitSurface<SegmentedLeaf, TauSegmentSplit>;

/// Multi-K_ref surface (outer split over K_refs of segmented inner)
using MultiKRefInner = SplitSurface<SegmentedPriceSurface, MultiKRefSplit>;

/// Multi-K_ref surface (satisfies PriceSurface concept)
using MultiKRefPriceSurface = BoundedSurface<MultiKRefInner>;


/// Create a StandardSurface from a pre-built EEP surface.
/// Reads K_ref and dividend_yield from surface metadata.
/// Requires SurfaceContent::EarlyExercisePremium; rejects NormalizedPrice.
[[nodiscard]] std::expected<StandardSurface, std::string>
make_standard_surface(
    std::shared_ptr<const PriceTableSurface> surface,
    OptionType type);

}  // namespace mango
