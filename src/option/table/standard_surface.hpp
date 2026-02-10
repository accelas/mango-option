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
// Keep old includes for transition (consumers may still use them)
#include "mango/option/table/eep_transform.hpp"
#include "mango/option/table/price_table_inner.hpp"
#include "mango/option/table/spliced_surface.hpp"
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

/// Standard surface wrapper (satisfies PriceSurface concept)
using StandardSurfaceWrapper = BoundedSurface<StandardLeaf>;

/// Leaf adapter for segmented surfaces (no EEP decomposition)
using SegmentedLeaf = EEPSurfaceAdapter<SharedBSplineInterp<4>,
                                         StandardTransform4D, IdentityEEP>;

/// Tau-segmented surface
using SegmentedPriceSurface = SplitSurface<SegmentedLeaf, TauSegmentSplit>;

/// Multi-K_ref surface (outer split over K_refs of segmented inner)
using MultiKRefPriceSurface = SplitSurface<SegmentedPriceSurface, MultiKRefSplit>;

/// Multi-K_ref wrapper (satisfies PriceSurface concept)
using MultiKRefPriceWrapper = BoundedSurface<MultiKRefPriceSurface>;

// ===========================================================================
// Legacy aliases for gradual migration
// ===========================================================================

/// Keep StandardSurface name for any code that references it.
using StandardSurface = StandardLeaf;

/// Create a StandardSurfaceWrapper from a pre-built EEP surface.
/// Reads K_ref and dividend_yield from surface metadata.
/// Requires SurfaceContent::EarlyExercisePremium; rejects NormalizedPrice.
[[nodiscard]] std::expected<StandardSurfaceWrapper, std::string>
make_standard_wrapper(
    std::shared_ptr<const PriceTableSurface> surface,
    OptionType type);

}  // namespace mango
