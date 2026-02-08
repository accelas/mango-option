// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/eep_transform.hpp"
#include "mango/option/table/price_table_inner.hpp"
#include "mango/option/table/spliced_surface.hpp"

namespace mango {

/// Standard (non-segmented) American price surface with EEP decomposition.
/// EEPPriceTableInner handles EEP reconstruction at query time (price and vega).
/// SingleBracket provides trivial 1-slice dispatch.
/// IdentityTransform passes through since reconstruction is in the Inner adapter.
using StandardSurface = SplicedSurface<EEPPriceTableInner, SingleBracket, IdentityTransform, WeightedSum>;
using StandardSurfaceWrapper = SplicedSurfaceWrapper<StandardSurface>;

/// Segmented surface types using PriceTableInner (replaces AmericanPriceSurfaceAdapter defaults)
using SegmentedSurfacePI = SegmentedSurface<PriceTableInner>;
using MultiKRefSurfacePI = MultiKRefSurface<SegmentedSurfacePI>;
using MultiKRefSurfaceWrapperPI = SplicedSurfaceWrapper<MultiKRefSurfacePI>;

}  // namespace mango
