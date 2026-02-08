// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/eep_transform.hpp"
#include "mango/option/table/price_table_inner.hpp"
#include "mango/option/table/spliced_surface.hpp"

namespace mango {

/// Standard (non-segmented) American price surface with EEP decomposition.
/// Single slice with EEPTransform handles decompose (build-time) and
/// reconstruct (query-time) of Early Exercise Premium.
using StandardSurface = SplicedSurface<PriceTableInner, SingleBracket, EEPTransform, WeightedSum>;
using StandardSurfaceWrapper = SplicedSurfaceWrapper<StandardSurface>;

}  // namespace mango
