// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/eep_transform.hpp"
#include "mango/option/table/price_table_inner.hpp"
#include "mango/option/table/spliced_surface.hpp"
#include <expected>
#include <memory>
#include <string>

namespace mango {

/// Standard (non-segmented) American price surface with EEP decomposition.
/// EEPPriceTableInner handles EEP reconstruction at query time (price and vega).
/// SingleBracket provides trivial 1-slice dispatch.
/// IdentityTransform passes through since reconstruction is in the Inner adapter.
using StandardSurface = SplicedSurface<EEPPriceTableInner, SingleBracket, IdentityTransform, WeightedSum>;
using StandardSurfaceWrapper = SplicedSurfaceWrapper<StandardSurface>;

/// Segmented surface types using PriceTableInner (NormalizedPrice content)
using SegmentedSurfacePI = SegmentedSurface<PriceTableInner>;
using MultiKRefSurfacePI = MultiKRefSurface<SegmentedSurfacePI>;
using MultiKRefSurfaceWrapperPI = SplicedSurfaceWrapper<MultiKRefSurfacePI>;

/// Create a StandardSurfaceWrapper from a pre-built EEP surface.
/// Reads K_ref and dividend_yield from surface metadata.
/// Requires SurfaceContent::EarlyExercisePremium; rejects NormalizedPrice.
[[nodiscard]] std::expected<StandardSurfaceWrapper, std::string>
make_standard_wrapper(
    std::shared_ptr<const PriceTableSurface<4>> surface,
    OptionType type);

}  // namespace mango
