// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include <memory>

namespace mango {

/// Get the pre-computed golden dimensionless EEP surface.
///
/// The surface covers all American options with q=0 across the standard
/// domain: S/K in [0.65, 1.50], sigma in [0.10, 0.80], rate in [0.005, 0.10],
/// tau in [7/365, 2.0]. It is reconstructed from embedded B-spline
/// coefficients on first call, then cached. Thread-safe.
///
/// @return Shared pointer to the segmented dimensionless surface
[[nodiscard]] std::shared_ptr<const SegmentedDimensionlessSurface>
golden_dimensionless_surface();

}  // namespace mango
