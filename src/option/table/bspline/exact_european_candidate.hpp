// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/bspline/bspline_types.hpp"
#include "mango/option/table/surface_bounds.hpp"
#include "mango/option/table/fixed_expiry.hpp"
#include <expected>
#include <optional>

namespace mango::detail {

/// Construct exact zero EEP only for an entire dividend-free CALL domain with
/// q==0 and nonnegative rates. Empty means the usual numerical builder applies.
/// The caller supplies its resolved coordinates/knots; they are retained exactly.
/// This is an untrusted numerical candidate, not public certificate/accuracy
/// admission. It performs no PDE work. Mixed-rate/cash families never take it.
[[nodiscard]] std::expected<std::optional<BSplineLeaf>, PriceTableError>
make_exact_european_bspline_candidate(
    BSplineND<double, 4>::GridArray grids,
    BSplineND<double, 4>::KnotArray knots,
    const SurfaceBounds& requested_bounds, double reference_strike,
    OptionType option_type, double dividend_yield,
    const std::optional<FixedExpiryMetadata>& model = std::nullopt);

}  // namespace mango::detail
