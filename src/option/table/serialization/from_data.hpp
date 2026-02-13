// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/serialization/price_table_data.hpp"
#include "mango/option/table/price_table.hpp"
#include "mango/support/error_types.hpp"

#include <expected>

namespace mango {

/// Reconstruct a PriceTable<Inner> from its serialized PriceTableData.
///
/// Validates that data.surface_type matches the expected type for Inner.
/// Explicit instantiations are provided for all 8 Inner types.
///
/// **Tucker surfaces are not directly reconstructible.** Tucker decomposition
/// loses rank/factor structure during serialization (values are expanded to
/// raw tensor). The specializations for ChebyshevLeaf and Chebyshev3DLeaf
/// always return InvalidConfig. Instead, use the Raw equivalents:
///
///   from_data<ChebyshevRawLeaf>(data)     — accepts "chebyshev_4d" or "chebyshev_4d_raw"
///   from_data<Chebyshev3DRawLeaf>(data)   — accepts "chebyshev_3d" or "chebyshev_3d_raw"
///
/// The Raw variants produce numerically identical results to the original
/// Tucker surfaces (they store the same expanded tensor values).
///
/// @tparam Inner  The inner surface type (e.g. BSplineLeaf, ChebyshevRawLeaf)
/// @param data    Serialized price table data
/// @return Reconstructed PriceTable or PriceTableError
template <typename Inner>
[[nodiscard]] std::expected<PriceTable<Inner>, PriceTableError>
from_data(const PriceTableData& data);

}  // namespace mango
