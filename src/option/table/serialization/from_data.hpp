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
/// Explicit instantiations are provided for all surface Inner types.
///
/// @tparam Inner  The inner surface type (e.g. BSplineLeaf, ChebyshevRawLeaf)
/// @param data    Serialized price table data
/// @return Reconstructed PriceTable or PriceTableError
template <typename Inner>
[[nodiscard]] std::expected<PriceTable<Inner>, PriceTableError>
from_data(const PriceTableData& data);

}  // namespace mango
