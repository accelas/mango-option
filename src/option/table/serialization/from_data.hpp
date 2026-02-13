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
/// **Note:** Tucker-based types (ChebyshevLeaf, Chebyshev3DLeaf) always
/// return InvalidConfig because Tucker decomposition is not recoverable
/// from serialized raw values. Use ChebyshevRawLeaf or Chebyshev3DRawLeaf
/// instead — they accept both Tucker and Raw surface_type strings.
///
/// @tparam Inner  The inner surface type (e.g. BSplineLeaf, ChebyshevRawLeaf)
/// @param data    Serialized price table data
/// @return Reconstructed PriceTable or PriceTableError
template <typename Inner>
[[nodiscard]] std::expected<PriceTable<Inner>, PriceTableError>
from_data(const PriceTableData& data);

}  // namespace mango
