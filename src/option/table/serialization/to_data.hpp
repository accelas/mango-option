// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/serialization/price_table_data.hpp"
#include "mango/option/table/price_table.hpp"

namespace mango {

// ============================================================================
// surface_type_string: maps Inner type to a human-readable string identifier.
// ============================================================================

/// Primary template (unspecialized) -- intentionally left undefined.
/// Each supported Inner type must provide an explicit specialization.
template <typename Inner>
[[nodiscard]] constexpr const char* surface_type_string();

// ============================================================================
// to_data: extract PriceTableData from any PriceTable<Inner>.
// ============================================================================

/// Convert a PriceTable<Inner> to a serializable PriceTableData.
/// Explicit instantiations are provided for all 8 Inner types.
template <typename Inner>
[[nodiscard]] PriceTableData to_data(const PriceTable<Inner>& table);

}  // namespace mango
