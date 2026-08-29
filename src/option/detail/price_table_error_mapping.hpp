// SPDX-License-Identifier: MIT
#pragma once

// INTERNAL, UNSTABLE — implementation detail of price_table_factory,
// exposed only so the PriceTableError -> ValidationError mapping is
// directly unit-testable.  Not part of the public API.

#include "mango/support/error_types.hpp"

namespace mango::detail {

/// Map a price-table build failure to the factory's public ValidationError.
/// Grid-shaped failures keep their specific codes; everything else becomes
/// the generic PriceTableBuildFailed (issue #441 item 7).
[[nodiscard]] ValidationError to_validation_error(const PriceTableError& error);

}  // namespace mango::detail
