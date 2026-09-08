// SPDX-License-Identifier: MIT
#pragma once

namespace mango {

/// Evidence from a bounded proof of the represented physical price expression.
/// This status is diagnostic metadata, not a price-table admission token.
enum class PriceProofStatus {
    NotRun,
    Certified,
    // Requires a reachable physical query and a strictly negative enclosure
    // for the final expression; a decreasing raw leaf alone is insufficient.
    NegativeWitness,
    // Includes exhausted budgets and unresolved arithmetic or domain bounds.
    Indeterminate,
};

} // namespace mango
