// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/chebyshev/modal_types.hpp"
#include "mango/option/table/price_table.hpp"

namespace mango {

using ChebyshevTransformLeaf = ChebyshevModalTransformLeaf;
using ChebyshevLeaf = ChebyshevModalLeaf;

using ChebyshevSurface = PriceTable<ChebyshevLeaf>;

// Back-compat aliases (used by adaptive builder result types)
using ChebyshevRawTransformLeaf = ChebyshevTransformLeaf;
using ChebyshevRawLeaf = ChebyshevLeaf;
using ChebyshevRawSurface = ChebyshevSurface;

/// Leaf for segmented Chebyshev surfaces (V/K_ref, no EEP decomposition).
/// Used with TauSegmentSplit for discrete dividend support.
using ChebyshevSegmentedLeaf = ChebyshevModalSegmentedLeaf;

}  // namespace mango
