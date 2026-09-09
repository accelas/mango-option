// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/chebyshev/modal_types.hpp"
#include "mango/option/table/price_table.hpp"

namespace mango {

using Chebyshev3DTransformLeaf = ChebyshevModal3DTransformLeaf;
using Chebyshev3DLeaf = ChebyshevModal3DLeaf;
using Chebyshev3DPriceTable = PriceTable<Chebyshev3DLeaf>;

// Back-compat aliases
using Chebyshev3DRawTransformLeaf = Chebyshev3DTransformLeaf;
using Chebyshev3DRawLeaf = Chebyshev3DLeaf;
using Chebyshev3DRawPriceTable = Chebyshev3DPriceTable;

}  // namespace mango
