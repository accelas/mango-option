// SPDX-License-Identifier: MIT
#pragma once

#include "mango/math/chebyshev/chebyshev_interpolant.hpp"
#include "mango/math/chebyshev/raw_tensor.hpp"
#include "mango/option/table/price_table.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/eep/eep_layer.hpp"
#include "mango/option/table/transform_leaf.hpp"
#include "mango/option/table/transforms/standard_4d.hpp"

namespace mango {

using ChebyshevTransformLeaf = TransformLeaf<
    ChebyshevInterpolant<4, RawTensor<4>>, StandardTransform4D>;
using ChebyshevLeaf = EEPLayer<ChebyshevTransformLeaf, AnalyticalEEP>;

using ChebyshevSurface = PriceTable<ChebyshevLeaf>;

// Back-compat aliases (used by adaptive builder result types)
using ChebyshevRawTransformLeaf = ChebyshevTransformLeaf;
using ChebyshevRawLeaf = ChebyshevLeaf;
using ChebyshevRawSurface = ChebyshevSurface;

/// Leaf for segmented Chebyshev surfaces (V/K_ref, no EEP decomposition).
/// Used with TauSegmentSplit for discrete dividend support.
using ChebyshevSegmentedLeaf = TransformLeaf<
    ChebyshevInterpolant<4, RawTensor<4>>, StandardTransform4D>;

}  // namespace mango
