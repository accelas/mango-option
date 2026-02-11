// SPDX-License-Identifier: MIT
#pragma once

#include "mango/math/chebyshev/chebyshev_interpolant.hpp"
#include "mango/math/chebyshev/raw_tensor.hpp"
#include "mango/math/chebyshev/tucker_tensor.hpp"
#include "mango/option/table/price_table.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/eep/eep_layer.hpp"
#include "mango/option/table/split_surface.hpp"
#include "mango/option/table/splits/tau_segment.hpp"
#include "mango/option/table/splits/multi_kref.hpp"
#include "mango/option/table/transform_leaf.hpp"
#include "mango/option/table/transforms/standard_4d.hpp"

namespace mango {

using ChebyshevTransformLeaf = TransformLeaf<
    ChebyshevInterpolant<4, TuckerTensor<4>>, StandardTransform4D>;
using ChebyshevLeaf = EEPLayer<ChebyshevTransformLeaf, AnalyticalEEP>;

using ChebyshevSurface = PriceTable<ChebyshevLeaf>;

using ChebyshevRawTransformLeaf = TransformLeaf<
    ChebyshevInterpolant<4, RawTensor<4>>, StandardTransform4D>;
using ChebyshevRawLeaf = EEPLayer<ChebyshevRawTransformLeaf, AnalyticalEEP>;

using ChebyshevRawSurface = PriceTable<ChebyshevRawLeaf>;

/// Leaf for segmented Chebyshev surfaces (V/K_ref, no EEP decomposition).
/// Used with TauSegmentSplit for discrete dividend support.
using ChebyshevSegmentedLeaf = TransformLeaf<
    ChebyshevInterpolant<4, RawTensor<4>>, StandardTransform4D>;

/// Tau-segmented Chebyshev surface (one leaf per inter-dividend interval)
using ChebyshevTauSegmented = SplitSurface<ChebyshevSegmentedLeaf, TauSegmentSplit>;

/// Multi-K_ref blended segmented Chebyshev surface
using ChebyshevMultiKRefInner = SplitSurface<ChebyshevTauSegmented, MultiKRefSplit>;

/// Multi-K_ref segmented Chebyshev price table (final queryable surface)
using ChebyshevMultiKRefSurface = PriceTable<ChebyshevMultiKRefInner>;

}  // namespace mango
