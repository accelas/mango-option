// SPDX-License-Identifier: MIT
#pragma once

#include "mango/math/chebyshev/chebyshev_interpolant.hpp"
#include "mango/math/chebyshev/raw_tensor.hpp"
#include "mango/math/chebyshev/tucker_tensor.hpp"
#include "mango/option/table/price_table.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/eep/identity_eep.hpp"
#include "mango/option/table/eep_surface_adapter.hpp"
#include "mango/option/table/transforms/standard_4d.hpp"

namespace mango {

using ChebyshevLeaf = EEPSurfaceAdapter<
    ChebyshevInterpolant<4, TuckerTensor<4>>,
    StandardTransform4D, AnalyticalEEP>;

using ChebyshevSurface = PriceTable<ChebyshevLeaf>;

using ChebyshevRawLeaf = EEPSurfaceAdapter<
    ChebyshevInterpolant<4, RawTensor<4>>,
    StandardTransform4D, AnalyticalEEP>;

using ChebyshevRawSurface = PriceTable<ChebyshevRawLeaf>;

/// Leaf for segmented Chebyshev surfaces (V/K_ref, no EEP decomposition).
/// Used with TauSegmentSplit for discrete dividend support.
using ChebyshevSegmentedLeaf = EEPSurfaceAdapter<
    ChebyshevInterpolant<4, RawTensor<4>>,
    StandardTransform4D, IdentityEEP>;

}  // namespace mango
