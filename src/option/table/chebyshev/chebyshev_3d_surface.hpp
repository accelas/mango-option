// SPDX-License-Identifier: MIT
#pragma once

#include "mango/math/chebyshev/chebyshev_interpolant.hpp"
#include "mango/math/chebyshev/raw_tensor.hpp"
#include "mango/option/table/transform_leaf.hpp"
#include "mango/option/table/eep/eep_layer.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/price_table.hpp"
#include "mango/option/table/transforms/dimensionless_3d.hpp"

namespace mango {

using Chebyshev3DTransformLeaf = TransformLeaf<
    ChebyshevInterpolant<3, RawTensor<3>>, DimensionlessTransform3D>;
using Chebyshev3DLeaf = EEPLayer<Chebyshev3DTransformLeaf, AnalyticalEEP>;
using Chebyshev3DPriceTable = PriceTable<Chebyshev3DLeaf>;

// Back-compat aliases
using Chebyshev3DRawTransformLeaf = Chebyshev3DTransformLeaf;
using Chebyshev3DRawLeaf = Chebyshev3DLeaf;
using Chebyshev3DRawPriceTable = Chebyshev3DPriceTable;

}  // namespace mango
