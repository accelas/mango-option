// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/transform_leaf.hpp"
#include "mango/option/table/eep/eep_layer.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/price_table.hpp"
#include "mango/option/table/transforms/dimensionless_3d.hpp"

namespace mango {

using BSpline3DTransformLeaf = TransformLeaf<SharedBSplineInterp<3>, DimensionlessTransform3D>;
using BSpline3DLeaf = EEPLayer<BSpline3DTransformLeaf, AnalyticalEEP>;
using BSpline3DPriceTable = PriceTable<BSpline3DLeaf>;

}  // namespace mango
