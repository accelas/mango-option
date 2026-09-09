// SPDX-License-Identifier: MIT
#pragma once
#include "mango/math/bspline/bspline_nd.hpp"
#include "mango/option/table/shared_interp.hpp"
#include "mango/option/table/transform_leaf.hpp"
#include "mango/option/table/eep/eep_layer.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/split_surface.hpp"
#include "mango/option/table/splits/tau_segment.hpp"
#include "mango/option/table/splits/multi_kref.hpp"
#include "mango/option/table/transforms/standard_4d.hpp"
#include "mango/option/table/transforms/dimensionless_3d.hpp"
namespace mango {
// Numerical layer types are independent of PriceTable publication/admission.
template <std::size_t N>
using SharedBSplineInterp = SharedInterp<BSplineND<double, N>, N>;
using BSplineTransformLeaf = TransformLeaf<SharedBSplineInterp<4>, StandardTransform4D>;
using BSplineLeaf = EEPLayer<BSplineTransformLeaf, AnalyticalEEP>;
using BSplineSegmentedLeaf = TransformLeaf<SharedBSplineInterp<4>, StandardTransform4D>;
using BSplineSegmentedSurface = SplitSurface<BSplineSegmentedLeaf, TauSegmentSplit>;
using BSplineMultiKRefInner = SplitSurface<BSplineSegmentedSurface, MultiKRefSplit>;
using BSpline3DTransformLeaf = TransformLeaf<SharedBSplineInterp<3>, DimensionlessTransform3D>;
using BSpline3DLeaf = EEPLayer<BSpline3DTransformLeaf, AnalyticalEEP>;
} // namespace mango
