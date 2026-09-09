// SPDX-License-Identifier: MIT
#pragma once
#include "mango/math/chebyshev/chebyshev_modal_interpolant.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/eep/eep_layer.hpp"
#include "mango/option/table/split_surface.hpp"
#include "mango/option/table/splits/multi_kref.hpp"
#include "mango/option/table/splits/tau_segment.hpp"
#include "mango/option/table/transform_leaf.hpp"
#include "mango/option/table/transforms/dimensionless_3d.hpp"
#include "mango/option/table/transforms/standard_4d.hpp"
namespace mango {
using ChebyshevModalTransformLeaf =
    TransformLeaf<ChebyshevModalInterpolant<4>, StandardTransform4D>;
using ChebyshevModalLeaf = EEPLayer<ChebyshevModalTransformLeaf, AnalyticalEEP>;
using ChebyshevModal3DTransformLeaf =
    TransformLeaf<ChebyshevModalInterpolant<3>, DimensionlessTransform3D>;
using ChebyshevModal3DLeaf = EEPLayer<ChebyshevModal3DTransformLeaf, AnalyticalEEP>;
using ChebyshevModalSegmentedLeaf = ChebyshevModalTransformLeaf;
using ChebyshevModalSegmentedSurface = SplitSurface<ChebyshevModalSegmentedLeaf, TauSegmentSplit>;
using ChebyshevModalMultiKRefInner = SplitSurface<ChebyshevModalSegmentedSurface, MultiKRefSplit>;
} // namespace mango
