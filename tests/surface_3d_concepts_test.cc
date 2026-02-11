// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/bspline/bspline_3d_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_3d_surface.hpp"
#include "mango/option/table/surface_concepts.hpp"

namespace mango {
namespace {

static_assert(SurfaceInterpolant<SharedBSplineInterp<3>, 3>);
static_assert(CoordinateTransform<DimensionlessTransform3D>);
static_assert(DimensionlessTransform3D::kDim == 3);

TEST(Surface3DConceptsTest, BSpline3DTypesExist) {
    EXPECT_EQ(sizeof(BSpline3DTransformLeaf), sizeof(BSpline3DTransformLeaf));
    EXPECT_EQ(sizeof(BSpline3DLeaf), sizeof(BSpline3DLeaf));
    EXPECT_EQ(sizeof(BSpline3DPriceTable), sizeof(BSpline3DPriceTable));
}

TEST(Surface3DConceptsTest, Chebyshev3DTypesExist) {
    EXPECT_EQ(sizeof(Chebyshev3DTransformLeaf), sizeof(Chebyshev3DTransformLeaf));
    EXPECT_EQ(sizeof(Chebyshev3DLeaf), sizeof(Chebyshev3DLeaf));
    EXPECT_EQ(sizeof(Chebyshev3DPriceTable), sizeof(Chebyshev3DPriceTable));
}

}  // namespace
}  // namespace mango
