// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/surface_concepts.hpp"
#include "mango/option/table/bspline/bspline_interpolant.hpp"

using namespace mango;

TEST(SurfaceConceptsTest, BSplineInterpolantSatisfiesConcept) {
    static_assert(SurfaceInterpolant<SharedBSplineInterp<4>, 4>);
    static_assert(SurfaceInterpolant<SharedBSplineInterp<3>, 3>);
}
