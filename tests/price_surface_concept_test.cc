// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/price_surface_concept.hpp"
#include "mango/option/table/standard_surface.hpp"

using namespace mango;

static_assert(PriceSurface<StandardSurface>,
    "StandardSurface must satisfy PriceSurface concept");

TEST(PriceSurfaceConceptTest, StandardSurfaceSatisfiesConcept) {
    // Compile-time check above is the real test.
    // This test exists so the test binary runs.
    SUCCEED();
}
