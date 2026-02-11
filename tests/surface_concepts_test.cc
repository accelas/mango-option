// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/surface_concepts.hpp"
#include "mango/option/table/bspline/bspline_interpolant.hpp"
#include "mango/option/table/transforms/standard_4d.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/eep/identity_eep.hpp"
#include "mango/option/table/eep/eep_surface_adapter.hpp"
#include "mango/option/table/price_table.hpp"
using namespace mango;

TEST(SurfaceConceptsTest, BSplineInterpolantSatisfiesConcept) {
    static_assert(SurfaceInterpolant<SharedBSplineInterp<4>, 4>);
    static_assert(SurfaceInterpolant<SharedBSplineInterp<3>, 3>);
}

TEST(SurfaceConceptsTest, StandardTransform4DSatisfiesConcept) {
    static_assert(CoordinateTransform<StandardTransform4D>);
    static_assert(StandardTransform4D::kDim == 4);
}

TEST(SurfaceConceptsTest, AnalyticalEEPSatisfiesConcept) {
    static_assert(EEPStrategy<AnalyticalEEP>);
}

TEST(SurfaceConceptsTest, IdentityEEPSatisfiesConcept) {
    static_assert(EEPStrategy<IdentityEEP>);
}

TEST(StandardTransform4DTest, ToCoordsReturnsLogMoneyness) {
    StandardTransform4D xform;
    auto c = xform.to_coords(110.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_NEAR(c[0], std::log(110.0 / 100.0), 1e-12);  // x = ln(S/K)
    EXPECT_NEAR(c[1], 0.5, 1e-12);   // tau
    EXPECT_NEAR(c[2], 0.20, 1e-12);  // sigma
    EXPECT_NEAR(c[3], 0.05, 1e-12);  // rate
}

TEST(StandardTransform4DTest, VegaWeightsOnlySigmaAxis) {
    StandardTransform4D xform;
    auto w = xform.vega_weights(110.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_EQ(w[0], 0.0);
    EXPECT_EQ(w[1], 0.0);
    EXPECT_EQ(w[2], 1.0);
    EXPECT_EQ(w[3], 0.0);
}

TEST(IdentityEEPTest, EuropeanPriceIsZero) {
    IdentityEEP eep;
    EXPECT_EQ(eep.european_price(100, 100, 0.5, 0.20, 0.05), 0.0);
    EXPECT_EQ(eep.european_vega(100, 100, 0.5, 0.20, 0.05), 0.0);
}

TEST(IdentityEEPTest, ScaleIsStrikeOverKRef) {
    IdentityEEP eep;
    EXPECT_NEAR(eep.scale(110.0, 100.0), 1.1, 1e-12);
}

TEST(AnalyticalEEPTest, ScaleIsStrikeOverKRef) {
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    EXPECT_NEAR(eep.scale(110.0, 100.0), 1.1, 1e-12);
}

TEST(AnalyticalEEPTest, EuropeanPriceIsPositiveForATMPut) {
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    double p = eep.european_price(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_GT(p, 0.0);
    EXPECT_LT(p, 20.0);  // Reasonable range
}

TEST(AnalyticalEEPTest, EuropeanVegaIsPositive) {
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    double v = eep.european_vega(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_GT(v, 0.0);
}

