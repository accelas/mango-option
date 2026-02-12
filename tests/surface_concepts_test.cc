// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/surface_concepts.hpp"
#include "mango/option/table/greek_types.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/transforms/standard_4d.hpp"
#include "mango/option/table/transforms/dimensionless_3d.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
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
    auto w = xform.greek_weights(Greek::Vega, 110.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_EQ(w[0], 0.0);
    EXPECT_EQ(w[1], 0.0);
    EXPECT_EQ(w[2], 1.0);
    EXPECT_EQ(w[3], 0.0);
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

TEST(SurfaceConceptsTest, StandardTransform4DGreekWeights) {
    StandardTransform4D xform;
    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, rate = 0.05;

    auto dw = xform.greek_weights(Greek::Delta, S, K, tau, sigma, rate);
    EXPECT_NEAR(dw[0], 1.0 / S, 1e-12);
    EXPECT_EQ(dw[1], 0.0);
    EXPECT_EQ(dw[2], 0.0);
    EXPECT_EQ(dw[3], 0.0);

    auto vw = xform.greek_weights(Greek::Vega, S, K, tau, sigma, rate);
    EXPECT_EQ(vw[0], 0.0);
    EXPECT_EQ(vw[1], 0.0);
    EXPECT_EQ(vw[2], 1.0);
    EXPECT_EQ(vw[3], 0.0);

    auto tw = xform.greek_weights(Greek::Theta, S, K, tau, sigma, rate);
    EXPECT_EQ(tw[0], 0.0);
    EXPECT_EQ(tw[1], -1.0);
    EXPECT_EQ(tw[2], 0.0);
    EXPECT_EQ(tw[3], 0.0);

    auto rw = xform.greek_weights(Greek::Rho, S, K, tau, sigma, rate);
    EXPECT_EQ(rw[0], 0.0);
    EXPECT_EQ(rw[1], 0.0);
    EXPECT_EQ(rw[2], 0.0);
    EXPECT_EQ(rw[3], 1.0);
}

TEST(SurfaceConceptsTest, DimensionlessTransform3DGreekWeights) {
    DimensionlessTransform3D xform;
    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, rate = 0.05;

    auto dw = xform.greek_weights(Greek::Delta, S, K, tau, sigma, rate);
    EXPECT_NEAR(dw[0], 1.0 / S, 1e-12);
    EXPECT_EQ(dw[1], 0.0);
    EXPECT_EQ(dw[2], 0.0);

    auto vw = xform.greek_weights(Greek::Vega, S, K, tau, sigma, rate);
    EXPECT_EQ(vw[0], 0.0);
    EXPECT_NEAR(vw[1], sigma * tau, 1e-12);
    EXPECT_NEAR(vw[2], -2.0 / sigma, 1e-12);

    auto tw = xform.greek_weights(Greek::Theta, S, K, tau, sigma, rate);
    EXPECT_EQ(tw[0], 0.0);
    EXPECT_NEAR(tw[1], sigma * sigma / 2.0, 1e-12);
    EXPECT_EQ(tw[2], 0.0);

    auto rw = xform.greek_weights(Greek::Rho, S, K, tau, sigma, rate);
    EXPECT_EQ(rw[0], 0.0);
    EXPECT_EQ(rw[1], 0.0);
    EXPECT_NEAR(rw[2], 1.0 / rate, 1e-12);
}
