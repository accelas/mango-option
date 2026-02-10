// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/surface_concepts.hpp"
#include "mango/option/table/bspline/bspline_interpolant.hpp"
#include "mango/option/table/transforms/standard_4d.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/eep/identity_eep.hpp"
#include "mango/option/table/eep_surface_adapter.hpp"
#include "mango/option/table/bounded_surface.hpp"
#include "mango/option/table/price_surface_concept.hpp"
#include "mango/option/table/price_table_builder.hpp"
#include "mango/option/table/eep_transform.hpp"

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

// Build a small test surface for comparison tests.
static std::shared_ptr<const mango::PriceTableSurface> make_test_surface() {
    using namespace mango;
    // Each axis needs >= 4 points for B-spline fitting
    auto setup = PriceTableBuilder::from_vectors(
        {-0.2, -0.1, 0.0, 0.1, 0.2},     // log-moneyness (5 pts)
        {0.25, 0.50, 0.75, 1.00},         // maturities (4 pts)
        {0.15, 0.20, 0.25, 0.30},         // vols (4 pts)
        {0.02, 0.03, 0.04, 0.05},         // rates (4 pts)
        100.0,                              // K_ref
        GridAccuracyParams{},
        OptionType::PUT, 0.02);
    EXPECT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;
    EEPDecomposer decomposer{OptionType::PUT, 100.0, 0.02};
    auto result = builder.build(axes, SurfaceContent::EarlyExercisePremium,
        [&](PriceTensor& t, const PriceTableAxes& a) { decomposer.decompose(t, a); });
    EXPECT_TRUE(result.has_value());
    return result->surface;
}

TEST(EEPSurfaceAdapterTest, MatchesOldEEPPriceTableInner) {
    auto surface = make_test_surface();
    double K_ref = surface->metadata().K_ref;

    // Old path
    EEPPriceTableInner old_inner(surface, OptionType::PUT, K_ref, 0.02);

    // New path
    SharedBSplineInterp<4> interp(surface);
    StandardTransform4D xform;
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    EEPSurfaceAdapter adapter(std::move(interp), xform, eep, K_ref);

    // Compare at several query points
    struct TestPoint { double spot, strike, tau, sigma, rate; };
    TestPoint points[] = {
        {100, 100, 0.5, 0.20, 0.04},  // ATM
        {110, 100, 0.5, 0.20, 0.04},  // ITM put
        {90,  100, 0.5, 0.20, 0.04},  // OTM put
        {100, 100, 0.3, 0.25, 0.03},  // Short maturity
        {100, 100, 0.9, 0.20, 0.04},  // Long maturity
    };

    for (const auto& p : points) {
        PriceQuery q{p.spot, p.strike, p.tau, p.sigma, p.rate};
        double old_price = old_inner.price(q);
        double new_price = adapter.price(p.spot, p.strike, p.tau, p.sigma, p.rate);
        EXPECT_NEAR(old_price, new_price, 1e-12)
            << "Mismatch at S=" << p.spot << " K=" << p.strike;

        double old_vega = old_inner.vega(q);
        double new_vega = adapter.vega(p.spot, p.strike, p.tau, p.sigma, p.rate);
        EXPECT_NEAR(old_vega, new_vega, 1e-12)
            << "Vega mismatch at S=" << p.spot << " K=" << p.strike;
    }
}

TEST(BoundedSurfaceTest, SatisfiesPriceSurfaceConcept) {
    using StandardAdapter = EEPSurfaceAdapter<SharedBSplineInterp<4>,
                                               StandardTransform4D, AnalyticalEEP>;
    static_assert(PriceSurface<BoundedSurface<StandardAdapter>>);
}
