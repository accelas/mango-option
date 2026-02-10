// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include "mango/option/table/price_surface_concept.hpp"
#include "mango/option/table/surface_concepts.hpp"

using namespace mango;

// Static assertions
static_assert(SurfaceInterpolant<ChebyshevInterpolant<4, TuckerTensor<4>>, 4>);
static_assert(SurfaceInterpolant<ChebyshevInterpolant<4, RawTensor<4>>, 4>);
static_assert(PriceSurface<ChebyshevSurface>);
static_assert(PriceSurface<ChebyshevRawSurface>);

TEST(ChebyshevSurfaceTest, ConstructAndQuery) {
    Domain<4> domain{
        .lo = {-0.5, 0.01, 0.05, 0.01},
        .hi = { 0.5, 2.00, 0.50, 0.10},
    };
    std::array<size_t, 4> num_pts = {5, 5, 5, 5};

    auto interp = ChebyshevInterpolant<4, TuckerTensor<4>>::build(
        [](std::array<double, 4>) { return 0.05; },
        domain, num_pts, 1e-8);

    ChebyshevLeaf leaf(
        std::move(interp),
        StandardTransform4D{},
        AnalyticalEEP(OptionType::PUT, 0.02),
        100.0);

    SurfaceBounds bounds{
        .m_min = -0.5, .m_max = 0.5,
        .tau_min = 0.01, .tau_max = 2.0,
        .sigma_min = 0.05, .sigma_max = 0.50,
        .rate_min = 0.01, .rate_max = 0.10,
    };

    ChebyshevSurface surface(std::move(leaf), bounds, OptionType::PUT, 0.02);

    double p = surface.price(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_GT(p, 0.0);
    EXPECT_LT(p, 50.0);

    double v = surface.vega(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_GT(v, 0.0);
}
