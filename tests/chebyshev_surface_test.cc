// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_table_builder.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/table/surface_concepts.hpp"

using namespace mango;

// Static assertions
static_assert(SurfaceInterpolant<ChebyshevInterpolant<4, RawTensor<4>>, 4>);

TEST(ChebyshevSurfaceTest, ConstructAndQuery) {
    Domain<4> domain{
        .lo = {-0.5, 0.01, 0.05, 0.01},
        .hi = { 0.5, 2.00, 0.50, 0.10},
    };
    std::array<size_t, 4> num_pts = {5, 5, 5, 5};

    auto interp = ChebyshevInterpolant<4, RawTensor<4>>::build(
        [](std::array<double, 4>) { return 0.05; },
        domain, num_pts);

    ChebyshevTransformLeaf tleaf(
        std::move(interp), StandardTransform4D{}, 100.0);
    ChebyshevLeaf leaf(std::move(tleaf),
        AnalyticalEEP(OptionType::PUT, 0.02));

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

TEST(ChebyshevTableBuilderTest, BuildSucceeds) {
    ChebyshevTableConfig config{
        .num_pts = {12, 8, 8, 5},
        .domain = Domain<4>{
            .lo = {-0.30, 0.02, 0.05, 0.01},
            .hi = { 0.30, 2.00, 0.50, 0.10},
        },
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
    };

    auto result = build_chebyshev_table(config);
    ASSERT_TRUE(result.has_value()) << "Builder should succeed";
    EXPECT_GT(result->n_pde_solves, 0u);
    EXPECT_GT(result->build_seconds, 0.0);

    // Query the surface at ATM
    double p = result->price(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_GT(p, 0.0);
    EXPECT_LT(p, 50.0);
}

TEST(ChebyshevTableBuilderTest, IVRoundTrip) {
    ChebyshevTableConfig config{
        .num_pts = {20, 14, 14, 8},
        .domain = Domain<4>{
            .lo = {-0.40, 0.02, 0.05, 0.01},
            .hi = { 0.40, 2.00, 0.50, 0.10},
        },
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
    };

    auto result = build_chebyshev_table(config);
    ASSERT_TRUE(result.has_value());

    // Get FDM reference price at sigma=0.20
    PricingParams ref_params(
        OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT},
        0.20);
    auto ref = solve_american_option(ref_params);
    ASSERT_TRUE(ref.has_value());

    // Chebyshev price should be close to FDM
    double cheb_price = result->price(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_NEAR(cheb_price, ref->value(), 0.50);  // within $0.50 for initial integration
}
