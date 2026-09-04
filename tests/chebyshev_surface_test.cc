// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_table_builder.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/table/surface_concepts.hpp"

#include <cmath>
#include <limits>

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
    ASSERT_TRUE(interp.has_value());

    ChebyshevTransformLeaf tleaf(
        std::move(*interp), StandardTransform4D{}, 100.0);
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

// ===========================================================================
// Regression tests for issue #466 (TransformLeaf masked NaN as 0.0)
// ===========================================================================

// Regression: TransformLeaf::price returned 0.0 for NaN interpolant output
// Bug: std::max(0.0, raw) returns 0.0 when raw is NaN
TEST(TransformLeafNaNTest, PricePropagatesNaNAndKeepsFloor) {
    auto f = [](std::array<double, 4>) { return -1.0; };  // always-negative raw
    Domain<4> dom{.lo = {-0.7, 0.05, 0.1, 0.0}, .hi = {0.7, 2.0, 0.5, 0.08}};
    std::array<size_t, 4> npts = {5, 5, 5, 5};
    auto interp = ChebyshevInterpolant<4, RawTensor<4>>::build(f, dom, npts);
    ASSERT_TRUE(interp.has_value());
    ChebyshevTransformLeaf leaf(std::move(*interp),
                                 StandardTransform4D{}, 100.0);

    // Finite query over a negative raw value: floored to +0.0 (not -0.0)
    double p = leaf.price(100.0, 100.0, 1.0, 0.2, 0.05);
    EXPECT_DOUBLE_EQ(p, 0.0);
    EXPECT_FALSE(std::signbit(p));

    // NaN spot propagates instead of masking to 0.0
    EXPECT_TRUE(std::isnan(leaf.price(std::nan(""), 100.0, 1.0, 0.2, 0.05)));

    // Inf spot still clamps to the domain edge (finite output)
    EXPECT_TRUE(std::isfinite(
        leaf.price(std::numeric_limits<double>::infinity(), 100.0, 1.0, 0.2, 0.05)));
}

// ===========================================================================
// Regression test for issue #480 (S3): gridless non-adaptive solve
// extrapolated outer moneyness nodes
// ===========================================================================

// Regression (#480, S3): build_chebyshev_table solved its batch gridless
// with the default accuracy (n_sigma = 5), so the PDE half-width was
// 5 * 0.20 * sqrt(0.1) ~= 0.32 while the moneyness nodes reach +-0.7:
// both endpoint nodes were cubic-spline extrapolations.  All queried
// coordinates are CGL nodes (endpoints), so only extraction is measured.
// Pre-fix max abs error on this branch's parent: 74.14 at m=-0.7,
// sigma=0.10 (deep-ITM put; also 74.14 at sigma=0.20).
TEST(ChebyshevTableBuilderTest, TailsMatchFdmAtExtremeMoneyness) {
    ChebyshevTableConfig config{
        .num_pts = {9, 5, 3, 3},
        .domain = {.lo = {-0.7, 0.01, 0.10, 0.02},
                   .hi = { 0.7, 0.10, 0.20, 0.06}},
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
    };
    auto result = build_chebyshev_table(config);
    ASSERT_TRUE(result.has_value());

    const double K = 100.0;
    const double tau = 0.10;   // tau node (domain hi)
    const double r = 0.06;     // rate node (domain hi)

    // Tolerance ($ per K=100): post-fix max |got-ref| measured 1.173e-6;
    // 10x that is 1.173e-5, rounded up to 2e-5 for cross-toolchain slack
    // (still ~3.7e6x under the 74.14 pre-fix error).
    constexpr double TOL = 2e-5;

    for (double m : {-0.7, 0.7}) {
        for (double sigma : {0.10, 0.20}) {
            const double S = K * std::exp(m);
            PricingParams p(
                OptionSpec{.spot = S, .strike = K, .maturity = tau,
                           .rate = r, .dividend_yield = 0.0,
                           .option_type = OptionType::PUT},
                sigma);
            auto solver = AmericanOptionSolver::create(
                p, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::High)});
            ASSERT_TRUE(solver.has_value());
            auto ref = solver->solve();
            ASSERT_TRUE(ref.has_value());
            const double got = result->surface.price(S, K, tau, sigma, r);
            EXPECT_NEAR(got, ref->value_at(S), TOL)
                << "m=" << m << " sigma=" << sigma;
        }
    }
}
