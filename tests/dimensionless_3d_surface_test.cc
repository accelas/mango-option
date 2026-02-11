// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/bspline/bspline_3d_surface.hpp"
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/american_option.hpp"
#include <cmath>

namespace mango {
namespace {

class Dimensionless3DSurfaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        DimensionlessAxes axes;
        axes.log_moneyness = {-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30};
        axes.tau_prime = {0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16};
        axes.ln_kappa = {-2.5, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.8};

        auto result = build_dimensionless_surface(axes, K_ref_, OptionType::PUT);
        ASSERT_TRUE(result.has_value())
            << "Build failed: code=" << static_cast<int>(result.error().code);

        // Build BSpline3DPriceTable via the layered architecture
        SharedBSplineInterp<3> interp(result->surface);
        DimensionlessTransform3D xform;
        BSpline3DTransformLeaf leaf(std::move(interp), xform, 1.0);
        AnalyticalEEP eep(OptionType::PUT, 0.0);
        BSpline3DLeaf eep_leaf(std::move(leaf), std::move(eep));

        // Compute bounds in physical coords from dimensionless axes.
        // tau' = sigma^2 * tau / 2, so tau = 2*tau'/sigma^2.
        // For sigma range [0.10, 0.80]:
        //   tau_min arises from smallest tau' with largest sigma
        //   tau_max arises from largest tau' with smallest sigma
        const double sigma_min = 0.10;
        const double sigma_max = 0.80;
        SurfaceBounds bounds{
            .m_min = axes.log_moneyness.front(),
            .m_max = axes.log_moneyness.back(),
            .tau_min = 2.0 * axes.tau_prime.front() / (sigma_max * sigma_max),
            .tau_max = 2.0 * axes.tau_prime.back() / (sigma_min * sigma_min),
            .sigma_min = sigma_min,
            .sigma_max = sigma_max,
            .rate_min = 0.005,
            .rate_max = 0.10,
        };

        table_ = std::make_unique<BSpline3DPriceTable>(
            std::move(eep_leaf), bounds, OptionType::PUT, 0.0);
    }

    static constexpr double K_ref_ = 100.0;
    std::unique_ptr<BSpline3DPriceTable> table_;
};

TEST_F(Dimensionless3DSurfaceTest, PriceMatchesPDE) {
    struct TestPoint { double S, K, tau, sigma, rate; };
    std::vector<TestPoint> points = {
        {100, 100, 1.0, 0.20, 0.05},   // ATM
        {90,  100, 0.5, 0.25, 0.03},   // OTM put
        {110, 100, 1.5, 0.15, 0.04},   // ITM put
    };

    for (const auto& p : points) {
        double surface_price = table_->price(p.S, p.K, p.tau, p.sigma, p.rate);

        auto ref = solve_american_option(PricingParams(
            OptionSpec{.spot = p.S, .strike = p.K, .maturity = p.tau,
                .rate = p.rate, .dividend_yield = 0.0,
                .option_type = OptionType::PUT}, p.sigma));
        ASSERT_TRUE(ref.has_value());
        double pde_price = ref->value_at(p.S);

        // The 3D approximation has ~312 bps RMS error from sigma/rate
        // coupling, so allow up to $0.50 absolute tolerance.
        EXPECT_NEAR(surface_price, pde_price, 0.50)
            << "S=" << p.S << " K=" << p.K << " tau=" << p.tau
            << " sigma=" << p.sigma << " r=" << p.rate;
    }
}

TEST_F(Dimensionless3DSurfaceTest, VegaIsPositive) {
    double v = table_->vega(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_GT(v, 0.0) << "Vega should be positive for ATM option";
}

}  // namespace
}  // namespace mango
