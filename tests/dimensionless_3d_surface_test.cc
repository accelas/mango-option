// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/bspline/bspline_3d_surface.hpp"
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/eep/dimensionless_3d_accessor.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/american_option.hpp"
#include "mango/math/bspline_nd_separable.hpp"
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

        // 1. PDE solve -> raw V/K
        auto pde = solve_dimensionless_pde(axes, K_ref_, OptionType::PUT);
        ASSERT_TRUE(pde.has_value())
            << "PDE solve failed: code=" << static_cast<int>(pde.error().code);

        // 2. EEP decompose
        Dimensionless3DAccessor accessor(pde->values, axes, K_ref_);
        eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, 0.0));

        // 3. Fit B-spline
        std::array<std::vector<double>, 3> grids = {
            axes.log_moneyness, axes.tau_prime, axes.ln_kappa};
        auto fitter = BSplineNDSeparable<double, 3>::create(grids);
        ASSERT_TRUE(fitter.has_value());
        auto fit = fitter->fit(std::move(pde->values));
        ASSERT_TRUE(fit.has_value());

        // 4. Build surface with actual K_ref
        PriceTableAxesND<3> surface_axes;
        surface_axes.grids[0] = axes.log_moneyness;
        surface_axes.grids[1] = axes.tau_prime;
        surface_axes.grids[2] = axes.ln_kappa;
        surface_axes.names = {"log_moneyness", "tau_prime", "ln_kappa"};

        auto surface = PriceTableSurfaceND<3>::build(
            std::move(surface_axes), std::move(fit->coefficients), K_ref_);
        ASSERT_TRUE(surface.has_value());

        // 5. Wrap in layered PriceTable
        SharedBSplineInterp<3> interp(std::move(surface.value()));
        DimensionlessTransform3D xform;
        BSpline3DTransformLeaf leaf(std::move(interp), xform, K_ref_);
        AnalyticalEEP eep(OptionType::PUT, 0.0);
        BSpline3DLeaf eep_leaf(std::move(leaf), std::move(eep));

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
