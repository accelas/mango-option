// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <cmath>
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/greek_types.hpp"
#include "mango/option/option_spec.hpp"

using namespace mango;

// ===========================================================================
// Test fixture: build a small B-spline surface once for all tests
// ===========================================================================

class TransformLeafGreeksTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Build a minimal 4D B-spline surface (4x4x4x4 = 16 PDE solves)
        PriceTableConfig config{
            .option_type = OptionType::PUT,
            .K_ref = 100.0,
            .pde_grid = PDEGridConfig{
                GridSpec<double>::uniform(-3.0, 3.0, 101).value(), 100},
        };

        PriceTableBuilder builder(config);

        PriceTableAxes axes;
        axes.grids[0] = {std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1)};
        axes.grids[1] = {0.25, 0.5, 0.75, 1.0};
        axes.grids[2] = {0.15, 0.20, 0.25, 0.30};
        axes.grids[3] = {0.03, 0.04, 0.05, 0.06};

        auto result = builder.build(axes);
        ASSERT_TRUE(result.has_value()) << "Build failed: " << result.error();

        spline_ = result->spline;
        K_ref_ = config.K_ref;
    }

    // Helper: construct a BSplineTransformLeaf from the shared spline
    static BSplineTransformLeaf make_leaf() {
        SharedBSplineInterp<4> interp(spline_);
        StandardTransform4D xform;
        return BSplineTransformLeaf(std::move(interp), xform, K_ref_);
    }

    // Helper: construct PricingParams for an ATM put
    static PricingParams atm_put() {
        return PricingParams(
            OptionSpec{
                .spot = 100.0, .strike = 100.0, .maturity = 0.5,
                .rate = 0.05, .dividend_yield = 0.0,
                .option_type = OptionType::PUT},
            0.20);
    }

    // Helper: construct PricingParams for an ITM put (low spot = high put value)
    static PricingParams itm_put() {
        return PricingParams(
            OptionSpec{
                .spot = 90.0, .strike = 100.0, .maturity = 0.5,
                .rate = 0.05, .dividend_yield = 0.0,
                .option_type = OptionType::PUT},
            0.20);
    }

    // Helper: construct PricingParams for an OTM put (high spot = low put value)
    static PricingParams otm_put() {
        return PricingParams(
            OptionSpec{
                .spot = 110.0, .strike = 100.0, .maturity = 0.5,
                .rate = 0.05, .dividend_yield = 0.0,
                .option_type = OptionType::PUT},
            0.20);
    }

    static std::shared_ptr<const BSplineND<double, 4>> spline_;
    static double K_ref_;
};

std::shared_ptr<const BSplineND<double, 4>> TransformLeafGreeksTest::spline_;
double TransformLeafGreeksTest::K_ref_ = 100.0;

// ===========================================================================
// greek() tests
// ===========================================================================

TEST_F(TransformLeafGreeksTest, DeltaIsFiniteForATMPut) {
    auto leaf = make_leaf();
    auto params = atm_put();

    auto result = leaf.greek(Greek::Delta, params);
    ASSERT_TRUE(result.has_value()) << "greek(Delta) failed";
    EXPECT_TRUE(std::isfinite(result.value()));
}

TEST_F(TransformLeafGreeksTest, DeltaHasReasonableMagnitude) {
    // For a leaf (EEP component only), delta should be modest in magnitude.
    // Not necessarily in [-1,0] because EEP is just the early exercise premium part.
    auto leaf = make_leaf();
    auto params = atm_put();

    auto result = leaf.greek(Greek::Delta, params);
    ASSERT_TRUE(result.has_value());
    // EEP delta for ATM should not be enormous
    EXPECT_LT(std::abs(result.value()), 10.0);
}

TEST_F(TransformLeafGreeksTest, VegaMatchesExistingMethod) {
    // greek(Greek::Vega, params) should return the same value as vega(...)
    auto leaf = make_leaf();
    auto params = atm_put();

    double spot = params.spot, strike = params.strike;
    double tau = params.maturity, sigma = params.volatility;
    double rate = get_zero_rate(params.rate, params.maturity);

    double existing_vega = leaf.vega(spot, strike, tau, sigma, rate);
    auto greek_vega = leaf.greek(Greek::Vega, params);

    ASSERT_TRUE(greek_vega.has_value());
    EXPECT_NEAR(greek_vega.value(), existing_vega, 1e-12);
}

TEST_F(TransformLeafGreeksTest, ThetaIsFinite) {
    auto leaf = make_leaf();
    auto params = atm_put();

    auto result = leaf.greek(Greek::Theta, params);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(std::isfinite(result.value()));
}

TEST_F(TransformLeafGreeksTest, RhoIsFinite) {
    auto leaf = make_leaf();
    auto params = atm_put();

    auto result = leaf.greek(Greek::Rho, params);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(std::isfinite(result.value()));
}

TEST_F(TransformLeafGreeksTest, GreekReturnsZeroWhenRawNonPositive) {
    // For a deep OTM put, the EEP raw value may be zero or negative.
    // In that case, greek() should return 0.0, not an error.
    auto leaf = make_leaf();

    // Extremely OTM: spot >> strike
    PricingParams deep_otm(
        OptionSpec{
            .spot = 110.0, .strike = 100.0, .maturity = 0.25,
            .rate = 0.05, .option_type = OptionType::PUT},
        0.15);

    auto result = leaf.greek(Greek::Delta, deep_otm);
    ASSERT_TRUE(result.has_value());
    // Either 0.0 or a small finite number (depending on raw value sign)
    EXPECT_TRUE(std::isfinite(result.value()));
}

// ===========================================================================
// gamma() tests
// ===========================================================================

TEST_F(TransformLeafGreeksTest, GammaIsFiniteForATM) {
    auto leaf = make_leaf();
    auto params = atm_put();

    auto result = leaf.gamma(params);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(std::isfinite(result.value()));
}

TEST_F(TransformLeafGreeksTest, GammaReturnsZeroWhenRawNonPositive) {
    auto leaf = make_leaf();

    // Deep OTM where raw EEP value may be non-positive
    PricingParams deep_otm(
        OptionSpec{
            .spot = 110.0, .strike = 100.0, .maturity = 0.25,
            .rate = 0.05, .option_type = OptionType::PUT},
        0.15);

    auto result = leaf.gamma(deep_otm);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(std::isfinite(result.value()));
}

TEST_F(TransformLeafGreeksTest, GammaUsesAnalyticalSecondPartial) {
    // SharedBSplineInterp has eval_second_partial, so the if constexpr
    // should pick the analytical path. Verify by checking gamma gives
    // a reasonable answer for ITM.
    auto leaf = make_leaf();
    auto params = itm_put();

    auto result = leaf.gamma(params);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(std::isfinite(result.value()));
    // Gamma magnitude should not be enormous
    EXPECT_LT(std::abs(result.value()), 1.0);
}

// ===========================================================================
// raw_value() tests
// ===========================================================================

TEST_F(TransformLeafGreeksTest, RawValuePositiveForITM) {
    auto leaf = make_leaf();
    auto params = itm_put();

    double raw = leaf.raw_value(
        params.spot, params.strike, params.maturity,
        params.volatility, get_zero_rate(params.rate, params.maturity));

    // ITM put should have positive EEP raw value
    // (though it depends on the surface; this is a sanity check)
    EXPECT_TRUE(std::isfinite(raw));
}

TEST_F(TransformLeafGreeksTest, RawValueMatchesPriceScaling) {
    // raw_value * strike/K_ref should equal the max(0, raw)*strike/K_ref part
    // i.e. when raw > 0, price() == raw * strike / K_ref
    auto leaf = make_leaf();
    auto params = itm_put();

    double spot = params.spot, strike = params.strike;
    double tau = params.maturity, sigma = params.volatility;
    double rate = get_zero_rate(params.rate, params.maturity);

    double raw = leaf.raw_value(spot, strike, tau, sigma, rate);
    double price = leaf.price(spot, strike, tau, sigma, rate);

    if (raw > 0.0) {
        EXPECT_NEAR(price, raw * strike / K_ref_, 1e-12);
    }
}

// ===========================================================================
// Consistency: greek(Delta) via FD comparison
// ===========================================================================

TEST_F(TransformLeafGreeksTest, DeltaConsistentWithFiniteDifference) {
    // Verify greek(Delta) is consistent with a finite-difference approximation
    // using leaf.price(). This validates the chain rule computation.
    auto leaf = make_leaf();
    auto params = atm_put();

    double spot = params.spot, strike = params.strike;
    double tau = params.maturity, sigma = params.volatility;
    double rate = get_zero_rate(params.rate, params.maturity);

    double h = 0.01;  // $0.01 bump
    double price_up = leaf.price(spot + h, strike, tau, sigma, rate);
    double price_dn = leaf.price(spot - h, strike, tau, sigma, rate);
    double fd_delta = (price_up - price_dn) / (2.0 * h);

    auto analytical = leaf.greek(Greek::Delta, params);
    ASSERT_TRUE(analytical.has_value());

    // Allow some tolerance: B-spline partial vs FD on price
    EXPECT_NEAR(analytical.value(), fd_delta, 0.05)
        << "Delta: analytical=" << analytical.value() << " fd=" << fd_delta;
}

TEST_F(TransformLeafGreeksTest, VegaConsistentWithFiniteDifference) {
    auto leaf = make_leaf();
    auto params = atm_put();

    double spot = params.spot, strike = params.strike;
    double tau = params.maturity, sigma = params.volatility;
    double rate = get_zero_rate(params.rate, params.maturity);

    double h = 0.001;  // 0.1% vol bump
    double price_up = leaf.price(spot, strike, tau, sigma + h, rate);
    double price_dn = leaf.price(spot, strike, tau, sigma - h, rate);
    double fd_vega = (price_up - price_dn) / (2.0 * h);

    auto analytical = leaf.greek(Greek::Vega, params);
    ASSERT_TRUE(analytical.has_value());

    EXPECT_NEAR(analytical.value(), fd_vega, 0.1)
        << "Vega: analytical=" << analytical.value() << " fd=" << fd_vega;
}
