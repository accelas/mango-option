// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <cmath>
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/greek_types.hpp"
#include "mango/option/option_spec.hpp"

using namespace mango;

// ===========================================================================
// Test fixture: build a small B-spline PriceTable once for all tests
// ===========================================================================

class PriceTableGreeksTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
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

        auto wrapper = make_bspline_surface(
            result->spline, result->K_ref,
            result->dividends.dividend_yield, OptionType::PUT);
        ASSERT_TRUE(wrapper.has_value()) << "make_bspline_surface failed: " << wrapper.error();
        surface_ = std::make_unique<BSplinePriceTable>(std::move(*wrapper));
    }

    static PricingParams atm_put() {
        return PricingParams(
            OptionSpec{
                .spot = 100.0, .strike = 100.0, .maturity = 0.5,
                .rate = 0.05, .dividend_yield = 0.0,
                .option_type = OptionType::PUT},
            0.20);
    }

    static PricingParams itm_put() {
        return PricingParams(
            OptionSpec{
                .spot = 90.0, .strike = 100.0, .maturity = 0.5,
                .rate = 0.05, .dividend_yield = 0.0,
                .option_type = OptionType::PUT},
            0.20);
    }

    static PricingParams otm_put() {
        return PricingParams(
            OptionSpec{
                .spot = 110.0, .strike = 100.0, .maturity = 0.5,
                .rate = 0.05, .dividend_yield = 0.0,
                .option_type = OptionType::PUT},
            0.20);
    }

    static std::unique_ptr<BSplinePriceTable> surface_;
};

std::unique_ptr<BSplinePriceTable> PriceTableGreeksTest::surface_;

// ===========================================================================
// Delta tests
// ===========================================================================

TEST_F(PriceTableGreeksTest, DeltaIsNegativeForATMPut) {
    auto params = atm_put();
    auto delta = surface_->delta(params);
    ASSERT_TRUE(delta.has_value()) << "delta() failed";
    EXPECT_LT(*delta, 0.0) << "Put delta should be negative";
    EXPECT_GT(*delta, -1.0) << "Put delta should be > -1";
}

TEST_F(PriceTableGreeksTest, DeltaIsMoreNegativeForITMPut) {
    auto atm = surface_->delta(atm_put());
    auto itm = surface_->delta(itm_put());
    ASSERT_TRUE(atm.has_value());
    ASSERT_TRUE(itm.has_value());
    // ITM put has more negative delta (closer to -1) than ATM
    EXPECT_LT(*itm, *atm);
}

TEST_F(PriceTableGreeksTest, DeltaIsFiniteForOTMPut) {
    auto delta = surface_->delta(otm_put());
    ASSERT_TRUE(delta.has_value());
    EXPECT_TRUE(std::isfinite(*delta));
    EXPECT_LT(*delta, 0.0);  // Still negative
    EXPECT_GT(*delta, -1.0);
}

// ===========================================================================
// Gamma tests
// ===========================================================================

TEST_F(PriceTableGreeksTest, GammaIsPositiveForATMPut) {
    auto gamma = surface_->gamma(atm_put());
    ASSERT_TRUE(gamma.has_value()) << "gamma() failed";
    EXPECT_GT(*gamma, 0.0) << "Gamma should be positive";
}

TEST_F(PriceTableGreeksTest, GammaIsFiniteForITM) {
    auto gamma = surface_->gamma(itm_put());
    ASSERT_TRUE(gamma.has_value());
    EXPECT_TRUE(std::isfinite(*gamma));
    EXPECT_GT(*gamma, 0.0);
}

TEST_F(PriceTableGreeksTest, GammaIsFiniteForOTM) {
    auto gamma = surface_->gamma(otm_put());
    ASSERT_TRUE(gamma.has_value());
    EXPECT_TRUE(std::isfinite(*gamma));
    // Gamma may be small for OTM but should be non-negative
    EXPECT_GE(*gamma, 0.0);
}

// ===========================================================================
// Theta tests
// ===========================================================================

TEST_F(PriceTableGreeksTest, ThetaIsNegativeForATMPut) {
    auto theta = surface_->theta(atm_put());
    ASSERT_TRUE(theta.has_value()) << "theta() failed";
    // Theta for ATM put is typically negative (time decay)
    EXPECT_LT(*theta, 0.0) << "Put theta should be negative (time decay)";
}

TEST_F(PriceTableGreeksTest, ThetaIsFinite) {
    for (auto params : {atm_put(), itm_put(), otm_put()}) {
        auto theta = surface_->theta(params);
        ASSERT_TRUE(theta.has_value());
        EXPECT_TRUE(std::isfinite(*theta));
    }
}

// ===========================================================================
// Rho tests
// ===========================================================================

TEST_F(PriceTableGreeksTest, RhoIsNegativeForPut) {
    auto rho = surface_->rho(atm_put());
    ASSERT_TRUE(rho.has_value()) << "rho() failed";
    // Rho for put is negative (higher rates reduce put value)
    EXPECT_LT(*rho, 0.0) << "Put rho should be negative";
}

TEST_F(PriceTableGreeksTest, RhoIsFinite) {
    for (auto params : {atm_put(), itm_put(), otm_put()}) {
        auto rho = surface_->rho(params);
        ASSERT_TRUE(rho.has_value());
        EXPECT_TRUE(std::isfinite(*rho));
    }
}

// ===========================================================================
// FD consistency: delta vs price bump
// ===========================================================================

TEST_F(PriceTableGreeksTest, DeltaConsistentWithFiniteDifference) {
    auto params = atm_put();
    double spot = params.spot, strike = params.strike;
    double tau = params.maturity, sigma = params.volatility;
    double rate = get_zero_rate(params.rate, params.maturity);

    double h = 0.01;
    double price_up = surface_->price(spot + h, strike, tau, sigma, rate);
    double price_dn = surface_->price(spot - h, strike, tau, sigma, rate);
    double fd_delta = (price_up - price_dn) / (2.0 * h);

    auto analytical = surface_->delta(params);
    ASSERT_TRUE(analytical.has_value());

    EXPECT_NEAR(*analytical, fd_delta, 0.05)
        << "Delta: analytical=" << *analytical << " fd=" << fd_delta;
}

TEST_F(PriceTableGreeksTest, GammaConsistentWithFiniteDifference) {
    auto params = atm_put();
    double spot = params.spot, strike = params.strike;
    double tau = params.maturity, sigma = params.volatility;
    double rate = get_zero_rate(params.rate, params.maturity);

    double h = 0.5;
    double price_up = surface_->price(spot + h, strike, tau, sigma, rate);
    double price_mid = surface_->price(spot, strike, tau, sigma, rate);
    double price_dn = surface_->price(spot - h, strike, tau, sigma, rate);
    double fd_gamma = (price_up - 2.0 * price_mid + price_dn) / (h * h);

    auto analytical = surface_->gamma(params);
    ASSERT_TRUE(analytical.has_value());

    // Gamma FD is noisier, allow wider tolerance
    EXPECT_NEAR(*analytical, fd_gamma, 0.01)
        << "Gamma: analytical=" << *analytical << " fd=" << fd_gamma;
}
