// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/american_price_surface.hpp"
#include "mango/option/table/price_table_surface.hpp"
#include "mango/option/table/price_table_axes.hpp"
#include "mango/option/table/price_table_metadata.hpp"
#include "mango/option/european_option.hpp"
#include <cmath>

namespace mango {
namespace {

// Helper to build a test surface
std::shared_ptr<const PriceTableSurface<4>> make_test_surface(
    SurfaceContent content, double eep_value = 2.0, double div_yield = 0.0) {
    PriceTableAxes<4> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1, 1.2};  // moneyness
    axes.grids[1] = {0.25, 0.5, 1.0, 2.0};       // maturity
    axes.grids[2] = {0.10, 0.20, 0.30, 0.40};    // volatility
    axes.grids[3] = {0.02, 0.04, 0.06, 0.08};    // rate

    std::vector<double> coeffs(5 * 4 * 4 * 4, eep_value);
    PriceTableMetadata meta{
        .K_ref = 100.0,
        .dividends = {.dividend_yield = div_yield},
        .m_min = 0.8,
        .m_max = 1.2,
        .content = content
    };

    auto result = PriceTableSurface<4>::build(axes, coeffs, meta);
    return result.value();
}

TEST(AmericanPriceSurfaceTest, AcceptsRawPriceSurface) {
    auto surface = make_test_surface(SurfaceContent::RawPrice);
    auto result = AmericanPriceSurface::create(surface, OptionType::PUT);
    EXPECT_TRUE(result.has_value());
}

TEST(AmericanPriceSurfaceTest, AcceptsEEPSurface) {
    auto surface = make_test_surface(SurfaceContent::EarlyExercisePremium);
    auto result = AmericanPriceSurface::create(surface, OptionType::PUT);
    EXPECT_TRUE(result.has_value());
}

TEST(AmericanPriceSurfaceTest, RejectsNullSurface) {
    auto result = AmericanPriceSurface::create(nullptr, OptionType::PUT);
    EXPECT_FALSE(result.has_value());
}

TEST(AmericanPriceSurfaceTest, RejectsZeroKRef) {
    auto surface = make_test_surface(SurfaceContent::EarlyExercisePremium);
    // Manually build a surface with K_ref=0 to trigger validation
    PriceTableAxes<4> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1, 1.2};
    axes.grids[1] = {0.25, 0.5, 1.0, 2.0};
    axes.grids[2] = {0.10, 0.20, 0.30, 0.40};
    axes.grids[3] = {0.02, 0.04, 0.06, 0.08};
    std::vector<double> coeffs(5 * 4 * 4 * 4, 2.0);
    PriceTableMetadata meta{
        .K_ref = 0.0,
        .m_min = 0.8,
        .m_max = 1.2,
        .content = SurfaceContent::EarlyExercisePremium
    };
    auto bad_surface = PriceTableSurface<4>::build(axes, coeffs, meta).value();
    auto result = AmericanPriceSurface::create(bad_surface, OptionType::PUT);
    EXPECT_FALSE(result.has_value());
}

TEST(AmericanPriceSurfaceTest, RejectsDiscreteDividends) {
    // EEP decomposition only supports continuous dividend yield
    PriceTableAxes<4> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1, 1.2};
    axes.grids[1] = {0.25, 0.5, 1.0, 2.0};
    axes.grids[2] = {0.10, 0.20, 0.30, 0.40};
    axes.grids[3] = {0.02, 0.04, 0.06, 0.08};

    std::vector<double> coeffs(5 * 4 * 4 * 4, 2.0);
    PriceTableMetadata meta{
        .K_ref = 100.0,
        .dividends = {.dividend_yield = 0.0, .discrete_dividends = {{.calendar_time = 0.25, .amount = 1.50}, {.calendar_time = 0.75, .amount = 1.50}}},
        .m_min = 0.8,
        .m_max = 1.2,
        .content = SurfaceContent::EarlyExercisePremium,
    };
    auto surface = PriceTableSurface<4>::build(axes, coeffs, meta).value();
    auto result = AmericanPriceSurface::create(surface, OptionType::PUT);
    EXPECT_FALSE(result.has_value());
}

TEST(AmericanPriceSurfaceTest, PriceReconstructsCorrectly) {
    auto surface = make_test_surface(SurfaceContent::EarlyExercisePremium, 2.0, 0.02);
    auto aps = AmericanPriceSurface::create(surface, OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, r = 0.05;
    double price = aps->price(S, K, tau, sigma, r);

    // Expected: EEP * (K/K_ref) + P_EU
    double m = S / K;
    double eep_interp = surface->value({m, tau, sigma, r});
    auto eu = EuropeanOptionSolver(PricingParams(OptionSpec{.spot = S, .strike = K, .maturity = tau, .rate = r, .dividend_yield = 0.02, .option_type = OptionType::PUT}, sigma)).solve().value();
    double expected = eep_interp * (K / 100.0) + eu.value();

    EXPECT_NEAR(price, expected, 1e-10);
}

TEST(AmericanPriceSurfaceTest, StrikeScaling) {
    // EEP scales with K/K_ref, European is exact for each K
    auto surface = make_test_surface(SurfaceContent::EarlyExercisePremium, 2.0, 0.0);
    auto aps = AmericanPriceSurface::create(surface, OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    // Same moneyness, different strikes
    double tau = 1.0, sigma = 0.20, r = 0.05;
    double p1 = aps->price(100.0, 100.0, tau, sigma, r);
    double p2 = aps->price(200.0, 200.0, tau, sigma, r);

    // Both have m=1.0 so EEP interpolation is same, but K/K_ref differs
    // EEP component: for K=100 -> 2.0 * 1.0, for K=200 -> 2.0 * 2.0
    // European component computed exactly for each K
    // The prices should NOT be equal (different K means different absolute values)
    EXPECT_NE(p1, p2);
    // But P(200)/P(100) should be close to 2 (homogeneity) -- approximate since EU isn't perfectly linear
    EXPECT_GT(p2 / p1, 1.5);
    EXPECT_LT(p2 / p1, 2.5);
}

TEST(AmericanPriceSurfaceTest, VegaMatchesFiniteDiff) {
    auto surface = make_test_surface(SurfaceContent::EarlyExercisePremium, 2.0, 0.0);
    auto aps = AmericanPriceSurface::create(surface, OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, r = 0.05;
    double eps = 1e-5;
    double p_up = aps->price(S, K, tau, sigma + eps, r);
    double p_dn = aps->price(S, K, tau, sigma - eps, r);
    double fd_vega = (p_up - p_dn) / (2.0 * eps);

    EXPECT_NEAR(aps->vega(S, K, tau, sigma, r), fd_vega, 0.01);
}

TEST(AmericanPriceSurfaceTest, ThetaMatchesFiniteDiff) {
    auto surface = make_test_surface(SurfaceContent::EarlyExercisePremium, 2.0, 0.0);
    auto aps = AmericanPriceSurface::create(surface, OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, r = 0.05;
    double eps = 1e-5;
    double p_up = aps->price(S, K, tau + eps, sigma, r);
    double p_dn = aps->price(S, K, tau - eps, sigma, r);
    // theta is dV/dt (calendar) = -(dP/d(tau))
    double fd_theta = -(p_up - p_dn) / (2.0 * eps);

    EXPECT_NEAR(aps->theta(S, K, tau, sigma, r), fd_theta, 0.01);
}

TEST(AmericanPriceSurfaceTest, DeltaMatchesFiniteDiff) {
    auto surface = make_test_surface(SurfaceContent::EarlyExercisePremium, 2.0, 0.0);
    auto aps = AmericanPriceSurface::create(surface, OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, r = 0.05;
    double eps = S * 1e-4;
    double p_up = aps->price(S + eps, K, tau, sigma, r);
    double p_dn = aps->price(S - eps, K, tau, sigma, r);
    double fd_delta = (p_up - p_dn) / (2.0 * eps);

    EXPECT_NEAR(aps->delta(S, K, tau, sigma, r), fd_delta, 1e-4);
}

TEST(AmericanPriceSurfaceTest, GammaMatchesFiniteDiff) {
    auto surface = make_test_surface(SurfaceContent::EarlyExercisePremium, 2.0, 0.0);
    auto aps = AmericanPriceSurface::create(surface, OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, r = 0.05;
    double eps = S * 1e-3;
    double p_up = aps->price(S + eps, K, tau, sigma, r);
    double p_mid = aps->price(S, K, tau, sigma, r);
    double p_dn = aps->price(S - eps, K, tau, sigma, r);
    double fd_gamma = (p_up - 2*p_mid + p_dn) / (eps * eps);

    EXPECT_NEAR(aps->gamma(S, K, tau, sigma, r), fd_gamma, 1e-3);
}

TEST(AmericanPriceSurfaceTest, MetadataAccess) {
    auto surface = make_test_surface(SurfaceContent::EarlyExercisePremium);
    auto aps = AmericanPriceSurface::create(surface, OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    EXPECT_DOUBLE_EQ(aps->metadata().K_ref, 100.0);
    EXPECT_EQ(aps->metadata().content, SurfaceContent::EarlyExercisePremium);
}

// ===========================================================================
// RawPrice content type tests
// ===========================================================================

TEST(AmericanPriceSurfaceTest, RejectsRawPriceWithDiscreteDividends) {
    PriceTableAxes<4> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1, 1.2};
    axes.grids[1] = {0.25, 0.5, 1.0, 2.0};
    axes.grids[2] = {0.10, 0.20, 0.30, 0.40};
    axes.grids[3] = {0.02, 0.04, 0.06, 0.08};

    std::vector<double> coeffs(5 * 4 * 4 * 4, 5.0);
    PriceTableMetadata meta{
        .K_ref = 100.0,
        .dividends = {.dividend_yield = 0.0, .discrete_dividends = {{.calendar_time = 0.25, .amount = 1.50}, {.calendar_time = 0.75, .amount = 1.50}}},
        .m_min = 0.8,
        .m_max = 1.2,
        .content = SurfaceContent::RawPrice,
    };
    auto surface = PriceTableSurface<4>::build(axes, coeffs, meta).value();
    auto result = AmericanPriceSurface::create(surface, OptionType::PUT);
    EXPECT_FALSE(result.has_value());
}

TEST(AmericanPriceSurfaceTest, RawPricePriceReturnsInterpolatedValue) {
    double val = 7.5;
    auto surface = make_test_surface(SurfaceContent::RawPrice, val);
    auto aps = AmericanPriceSurface::create(surface, OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    // For RawPrice: price = surface_->value({spot/K_ref, tau, sigma, rate})
    // With constant coefficients, the spline returns approximately val
    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, r = 0.05;
    double price = aps->price(S, K, tau, sigma, r);
    double direct = surface->value({S / 100.0, tau, sigma, r});
    EXPECT_DOUBLE_EQ(price, direct);
}

TEST(AmericanPriceSurfaceTest, RawPriceVegaReturnsNaN) {
    auto surface = make_test_surface(SurfaceContent::RawPrice, 5.0);
    auto aps = AmericanPriceSurface::create(surface, OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    double vega = aps->vega(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_TRUE(std::isnan(vega));
}

// ===========================================================================
// Bounds accessor tests
// ===========================================================================

TEST(AmericanPriceSurfaceTest, BoundsAccessorsEEP) {
    auto surface = make_test_surface(SurfaceContent::EarlyExercisePremium);
    auto aps = AmericanPriceSurface::create(surface, OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    // Moneyness bounds from metadata
    EXPECT_DOUBLE_EQ(aps->m_min(), 0.8);
    EXPECT_DOUBLE_EQ(aps->m_max(), 1.2);

    // Maturity bounds from axes.grids[1]
    EXPECT_DOUBLE_EQ(aps->tau_min(), 0.25);
    EXPECT_DOUBLE_EQ(aps->tau_max(), 2.0);

    // Volatility bounds from axes.grids[2]
    EXPECT_DOUBLE_EQ(aps->sigma_min(), 0.10);
    EXPECT_DOUBLE_EQ(aps->sigma_max(), 0.40);

    // Rate bounds from axes.grids[3]
    EXPECT_DOUBLE_EQ(aps->rate_min(), 0.02);
    EXPECT_DOUBLE_EQ(aps->rate_max(), 0.08);
}

TEST(AmericanPriceSurfaceTest, BoundsAccessorsRawPrice) {
    auto surface = make_test_surface(SurfaceContent::RawPrice);
    auto aps = AmericanPriceSurface::create(surface, OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    EXPECT_DOUBLE_EQ(aps->m_min(), 0.8);
    EXPECT_DOUBLE_EQ(aps->m_max(), 1.2);
    EXPECT_DOUBLE_EQ(aps->tau_min(), 0.25);
    EXPECT_DOUBLE_EQ(aps->tau_max(), 2.0);
    EXPECT_DOUBLE_EQ(aps->sigma_min(), 0.10);
    EXPECT_DOUBLE_EQ(aps->sigma_max(), 0.40);
    EXPECT_DOUBLE_EQ(aps->rate_min(), 0.02);
    EXPECT_DOUBLE_EQ(aps->rate_max(), 0.08);
}

// ===========================================================================
// Regression: create() still accepts EEP (guard against accidental breakage)
// ===========================================================================

TEST(AmericanPriceSurfaceTest, RegressionEEPStillAccepted) {
    auto surface = make_test_surface(SurfaceContent::EarlyExercisePremium, 3.0, 0.01);
    auto result = AmericanPriceSurface::create(surface, OptionType::PUT);
    ASSERT_TRUE(result.has_value());

    // Verify EEP reconstruction still works (price != raw interpolation)
    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, r = 0.05;
    double price = result->price(S, K, tau, sigma, r);
    double raw = surface->value({S / K, tau, sigma, r});
    // EEP price = raw * (K/K_ref) + European > raw for a put with positive European value
    EXPECT_GT(price, raw);
}

}  // namespace
}  // namespace mango
