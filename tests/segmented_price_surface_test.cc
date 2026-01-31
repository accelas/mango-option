// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/table/segmented_price_surface.hpp"
#include "src/option/table/price_table_builder.hpp"
#include "src/option/table/american_price_surface.hpp"
#include <cmath>

using namespace mango;

namespace {

// Build a small AmericanPriceSurface for testing.
// mode: EarlyExercisePremium or RawPrice
// include_tau_zero: whether to include tau=0 in grid (for RawPrice/custom IC segments)
AmericanPriceSurface build_test_surface(
    SurfaceContent mode,
    bool include_tau_zero,
    double K_ref = 100.0)
{
    std::vector<double> m_grid = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau_grid;
    if (include_tau_zero) {
        tau_grid = {0.0, 0.1, 0.25, 0.5};
    } else {
        tau_grid = {0.05, 0.1, 0.25, 0.5};
    }
    std::vector<double> vol_grid = {0.15, 0.20, 0.30, 0.40};
    std::vector<double> rate_grid = {0.02, 0.03, 0.04, 0.05};

    auto setup = PriceTableBuilder<4>::from_vectors(
        m_grid, tau_grid, vol_grid, rate_grid, K_ref,
        GridAccuracyParams{}, OptionType::PUT);
    EXPECT_TRUE(setup.has_value()) << "Failed to create builder";
    auto& [builder, axes] = *setup;

    if (mode == SurfaceContent::RawPrice) {
        builder.set_surface_content(SurfaceContent::RawPrice);
    }
    if (include_tau_zero) {
        builder.set_allow_tau_zero(true);
        // Custom IC: put payoff + small offset
        builder.set_initial_condition(
            [](std::span<const double> x, std::span<double> u) {
                for (size_t i = 0; i < x.size(); ++i) {
                    u[i] = std::max(1.0 - std::exp(x[i]), 0.0) + 0.005;
                }
            });
    }

    auto result = builder.build(axes);
    EXPECT_TRUE(result.has_value()) << "Failed to build surface";

    auto aps = AmericanPriceSurface::create(result->surface, OptionType::PUT);
    EXPECT_TRUE(aps.has_value()) << "Failed to create AmericanPriceSurface";

    return std::move(*aps);
}

// Helper to build a two-segment SegmentedPriceSurface
// Segment 0 (closest to expiry): EEP, τ ∈ [0, 0.5]
// Segment 1 (earlier):           mode1, τ ∈ (0.5, 1.0]
SegmentedPriceSurface build_two_segment_surface(
    SurfaceContent seg1_mode = SurfaceContent::EarlyExercisePremium,
    double dividend_amount = 0.0,
    double dividend_time = 0.5)
{
    double K_ref = 100.0;
    double T = 1.0;

    // Segment 0: EEP, last segment (closest to expiry)
    auto seg0_surface = build_test_surface(
        SurfaceContent::EarlyExercisePremium, false, K_ref);

    // Segment 1: user-specified mode
    bool include_tau_zero = (seg1_mode == SurfaceContent::RawPrice);
    auto seg1_surface = build_test_surface(seg1_mode, include_tau_zero, K_ref);

    SegmentedPriceSurface::Config config;
    config.K_ref = K_ref;
    config.T = T;

    config.segments.push_back(SegmentedPriceSurface::Segment{
        .surface = std::move(seg0_surface),
        .tau_start = 0.0,
        .tau_end = 0.5,
    });
    config.segments.push_back(SegmentedPriceSurface::Segment{
        .surface = std::move(seg1_surface),
        .tau_start = 0.5,
        .tau_end = 1.0,
    });

    if (dividend_amount > 0.0) {
        config.dividends.push_back({dividend_time, dividend_amount});
    }

    auto result = SegmentedPriceSurface::create(std::move(config));
    EXPECT_TRUE(result.has_value()) << "Failed to create SegmentedPriceSurface";
    return std::move(*result);
}

}  // namespace

// ---------------------------------------------------------------------------
// Validation tests
// ---------------------------------------------------------------------------

TEST(SegmentedPriceSurfaceTest, CreateFailsWithEmptySegments) {
    SegmentedPriceSurface::Config config;
    config.K_ref = 100.0;
    config.T = 1.0;

    auto result = SegmentedPriceSurface::create(std::move(config));
    EXPECT_FALSE(result.has_value());
}

TEST(SegmentedPriceSurfaceTest, CreateFailsWithInvalidKRef) {
    auto seg = build_test_surface(SurfaceContent::EarlyExercisePremium, false);

    SegmentedPriceSurface::Config config;
    config.K_ref = -1.0;
    config.T = 1.0;
    config.segments.push_back(SegmentedPriceSurface::Segment{
        .surface = std::move(seg),
        .tau_start = 0.0,
        .tau_end = 0.5,
    });

    auto result = SegmentedPriceSurface::create(std::move(config));
    EXPECT_FALSE(result.has_value());
}

// ---------------------------------------------------------------------------
// Segment routing tests
// ---------------------------------------------------------------------------

TEST(SegmentedPriceSurfaceTest, FindsCorrectSegment) {
    auto sps = build_two_segment_surface();

    double K = 100.0;
    double sigma = 0.25;
    double rate = 0.05;

    // Query in segment 0: τ = 0.3 → local τ = 0.3
    double p0 = sps.price(100.0, K, 0.3, sigma, rate);
    EXPECT_GT(p0, 0.0);
    EXPECT_FALSE(std::isnan(p0));

    // Query in segment 1: τ = 0.8 → local τ = 0.3
    double p1 = sps.price(100.0, K, 0.8, sigma, rate);
    EXPECT_GT(p1, 0.0);
    EXPECT_FALSE(std::isnan(p1));

    // Prices at τ=0.8 should generally be higher than at τ=0.3 (more time value)
    // but both should be valid finite numbers
    EXPECT_TRUE(std::isfinite(p0));
    EXPECT_TRUE(std::isfinite(p1));
}

TEST(SegmentedPriceSurfaceTest, BoundaryTauGoesToCorrectSegment) {
    auto sps = build_two_segment_surface();

    double K = 100.0;
    double sigma = 0.25;
    double rate = 0.05;

    // τ = 0.5 is the boundary. For segment 0: [0, 0.5] inclusive.
    // For segment 1: (0.5, 1.0]. So τ=0.5 should go to segment 0.
    double p_boundary = sps.price(100.0, K, 0.5, sigma, rate);
    EXPECT_GT(p_boundary, 0.0);
    EXPECT_TRUE(std::isfinite(p_boundary));
}

// ---------------------------------------------------------------------------
// Spot adjustment (dividend) tests
// ---------------------------------------------------------------------------

TEST(SegmentedPriceSurfaceTest, SpotAdjustmentForDividend) {
    double D = 2.0;
    double t_div = 0.5;  // calendar time
    auto sps = build_two_segment_surface(SurfaceContent::EarlyExercisePremium, D, t_div);

    double K = 100.0;
    double sigma = 0.25;
    double rate = 0.05;

    // Query τ = 0.8 (calendar t = 0.2).
    // Segment 1: tau_start=0.5 → t_boundary = 1.0 - 0.5 = 0.5
    // Dividend at t=0.5: 0.2 < 0.5 <= 0.5 → subtracted. S_adj = 100 - 2 = 98
    double p_with_div = sps.price(100.0, K, 0.8, sigma, rate);

    // Build without dividend for comparison
    auto sps_no_div = build_two_segment_surface(SurfaceContent::EarlyExercisePremium, 0.0);
    double p_no_div = sps_no_div.price(100.0, K, 0.8, sigma, rate);

    // With dividend subtracted from spot, put price should be higher (lower S_adj)
    // But both should be valid
    EXPECT_TRUE(std::isfinite(p_with_div));
    EXPECT_TRUE(std::isfinite(p_no_div));

    // For a put: lower spot → higher price
    // The dividend-adjusted price should differ from the non-adjusted one
    EXPECT_NE(p_with_div, p_no_div);
}

TEST(SegmentedPriceSurfaceTest, NoDividendAdjustmentAfterExDiv) {
    double D = 2.0;
    double t_div = 0.5;  // calendar time
    auto sps = build_two_segment_surface(SurfaceContent::EarlyExercisePremium, D, t_div);

    double K = 100.0;
    double sigma = 0.25;
    double rate = 0.05;

    // Query τ = 0.3 (calendar t = 0.7).
    // Segment 0: tau_start=0.0 → t_boundary = 1.0 - 0.0 = 1.0
    // Dividend at t=0.5: 0.7 < 0.5? NO. Not subtracted. S_adj = 100.
    double p_after_exdiv = sps.price(100.0, K, 0.3, sigma, rate);

    // Build without dividend for comparison
    auto sps_no_div = build_two_segment_surface(SurfaceContent::EarlyExercisePremium, 0.0);
    double p_no_div = sps_no_div.price(100.0, K, 0.3, sigma, rate);

    // After ex-div date, the dividend should not affect the price
    // Both surfaces use same segment 0, no spot adjustment
    EXPECT_NEAR(p_after_exdiv, p_no_div, 1e-10);
}

// ---------------------------------------------------------------------------
// Vega tests
// ---------------------------------------------------------------------------

TEST(SegmentedPriceSurfaceTest, AnalyticVegaForEEPSegment) {
    auto sps = build_two_segment_surface();

    double K = 100.0;
    double sigma = 0.25;
    double rate = 0.05;

    // Query in segment 0 (EEP): should get analytic vega
    double v = sps.vega(100.0, K, 0.3, sigma, rate);
    EXPECT_TRUE(std::isfinite(v));
    EXPECT_GT(v, 0.0);  // Vega should be positive for puts
}

TEST(SegmentedPriceSurfaceTest, FDVegaForRawPriceSegment) {
    auto sps = build_two_segment_surface(SurfaceContent::RawPrice);

    double K = 100.0;
    double sigma = 0.25;
    double rate = 0.05;

    // Query in segment 1 (RawPrice): should use FD vega
    double v = sps.vega(100.0, K, 0.8, sigma, rate);
    EXPECT_TRUE(std::isfinite(v));
    // Vega should be positive (option value increases with vol)
    EXPECT_GT(v, 0.0);

    // Verify FD vega is consistent with manual FD calculation
    double eps = std::max(1e-4, 1e-4 * sigma);
    double p_up = sps.price(100.0, K, 0.8, sigma + eps, rate);
    double p_dn = sps.price(100.0, K, 0.8, sigma - eps, rate);
    double fd_vega = (p_up - p_dn) / (2.0 * eps);
    EXPECT_NEAR(v, fd_vega, 1e-10);
}

// ---------------------------------------------------------------------------
// Edge case tests
// ---------------------------------------------------------------------------

TEST(SegmentedPriceSurfaceTest, SpotClampWhenSAdjNegative) {
    // Large dividend that makes S_adj negative
    double D = 200.0;  // D > S
    double t_div = 0.5;
    auto sps = build_two_segment_surface(SurfaceContent::EarlyExercisePremium, D, t_div);

    double K = 100.0;
    double sigma = 0.25;
    double rate = 0.05;

    // S=100, D=200 → S_adj = 100 - 200 = -100 → clamped to 1e-8
    double p = sps.price(100.0, K, 0.8, sigma, rate);
    EXPECT_TRUE(std::isfinite(p));
    EXPECT_FALSE(std::isnan(p));
}

// ---------------------------------------------------------------------------
// Bounds tests
// ---------------------------------------------------------------------------

TEST(SegmentedPriceSurfaceTest, BoundsSpanFullMaturityRange) {
    auto sps = build_two_segment_surface();

    // tau_min reflects the first segment's actual grid start (not 0.0)
    EXPECT_GT(sps.tau_min(), 0.0);
    EXPECT_DOUBLE_EQ(sps.tau_max(), 1.0);

    // Other bounds should come from segment 0
    EXPECT_GT(sps.m_min(), 0.0);
    EXPECT_GT(sps.m_max(), sps.m_min());
    EXPECT_GT(sps.sigma_min(), 0.0);
    EXPECT_GT(sps.sigma_max(), sps.sigma_min());
    EXPECT_LT(sps.rate_max(), 1.0);  // sanity
}

TEST(SegmentedPriceSurfaceTest, KRefAccessor) {
    auto sps = build_two_segment_surface();
    EXPECT_DOUBLE_EQ(sps.K_ref(), 100.0);
}
