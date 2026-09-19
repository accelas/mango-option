// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/surface_inversion.hpp"
#include <cmath>

using namespace mango;

// A monotone synthetic "surface": price = 10 + 40*(sigma - 0.2).
// Closures, not free functions: ObjectiveRef stores `const void*`, which a
// function pointer does not convert to.
static const auto lin_price = [](double s) { return 10.0 + 40.0 * (s - 0.2); };
static const auto lin_vega = [](double) { return 40.0; };

TEST(SurfaceInversion, EffectiveBracketAppliesCapConfigPublishedAndFallback) {
    SurfaceInversionPolicy pol{.published_sigma_min = 0.1, .published_sigma_max = 0.5};
    // time_value_pct = (12-0)/12 = 1 > 0.5 -> cap 3.0 -> [0.1, 0.5]
    auto b = effective_sigma_bracket(100.0, 100.0, OptionType::PUT, 12.0, pol);
    EXPECT_DOUBLE_EQ(b.first, 0.1);
    EXPECT_DOUBLE_EQ(b.second, 0.5);
    // Config narrower than published wins.
    pol.config_sigma_min = 0.15;
    pol.config_sigma_max = 0.4;
    b = effective_sigma_bracket(100.0, 100.0, OptionType::PUT, 12.0, pol);
    EXPECT_DOUBLE_EQ(b.first, 0.15);
    EXPECT_DOUBLE_EQ(b.second, 0.4);
    // Disjoint -> fallback to the published range.
    pol.config_sigma_min = 0.6;
    pol.config_sigma_max = 0.9;
    b = effective_sigma_bracket(100.0, 100.0, OptionType::PUT, 12.0, pol);
    EXPECT_DOUBLE_EQ(b.first, 0.1);
    EXPECT_DOUBLE_EQ(b.second, 0.5);
}

// The cap ladder only shows up when neither the caller's nor the surface's
// limits bind, so open both out to 5.0 and vary the time-value fraction.
TEST(SurfaceInversion, TimeValueCapLadderSetsTheUpperBound) {
    SurfaceInversionPolicy pol{.config_sigma_min = 0.01, .config_sigma_max = 5.0,
                               .published_sigma_min = 0.1, .published_sigma_max = 5.0};

    // ATM put, all time value: (12-0)/12 = 1 > 0.5 -> 3.0.
    auto b = effective_sigma_bracket(100.0, 100.0, OptionType::PUT, 12.0, pol);
    EXPECT_DOUBLE_EQ(b.first, 0.1);
    EXPECT_DOUBLE_EQ(b.second, 3.0);

    // ITM put worth 16 on 10 of intrinsic: 6/16 = 0.375 in (0.2, 0.5] -> 2.0.
    b = effective_sigma_bracket(90.0, 100.0, OptionType::PUT, 16.0, pol);
    EXPECT_DOUBLE_EQ(b.first, 0.1);
    EXPECT_DOUBLE_EQ(b.second, 2.0);

    // Same put worth 12: 2/12 = 0.1667 < 0.2 -> 1.5.
    b = effective_sigma_bracket(90.0, 100.0, OptionType::PUT, 12.0, pol);
    EXPECT_DOUBLE_EQ(b.first, 0.1);
    EXPECT_DOUBLE_EQ(b.second, 1.5);
}

TEST(SurfaceInversion, InvertsMonotoneSurface) {
    SurfaceInversionPolicy pol{.published_sigma_min = 0.1, .published_sigma_max = 0.5};
    auto r = invert_price_on_surface(lin_price, lin_vega, lin_price(0.31),
                                     {0.1, 0.5}, 100.0, pol);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(r->implied_vol, 0.31, 1e-7);
    EXPECT_FALSE(r->used_rate_approximation);
}

TEST(SurfaceInversion, ReportsProductErrorCodes) {
    SurfaceInversionPolicy pol{.published_sigma_min = 0.1, .published_sigma_max = 0.5};
    // Target above the surface's range -> BracketingFailed.
    auto no_root = invert_price_on_surface(lin_price, lin_vega, lin_price(0.9),
                                           {0.1, 0.5}, 100.0, pol);
    ASSERT_FALSE(no_root.has_value());
    EXPECT_EQ(no_root.error().code, IVErrorCode::BracketingFailed);

    // Flat surface -> VegaTooSmall from the quartile pre-check.
    auto flat_price = [](double) { return 10.0; };
    auto zero_vega = [](double) { return 0.0; };
    auto flat = invert_price_on_surface(flat_price, zero_vega, 10.0,
                                        {0.1, 0.5}, 100.0, pol);
    ASSERT_FALSE(flat.has_value());
    EXPECT_EQ(flat.error().code, IVErrorCode::VegaTooSmall);

    // Non-monotone with two crossings -> MultipleRoots.
    auto bump = [](double s) {
        return 10.0 + 40.0 * (s - 0.2) - 30.0 * (s - 0.2) * (s - 0.2) * 10.0;
    };
    auto unit_vega = [](double) { return 1.0; };
    // Both roots (sigma ~ 0.2140 and ~ 0.3194) lie inside [0.1, 0.5], so the
    // 17-point screen must refuse rather than pick one.
    auto multi = invert_price_on_surface(bump, unit_vega, 10.5, {0.1, 0.5}, 100.0, pol);
    ASSERT_FALSE(multi.has_value());
    EXPECT_EQ(multi.error().code, IVErrorCode::MultipleRoots);

    // NaN interior -> NumericalInstability.
    auto nan_price = [](double s) { return s > 0.3 ? std::nan("") : lin_price(s); };
    auto nf = invert_price_on_surface(nan_price, lin_vega, lin_price(0.25),
                                      {0.1, 0.5}, 100.0, pol);
    ASSERT_FALSE(nf.has_value());
    EXPECT_EQ(nf.error().code, IVErrorCode::NumericalInstability);
}
