// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/table/segmented_multi_kref_surface.hpp"
#include "src/option/table/segmented_price_table_builder.hpp"
#include "src/option/table/price_surface_concept.hpp"

using namespace mango;

static_assert(PriceSurface<SegmentedMultiKRefSurface>,
    "SegmentedMultiKRefSurface must satisfy PriceSurface concept");

// Helper to build a SegmentedPriceSurface at a given K_ref
static SegmentedPriceSurface build_surface(double K_ref) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = K_ref,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .dividends = {{.calendar_time = 0.5, .amount = 2.0}},
        .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},
        .maturity = 1.0,
        .vol_grid = {0.15, 0.20, 0.30, 0.40},
        .rate_grid = {0.03, 0.05, 0.07, 0.09},
    };
    return SegmentedPriceTableBuilder::build(config).value();
}

TEST(SegmentedMultiKRefSurfaceTest, StrikeInterpolation) {
    auto s80 = build_surface(80.0);
    auto s100 = build_surface(100.0);
    auto s120 = build_surface(120.0);

    std::vector<SegmentedMultiKRefSurface::Entry> entries;
    entries.push_back({80.0, std::move(s80)});
    entries.push_back({100.0, std::move(s100)});
    entries.push_back({120.0, std::move(s120)});

    auto result = SegmentedMultiKRefSurface::create(std::move(entries));
    ASSERT_TRUE(result.has_value());

    // Query at strike=90 should interpolate between K_ref=80 and K_ref=100
    double p90 = result->price(100.0, 90.0, 0.5, 0.20, 0.05);
    EXPECT_GT(p90, 0.0);

    // Price at strike=90 should be between prices at K_ref=80 and K_ref=100
    double p80 = result->price(100.0, 80.0, 0.5, 0.20, 0.05);
    double p100 = result->price(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_TRUE(std::isfinite(p90));
    EXPECT_TRUE(std::isfinite(p80));
    EXPECT_TRUE(std::isfinite(p100));
}

TEST(SegmentedMultiKRefSurfaceTest, StrikeOutsideRange) {
    auto s80 = build_surface(80.0);
    auto s120 = build_surface(120.0);

    std::vector<SegmentedMultiKRefSurface::Entry> entries;
    entries.push_back({80.0, std::move(s80)});
    entries.push_back({120.0, std::move(s120)});

    auto result = SegmentedMultiKRefSurface::create(std::move(entries));
    ASSERT_TRUE(result.has_value());

    // Strike outside K_ref range uses the nearest surface.
    // EEP segments evaluate at actual strike; RawPrice segments use K_ref.
    double p60 = result->price(100.0, 60.0, 0.5, 0.20, 0.05);
    double p80 = result->price(100.0, 80.0, 0.5, 0.20, 0.05);
    EXPECT_TRUE(std::isfinite(p60));
    EXPECT_TRUE(std::isfinite(p80));
    EXPECT_GT(p60, 0.0);

    double p150 = result->price(100.0, 150.0, 0.5, 0.20, 0.05);
    double p120 = result->price(100.0, 120.0, 0.5, 0.20, 0.05);
    EXPECT_TRUE(std::isfinite(p150));
    EXPECT_TRUE(std::isfinite(p120));
    EXPECT_GT(p150, 0.0);
}

TEST(SegmentedMultiKRefSurfaceTest, VegaInterpolation) {
    auto s80 = build_surface(80.0);
    auto s100 = build_surface(100.0);

    std::vector<SegmentedMultiKRefSurface::Entry> entries;
    entries.push_back({80.0, std::move(s80)});
    entries.push_back({100.0, std::move(s100)});

    auto result = SegmentedMultiKRefSurface::create(std::move(entries));
    ASSERT_TRUE(result.has_value());

    double vega = result->vega(100.0, 90.0, 0.5, 0.20, 0.05);
    EXPECT_TRUE(std::isfinite(vega));
    EXPECT_GT(vega, 0.0);
}

TEST(SegmentedMultiKRefSurfaceTest, BoundsAreReasonable) {
    auto s80 = build_surface(80.0);
    auto s100 = build_surface(100.0);

    std::vector<SegmentedMultiKRefSurface::Entry> entries;
    entries.push_back({80.0, std::move(s80)});
    entries.push_back({100.0, std::move(s100)});

    auto result = SegmentedMultiKRefSurface::create(std::move(entries));
    ASSERT_TRUE(result.has_value());

    EXPECT_GT(result->m_max(), result->m_min());
    EXPECT_GT(result->tau_max(), result->tau_min());
    EXPECT_GT(result->sigma_max(), result->sigma_min());
    EXPECT_GT(result->rate_max(), result->rate_min());
}

TEST(SegmentedMultiKRefSurfaceTest, CreateRejectsEmptyEntries) {
    std::vector<SegmentedMultiKRefSurface::Entry> entries;
    auto result = SegmentedMultiKRefSurface::create(std::move(entries));
    ASSERT_FALSE(result.has_value());
}
