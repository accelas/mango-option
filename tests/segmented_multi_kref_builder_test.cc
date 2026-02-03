// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/segmented_multi_kref_builder.hpp"
#include "mango/option/table/segmented_price_table_builder.hpp"

using namespace mango;

TEST(SegmentedMultiKRefBuilderTest, BuildWithExplicitKRefs) {
    SegmentedMultiKRefBuilder::Config config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.0, .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}}},
        .moneyness_grid = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
        .maturity = 1.0,
        .vol_grid = {0.15, 0.20, 0.30, 0.40},
        .rate_grid = {0.03, 0.05, 0.07, 0.09},
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    };

    auto result = SegmentedMultiKRefBuilder::build(config);
    ASSERT_TRUE(result.has_value())
        << "error code: " << static_cast<int>(result.error().code)
        << " value: " << result.error().value;

    double price = result->price(100.0, 95.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
    EXPECT_LT(price, 50.0);
}

TEST(SegmentedMultiKRefBuilderTest, AutoKRefSelection) {
    SegmentedMultiKRefBuilder::Config config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.0, .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}}},
        .moneyness_grid = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
        .maturity = 1.0,
        .vol_grid = {0.15, 0.20, 0.30, 0.40},
        .rate_grid = {0.03, 0.05, 0.07, 0.09},
        .kref_config = {.K_ref_count = 3, .K_ref_span = 0.2},
    };

    auto result = SegmentedMultiKRefBuilder::build(config);
    ASSERT_TRUE(result.has_value())
        << "error code: " << static_cast<int>(result.error().code)
        << " value: " << result.error().value;

    double price = result->price(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
}

TEST(SegmentedMultiKRefBuilderTest, NoDividendsFallback) {
    SegmentedMultiKRefBuilder::Config config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.02},
        .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},
        .maturity = 1.0,
        .vol_grid = {0.15, 0.20, 0.30, 0.40},
        .rate_grid = {0.03, 0.05, 0.07, 0.09},
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    };

    auto result = SegmentedMultiKRefBuilder::build(config);
    ASSERT_TRUE(result.has_value())
        << "error code: " << static_cast<int>(result.error().code)
        << " value: " << result.error().value;

    double price = result->price(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
}

TEST(SegmentedPriceTableBuilderTest, SkipMoneyExpansionFlag) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.02,
                      .discrete_dividends = {{.calendar_time = 0.5, .amount = 5.0}}},
        .moneyness_grid = {0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
        .maturity = 1.0,
        .vol_grid = {0.10, 0.15, 0.20, 0.30},
        .rate_grid = {0.02, 0.03, 0.05, 0.07},
        .skip_moneyness_expansion = true,
    };

    auto surface = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(surface.has_value());

    // When expansion is skipped, the surface's moneyness range should
    // match the input grid exactly (no extra points below 0.6).
    // Query at the lowest input moneyness should work.
    double price = surface->price(100.0, 100.0 / 0.6, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
    EXPECT_TRUE(std::isfinite(price));
}
