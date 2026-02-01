// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/table/segmented_multi_kref_builder.hpp"

using namespace mango;

TEST(SegmentedMultiKRefBuilderTest, BuildWithExplicitKRefs) {
    SegmentedMultiKRefBuilder::Config config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .dividends = {{.calendar_time = 0.5, .amount = 2.0}},
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
        .dividend_yield = 0.0,
        .dividends = {{.calendar_time = 0.5, .amount = 2.0}},
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
        .dividend_yield = 0.02,
        .dividends = {},
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
