// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/table/segmented_price_table_builder.hpp"
#include <cmath>

using namespace mango;

TEST(SegmentedPriceTableBuilderTest, BuildWithOneDividend) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}},
        .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},
        .maturity = 1.0,
        .vol_grid = {0.15, 0.20, 0.30, 0.40},
        .rate_grid = {0.03, 0.05, 0.07, 0.09},
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value()) << "Build should succeed";

    // Verify price is reasonable for ATM put
    double price = result->price(100.0, 100.0, 0.8, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
    EXPECT_LT(price, 50.0);  // sanity check

    // Verify vega is finite
    double vega = result->vega(100.0, 100.0, 0.8, 0.20, 0.05);
    EXPECT_TRUE(std::isfinite(vega));
    EXPECT_GT(vega, 0.0);
}

TEST(SegmentedPriceTableBuilderTest, BuildWithNoDividends) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {},
        .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},
        .maturity = 1.0,
        .vol_grid = {0.15, 0.20, 0.30, 0.40},
        .rate_grid = {0.03, 0.05, 0.07, 0.09},
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());

    double price = result->price(100.0, 100.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
}

TEST(SegmentedPriceTableBuilderTest, DividendAtExpiryIgnored) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .discrete_dividends = {{.calendar_time = 1.0, .amount = 5.0}},  // at expiry — should be filtered out
        .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},
        .maturity = 1.0,
        .vol_grid = {0.15, 0.20, 0.30, 0.40},
        .rate_grid = {0.03, 0.05, 0.07, 0.09},
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());
}

TEST(SegmentedPriceTableBuilderTest, DividendAtTimeZeroIgnored) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .discrete_dividends = {{.calendar_time = 0.0, .amount = 3.0}},  // at time 0 — should be filtered out
        .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},
        .maturity = 1.0,
        .vol_grid = {0.15, 0.20, 0.30, 0.40},
        .rate_grid = {0.03, 0.05, 0.07, 0.09},
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());
}

TEST(SegmentedPriceTableBuilderTest, InvalidKRefFails) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = -100.0,
        .option_type = OptionType::PUT,
        .discrete_dividends = {},
        .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},
        .maturity = 1.0,
        .vol_grid = {0.15, 0.20, 0.30, 0.40},
        .rate_grid = {0.03, 0.05, 0.07, 0.09},
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    EXPECT_FALSE(result.has_value());
}

TEST(SegmentedPriceTableBuilderTest, InvalidMaturityFails) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .discrete_dividends = {},
        .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},
        .maturity = 0.0,
        .vol_grid = {0.15, 0.20, 0.30, 0.40},
        .rate_grid = {0.03, 0.05, 0.07, 0.09},
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    EXPECT_FALSE(result.has_value());
}
