// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/segmented_price_table_builder.hpp"
#include <cmath>

using namespace mango;

TEST(SegmentedPriceTableBuilderTest, BuildWithOneDividend) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.0, .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}}},
        .grid = ManualGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value()) << "Build should succeed";

    // Verify price is reasonable for ATM put
    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.8, .sigma = 0.20, .rate = 0.05};
    double price = result->price(q);
    EXPECT_GT(price, 0.0);
    EXPECT_LT(price, 50.0);  // sanity check

    // Verify vega is finite
    double vega = result->vega(q);
    EXPECT_TRUE(std::isfinite(vega));
    EXPECT_GT(vega, 0.0);
}

TEST(SegmentedPriceTableBuilderTest, BuildWithNoDividends) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.dividend_yield = 0.02},
        .grid = ManualGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());

    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.5, .sigma = 0.20, .rate = 0.05};
    double price = result->price(q);
    EXPECT_GT(price, 0.0);
}

TEST(SegmentedPriceTableBuilderTest, DividendAtExpiryIgnored) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.discrete_dividends = {{.calendar_time = 1.0, .amount = 5.0}}},  // at expiry — should be filtered out
        .grid = ManualGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());
}

TEST(SegmentedPriceTableBuilderTest, DividendAtTimeZeroIgnored) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.discrete_dividends = {{.calendar_time = 0.0, .amount = 3.0}}},  // at time 0 — should be filtered out
        .grid = ManualGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());
}

TEST(SegmentedPriceTableBuilderTest, InvalidKRefFails) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = -100.0,
        .option_type = OptionType::PUT,
        .grid = ManualGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 1.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    EXPECT_FALSE(result.has_value());
}

TEST(SegmentedPriceTableBuilderTest, InvalidMaturityFails) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .grid = ManualGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.03, 0.05, 0.07, 0.09},
        },
        .maturity = 0.0,
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    EXPECT_FALSE(result.has_value());
}
