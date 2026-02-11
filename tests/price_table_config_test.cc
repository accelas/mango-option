// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/bspline/bspline_builder.hpp"

namespace mango {
namespace {

TEST(PriceTableConfigTest, DefaultValues) {
    PriceTableConfig config;
    EXPECT_EQ(config.option_type, OptionType::PUT);
    EXPECT_DOUBLE_EQ(config.dividends.dividend_yield, 0.0);
    EXPECT_TRUE(config.dividends.discrete_dividends.empty());
    // Default PDE grid is GridAccuracyParams (auto-estimated)
    ASSERT_TRUE(std::holds_alternative<GridAccuracyParams>(config.pde_grid));
}

TEST(PriceTableConfigTest, WithDiscreteDividends) {
    PriceTableConfig config{
        .option_type = OptionType::CALL,
        .dividends = {.dividend_yield = 0.01, .discrete_dividends = {{.calendar_time = 0.25, .amount = 2.0}, {.calendar_time = 0.75, .amount = 2.0}}}
    };

    EXPECT_EQ(config.option_type, OptionType::CALL);
    EXPECT_EQ(config.dividends.discrete_dividends.size(), 2);
}

} // namespace
} // namespace mango
