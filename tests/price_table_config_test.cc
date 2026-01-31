// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/table/price_table_config.hpp"

namespace mango {
namespace {

TEST(PriceTableConfigTest, DefaultValues) {
    PriceTableConfig config;
    EXPECT_EQ(config.option_type, OptionType::PUT);
    EXPECT_DOUBLE_EQ(config.dividend_yield, 0.0);
    EXPECT_TRUE(config.discrete_dividends.empty());
    // Default PDE grid is ExplicitPDEGrid with 101 points and 1000 time steps
    ASSERT_TRUE(std::holds_alternative<ExplicitPDEGrid>(config.pde_grid));
    auto& grid = std::get<ExplicitPDEGrid>(config.pde_grid);
    EXPECT_EQ(grid.n_time, 1000);
    EXPECT_EQ(grid.grid_spec.n_points(), 101);
}

TEST(PriceTableConfigTest, WithDiscreteDividends) {
    PriceTableConfig config{
        .option_type = OptionType::CALL,
        .dividend_yield = 0.01,
        .discrete_dividends = {{0.25, 2.0}, {0.75, 2.0}}
    };

    EXPECT_EQ(config.option_type, OptionType::CALL);
    EXPECT_EQ(config.discrete_dividends.size(), 2);
}

TEST(PriceTableConfigTest, DefaultStoreEepIsTrue) {
    PriceTableConfig config;
    EXPECT_TRUE(config.store_eep);
}

TEST(PriceTableConfigTest, CanDisableStoreEep) {
    PriceTableConfig config;
    config.store_eep = false;
    EXPECT_FALSE(config.store_eep);
}

} // namespace
} // namespace mango
