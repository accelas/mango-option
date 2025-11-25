#include <gtest/gtest.h>
#include "src/option/table/price_table_config.hpp"

namespace mango {
namespace {

TEST(PriceTableConfigTest, DefaultValues) {
    PriceTableConfig config;
    EXPECT_EQ(config.option_type, OptionType::PUT);
    EXPECT_EQ(config.n_time, 1000);
    EXPECT_DOUBLE_EQ(config.dividend_yield, 0.0);
    EXPECT_TRUE(config.discrete_dividends.empty());
}

TEST(PriceTableConfigTest, WithDiscreteDividends) {
    PriceTableConfig config{
        .option_type = OptionType::CALL,
        .n_time = 500,
        .dividend_yield = 0.01,
        .discrete_dividends = {{0.25, 2.0}, {0.75, 2.0}}
    };

    EXPECT_EQ(config.option_type, OptionType::CALL);
    EXPECT_EQ(config.discrete_dividends.size(), 2);
}

} // namespace
} // namespace mango
