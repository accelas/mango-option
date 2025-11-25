#include <gtest/gtest.h>
#include "src/option/table/price_table_metadata.hpp"

namespace mango {
namespace {

TEST(PriceTableMetadataTest, DefaultConstruction) {
    PriceTableMetadata meta;
    EXPECT_DOUBLE_EQ(meta.K_ref, 0.0);
    EXPECT_DOUBLE_EQ(meta.dividend_yield, 0.0);
    EXPECT_TRUE(meta.discrete_dividends.empty());
}

TEST(PriceTableMetadataTest, WithDiscreteDividends) {
    PriceTableMetadata meta{
        .K_ref = 100.0,
        .dividend_yield = 0.02,
        .discrete_dividends = {{0.25, 2.50}, {0.75, 2.50}}
    };

    EXPECT_DOUBLE_EQ(meta.K_ref, 100.0);
    EXPECT_DOUBLE_EQ(meta.dividend_yield, 0.02);
    EXPECT_EQ(meta.discrete_dividends.size(), 2);
    EXPECT_DOUBLE_EQ(meta.discrete_dividends[0].first, 0.25);
    EXPECT_DOUBLE_EQ(meta.discrete_dividends[0].second, 2.50);
}

} // namespace
} // namespace mango
