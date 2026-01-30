// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/simple/chain_builder.hpp"
#include "src/simple/sources/yfinance.hpp"
#include "src/simple/sources/databento.hpp"

using namespace mango::simple;

TEST(ChainBuilderTest, YFinanceBasic) {
    auto chain = ChainBuilder<YFinanceSource>{}
        .symbol("SPY")
        .spot(580.50)
        .quote_time("2024-06-21T10:30:00")
        .build();

    EXPECT_EQ(chain.symbol, "SPY");
    ASSERT_TRUE(chain.spot.has_value());
    EXPECT_DOUBLE_EQ(chain.spot->to_double(), 580.50);
}

TEST(ChainBuilderTest, DabentoBasic) {
    auto chain = ChainBuilder<DatabentSource>{}
        .symbol("SPY")
        .spot(580500000000LL)  // Fixed-point
        .quote_time(1718972400000000000ULL)  // Nanoseconds
        .build();

    EXPECT_EQ(chain.symbol, "SPY");
    ASSERT_TRUE(chain.spot.has_value());
    EXPECT_DOUBLE_EQ(chain.spot->to_double(), 580.50);
    EXPECT_TRUE(chain.spot->is_fixed_point());
}

TEST(ChainBuilderTest, AddOptions) {
    Converter<YFinanceSource>::RawOption call_opt{
        .expiry = "2024-06-21",
        .strike = 580.0,
        .bid = 2.85,
        .ask = 2.92,
        .lastPrice = 2.88,
        .volume = 42150,
        .openInterest = 51200,
        .impliedVolatility = 0.128
    };

    auto chain = ChainBuilder<YFinanceSource>{}
        .symbol("SPY")
        .spot(580.50)
        .add_call("2024-06-21", call_opt)
        .settlement(Settlement::PM)
        .build();

    EXPECT_EQ(chain.expiries.size(), 1);
    EXPECT_EQ(chain.expiries[0].options.size(), 1);
    EXPECT_EQ(std::ranges::distance(chain.expiries[0].calls()), 1);
    EXPECT_EQ(chain.expiries[0].options[0].type, OptionType::CALL);
    EXPECT_DOUBLE_EQ(chain.expiries[0].options[0].strike.to_double(), 580.0);
}
