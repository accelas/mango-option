// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/simple/option_types.hpp"

using namespace mango::simple;

TEST(SimpleOptionTypesTest, SettlementDefaults) {
    OptionLeg leg;
    EXPECT_FALSE(leg.settlement.has_value());  // Unknown by default
}

TEST(SimpleOptionTypesTest, OptionLegWithOptionalFields) {
    OptionLeg leg;
    leg.strike = Price{580.0};
    leg.bid = Price{1.45};
    leg.ask = Price{1.52};
    // last, volume, open_interest are optional

    EXPECT_TRUE(leg.bid.has_value());
    EXPECT_TRUE(leg.ask.has_value());
    EXPECT_FALSE(leg.last.has_value());
    EXPECT_FALSE(leg.volume.has_value());
}

TEST(SimpleOptionTypesTest, OptionLegMid) {
    OptionLeg leg;
    leg.bid = Price{1.45};
    leg.ask = Price{1.52};

    auto mid = leg.mid();
    ASSERT_TRUE(mid.has_value());
    EXPECT_NEAR(mid->to_double(), 1.485, 1e-6);
}

TEST(SimpleOptionTypesTest, OptionLegMidWithoutBothPrices) {
    OptionLeg leg;
    leg.bid = Price{1.45};
    // No ask

    auto mid = leg.mid();
    EXPECT_FALSE(mid.has_value());
}

TEST(SimpleOptionTypesTest, OptionLegPriceForIV) {
    OptionLeg leg;
    leg.bid = Price{1.45};
    leg.ask = Price{1.52};
    leg.last = Price{1.50};

    // Prefer mid over last
    auto price = leg.price_for_iv();
    ASSERT_TRUE(price.has_value());
    EXPECT_NEAR(price->to_double(), 1.485, 1e-6);
}

TEST(SimpleOptionTypesTest, OptionLegPriceForIVFallback) {
    OptionLeg leg;
    leg.last = Price{1.50};
    // No bid/ask

    auto price = leg.price_for_iv();
    ASSERT_TRUE(price.has_value());
    EXPECT_DOUBLE_EQ(price->to_double(), 1.50);
}
