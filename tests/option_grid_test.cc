// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/option_grid.hpp"

TEST(OptionGridTest, DefaultConstruction) {
    mango::OptionGrid chain;
    EXPECT_TRUE(chain.ticker.empty());
    EXPECT_EQ(chain.spot, 0.0);
    EXPECT_TRUE(chain.strikes.empty());
    EXPECT_TRUE(chain.maturities.empty());
    EXPECT_TRUE(chain.implied_vols.empty());
    EXPECT_TRUE(chain.rates.empty());
    EXPECT_EQ(chain.dividend_yield, 0.0);
}

TEST(OptionGridTest, FieldPopulation) {
    mango::OptionGrid chain;
    chain.ticker = "AAPL";
    chain.spot = 150.0;
    chain.strikes = {140.0, 150.0, 160.0};
    chain.maturities = {0.25, 0.5, 1.0};
    chain.implied_vols = {0.20, 0.22, 0.25};
    chain.rates = {0.05, 0.05, 0.05};
    chain.dividend_yield = 0.01;

    EXPECT_EQ(chain.ticker, "AAPL");
    EXPECT_EQ(chain.spot, 150.0);
    EXPECT_EQ(chain.strikes.size(), 3);
    EXPECT_EQ(chain.dividend_yield, 0.01);
}
