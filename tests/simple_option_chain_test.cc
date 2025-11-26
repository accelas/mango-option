#include <gtest/gtest.h>
#include "src/simple/option_chain.hpp"

using namespace mango::simple;

TEST(SimpleOptionChainTest, EmptyChain) {
    OptionChain chain;
    EXPECT_TRUE(chain.expiries.empty());
    EXPECT_FALSE(chain.spot.has_value());
}

TEST(SimpleOptionChainTest, ChainWithData) {
    OptionChain chain;
    chain.symbol = "SPY";
    chain.spot = Price{580.50};
    chain.quote_time = Timestamp{"2024-06-21T10:30:00"};

    ExpirySlice slice;
    slice.expiry = Timestamp{"2024-06-21"};
    slice.settlement = Settlement::PM;

    OptionLeg call;
    call.strike = Price{580.0};
    call.bid = Price{2.85};
    call.ask = Price{2.92};
    slice.calls.push_back(call);

    chain.expiries.push_back(std::move(slice));

    EXPECT_EQ(chain.symbol, "SPY");
    EXPECT_EQ(chain.expiries.size(), 1);
    EXPECT_EQ(chain.expiries[0].calls.size(), 1);
}

TEST(SimpleMarketContextTest, DefaultContext) {
    MarketContext ctx;
    EXPECT_FALSE(ctx.rate.has_value());
    EXPECT_FALSE(ctx.valuation_time.has_value());
}

TEST(SimpleMarketContextTest, ContextWithRate) {
    MarketContext ctx;
    ctx.rate = 0.053;  // Flat 5.3%

    EXPECT_TRUE(ctx.rate.has_value());
    EXPECT_DOUBLE_EQ(std::get<double>(*ctx.rate), 0.053);
}

TEST(SimpleMarketContextTest, ContextWithYieldCurve) {
    auto curve = mango::YieldCurve::flat(0.05);
    MarketContext ctx;
    ctx.rate = curve;

    EXPECT_TRUE(ctx.rate.has_value());
    EXPECT_TRUE(std::holds_alternative<mango::YieldCurve>(*ctx.rate));
}
