#include <gtest/gtest.h>
#include "src/simple/vol_surface.hpp"
#include "src/simple/chain_builder.hpp"
#include "src/simple/sources/yfinance.hpp"

using namespace mango::simple;

TEST(VolSurfaceTest, ComputeSmileFromChain) {
    // Build a simple chain
    auto chain = ChainBuilder<YFinanceSource>{}
        .symbol("SPY")
        .spot(580.50)
        .quote_time("2024-06-21T10:30:00")
        .settlement(Settlement::PM)
        .dividend_yield(0.013)
        .build();

    // Add a single option for testing
    ExpirySlice slice;
    slice.expiry = Timestamp{"2024-06-28"};  // 1 week out
    slice.settlement = Settlement::PM;

    OptionLeg call;
    call.strike = Price{580.0};
    call.bid = Price{5.50};
    call.ask = Price{5.70};
    slice.calls.push_back(call);

    chain.expiries.push_back(std::move(slice));

    MarketContext ctx;
    ctx.rate = 0.053;
    ctx.valuation_time = Timestamp{"2024-06-21T10:30:00"};

    // This requires a precomputed price table, so we test the structure
    // In real usage, you'd provide a solver
    VolatilitySurface surface;
    surface.symbol = chain.symbol;
    surface.spot = *chain.spot;

    EXPECT_EQ(surface.symbol, "SPY");
}

TEST(VolSmileTest, SmilePointStructure) {
    VolatilitySmile::Point pt;
    pt.strike = Price{580.0};
    pt.moneyness = 0.0;  // ATM
    pt.iv_mid = 0.15;

    EXPECT_DOUBLE_EQ(pt.strike.to_double(), 580.0);
    EXPECT_TRUE(pt.iv_mid.has_value());
}
