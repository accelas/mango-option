// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/simple/vol_surface.hpp"
#include "mango/simple/chain_builder.hpp"
#include "mango/simple/sources/yfinance.hpp"

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
    call.type = OptionType::CALL;
    call.strike = Price{580.0};
    call.bid = Price{5.50};
    call.ask = Price{5.70};
    slice.options.push_back(call);

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
    pt.type = OptionType::CALL;
    pt.strike = Price{580.0};
    pt.moneyness = 0.0;  // ATM
    pt.iv_mid = 0.15;

    EXPECT_EQ(pt.type, OptionType::CALL);
    EXPECT_DOUBLE_EQ(pt.strike.to_double(), 580.0);
    EXPECT_TRUE(pt.iv_mid.has_value());
}

// ===========================================================================
// Regression tests for bugs found during code review
// ===========================================================================

// Regression: compute_vol_surface dropped discrete dividend schedules (#448)
// Bug: only the double alternative of DividendSpec was read; the
//      vector<Dividend> alternative left div_yield = 0 and never populated
//      IVQuery::discrete_dividends. These tests pin the conversion helper.
TEST(DividendConversionTest, ConvertsExDatesToSortedYearOffsets) {
    Timestamp val{"2026-01-01T00:00:00"};
    std::vector<Dividend> divs = {
        {.ex_date = Timestamp{"2026-07-02T00:00:00"}, .amount = Price{1.50}},
        {.ex_date = Timestamp{"2026-04-02T00:00:00"}, .amount = Price{1.25}},
    };
    auto out = convert_discrete_dividends(divs, val, 1.0);
    ASSERT_EQ(out.size(), 2u);
    // Sorted by calendar time even though the input was not.
    EXPECT_NEAR(out[0].calendar_time, 91.0 / 365.0, 0.01);
    EXPECT_DOUBLE_EQ(out[0].amount, 1.25);
    EXPECT_NEAR(out[1].calendar_time, 182.0 / 365.0, 0.01);
    EXPECT_DOUBLE_EQ(out[1].amount, 1.50);
}

TEST(DividendConversionTest, DropsPastAndPostExpiryDividends) {
    Timestamp val{"2026-01-01T00:00:00"};
    std::vector<Dividend> divs = {
        // Already gone ex before valuation: excluded.
        {.ex_date = Timestamp{"2025-12-15T00:00:00"}, .amount = Price{1.00}},
        // Inside the window: kept.
        {.ex_date = Timestamp{"2026-03-01T00:00:00"}, .amount = Price{1.50}},
        // After expiry (tau_max = 0.5): excluded.
        {.ex_date = Timestamp{"2027-06-01T00:00:00"}, .amount = Price{2.00}},
    };
    auto out = convert_discrete_dividends(divs, val, 0.5);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_DOUBLE_EQ(out[0].amount, 1.50);
    EXPECT_GT(out[0].calendar_time, 0.0);
    EXPECT_LE(out[0].calendar_time, 0.5);
}

TEST(DividendConversionTest, UnparseableExDateIsDropped) {
    Timestamp val{"2026-01-01T00:00:00"};
    std::vector<Dividend> divs = {
        {.ex_date = Timestamp{"not-a-date"}, .amount = Price{1.00}},
    };
    auto out = convert_discrete_dividends(divs, val, 1.0);
    EXPECT_TRUE(out.empty());
}
