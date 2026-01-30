// SPDX-License-Identifier: MIT
/**
 * @file simple_yfinance_test.cc
 * @brief Tests for yfinance data pipeline (converted from simple_yfinance_example.cpp)
 *
 * Validates:
 * - ChainBuilder with YFinanceSource data
 * - Option chain construction from simulated yfinance data
 * - Chain structure (expiries, calls, puts)
 * - Data preservation through pipeline
 */

#include "src/simple/simple.hpp"
#include <gtest/gtest.h>
#include <ranges>

using namespace mango::simple;

class SimpleYFinanceTest : public ::testing::Test {
  protected:
    OptionChain build_spy_chain() {
        Converter<YFinanceSource>::RawOption spy_calls[] = {
            {.expiry = "2024-06-21", .strike = 575.0, .bid = 6.10, .ask = 6.25, .lastPrice = 6.15, .volume = 15420, .openInterest = 28300, .impliedVolatility = 0.142},
            {.expiry = "2024-06-21", .strike = 580.0, .bid = 2.85, .ask = 2.92, .lastPrice = 2.88, .volume = 42150, .openInterest = 51200, .impliedVolatility = 0.128},
            {.expiry = "2024-06-21", .strike = 585.0, .bid = 0.95, .ask = 1.02, .lastPrice = 0.98, .volume = 31200, .openInterest = 39100, .impliedVolatility = 0.135},
            {.expiry = "2024-06-21", .strike = 590.0, .bid = 0.22, .ask = 0.28, .lastPrice = 0.25, .volume = 18900, .openInterest = 22400, .impliedVolatility = 0.148},
        };

        Converter<YFinanceSource>::RawOption spy_puts[] = {
            {.expiry = "2024-06-21", .strike = 570.0, .bid = 0.18, .ask = 0.24, .lastPrice = 0.21, .volume = 12300, .openInterest = 18700, .impliedVolatility = 0.152},
            {.expiry = "2024-06-21", .strike = 575.0, .bid = 0.52, .ask = 0.58, .lastPrice = 0.55, .volume = 28400, .openInterest = 35600, .impliedVolatility = 0.138},
            {.expiry = "2024-06-21", .strike = 580.0, .bid = 2.30, .ask = 2.42, .lastPrice = 2.35, .volume = 38700, .openInterest = 48200, .impliedVolatility = 0.126},
            {.expiry = "2024-06-21", .strike = 585.0, .bid = 5.40, .ask = 5.55, .lastPrice = 5.48, .volume = 21500, .openInterest = 31400, .impliedVolatility = 0.132},
        };

        auto builder = ChainBuilder<YFinanceSource>{}
            .symbol("SPY")
            .spot(580.50)
            .quote_time("2024-06-21T10:30:00")
            .settlement(Settlement::PM)
            .dividend_yield(0.013);

        for (const auto& call : spy_calls) {
            builder.add_call(call.expiry, call);
        }
        for (const auto& put : spy_puts) {
            builder.add_put(put.expiry, put);
        }

        return builder.build();
    }
};

TEST_F(SimpleYFinanceTest, ChainBuildsSuccessfully) {
    auto chain = build_spy_chain();

    EXPECT_EQ(chain.symbol, "SPY");
    ASSERT_TRUE(chain.spot.has_value());
    EXPECT_NEAR(chain.spot->to_double(), 580.50, 1e-6);
    ASSERT_TRUE(chain.quote_time.has_value());
    ASSERT_TRUE(chain.dividends.has_value());
    EXPECT_NEAR(std::get<double>(*chain.dividends), 0.013, 1e-6);
}

TEST_F(SimpleYFinanceTest, ExpiriesCorrect) {
    auto chain = build_spy_chain();
    EXPECT_EQ(chain.expiries.size(), 1u);
    ASSERT_TRUE(chain.expiries[0].settlement.has_value());
    EXPECT_EQ(*chain.expiries[0].settlement, Settlement::PM);
}

TEST_F(SimpleYFinanceTest, CallsAndPutsCount) {
    auto chain = build_spy_chain();
    ASSERT_EQ(chain.expiries.size(), 1u);

    auto n_calls = std::ranges::distance(chain.expiries[0].calls());
    auto n_puts = std::ranges::distance(chain.expiries[0].puts());

    EXPECT_EQ(n_calls, 4);
    EXPECT_EQ(n_puts, 4);
}

TEST_F(SimpleYFinanceTest, OptionDataPreserved) {
    auto chain = build_spy_chain();
    ASSERT_EQ(chain.expiries.size(), 1u);

    // Check first call preserves data
    auto calls_range = chain.expiries[0].calls();
    auto it = calls_range.begin();
    ASSERT_NE(it, calls_range.end());

    EXPECT_NEAR(it->strike.to_double(), 575.0, 1e-6);
    ASSERT_TRUE(it->bid.has_value());
    EXPECT_NEAR(it->bid->to_double(), 6.10, 1e-6);
    ASSERT_TRUE(it->ask.has_value());
    EXPECT_NEAR(it->ask->to_double(), 6.25, 1e-6);
    ASSERT_TRUE(it->volume.has_value());
    EXPECT_EQ(*it->volume, 15420);
    ASSERT_TRUE(it->open_interest.has_value());
    EXPECT_EQ(*it->open_interest, 28300);
}

TEST_F(SimpleYFinanceTest, TauComputation) {
    auto chain = build_spy_chain();

    MarketContext ctx;
    ctx.rate = 0.053;
    // Quote time before expiry to get positive tau
    ctx.valuation_time = Timestamp{"2024-06-14T10:30:00"};

    double tau = compute_tau(*ctx.valuation_time, chain.expiries[0].expiry);
    EXPECT_GT(tau, 0.0);
    EXPECT_LT(tau, 1.0);  // Less than 1 year
}
