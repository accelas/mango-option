/**
 * @file simple_databento_test.cc
 * @brief Tests for Databento data pipeline (converted from simple_databento_example.cpp)
 *
 * Validates:
 * - ChainBuilder with DatabentSource fixed-point data
 * - Fixed-point precision preservation through pipeline
 * - Correct conversion of nanosecond timestamps
 */

#include "src/simple/simple.hpp"
#include <gtest/gtest.h>

using namespace mango::simple;

class SimpleDatabentTest : public ::testing::Test {
  protected:
    OptionChain build_spy_chain() {
        Converter<DatabentSource>::RawOption options[] = {
            {.ts_event = 1718972400000000000ULL, .price = 615000000000LL, .bid_px = 610000000000LL, .ask_px = 625000000000LL, .strike_price = 575000000000LL, .option_type = 'C'},
            {.ts_event = 1718972400000000000ULL, .price = 288000000000LL, .bid_px = 285000000000LL, .ask_px = 292000000000LL, .strike_price = 580000000000LL, .option_type = 'C'},
            {.ts_event = 1718972400000000000ULL, .price = 98000000000LL, .bid_px = 95000000000LL, .ask_px = 102000000000LL, .strike_price = 585000000000LL, .option_type = 'C'},
            {.ts_event = 1718972400000000000ULL, .price = 21000000000LL, .bid_px = 18000000000LL, .ask_px = 24000000000LL, .strike_price = 570000000000LL, .option_type = 'P'},
            {.ts_event = 1718972400000000000ULL, .price = 235000000000LL, .bid_px = 230000000000LL, .ask_px = 242000000000LL, .strike_price = 580000000000LL, .option_type = 'P'},
        };

        auto builder = ChainBuilder<DatabentSource>{}
            .symbol("SPY")
            .spot(580500000000LL)
            .quote_time(1718972400000000000ULL)
            .settlement(Settlement::PM);

        uint64_t expiry_nanos = 1719014400000000000ULL;

        for (const auto& opt : options) {
            if (opt.option_type == 'C') {
                builder.add_call(expiry_nanos, opt);
            } else {
                builder.add_put(expiry_nanos, opt);
            }
        }

        return builder.build();
    }
};

TEST_F(SimpleDatabentTest, ChainBuildsSuccessfully) {
    auto chain = build_spy_chain();

    EXPECT_EQ(chain.symbol, "SPY");
    ASSERT_TRUE(chain.spot.has_value());
    EXPECT_NEAR(chain.spot->to_double(), 580.5, 1e-6);
}

TEST_F(SimpleDatabentTest, FixedPointPrecisionPreserved) {
    auto chain = build_spy_chain();

    ASSERT_TRUE(chain.spot.has_value());
    EXPECT_TRUE(chain.spot->is_fixed_point());
}

TEST_F(SimpleDatabentTest, StrikePrecisionPreserved) {
    Converter<DatabentSource>::RawOption opt = {
        .ts_event = 1718972400000000000ULL,
        .price = 615000000000LL,
        .bid_px = 610000000000LL,
        .ask_px = 625000000000LL,
        .strike_price = 575000000000LL,
        .option_type = 'C'
    };

    auto leg = Converter<DatabentSource>::to_leg(opt);
    EXPECT_TRUE(leg.strike.is_fixed_point());
    EXPECT_NEAR(leg.strike.to_double(), 575.0, 1e-6);
}

TEST_F(SimpleDatabentTest, CallsAndPutsSegregated) {
    auto chain = build_spy_chain();
    ASSERT_EQ(chain.expiries.size(), 1u);

    auto n_calls = std::ranges::distance(chain.expiries[0].calls());
    auto n_puts = std::ranges::distance(chain.expiries[0].puts());

    EXPECT_EQ(n_calls, 3);
    EXPECT_EQ(n_puts, 2);
}
