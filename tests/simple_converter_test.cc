// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/simple/converter.hpp"
#include "src/simple/sources/yfinance.hpp"
#include "src/simple/sources/databento.hpp"
#include "src/simple/sources/ibkr.hpp"

using namespace mango::simple;

// Test yfinance converter
TEST(ConverterTest, YFinancePrice) {
    auto p = Converter<YFinanceSource>::to_price(100.50);
    EXPECT_DOUBLE_EQ(p.to_double(), 100.50);
}

TEST(ConverterTest, YFinanceTimestamp) {
    auto ts = Converter<YFinanceSource>::to_timestamp("2024-06-21");
    auto tp = ts.to_timepoint();
    EXPECT_TRUE(tp.has_value());
}

TEST(ConverterTest, YFinanceOptionType) {
    EXPECT_EQ(Converter<YFinanceSource>::to_option_type("call"), mango::OptionType::CALL);
    EXPECT_EQ(Converter<YFinanceSource>::to_option_type("put"), mango::OptionType::PUT);
}

// Test Databento converter
TEST(ConverterTest, DatabentoPriceFixedPoint) {
    int64_t fixed = 100500000000LL;  // 100.50 * 10^9
    auto p = Converter<DatabentSource>::to_price(fixed);
    EXPECT_DOUBLE_EQ(p.to_double(), 100.50);
    EXPECT_TRUE(p.is_fixed_point());
}

TEST(ConverterTest, DabentoTimestamp) {
    uint64_t nanos = 1718928000000000000ULL;
    auto ts = Converter<DatabentSource>::to_timestamp(nanos);
    auto tp = ts.to_timepoint();
    EXPECT_TRUE(tp.has_value());
}

TEST(ConverterTest, DabentoOptionType) {
    EXPECT_EQ(Converter<DatabentSource>::to_option_type('C'), mango::OptionType::CALL);
    EXPECT_EQ(Converter<DatabentSource>::to_option_type('P'), mango::OptionType::PUT);
}

// Test IBKR converter
TEST(ConverterTest, IBKRPrice) {
    auto p = Converter<IBKRSource>::to_price(100.50);
    EXPECT_DOUBLE_EQ(p.to_double(), 100.50);
}

TEST(ConverterTest, IBKRTimestamp) {
    auto ts = Converter<IBKRSource>::to_timestamp("20240621");
    auto tp = ts.to_timepoint();
    EXPECT_TRUE(tp.has_value());
}

TEST(ConverterTest, IBKROptionType) {
    EXPECT_EQ(Converter<IBKRSource>::to_option_type("C"), mango::OptionType::CALL);
    EXPECT_EQ(Converter<IBKRSource>::to_option_type("P"), mango::OptionType::PUT);
    EXPECT_EQ(Converter<IBKRSource>::to_option_type("CALL"), mango::OptionType::CALL);
}
