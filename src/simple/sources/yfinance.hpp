// SPDX-License-Identifier: MIT
/**
 * @file yfinance.hpp
 * @brief Converter for yfinance data format
 */

#pragma once

#include "mango/simple/converter.hpp"

namespace mango::simple {

template<>
struct Converter<YFinanceSource> {
    static Price to_price(double v) {
        return Price{v};
    }

    static Timestamp to_timestamp(const std::string& s) {
        return Timestamp{s, TimestampFormat::ISO};
    }

    static mango::OptionType to_option_type(const std::string& s) {
        if (s == "call" || s == "Call" || s == "CALL") {
            return mango::OptionType::CALL;
        }
        if (s == "put" || s == "Put" || s == "PUT") {
            return mango::OptionType::PUT;
        }
        throw ConversionError("Invalid option type: " + s);
    }

    /// Convert yfinance option data to OptionLeg
    struct RawOption {
        std::string expiry;
        double strike;
        double bid;
        double ask;
        double lastPrice;
        int64_t volume;
        int64_t openInterest;
        double impliedVolatility;
    };

    static OptionLeg to_leg(const RawOption& src) {
        OptionLeg leg;
        leg.strike = to_price(src.strike);
        leg.bid = to_price(src.bid);
        leg.ask = to_price(src.ask);
        leg.last = to_price(src.lastPrice);
        leg.volume = src.volume;
        leg.open_interest = src.openInterest;
        leg.source_iv = src.impliedVolatility;
        return leg;
    }
};

}  // namespace mango::simple
