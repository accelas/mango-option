// SPDX-License-Identifier: MIT
/**
 * @file ibkr.hpp
 * @brief Converter for Interactive Brokers data format
 */

#pragma once

#include "mango/simple/converter.hpp"

namespace mango::simple {

template<>
struct Converter<IBKRSource> {
    static Price to_price(double v) {
        return Price{v};
    }

    static Timestamp to_timestamp(const std::string& s) {
        // IBKR uses compact format: "20240621"
        return Timestamp{s, TimestampFormat::Compact};
    }

    static mango::OptionType to_option_type(const std::string& s) {
        if (s == "C" || s == "CALL" || s == "Call") {
            return mango::OptionType::CALL;
        }
        if (s == "P" || s == "PUT" || s == "Put") {
            return mango::OptionType::PUT;
        }
        throw ConversionError("Invalid option type: " + s);
    }

    /// IBKR raw option data
    struct RawOption {
        std::string expiry;
        double strike;
        double bid;
        double ask;
        double last;
        std::string right;
        int64_t volume;
    };

    static OptionLeg to_leg(const RawOption& src) {
        OptionLeg leg;
        leg.strike = to_price(src.strike);
        leg.bid = to_price(src.bid);
        leg.ask = to_price(src.ask);
        leg.last = to_price(src.last);
        leg.volume = src.volume;
        return leg;
    }
};

}  // namespace mango::simple
