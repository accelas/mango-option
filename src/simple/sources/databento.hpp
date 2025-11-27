/**
 * @file databento.hpp
 * @brief Converter for Databento data format
 */

#pragma once

#include "src/simple/converter.hpp"

namespace mango::simple {

template<>
struct Converter<DatabentSource> {
    static constexpr double PRICE_SCALE = 1e-9;

    static Price to_price(int64_t v) {
        return Price{v, PriceFormat::FixedPoint9};
    }

    static Timestamp to_timestamp(uint64_t nanos) {
        return Timestamp{nanos};
    }

    static mango::OptionType to_option_type(char c) {
        if (c == 'C') return mango::OptionType::CALL;
        if (c == 'P') return mango::OptionType::PUT;
        throw ConversionError(std::string("Invalid option type: ") + c);
    }

    /// Databento raw option message
    struct RawOption {
        uint64_t ts_event;
        int64_t price;
        int64_t bid_px;
        int64_t ask_px;
        int64_t strike_price;
        char option_type;
    };

    static OptionLeg to_leg(const RawOption& src) {
        OptionLeg leg;
        leg.strike = to_price(src.strike_price);
        leg.bid = to_price(src.bid_px);
        leg.ask = to_price(src.ask_px);
        leg.last = to_price(src.price);
        return leg;
    }
};

}  // namespace mango::simple
