// SPDX-License-Identifier: MIT
/**
 * @file chain_builder.hpp
 * @brief Type-safe builder for OptionChain
 */

#pragma once

#include "src/simple/option_chain.hpp"
#include "src/simple/converter.hpp"
#include <map>

namespace mango::simple {

/// Type-safe chain builder
///
/// Uses Converter<Source> to ensure correct types at compile time.
template<typename Source>
class ChainBuilder {
    using Conv = Converter<Source>;

public:
    ChainBuilder& symbol(std::string sym) {
        chain_.symbol = std::move(sym);
        return *this;
    }

    template<typename T>
    ChainBuilder& spot(T&& v) {
        chain_.spot = Conv::to_price(std::forward<T>(v));
        return *this;
    }

    template<typename T>
    ChainBuilder& quote_time(T&& v) {
        chain_.quote_time = Conv::to_timestamp(std::forward<T>(v));
        return *this;
    }

    ChainBuilder& settlement(Settlement s) {
        default_settlement_ = s;
        return *this;
    }

    ChainBuilder& dividend_yield(double yield) {
        chain_.dividends = yield;
        return *this;
    }

    template<typename T, typename RawOpt>
    ChainBuilder& add_call(T&& expiry, const RawOpt& opt) {
        auto ts = Conv::to_timestamp(std::forward<T>(expiry));
        auto& slice = get_or_create_slice(ts);
        auto leg = Conv::to_leg(opt);
        leg.type = OptionType::CALL;
        slice.options.push_back(std::move(leg));
        return *this;
    }

    template<typename T, typename RawOpt>
    ChainBuilder& add_put(T&& expiry, const RawOpt& opt) {
        auto ts = Conv::to_timestamp(std::forward<T>(expiry));
        auto& slice = get_or_create_slice(ts);
        auto leg = Conv::to_leg(opt);
        leg.type = OptionType::PUT;
        slice.options.push_back(std::move(leg));
        return *this;
    }

    OptionChain build() {
        // Apply default settlement to slices without one
        for (auto& slice : chain_.expiries) {
            if (!slice.settlement.has_value() && default_settlement_.has_value()) {
                slice.settlement = default_settlement_;
            }
        }
        return std::move(chain_);
    }

private:
    ExpirySlice& get_or_create_slice(const Timestamp& expiry) {
        // Find existing slice with same expiry (by string comparison for simplicity)
        auto expiry_str = expiry.to_string();
        for (auto& slice : chain_.expiries) {
            if (slice.expiry.to_string() == expiry_str) {
                return slice;
            }
        }
        // Create new slice
        chain_.expiries.push_back(ExpirySlice{expiry});
        return chain_.expiries.back();
    }

    OptionChain chain_;
    std::optional<Settlement> default_settlement_;
};

}  // namespace mango::simple
