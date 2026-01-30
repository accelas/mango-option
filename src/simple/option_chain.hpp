// SPDX-License-Identifier: MIT
/**
 * @file option_chain.hpp
 * @brief Option chain and market context types
 */

#pragma once

#include "src/simple/option_types.hpp"
#include "src/math/yield_curve.hpp"
#include "src/option/option_spec.hpp"  // For RateSpec
#include <optional>
#include <string>
#include <vector>

namespace mango::simple {

/// Dividend specification
///
/// Either a continuous yield (for indices) or discrete dividends (for stocks).
struct Dividend {
    Timestamp ex_date;
    Price amount;
};

using DividendSpec = std::variant<double, std::vector<Dividend>>;

/// Full option chain for an underlying
struct OptionChain {
    std::string symbol;
    std::optional<Price> spot;
    std::optional<Timestamp> quote_time;
    std::optional<DividendSpec> dividends;
    std::optional<std::string> exchange;

    std::vector<ExpirySlice> expiries;

    /// Get all expiry timestamps
    [[nodiscard]] std::vector<Timestamp> expiry_dates() const {
        std::vector<Timestamp> dates;
        dates.reserve(expiries.size());
        for (const auto& slice : expiries) {
            dates.push_back(slice.expiry);
        }
        return dates;
    }
};

/// Market context for IV computation
///
/// Contains rate, valuation time, and optional dividend override.
struct MarketContext {
    std::optional<mango::RateSpec> rate;
    std::optional<Timestamp> valuation_time;
    std::optional<DividendSpec> dividends;  // Override chain's dividends
};

}  // namespace mango::simple
