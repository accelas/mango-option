/**
 * @file option_types.hpp
 * @brief Option-related types for mango::simple namespace
 */

#pragma once

#include "src/simple/price.hpp"
#include "src/simple/timestamp.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mango::simple {

/// Option settlement type
enum class Settlement {
    AM,   // AM-settled (expires at market open) - SPX, VIX
    PM    // PM-settled (expires at 4:00 PM ET) - SPY, AAPL, etc.
};

/// Single option leg with optional fields
///
/// All price/volume fields are optional since data sources
/// may not provide complete information.
struct OptionLeg {
    Price strike{0.0};

    // Price data - at least one should be present for IV calculation
    std::optional<Price> bid;
    std::optional<Price> ask;
    std::optional<Price> last;

    // Volume data - often missing for illiquid options
    std::optional<int64_t> volume;
    std::optional<int64_t> open_interest;

    // Source-provided values (for comparison/validation)
    std::optional<double> source_iv;
    std::optional<double> source_delta;
    std::optional<double> source_gamma;
    std::optional<double> source_theta;
    std::optional<double> source_vega;

    // Settlement type (may be unknown from some sources)
    std::optional<Settlement> settlement;

    /// Compute mid price if both bid and ask are present
    [[nodiscard]] std::optional<Price> mid() const {
        if (bid && ask) {
            return Price::midpoint(*bid, *ask);
        }
        return std::nullopt;
    }

    /// Best available price for IV calculation
    /// Priority: mid > last
    [[nodiscard]] std::optional<Price> price_for_iv() const {
        if (auto m = mid()) {
            return m;
        }
        return last;
    }
};

/// Options for a single expiry date
struct ExpirySlice {
    Timestamp expiry{""};
    std::optional<Settlement> settlement;
    std::vector<OptionLeg> calls;
    std::vector<OptionLeg> puts;
};

}  // namespace mango::simple
