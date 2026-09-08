// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/dividend_utils.hpp"
#include <cmath>
#include <vector>

namespace mango {

/// Numerical identity of a fixed-expiry cash-dividend surface. Dates are
/// offsets from its numerical anchor, not wall-clock timestamps.
struct FixedExpiryMetadata {
    double reference_maturity = 0.0;
    std::vector<Dividend> discrete_dividends;

    [[nodiscard]] bool valid(double published_tau_max) const noexcept {
        if (!std::isfinite(reference_maturity) || reference_maturity <= 0.0 ||
            !std::isfinite(published_tau_max) || published_tau_max > reference_maturity) {
            return false;
        }
        double previous = 0.0;
        for (const auto& dividend : discrete_dividends) {
            if (!std::isfinite(dividend.calendar_time) || !std::isfinite(dividend.amount) ||
                dividend.calendar_time <= previous || dividend.calendar_time >= reference_maturity ||
                dividend.amount <= 0.0) return false;
            previous = dividend.calendar_time;
        }
        return true;
    }
};

/// Use the same canonical schedule as direct fixed-expiry pricing.
inline FixedExpiryMetadata make_fixed_expiry_metadata(
    double reference_maturity, const std::vector<Dividend>& dividends)
{
    return {reference_maturity, filter_and_merge_dividends(dividends, reference_maturity)};
}

}  // namespace mango
