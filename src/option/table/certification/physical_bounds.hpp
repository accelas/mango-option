// SPDX-License-Identifier: MIT
#pragma once
#include "mango/math/proof/interval.hpp"
#include "mango/option/option_spec.hpp"
#include <span>

namespace mango::detail::certification {
using proof::Interval;

/// Physical parameters after dividing quote price by positive query strike.
struct QuoteBox {
    Interval ratio{1}, tau{1}, sigma{.2}, rate{0};
};
struct PriceBounds {
    Interval value{0};
    Interval sigma_partial{0};
    [[nodiscard]] bool finite() const { return value.finite() && sigma_partial.finite(); }
};

/// These internal enclosures compose the represented financial expression;
/// none of them creates or publishes a price-table certificate by itself.
PriceBounds european(const QuoteBox &box, OptionType type, double dividend_yield);
PriceBounds zero_floor(const PriceBounds &raw);
PriceBounds continuous_eep(const PriceBounds &raw_dollar_eep, const QuoteBox &box,
                           double reference_strike, OptionType type, double dividend_yield);
PriceBounds weighted_sum(std::span<const PriceBounds> pieces, std::span<const Interval> weights);
PriceBounds intrinsic_floor(const PriceBounds &continuation, const Interval &ratio,
                            OptionType type);
PriceBounds dimensionless_eep(const Interval &raw_value, const Interval &partial_time,
                              const Interval &partial_log_kappa, const QuoteBox &box,
                              double reference_strike, OptionType type);
} // namespace mango::detail::certification
