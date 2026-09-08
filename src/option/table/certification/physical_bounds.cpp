// SPDX-License-Identifier: MIT
#include "mango/option/table/certification/physical_bounds.hpp"
#include <cmath>
#include <limits>
namespace mango::detail::certification {
namespace {
PriceBounds invalid() {
    const Interval nan(std::numeric_limits<double>::quiet_NaN());
    return {nan, nan};
}
Interval cdf(const Interval &x) {
    return erfc((Interval(0) - x) / sqrt(Interval(2))) / Interval(2);
}
} // namespace
PriceBounds european(const QuoteBox &box, OptionType type, double dividend_yield) {
    if (!box.ratio.strictly_positive() || !box.tau.nonnegative() ||
        !box.sigma.strictly_positive() || !box.rate.finite() || !std::isfinite(dividend_yield) ||
        (type != OptionType::PUT && type != OptionType::CALL))
        return invalid();
    const auto intrinsic =
        positive_part(type == OptionType::PUT ? Interval(1) - box.ratio : box.ratio - Interval(1));
    if (box.tau.exact_zero())
        return {intrinsic, Interval(0)};
    const auto forward = box.ratio * exp(Interval(-dividend_yield) * box.tau);
    const auto discount = exp((Interval(0) - box.rate) * box.tau);
    const auto root_tau = sqrt(box.tau);
    const auto gaussian_max = Interval(1) / sqrt(Interval(2) * Interval::pi());
    const auto cap = type == OptionType::PUT ? discount : forward;
    if (!box.tau.strictly_positive()) {
        // Include the continuous expiry limit while the admitted domain is
        // positive tau. No division by a box containing zero is performed.
        const auto lower =
            positive_part(type == OptionType::PUT ? discount - forward : forward - discount);
        return {hull(lower, cap), hull(Interval(0), forward * root_tau * gaussian_max)};
    }
    const auto sigma_tau = box.sigma * root_tau;
    const auto d1 =
        (log(box.ratio) +
         (box.rate - Interval(dividend_yield) + square(box.sigma) / Interval(2)) * box.tau) /
        sigma_tau;
    const auto d2 = d1 - sigma_tau;
    const auto value = type == OptionType::PUT
                           ? discount * cdf(Interval(0) - d2) - forward * cdf(Interval(0) - d1)
                           : forward * cdf(d1) - discount * cdf(d2);
    const auto vega = forward * root_tau * gaussian_max * exp(Interval(-.5) * square(d1));
    // Mathematical European no-arbitrage bounds tighten dependency loss in
    // the two CDF terms. An empty intersection stays invalid, never zero.
    return {intersection(value, hull(Interval(0), cap)), vega};
}
PriceBounds zero_floor(const PriceBounds &raw) {
    if (!raw.value.finite())
        return invalid();
    if (raw.value.nonpositive())
        return {};
    const auto derivative =
        raw.value.strictly_positive() ? raw.sigma_partial : hull(Interval(0), raw.sigma_partial);
    return {positive_part(raw.value), derivative};
}
PriceBounds continuous_eep(const PriceBounds &raw, const QuoteBox &box, double reference_strike,
                           OptionType type, double dividend_yield) {
    if (!std::isfinite(reference_strike) || reference_strike <= 0)
        return invalid();
    const auto premium = zero_floor(raw);
    const auto eu = european(box, type, dividend_yield);
    const Interval scale(reference_strike);
    return {premium.value / scale + eu.value, premium.sigma_partial / scale + eu.sigma_partial};
}
PriceBounds weighted_sum(std::span<const PriceBounds> pieces, std::span<const Interval> weights) {
    if (pieces.empty() || pieces.size() != weights.size())
        return invalid();
    PriceBounds result;
    for (std::size_t i = 0; i < pieces.size(); ++i) {
        if (!weights[i].nonnegative())
            return invalid();
        // An exact reference endpoint does not evaluate an inactive neighbor.
        if (weights[i].exact_zero())
            continue;
        result.value = result.value + weights[i] * pieces[i].value;
        result.sigma_partial = result.sigma_partial + weights[i] * pieces[i].sigma_partial;
    }
    return result;
}
PriceBounds intrinsic_floor(const PriceBounds &continuation, const Interval &ratio,
                            OptionType type) {
    if (!continuation.value.finite() || !ratio.strictly_positive() ||
        (type != OptionType::PUT && type != OptionType::CALL))
        return invalid();
    const auto intrinsic =
        positive_part(type == OptionType::PUT ? Interval(1) - ratio : ratio - Interval(1));
    const auto excess = continuation.value - intrinsic;
    if (excess.nonpositive())
        return {intrinsic, Interval(0)};
    if (excess.strictly_positive())
        return continuation;
    // A crossing box includes a sigma-flat branch; it is not a rigorous
    // negative-vega witness even when the continuation branch decreases.
    return {intrinsic + positive_part(excess), hull(Interval(0), continuation.sigma_partial)};
}
PriceBounds dimensionless_eep(const Interval &raw_value, const Interval &partial_time,
                              const Interval &partial_log_kappa, const QuoteBox &box,
                              double reference_strike, OptionType type) {
    if (!box.rate.strictly_positive())
        return invalid();
    const auto sigma_derivative =
        box.sigma * box.tau * partial_time - Interval(2) / box.sigma * partial_log_kappa;
    return continuous_eep({raw_value, sigma_derivative}, box, reference_strike, type, 0);
}
} // namespace mango::detail::certification
