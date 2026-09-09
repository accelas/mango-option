// SPDX-License-Identifier: MIT
#pragma once
#include <cmath>

namespace mango::detail {

/// Compute value * numerator / denominator without losing a representable
/// result to the intermediate currency scale. Ordinary normalized values use
/// two arithmetic operations; extreme quotients use bounded mantissas.
inline double scale_quote(double value, double numerator, double denominator) noexcept {
    const double normalized = value / denominator;
    if (std::isnormal(normalized) || value == 0.0 || !std::isfinite(value) ||
        !std::isfinite(numerator) || !std::isfinite(denominator) || denominator == 0.0)
        return normalized * numerator;
    int value_exp = 0, numerator_exp = 0, denominator_exp = 0;
    const double value_mantissa = std::frexp(value, &value_exp);
    const double numerator_mantissa = std::frexp(numerator, &numerator_exp);
    const double denominator_mantissa = std::frexp(denominator, &denominator_exp);
    return std::scalbn((value_mantissa / denominator_mantissa) * numerator_mantissa,
                       value_exp + numerator_exp - denominator_exp);
}

} // namespace mango::detail
