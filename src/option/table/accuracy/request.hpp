// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>
#include <cstdint>
#include <optional>

namespace mango {

inline constexpr double kDefaultMaxPriceError = 0.01;
inline constexpr double kDefaultMaxIvError = 2e-5;  // 0.2 absolute-IV bp.

enum class AccuracyPolicy : uint8_t { Strict, BestEffort };

/// One accuracy request shared by manual and adaptive factory admission.
/// An absent IV target explicitly requests price-only acceptance.
struct AccuracyRequest {
    double max_price_error = kDefaultMaxPriceError;
    std::optional<double> max_iv_error = kDefaultMaxIvError;
    AccuracyPolicy policy = AccuracyPolicy::Strict;

    [[nodiscard]] bool valid() const noexcept {
        return std::isfinite(max_price_error) && max_price_error > 0.0 &&
            (!max_iv_error || (std::isfinite(*max_iv_error) && *max_iv_error > 0.0)) &&
            (policy == AccuracyPolicy::Strict || policy == AccuracyPolicy::BestEffort);
    }
};

} // namespace mango
