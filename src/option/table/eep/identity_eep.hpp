// SPDX-License-Identifier: MIT
#pragma once

namespace mango {

/// No EEP decomposition. Surface stores V/K_ref directly.
/// european_price/vega return 0, scale returns K/K_ref.
/// Used for segmented dividend segments (NormalizedPrice content).
struct IdentityEEP {
    [[nodiscard]] double european_price(
        double, double, double, double, double) const noexcept { return 0.0; }

    [[nodiscard]] double european_vega(
        double, double, double, double, double) const noexcept { return 0.0; }

    [[nodiscard]] double scale(double strike, double K_ref) const noexcept {
        return strike / K_ref;
    }
};

}  // namespace mango
