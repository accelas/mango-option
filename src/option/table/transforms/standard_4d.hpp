// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cmath>

namespace mango {

/// Identity coordinate transform for 4D (x, tau, sigma, rate) surfaces.
/// Direct sigma axis -- vega is a single partial derivative.
struct StandardTransform4D {
    static constexpr size_t kDim = 4;

    [[nodiscard]] std::array<double, 4> to_coords(
        double spot, double strike, double tau, double sigma, double rate) const noexcept {
        return {std::log(spot / strike), tau, sigma, rate};
    }

    [[nodiscard]] std::array<double, 4> vega_weights(
        double /*spot*/, double /*strike*/, double /*tau*/,
        double /*sigma*/, double /*rate*/) const noexcept {
        return {0.0, 0.0, 1.0, 0.0};
    }
};

}  // namespace mango
