// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/greek_types.hpp"
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

    [[nodiscard]] std::array<double, 4> greek_weights(
        Greek greek, double spot, double /*strike*/, double /*tau*/,
        double /*sigma*/, double /*rate*/) const noexcept {
        switch (greek) {
            case Greek::Delta: return {1.0 / spot, 0.0, 0.0, 0.0};
            case Greek::Vega:  return {0.0, 0.0, 1.0, 0.0};
            case Greek::Theta: return {0.0, -1.0, 0.0, 0.0};
            case Greek::Rho:   return {0.0, 0.0, 0.0, 1.0};
        }
        __builtin_unreachable();
    }
};

}  // namespace mango
