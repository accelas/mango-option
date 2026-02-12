// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/greek_types.hpp"
#include <array>
#include <cmath>

namespace mango {

/// Dimensionless coordinate transform for 3D (x, τ', ln κ) surfaces.
///
/// Maps physical parameters:
///   x   = ln(S/K)             — log-moneyness
///   τ'  = σ²τ/2              — dimensionless time
///   ln κ = ln(2r/σ²)          — dimensionless rate
///
/// Greek weights encode chain rule ∂/∂(physical) through dimensionless coords:
///   Delta: ∂x/∂S = 1/S
///   Vega:  ∂τ'/∂σ = στ,   ∂(ln κ)/∂σ = -2/σ
///   Theta: ∂τ'/∂τ = σ²/2
///   Rho:   ∂(ln κ)/∂r = 1/r
struct DimensionlessTransform3D {
    static constexpr size_t kDim = 3;

    [[nodiscard]] std::array<double, 3> to_coords(
        double spot, double strike, double tau, double sigma, double rate) const noexcept {
        return {std::log(spot / strike),
                sigma * sigma * tau / 2.0,
                std::log(2.0 * rate / (sigma * sigma))};
    }

    [[nodiscard]] std::array<double, 3> greek_weights(
        Greek greek, double spot, double /*strike*/, double tau,
        double sigma, double rate) const noexcept {
        switch (greek) {
            case Greek::Delta: return {1.0 / spot, 0.0, 0.0};
            case Greek::Vega:  return {0.0, sigma * tau, -2.0 / sigma};
            case Greek::Theta: return {0.0, sigma * sigma / 2.0, 0.0};
            case Greek::Rho:   return {0.0, 0.0, 1.0 / rate};
        }
        __builtin_unreachable();
    }
};

}  // namespace mango
