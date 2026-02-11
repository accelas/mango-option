// SPDX-License-Identifier: MIT
#pragma once

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
/// Vega weights encode the chain rule ∂/∂σ through dimensionless coords:
///   ∂τ'/∂σ = στ,   ∂(ln κ)/∂σ = -2/σ
struct DimensionlessTransform3D {
    static constexpr size_t kDim = 3;

    [[nodiscard]] std::array<double, 3> to_coords(
        double spot, double strike, double tau, double sigma, double rate) const noexcept {
        return {std::log(spot / strike),
                sigma * sigma * tau / 2.0,
                std::log(2.0 * rate / (sigma * sigma))};
    }

    [[nodiscard]] std::array<double, 3> vega_weights(
        double /*spot*/, double /*strike*/, double tau,
        double sigma, double /*rate*/) const noexcept {
        return {0.0, sigma * tau, -2.0 / sigma};
    }
};

}  // namespace mango
