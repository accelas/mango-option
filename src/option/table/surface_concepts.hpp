// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <concepts>

namespace mango {

/// Raw interpolation engine: eval + partial derivative at N-dim coordinates.
/// Implementations: SharedBSplineInterp<N>, ChebyshevTuckerND<N> (future).
template <typename S, size_t N>
concept SurfaceInterpolant = requires(const S& s, std::array<double, N> coords) {
    { s.eval(coords) } -> std::same_as<double>;
    { s.partial(size_t{}, coords) } -> std::same_as<double>;
};

/// Maps 5-param price query to N-dim interpolation coordinates + vega weights.
/// Implementations: StandardTransform4D, DimensionlessTransform3D (future).
template <typename T>
concept CoordinateTransform = requires(const T& t, double spot, double strike,
                                        double tau, double sigma, double rate) {
    { T::kDim } -> std::convertible_to<size_t>;
    { t.to_coords(spot, strike, tau, sigma, rate) }
        -> std::same_as<std::array<double, T::kDim>>;
    { t.vega_weights(spot, strike, tau, sigma, rate) }
        -> std::same_as<std::array<double, T::kDim>>;
};

/// European price/vega for EEP decomposition: American = leaf + European.
/// Implementation: AnalyticalEEP.
template <typename E>
concept EEPStrategy = requires(const E& e, double spot, double strike,
                                double tau, double sigma, double rate) {
    { e.european_price(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { e.european_vega(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
};

}  // namespace mango
