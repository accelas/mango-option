// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/surface_concepts.hpp"
#include <algorithm>

namespace mango {

/// Coordinate transform + raw interpolation + K/K_ref scaling.
/// Produces: max(0, interp(coords)) * strike/K_ref.
///
/// Used directly for segmented leaves (no EEP decomposition).
/// Wrapped by EEPLayer for standard leaves (European add-back).
template <typename Interp, CoordinateTransform Xform>
    requires SurfaceInterpolant<Interp, Xform::kDim>
class TransformLeaf {
public:
    TransformLeaf(Interp interp, Xform xform, double K_ref)
        : interp_(std::move(interp))
        , xform_(std::move(xform))
        , K_ref_(K_ref)
    {}

    [[nodiscard]] double price(double spot, double strike,
                                double tau, double sigma, double rate) const {
        auto coords = xform_.to_coords(spot, strike, tau, sigma, rate);
        double raw = interp_.eval(coords);
        return std::max(0.0, raw) * strike / K_ref_;
    }

    [[nodiscard]] double vega(double spot, double strike,
                               double tau, double sigma, double rate) const {
        auto coords = xform_.to_coords(spot, strike, tau, sigma, rate);
        double raw = interp_.eval(coords);
        if (raw <= 0.0) return 0.0;
        auto w = xform_.vega_weights(spot, strike, tau, sigma, rate);
        double v = 0.0;
        for (size_t i = 0; i < Xform::kDim; ++i)
            if (w[i] != 0.0)
                v += w[i] * interp_.partial(i, coords);
        return v * strike / K_ref_;
    }

    [[nodiscard]] const Interp& interpolant() const noexcept { return interp_; }
    [[nodiscard]] double K_ref() const noexcept { return K_ref_; }

private:
    Interp interp_;
    Xform xform_;
    double K_ref_;
};

}  // namespace mango
