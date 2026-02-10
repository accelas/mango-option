// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/surface_concepts.hpp"
#include <algorithm>

namespace mango {

/// Composes interpolant + coordinate transform + EEP strategy into a
/// complete price surface with price() and vega() methods.
///
/// Replaces EEPPriceTableInner (with AnalyticalEEP) and PriceTableInner
/// (with IdentityEEP). Any combination of interpolant, transform, and
/// EEP strategy works without code duplication.
template <typename Interp, CoordinateTransform Xform, EEPStrategy EEP>
    requires SurfaceInterpolant<Interp, Xform::kDim>
class EEPSurfaceAdapter {
public:
    EEPSurfaceAdapter(Interp interp, Xform xform, EEP eep, double K_ref)
        : interp_(std::move(interp))
        , xform_(std::move(xform))
        , eep_(std::move(eep))
        , K_ref_(K_ref)
    {}

    [[nodiscard]] double price(double spot, double strike,
                                double tau, double sigma, double rate) const {
        auto coords = xform_.to_coords(spot, strike, tau, sigma, rate);
        double raw = interp_.eval(coords);
        double eep_val = std::max(0.0, raw);
        return eep_val * eep_.scale(strike, K_ref_)
             + eep_.european_price(spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] double vega(double spot, double strike,
                               double tau, double sigma, double rate) const {
        auto coords = xform_.to_coords(spot, strike, tau, sigma, rate);
        double raw = interp_.eval(coords);
        if (raw <= 0.0) {
            // EEP clamped to zero — EEP vega is zero (consistent with price clamp)
            return eep_.european_vega(spot, strike, tau, sigma, rate);
        }
        auto w = xform_.vega_weights(spot, strike, tau, sigma, rate);
        double eep_vega = 0.0;
        for (size_t i = 0; i < Xform::kDim; ++i) {
            eep_vega += w[i] * interp_.partial(i, coords);
        }
        return eep_vega * eep_.scale(strike, K_ref_)
             + eep_.european_vega(spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] const Interp& interpolant() const noexcept { return interp_; }
    [[nodiscard]] double K_ref() const noexcept { return K_ref_; }

private:
    Interp interp_;
    Xform xform_;
    EEP eep_;
    double K_ref_;
};

}  // namespace mango
