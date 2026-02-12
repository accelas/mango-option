// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/greek_types.hpp"
#include "mango/option/table/surface_concepts.hpp"
#include "mango/option/option_spec.hpp"
#include <algorithm>
#include <cmath>
#include <expected>

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
        auto w = xform_.greek_weights(Greek::Vega, spot, strike, tau, sigma, rate);
        double v = 0.0;
        for (size_t i = 0; i < Xform::kDim; ++i)
            if (w[i] != 0.0)
                v += w[i] * interp_.partial(i, coords);
        return v * strike / K_ref_;
    }

    /// Compute a first-order Greek (delta, vega, theta, rho).
    /// Returns the leaf contribution only (no European add-back).
    [[nodiscard]] std::expected<double, GreekError>
    greek(Greek g, const PricingParams& params) const {
        double spot = params.spot, strike = params.strike;
        double tau = params.maturity, sigma = params.volatility;
        double rate = get_zero_rate(params.rate, params.maturity);

        auto coords = xform_.to_coords(spot, strike, tau, sigma, rate);
        double raw = interp_.eval(coords);
        if (raw <= 0.0) return 0.0;

        auto w = xform_.greek_weights(g, spot, strike, tau, sigma, rate);
        double result = 0.0;
        for (size_t i = 0; i < Xform::kDim; ++i)
            if (w[i] != 0.0)
                result += w[i] * interp_.partial(i, coords);
        return result * strike / K_ref_;
    }

    /// Compute gamma = d^2V/dS^2.
    /// Uses analytical second partial if available, FD fallback otherwise.
    [[nodiscard]] std::expected<double, GreekError>
    gamma(const PricingParams& params) const {
        double spot = params.spot, strike = params.strike;
        double tau = params.maturity, sigma = params.volatility;
        double rate = get_zero_rate(params.rate, params.maturity);

        auto coords = xform_.to_coords(spot, strike, tau, sigma, rate);
        double raw = interp_.eval(coords);
        if (raw <= 0.0) return 0.0;

        double df_dx = interp_.partial(0, coords);
        double d2f_dx2 = compute_second_partial_x(coords);

        // d^2V/dS^2 = (d^2f/dx^2 - df/dx) / S^2 * strike/K_ref
        return (d2f_dx2 - df_dx) / (spot * spot) * strike / K_ref_;
    }

    /// Expose raw interpolant value for EEP layer guard.
    [[nodiscard]] double raw_value(double spot, double strike,
                                    double tau, double sigma, double rate) const {
        auto coords = xform_.to_coords(spot, strike, tau, sigma, rate);
        return interp_.eval(coords);
    }

    [[nodiscard]] const Interp& interpolant() const noexcept { return interp_; }
    [[nodiscard]] double K_ref() const noexcept { return K_ref_; }

private:
    /// Compute d^2f/dx^2 along moneyness axis.
    /// Analytical for interpolants with eval_second_partial, FD otherwise.
    [[nodiscard]] double compute_second_partial_x(
        const std::array<double, Xform::kDim>& coords) const {
        if constexpr (requires { interp_.eval_second_partial(size_t{0}, coords); }) {
            return interp_.eval_second_partial(0, coords);
        } else {
            // Central FD fallback
            double h = 1e-4;  // Fixed step in log-moneyness
            auto coords_up = coords;
            auto coords_dn = coords;
            coords_up[0] = coords[0] + h;
            coords_dn[0] = coords[0] - h;
            double f_up = interp_.eval(coords_up);
            double f_dn = interp_.eval(coords_dn);
            double f_mid = interp_.eval(coords);
            return (f_up - 2.0 * f_mid + f_dn) / (h * h);
        }
    }

    Interp interp_;
    Xform xform_;
    double K_ref_;
};

}  // namespace mango
