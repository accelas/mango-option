// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/greek_types.hpp"
#include "mango/option/table/surface_concepts.hpp"
#include "mango/option/option_spec.hpp"
#include <expected>

namespace mango {

/// Adds European price/vega/Greeks on top of any leaf with price()/vega().
/// Used for EEP decomposition: American = leaf_price + European.
template <typename Leaf, EEPStrategy EEP>
class EEPLayer {
public:
    EEPLayer(Leaf leaf, EEP eep)
        : leaf_(std::move(leaf))
        , eep_(std::move(eep))
    {}

    [[nodiscard]] double price(double spot, double strike,
                                double tau, double sigma, double rate) const {
        return leaf_.price(spot, strike, tau, sigma, rate)
             + eep_.european_price(spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] double vega(double spot, double strike,
                               double tau, double sigma, double rate) const {
        return leaf_.vega(spot, strike, tau, sigma, rate)
             + eep_.european_vega(spot, strike, tau, sigma, rate);
    }

    /// First-order Greek with EEP decomposition.
    /// When leaf EEP is zero (deep OTM), returns European Greek only.
    [[nodiscard]] std::expected<double, GreekError>
    greek(Greek g, const PricingParams& params) const {
        double spot = params.spot, strike = params.strike;
        double tau = params.maturity, sigma = params.volatility;
        double rate = get_zero_rate(params.rate, params.maturity);

        // Early guard: if EEP surface reads zero, return European only
        double raw = leaf_.raw_value(spot, strike, tau, sigma, rate);
        double european = [&] {
            switch (g) {
                case Greek::Delta: return eep_.european_delta(spot, strike, tau, sigma, rate);
                case Greek::Vega:  return eep_.european_vega(spot, strike, tau, sigma, rate);
                case Greek::Theta: return eep_.european_theta(spot, strike, tau, sigma, rate);
                case Greek::Rho:   return eep_.european_rho(spot, strike, tau, sigma, rate);
            }
            __builtin_unreachable();
        }();

        if (raw <= 0.0) return european;

        auto leaf_greek = leaf_.greek(g, params);
        if (!leaf_greek.has_value()) return std::unexpected(leaf_greek.error());
        return *leaf_greek + european;
    }

    /// Gamma with EEP decomposition.
    [[nodiscard]] std::expected<double, GreekError>
    gamma(const PricingParams& params) const {
        double spot = params.spot, strike = params.strike;
        double tau = params.maturity, sigma = params.volatility;
        double rate = get_zero_rate(params.rate, params.maturity);

        double raw = leaf_.raw_value(spot, strike, tau, sigma, rate);
        double european_gamma = eep_.european_gamma(spot, strike, tau, sigma, rate);

        if (raw <= 0.0) return european_gamma;

        auto leaf_gamma = leaf_.gamma(params);
        if (!leaf_gamma.has_value()) return std::unexpected(leaf_gamma.error());
        return *leaf_gamma + european_gamma;
    }

    [[nodiscard]] const Leaf& leaf() const noexcept { return leaf_; }
    [[nodiscard]] auto& interpolant() const noexcept { return leaf_.interpolant(); }
    [[nodiscard]] double K_ref() const noexcept { return leaf_.K_ref(); }

private:
    Leaf leaf_;
    EEP eep_;
};

}  // namespace mango
