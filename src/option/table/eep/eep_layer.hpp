// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/greek_types.hpp"
#include "mango/option/table/surface_concepts.hpp"
#include "mango/option/option_spec.hpp"
#include <expected>

namespace mango {

/// Adds European price/vega/Greeks on top of any leaf with price()/vega().
/// Used for EEP decomposition: American = leaf_price + European.
///
/// Leaf contract for greek()/gamma(): when the leaf's raw interpolant value
/// is <= 0 (deep OTM, zero early-exercise premium) the leaf must return
/// exactly 0.0 — not an error — without derivative work, so this layer can
/// call it unconditionally and the sum degenerates to the European Greek.
/// TransformLeaf satisfies this (pinned by TransformLeafZeroContractTest).
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
    /// Deep OTM the leaf's Greek is exactly 0.0 (see leaf contract in the class comment above),
    /// so the result degenerates to the European Greek.
    [[nodiscard]] std::expected<double, GreekError>
    greek(Greek g, const PricingParams& params) const {
        double spot = params.spot, strike = params.strike;
        double tau = params.maturity, sigma = params.volatility;
        double rate = get_zero_rate(params.rate, params.maturity);

        double european = [&] {
            switch (g) {
                case Greek::Delta: return eep_.european_delta(spot, strike, tau, sigma, rate);
                case Greek::Vega:  return eep_.european_vega(spot, strike, tau, sigma, rate);
                case Greek::Theta: return eep_.european_theta(spot, strike, tau, sigma, rate);
                case Greek::Rho:   return eep_.european_rho(spot, strike, tau, sigma, rate);
            }
            __builtin_unreachable();
        }();

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

        double european_gamma = eep_.european_gamma(spot, strike, tau, sigma, rate);

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
