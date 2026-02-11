// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/surface_concepts.hpp"

namespace mango {

/// Adds European price/vega on top of any leaf with price()/vega().
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

    [[nodiscard]] const Leaf& leaf() const noexcept { return leaf_; }
    [[nodiscard]] auto& interpolant() const noexcept { return leaf_.interpolant(); }
    [[nodiscard]] double K_ref() const noexcept { return leaf_.K_ref(); }

private:
    Leaf leaf_;
    EEP eep_;
};

}  // namespace mango
