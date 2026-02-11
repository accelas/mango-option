// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cmath>

namespace mango {

/// Wraps any inner surface, delegating price() but computing vega via
/// central finite differences on price().  American options have no
/// closed-form vega, so FD on the interpolated price is the correct
/// approach for Newton-based IV solving.
template <typename Inner>
class FDVegaAdapter {
public:
    explicit FDVegaAdapter(Inner inner) : inner_(std::move(inner)) {}

    [[nodiscard]] double price(double spot, double strike,
                                double tau, double sigma, double rate) const {
        return inner_.price(spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] double vega(double spot, double strike,
                               double tau, double sigma, double rate) const {
        double eps = std::max(1e-4, 0.01 * sigma);
        double sigma_up = sigma + eps;
        double sigma_dn = std::max(1e-4, sigma - eps);
        double eff_eps = (sigma_up - sigma_dn) / 2.0;
        return (inner_.price(spot, strike, tau, sigma_up, rate) -
                inner_.price(spot, strike, tau, sigma_dn, rate)) / (2.0 * eff_eps);
    }

private:
    Inner inner_;
};

}  // namespace mango
