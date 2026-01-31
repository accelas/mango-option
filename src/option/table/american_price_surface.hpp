// SPDX-License-Identifier: MIT
#pragma once

#include "src/option/table/price_table_surface.hpp"
#include "src/option/table/price_table_metadata.hpp"
#include "src/option/option_spec.hpp"
#include "src/support/error_types.hpp"
#include <memory>
#include <expected>

namespace mango {

/// Wrapper around PriceTableSurface<4> that reconstructs full American option
/// prices from Early Exercise Premium (EEP) data.
///
/// Reconstruction formula:
///   P_American = (K/K_ref) * EEP_spline(m, tau, sigma, r) + P_European(S, K, tau, sigma, r, q)
///
/// Thread-safe after construction.
class AmericanPriceSurface {
public:
    /// Create from EEP surface. Validates metadata.content == EarlyExercisePremium.
    static std::expected<AmericanPriceSurface, ValidationError> create(
        std::shared_ptr<const PriceTableSurface<4>> eep_surface,
        OptionType type);

    /// P = (K/K_ref) * E(m,tau,sigma,r) + P_Eu(S,K,tau,sigma,r,q)
    double price(double spot, double strike, double tau,
                 double sigma, double rate) const;

    /// Delta = (1/K_ref) * partial_0(E) + Delta_Eu
    double delta(double spot, double strike, double tau,
                 double sigma, double rate) const;

    /// Gamma via finite difference of delta (BSplineND lacks second derivatives)
    double gamma(double spot, double strike, double tau,
                 double sigma, double rate) const;

    /// Vega = (K/K_ref) * partial_2(E) + Vega_Eu
    double vega(double spot, double strike, double tau,
                double sigma, double rate) const;

    /// dP/d(tau) = (K/K_ref) * partial_1(E) - Theta_Eu (sign: EU theta is dV/dt)
    double theta(double spot, double strike, double tau,
                 double sigma, double rate) const;

    /// Access underlying EEP surface
    const PriceTableSurface<4>& eep_surface() const;
    const PriceTableMetadata& metadata() const;

private:
    AmericanPriceSurface(std::shared_ptr<const PriceTableSurface<4>> surface,
                         OptionType type, double K_ref, double dividend_yield);

    std::shared_ptr<const PriceTableSurface<4>> surface_;
    OptionType type_;
    double K_ref_;
    double dividend_yield_;
};

}  // namespace mango
