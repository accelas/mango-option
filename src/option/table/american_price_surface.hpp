// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/price_table_surface.hpp"
#include "mango/option/table/price_table_metadata.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/support/error_types.hpp"
#include <memory>
#include <expected>

namespace mango {

/// Wrapper around PriceTableSurface<4> that reconstructs full American option
/// prices from Early Exercise Premium (EEP) data.
///
/// Reconstruction formula:
///   P_American = (K/K_ref) * EEP_spline(x, tau, sigma, r) + P_European(S, K, tau, sigma, r, q)
///   where x = ln(S/K)
///
/// Thread-safe after construction.
class AmericanPriceSurface {
public:
    /// Create from price surface. Accepts EarlyExercisePremium or NormalizedPrice content.
    static std::expected<AmericanPriceSurface, ValidationError> create(
        std::shared_ptr<const PriceTableSurface<4>> eep_surface,
        OptionType type);

    /// P = (K/K_ref) * E(x,tau,sigma,r) + P_Eu(S,K,tau,sigma,r,q)  where x = ln(S/K)
    double price(double spot, double strike, double tau,
                 double sigma, double rate) const;

    /// Delta = (K/(K_ref·S)) * ∂E/∂x + Delta_Eu  where x = ln(S/K)
    double delta(double spot, double strike, double tau,
                 double sigma, double rate) const;

    /// Gamma = (K/K_ref) * (∂²E/∂x² - ∂E/∂x) / S² + Gamma_Eu
    double gamma(double spot, double strike, double tau,
                 double sigma, double rate) const;

    /// Vega = (K/K_ref) * partial_2(E) + Vega_Eu
    double vega(double spot, double strike, double tau,
                double sigma, double rate) const;

    /// Theta = dV/dt (calendar time, negative for decay)
    /// = -(K/K_ref) * ∂E/∂τ + Theta_Eu
    double theta(double spot, double strike, double tau,
                 double sigma, double rate) const;


    [[nodiscard]] OptionType option_type() const noexcept;
    [[nodiscard]] double dividend_yield() const noexcept;
    const PriceTableMetadata& metadata() const;

    /// Bounds accessors for PriceSurface concept
    [[nodiscard]] double m_min() const noexcept;
    [[nodiscard]] double m_max() const noexcept;
    [[nodiscard]] double tau_min() const noexcept;
    [[nodiscard]] double tau_max() const noexcept;
    [[nodiscard]] double sigma_min() const noexcept;
    [[nodiscard]] double sigma_max() const noexcept;
    [[nodiscard]] double rate_min() const noexcept;
    [[nodiscard]] double rate_max() const noexcept;

private:
    AmericanPriceSurface(std::shared_ptr<const PriceTableSurface<4>> surface,
                         OptionType type, double K_ref, double dividend_yield);

    std::shared_ptr<const PriceTableSurface<4>> surface_;
    OptionType type_;
    double K_ref_;
    double dividend_yield_;
};

}  // namespace mango
