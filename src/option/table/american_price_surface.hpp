// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/standard_surface.hpp"
#include "mango/option/table/price_table_surface.hpp"
#include "mango/option/table/price_table_metadata.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/support/error_types.hpp"
#include <memory>
#include <expected>

namespace mango {

/// Wrapper around StandardSurfaceWrapper that adds delta/gamma/theta Greeks.
///
/// For EEP content: delegates price()/vega() to StandardSurfaceWrapper
/// (which uses EEPPriceTableInner for EEP reconstruction).
/// For NormalizedPrice content: uses direct B-spline lookup (strike must == K_ref).
///
/// Delta, gamma, and theta require direct B-spline partial derivatives
/// and are only supported for EEP content.
///
/// Thread-safe after construction.
class AmericanPriceSurface {
public:
    /// Create from price surface. Accepts EarlyExercisePremium or NormalizedPrice content.
    static std::expected<AmericanPriceSurface, ValidationError> create(
        std::shared_ptr<const PriceTableSurface<4>> eep_surface,
        OptionType type);

    /// P = (K/K_ref) * E(x,tau,sigma,r) + P_Eu(S,K,tau,sigma,r,q)
    double price(double spot, double strike, double tau,
                 double sigma, double rate) const;

    /// Delta = (K/(K_ref*S)) * dE/dx + Delta_Eu (EEP content only)
    double delta(double spot, double strike, double tau,
                 double sigma, double rate) const;

    /// Gamma = (K/K_ref) * (d2E/dx2 - dE/dx) / S^2 + Gamma_Eu (EEP content only)
    double gamma(double spot, double strike, double tau,
                 double sigma, double rate) const;

    /// Vega = (K/K_ref) * partial_2(E) + Vega_Eu
    double vega(double spot, double strike, double tau,
                double sigma, double rate) const;

    /// Theta = -(K/K_ref) * dE/dtau + Theta_Eu (EEP content only)
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

    /// Access the underlying StandardSurfaceWrapper for use with InterpolatedIVSolver.
    /// Only valid for EEP content.
    [[nodiscard]] const StandardSurfaceWrapper& wrapper() const noexcept { return wrapper_; }
    [[nodiscard]] StandardSurfaceWrapper take_wrapper() && { return std::move(wrapper_); }

private:
    AmericanPriceSurface(StandardSurfaceWrapper wrapper,
                         std::shared_ptr<const PriceTableSurface<4>> surface,
                         OptionType type, double K_ref, double dividend_yield,
                         bool is_eep);

    StandardSurfaceWrapper wrapper_;
    std::shared_ptr<const PriceTableSurface<4>> surface_;  // direct B-spline access for Greeks
    OptionType type_;
    double K_ref_;
    double dividend_yield_;
    bool is_eep_;
};

}  // namespace mango
