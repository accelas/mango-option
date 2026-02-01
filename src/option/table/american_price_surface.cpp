// SPDX-License-Identifier: MIT
#include "src/option/table/american_price_surface.hpp"
#include "src/option/european_option.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace mango {

AmericanPriceSurface::AmericanPriceSurface(
    std::shared_ptr<const PriceTableSurface<4>> surface,
    OptionType type, double K_ref, double dividend_yield)
    : surface_(std::move(surface))
    , type_(type)
    , K_ref_(K_ref)
    , dividend_yield_(dividend_yield) {}

std::expected<AmericanPriceSurface, ValidationError>
AmericanPriceSurface::create(
    std::shared_ptr<const PriceTableSurface<4>> eep_surface,
    OptionType type)
{
    if (!eep_surface) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidBounds, 0.0, 0});
    }

    const auto& meta = eep_surface->metadata();
    if (meta.content != SurfaceContent::EarlyExercisePremium &&
        meta.content != SurfaceContent::RawPrice) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidBounds, 0.0, 0});
    }

    // Discrete dividends are not supported by either content type currently.
    // EEP decomposition assumes continuous dividend yield only;
    // RawPrice surfaces with discrete dividends require jump-condition handling
    // that is not yet implemented.
    if (!meta.discrete_dividends.empty()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidBounds, 0.0, 0});
    }

    if (meta.K_ref <= 0.0) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidBounds, meta.K_ref, 0});
    }

    return AmericanPriceSurface(
        std::move(eep_surface), type, meta.K_ref, meta.dividend_yield);
}

double AmericanPriceSurface::price(double spot, double strike, double tau,
                                   double sigma, double rate) const {
    if (surface_->metadata().content == SurfaceContent::RawPrice) {
        assert(strike == K_ref_ && "RawPrice surfaces require strike == K_ref");
        double m = spot / K_ref_;
        return surface_->value({m, tau, sigma, rate});
    }
    double m = spot / strike;
    double eep = surface_->value({m, tau, sigma, rate});
    auto eu = EuropeanOptionSolver(
        PricingParams(OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = dividend_yield_, .type = type_}, sigma)).solve().value();
    return eep * (strike / K_ref_) + eu.value();
}

double AmericanPriceSurface::delta(double spot, double strike, double tau,
                                   double sigma, double rate) const {
    double m = spot / strike;
    // partial(0, ...) returns dE/dm (chain rule already applied in PriceTableSurface)
    // delta_eep = (K/K_ref) * dE/dm * dm/dS = (K/K_ref) * dE/dm * (1/K) = (1/K_ref) * dE/dm
    double eep_delta = (1.0 / K_ref_) * surface_->partial(0, {m, tau, sigma, rate});
    auto eu = EuropeanOptionSolver(
        PricingParams(OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = dividend_yield_, .type = type_}, sigma)).solve().value();
    return eep_delta + eu.delta();
}

double AmericanPriceSurface::gamma(double spot, double strike, double tau,
                                   double sigma, double rate) const {
    double m = spot / strike;
    // γ_eep = (K/K_ref) · ∂²EEP/∂m² · (1/K)² = 1/(K_ref·K) · ∂²EEP/∂m²
    double eep_gamma = (1.0 / (K_ref_ * strike)) *
        surface_->second_partial(0, {m, tau, sigma, rate});
    auto eu = EuropeanOptionSolver(
        PricingParams(OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = dividend_yield_, .type = type_}, sigma)).solve().value();
    return eep_gamma + eu.gamma();
}

double AmericanPriceSurface::vega(double spot, double strike, double tau,
                                  double sigma, double rate) const {
    if (surface_->metadata().content == SurfaceContent::RawPrice) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double m = spot / strike;
    double eep_vega = (strike / K_ref_) * surface_->partial(2, {m, tau, sigma, rate});
    auto eu = EuropeanOptionSolver(
        PricingParams(OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = dividend_yield_, .type = type_}, sigma)).solve().value();
    return eep_vega + eu.vega();
}

double AmericanPriceSurface::theta(double spot, double strike, double tau,
                                   double sigma, double rate) const {
    double m = spot / strike;
    // partial(1, ...) gives dE/d(tau) in time-to-expiry space.
    // Convert to calendar time: dV/dt = -dV/d(tau).
    // European theta() already returns dV/dt (calendar time).
    // theta = -(K/K_ref) * dE/d(tau) + eu.theta()
    double eep_dtau = (strike / K_ref_) * surface_->partial(1, {m, tau, sigma, rate});
    auto eu = EuropeanOptionSolver(
        PricingParams(OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = dividend_yield_, .type = type_}, sigma)).solve().value();
    return -eep_dtau + eu.theta();
}

const PriceTableSurface<4>& AmericanPriceSurface::eep_surface() const {
    return *surface_;
}

const PriceTableMetadata& AmericanPriceSurface::metadata() const {
    return surface_->metadata();
}

double AmericanPriceSurface::m_min() const noexcept {
    return surface_->metadata().m_min;
}

double AmericanPriceSurface::m_max() const noexcept {
    return surface_->metadata().m_max;
}

double AmericanPriceSurface::tau_min() const noexcept {
    return surface_->axes().grids[1].front();
}

double AmericanPriceSurface::tau_max() const noexcept {
    return surface_->axes().grids[1].back();
}

double AmericanPriceSurface::sigma_min() const noexcept {
    return surface_->axes().grids[2].front();
}

double AmericanPriceSurface::sigma_max() const noexcept {
    return surface_->axes().grids[2].back();
}

double AmericanPriceSurface::rate_min() const noexcept {
    return surface_->axes().grids[3].front();
}

double AmericanPriceSurface::rate_max() const noexcept {
    return surface_->axes().grids[3].back();
}


OptionType AmericanPriceSurface::option_type() const noexcept {
    return type_;
}

double AmericanPriceSurface::dividend_yield() const noexcept {
    return dividend_yield_;
}
}  // namespace mango
