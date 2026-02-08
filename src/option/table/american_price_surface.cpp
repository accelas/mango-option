// SPDX-License-Identifier: MIT
#include "mango/option/table/american_price_surface.hpp"
#include "mango/option/european_option.hpp"
#include <cassert>
#include <cmath>

namespace mango {

AmericanPriceSurface::AmericanPriceSurface(
    StandardSurfaceWrapper wrapper,
    std::shared_ptr<const PriceTableSurface<4>> surface,
    OptionType type, double K_ref, double dividend_yield, bool is_eep)
    : wrapper_(std::move(wrapper))
    , surface_(std::move(surface))
    , type_(type)
    , K_ref_(K_ref)
    , dividend_yield_(dividend_yield)
    , is_eep_(is_eep) {}

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
        meta.content != SurfaceContent::NormalizedPrice) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidBounds, 0.0, 0});
    }

    if (!meta.dividends.discrete_dividends.empty()) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidBounds, 0.0, 0});
    }

    if (meta.K_ref <= 0.0) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidBounds, meta.K_ref, 0});
    }

    double K_ref = meta.K_ref;
    double dividend_yield = meta.dividends.dividend_yield;
    const auto& axes = eep_surface->axes();
    bool is_eep = (meta.content == SurfaceContent::EarlyExercisePremium);

    // Build StandardSurfaceWrapper using EEPPriceTableInner for EEP content.
    // For NormalizedPrice, the wrapper is constructed but price()/vega() bypass it.
    EEPPriceTableInner inner(eep_surface, type, K_ref, dividend_yield);
    StandardSurface std_surface({std::move(inner)}, SingleBracket{}, IdentityTransform{}, WeightedSum{});

    SplicedSurfaceWrapper<StandardSurface>::Bounds bounds{
        .m_min = meta.m_min,
        .m_max = meta.m_max,
        .tau_min = axes.grids[1].front(),
        .tau_max = axes.grids[1].back(),
        .sigma_min = axes.grids[2].front(),
        .sigma_max = axes.grids[2].back(),
        .rate_min = axes.grids[3].front(),
        .rate_max = axes.grids[3].back(),
    };

    StandardSurfaceWrapper wrapper(std::move(std_surface), bounds, type, dividend_yield);

    return AmericanPriceSurface(
        std::move(wrapper), std::move(eep_surface), type, K_ref, dividend_yield, is_eep);
}

double AmericanPriceSurface::price(double spot, double strike, double tau,
                                   double sigma, double rate) const {
    if (!is_eep_) {
        // NormalizedPrice: direct B-spline lookup (V/K_ref at K_ref)
        assert(strike == K_ref_ && "NormalizedPrice surfaces require strike == K_ref");
        double x = std::log(spot / K_ref_);
        return surface_->value({x, tau, sigma, rate});
    }
    return wrapper_.price(spot, strike, tau, sigma, rate);
}

double AmericanPriceSurface::vega(double spot, double strike, double tau,
                                  double sigma, double rate) const {
    if (!is_eep_) {
        // NormalizedPrice: finite-difference vega
        constexpr double eps = 1e-4;
        double up = price(spot, strike, tau, sigma + eps, rate);
        double dn = price(spot, strike, tau, sigma - eps, rate);
        return (up - dn) / (2.0 * eps);
    }
    return wrapper_.vega(spot, strike, tau, sigma, rate);
}

const StandardSurfaceWrapper& AmericanPriceSurface::wrapper() const noexcept {
    assert(is_eep_ && "wrapper() requires EEP content");
    return wrapper_;
}

StandardSurfaceWrapper AmericanPriceSurface::take_wrapper() && {
    assert(is_eep_ && "take_wrapper() requires EEP content");
    return std::move(wrapper_);
}

double AmericanPriceSurface::delta(double spot, double strike, double tau,
                                   double sigma, double rate) const {
    assert(is_eep_ && "delta() requires EEP content");
    double x = std::log(spot / strike);
    double dEdx = surface_->partial(0, {x, tau, sigma, rate});
    double eep_delta = (strike / (K_ref_ * spot)) * dEdx;
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = dividend_yield_, .option_type = type_}, sigma).solve().value();
    return eep_delta + eu.delta();
}

double AmericanPriceSurface::gamma(double spot, double strike, double tau,
                                   double sigma, double rate) const {
    assert(is_eep_ && "gamma() requires EEP content");
    double x = std::log(spot / strike);
    double dEdx = surface_->partial(0, {x, tau, sigma, rate});
    double d2Edx2 = surface_->second_partial(0, {x, tau, sigma, rate});
    double eep_gamma = (strike / K_ref_) * (d2Edx2 - dEdx) / (spot * spot);
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = dividend_yield_, .option_type = type_}, sigma).solve().value();
    return eep_gamma + eu.gamma();
}

double AmericanPriceSurface::theta(double spot, double strike, double tau,
                                   double sigma, double rate) const {
    assert(is_eep_ && "theta() requires EEP content");
    double x = std::log(spot / strike);
    double eep_dtau = (strike / K_ref_) * surface_->partial(1, {x, tau, sigma, rate});
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = dividend_yield_, .option_type = type_}, sigma).solve().value();
    return -eep_dtau + eu.theta();
}

const PriceTableMetadata& AmericanPriceSurface::metadata() const {
    return surface_->metadata();
}

double AmericanPriceSurface::m_min() const noexcept { return wrapper_.m_min(); }
double AmericanPriceSurface::m_max() const noexcept { return wrapper_.m_max(); }
double AmericanPriceSurface::tau_min() const noexcept { return wrapper_.tau_min(); }
double AmericanPriceSurface::tau_max() const noexcept { return wrapper_.tau_max(); }
double AmericanPriceSurface::sigma_min() const noexcept { return wrapper_.sigma_min(); }
double AmericanPriceSurface::sigma_max() const noexcept { return wrapper_.sigma_max(); }
double AmericanPriceSurface::rate_min() const noexcept { return wrapper_.rate_min(); }
double AmericanPriceSurface::rate_max() const noexcept { return wrapper_.rate_max(); }
OptionType AmericanPriceSurface::option_type() const noexcept { return type_; }
double AmericanPriceSurface::dividend_yield() const noexcept { return dividend_yield_; }

}  // namespace mango
