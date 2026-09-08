// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"
#include "mango/option/table/greek_types.hpp"
#include "mango/option/table/strike_bounds.hpp"
#include "mango/option/table/fixed_expiry.hpp"
#include <expected>
#include <cmath>

namespace mango {

/// Bounds metadata for a price surface.
struct SurfaceBounds {
    double m_min, m_max;
    double tau_min, tau_max;
    double sigma_min, sigma_max;
    double rate_min, rate_max;
    /// Required for segmented publication; omitted for homogeneous tables.
    std::optional<StrikeBounds> strike_bounds = std::nullopt;
};

/// Top-level queryable price surface with runtime metadata.
/// Used directly by InterpolatedIVSolver.
template <typename Inner>
class PriceTable {
public:
    using inner_type = Inner;
    static constexpr bool requires_fixed_expiry = [] {
        if constexpr (requires { Inner::requires_strike_bounds; }) return Inner::requires_strike_bounds;
        return false;
    }();

    PriceTable(Inner inner, const SurfaceBounds& bounds,
                   OptionType option_type, double dividend_yield,
                   std::optional<FixedExpiryMetadata> fixed_expiry = std::nullopt)
        : inner_(std::move(inner))
        , bounds_(bounds)
        , option_type_(option_type)
        , dividend_yield_(dividend_yield)
        , ratio_min_(std::exp(bounds.m_min))
        , ratio_max_(std::exp(bounds.m_max))
        , fixed_expiry_(std::move(fixed_expiry))
        , fixed_expiry_valid_(fixed_expiry_ ? fixed_expiry_->valid(bounds.tau_max)
                                           : !requires_fixed_expiry)
    {}

    /// Unchecked numerical primitive; requires admitted query/model metadata.
    [[nodiscard]] double price(double spot, double strike,
                                double tau, double sigma, double rate) const {
        return inner_.price(spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] double vega(double spot, double strike,
                               double tau, double sigma, double rate) const {
        return inner_.vega(spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] std::expected<double, GreekError>
    delta(const PricingParams& params) const {
        if (!contains_strike(params.strike) || !contains_moneyness(params.spot, params.strike)
            || !contains_maturity(params.maturity)) {
            return std::unexpected(GreekError::OutOfDomain);
        }
        return inner_.greek(Greek::Delta, params);
    }

    [[nodiscard]] std::expected<double, GreekError>
    gamma(const PricingParams& params) const {
        if (!contains_strike(params.strike) || !contains_moneyness(params.spot, params.strike)
            || !contains_maturity(params.maturity)) {
            return std::unexpected(GreekError::OutOfDomain);
        }
        return inner_.gamma(params);
    }

    [[nodiscard]] std::expected<double, GreekError>
    theta(const PricingParams& params) const {
        if (!contains_strike(params.strike) || !contains_moneyness(params.spot, params.strike)
            || !contains_maturity(params.maturity)) {
            return std::unexpected(GreekError::OutOfDomain);
        }
        return inner_.greek(Greek::Theta, params);
    }

    [[nodiscard]] std::expected<double, GreekError>
    rho(const PricingParams& params) const {
        if (!contains_strike(params.strike) || !contains_moneyness(params.spot, params.strike)
            || !contains_maturity(params.maturity)) {
            return std::unexpected(GreekError::OutOfDomain);
        }
        return inner_.greek(Greek::Rho, params);
    }

    [[nodiscard]] double m_min() const noexcept { return bounds_.m_min; }
    [[nodiscard]] double m_max() const noexcept { return bounds_.m_max; }
    [[nodiscard]] double tau_min() const noexcept { return bounds_.tau_min; }
    [[nodiscard]] double tau_max() const noexcept { return bounds_.tau_max; }
    [[nodiscard]] bool contains_maturity(double tau) const noexcept {
        if (!fixed_expiry_valid_ || !std::isfinite(tau) || tau < tau_min() || tau > tau_max()) return false;
        if constexpr (requires { inner_.contains_maturity(tau); }) {
            return inner_.contains_maturity(tau);
        }
        return true;
    }
    [[nodiscard]] double sigma_min() const noexcept { return bounds_.sigma_min; }
    [[nodiscard]] double sigma_max() const noexcept { return bounds_.sigma_max; }
    [[nodiscard]] double rate_min() const noexcept { return bounds_.rate_min; }
    [[nodiscard]] double rate_max() const noexcept { return bounds_.rate_max; }
    [[nodiscard]] OptionType option_type() const noexcept { return option_type_; }
    [[nodiscard]] double dividend_yield() const noexcept { return dividend_yield_; }

    [[nodiscard]] const std::optional<FixedExpiryMetadata>& fixed_expiry() const noexcept {
        return fixed_expiry_;
    }

    [[nodiscard]] const std::optional<StrikeBounds>& strike_bounds() const noexcept {
        return bounds_.strike_bounds;
    }
    /// Compare in quote space, using the same rounded S=K*exp(x) endpoint
    /// construction as callers. This avoids log(S/K) round-trip ULP errors
    /// without admitting the next representable spot outside either bound.
    [[nodiscard]] bool contains_moneyness(double spot, double strike) const noexcept {
        return std::isfinite(spot) && std::isfinite(strike) && spot > 0.0 && strike > 0.0
            && spot >= strike * ratio_min_
            && spot <= strike * ratio_max_;
    }

    [[nodiscard]] bool contains_strike(double strike) const noexcept {
        if (!std::isfinite(strike) || strike <= 0.0) return false;
        if (!bounds_.strike_bounds) {
            if constexpr (requires { Inner::requires_strike_bounds; }) {
                if (Inner::requires_strike_bounds) return false;
            }
        } else if (!bounds_.strike_bounds->contains(strike)) {
            return false;
        }
        if constexpr (requires { inner_.contains_strike(strike); }) {
            return inner_.contains_strike(strike);
        }
        return true;
    }

    [[nodiscard]] const Inner& inner() const noexcept { return inner_; }

private:
    Inner inner_;
    SurfaceBounds bounds_;
    OptionType option_type_;
    double dividend_yield_;
    double ratio_min_, ratio_max_;
    std::optional<FixedExpiryMetadata> fixed_expiry_;
    bool fixed_expiry_valid_;
};

}  // namespace mango
