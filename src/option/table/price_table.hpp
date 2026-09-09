// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"
#include "mango/option/table/greek_types.hpp"
#include "mango/option/table/strike_bounds.hpp"
#include "mango/option/table/moneyness_bounds.hpp"
#include "mango/option/table/surface_bounds.hpp"
#include "mango/option/table/fixed_expiry.hpp"
#include "mango/option/table/certification/publication.hpp"
#include <expected>
#include <concepts>
#include <cmath>
#include <memory>

namespace mango {

/// Top-level queryable price surface with runtime metadata.
/// Publication detaches numerical storage and model metadata from caller
/// aliases. Later copies share that immutable payload, including in IV solvers.
/// Used directly by InterpolatedIVSolver.
template <typename Inner>
class PriceTable {
public:
    using inner_type = Inner;
    static constexpr bool requires_fixed_expiry = [] {
        if constexpr (requires { Inner::requires_strike_bounds; }) return Inner::requires_strike_bounds;
        return false;
    }();

    /// Explicit certified creation. All numeric storage and model/domain
    /// metadata are detached before proving the exact retained payload.
    /// The certificate concerns the represented real price's sigma shape.
    /// A genuinely unrepresentable output at an extreme currency scale can
    /// still require NumericalFailure at the checked query boundary.
    [[nodiscard]] static std::expected<PriceTable, PriceTableError> create(
        const Inner& inner, const SurfaceBounds& bounds, OptionType type, double q,
        const std::optional<FixedExpiryMetadata>& fixed_expiry = std::nullopt,
        PriceTableProofBudget budget = {}) {
        if constexpr (std::same_as<Inner, BSplineLeaf> || std::same_as<Inner, BSpline3DLeaf> ||
            std::same_as<Inner, BSplineMultiKRefInner> || std::same_as<Inner, ChebyshevModalLeaf> ||
            std::same_as<Inner, ChebyshevModal3DLeaf> ||
            std::same_as<Inner, ChebyshevModalMultiKRefInner>) {
            auto owned = std::make_shared<Payload>(freeze_inner(inner), bounds, type, q, fixed_expiry);
            auto evidence = detail::certification::certify_payload(owned->inner, owned->bounds,
                owned->option_type, owned->dividend_yield, owned->fixed_expiry, budget);
            if (!evidence) return std::unexpected(evidence.error());
            if (evidence->status != PriceProofStatus::Certified) {
                return std::unexpected(PriceTableError{
                    PriceTableErrorCode::CertificationIndeterminate, 0, evidence->work});
            }
            owned->certificate = *evidence;
            return PriceTable(std::move(owned), AdoptPayload{});
        } else {
            return std::unexpected(PriceTableError{PriceTableErrorCode::UnsupportedRepresentation});
        }
    }
    [[nodiscard]] PriceProofStatus proof_status() const noexcept {
        return payload_ ? payload_->certificate.status : PriceProofStatus::NotRun;
    }
    [[nodiscard]] std::size_t proof_work() const noexcept { return payload_ ? payload_->certificate.work : 0; }

    /// Unchecked numerical primitive; requires admitted query/model metadata.
    [[nodiscard]] double price(double spot, double strike,
                                double tau, double sigma, double rate) const {
        return payload_->inner.price(spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] double vega(double spot, double strike,
                               double tau, double sigma, double rate) const {
        return payload_->inner.vega(spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] std::expected<double, GreekError>
    delta(const PricingParams& params) const {
        if (!contains_pricing_params(params)) {
            return std::unexpected(GreekError::OutOfDomain);
        }
        return checked_greek(payload_->inner.greek(Greek::Delta, params));
    }

    [[nodiscard]] std::expected<double, GreekError>
    gamma(const PricingParams& params) const {
        if (!contains_pricing_params(params)) {
            return std::unexpected(GreekError::OutOfDomain);
        }
        return checked_greek(payload_->inner.gamma(params));
    }

    [[nodiscard]] std::expected<double, GreekError>
    theta(const PricingParams& params) const {
        if (!contains_pricing_params(params)) {
            return std::unexpected(GreekError::OutOfDomain);
        }
        return checked_greek(payload_->inner.greek(Greek::Theta, params));
    }

    [[nodiscard]] std::expected<double, GreekError>
    rho(const PricingParams& params) const {
        if (!contains_pricing_params(params)) {
            return std::unexpected(GreekError::OutOfDomain);
        }
        return checked_greek(payload_->inner.greek(Greek::Rho, params));
    }

    [[nodiscard]] double m_min() const noexcept { return payload_->bounds.m_min; }
    [[nodiscard]] double m_max() const noexcept { return payload_->bounds.m_max; }
    [[nodiscard]] double tau_min() const noexcept { return payload_->bounds.tau_min; }
    [[nodiscard]] double tau_max() const noexcept { return payload_->bounds.tau_max; }
    [[nodiscard]] bool contains_maturity(double tau) const noexcept {
        if (!payload_->fixed_expiry_valid || !std::isfinite(tau) || tau <= 0.0 || tau < tau_min() || tau > tau_max()) return false;
        if constexpr (requires { payload_->inner.contains_maturity(tau); }) {
            return payload_->inner.contains_maturity(tau);
        }
        return true;
    }
    [[nodiscard]] double sigma_min() const noexcept { return payload_->bounds.sigma_min; }
    [[nodiscard]] double sigma_max() const noexcept { return payload_->bounds.sigma_max; }
    [[nodiscard]] double rate_min() const noexcept { return payload_->bounds.rate_min; }
    [[nodiscard]] double rate_max() const noexcept { return payload_->bounds.rate_max; }
    [[nodiscard]] OptionType option_type() const noexcept { return payload_->option_type; }
    [[nodiscard]] double dividend_yield() const noexcept { return payload_->dividend_yield; }

    [[nodiscard]] const std::optional<FixedExpiryMetadata>& fixed_expiry() const noexcept {
        return payload_->fixed_expiry;
    }

    [[nodiscard]] const std::optional<StrikeBounds>& strike_bounds() const noexcept {
        return payload_->bounds.strike_bounds;
    }
    /// Requested ratio endpoints and conservative real S/K enclosure are
    /// carried separately from interpolation coordinates and support headroom.
    [[nodiscard]] const MoneynessBounds& ratio_bounds() const noexcept {
        return payload_->moneyness_domain.requested();
    }
    [[nodiscard]] const MoneynessBounds& ratio_enclosure() const noexcept {
        return payload_->moneyness_domain.enclosure();
    }
    [[nodiscard]] bool contains_moneyness(double spot, double strike) const noexcept {
        return payload_->moneyness_domain.contains_quote(spot, strike);
    }

    [[nodiscard]] bool contains_strike(double strike) const noexcept {
        if (!std::isfinite(strike) || strike <= 0.0) return false;
        if (!payload_->bounds.strike_bounds) {
            if constexpr (requires { Inner::requires_strike_bounds; }) {
                if (Inner::requires_strike_bounds) return false;
            }
        } else if (!payload_->bounds.strike_bounds->contains(strike)) {
            return false;
        }
        if constexpr (requires { payload_->inner.contains_strike(strike); }) {
            return payload_->inner.contains_strike(strike);
        }
        return true;
    }

    [[nodiscard]] const Inner& inner() const noexcept { return payload_->inner; }

private:
    static std::expected<double, GreekError> checked_greek(std::expected<double, GreekError> result) {
        if (result && !std::isfinite(*result)) return std::unexpected(GreekError::NumericalFailure);
        return result;
    }
    [[nodiscard]] bool contains_pricing_params(const PricingParams& params) const {
        if (!mango::validate_pricing_params(params) || params.option_type != payload_->option_type ||
            std::abs(params.dividend_yield - payload_->dividend_yield) > 1e-10 ||
            !contains_strike(params.strike) || !contains_moneyness(params.spot, params.strike) ||
            !contains_maturity(params.maturity)) return false;
        const double rate = get_zero_rate(params.rate, params.maturity);
        return std::isfinite(rate) && params.volatility >= sigma_min() &&
            params.volatility <= sigma_max() && rate >= rate_min() && rate <= rate_max();
    }

    static Inner freeze_inner(const Inner& inner) {
        if constexpr (requires { inner.immutable_snapshot(); }) {
            return inner.immutable_snapshot();
        } else {
            return inner;
        }
    }

    struct Payload {
        Inner inner;
        SurfaceBounds bounds;
        OptionType option_type;
        double dividend_yield;
        MoneynessDomain moneyness_domain;
        std::optional<FixedExpiryMetadata> fixed_expiry;
        bool fixed_expiry_valid;
        detail::certification::PublicationEvidence certificate;

        Payload(Inner value, const SurfaceBounds& domain, OptionType type, double yield,
                const std::optional<FixedExpiryMetadata>& model)
            : inner(std::move(value)), bounds(domain), option_type(type), dividend_yield(yield)
            , moneyness_domain(domain.ratio_bounds ? *domain.ratio_bounds
                : MoneynessBounds{std::exp(domain.m_min), std::exp(domain.m_max)})
            // Moving the caller's vector could preserve a mutable data() alias.
            , fixed_expiry(model)
            , fixed_expiry_valid(fixed_expiry ? fixed_expiry->valid(domain.tau_max)
                                             : !requires_fixed_expiry)
        {}
    };

    struct AdoptPayload {};
    PriceTable(std::shared_ptr<const Payload> payload,AdoptPayload) : payload_(std::move(payload)) {}

    // Copying a published table copies only this immutable handle. Numeric
    // storage is detached once by certified creation above.
    std::shared_ptr<const Payload> payload_;
};

}  // namespace mango
