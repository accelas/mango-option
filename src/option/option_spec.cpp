// SPDX-License-Identifier: MIT
#include "mango/option/option_spec.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace mango {

std::expected<void, ValidationError> validate_option_spec(const OptionSpec& spec) {
    // Validate spot price
    if (spec.spot <= 0.0 || !std::isfinite(spec.spot)) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidSpotPrice,
            spec.spot));
    }

    // Validate strike price
    if (spec.strike <= 0.0 || !std::isfinite(spec.strike)) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidStrike,
            spec.strike));
    }

    // Validate maturity
    if (spec.maturity <= 0.0 || !std::isfinite(spec.maturity)) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidMaturity,
            spec.maturity));
    }

    // Validate rate (allow negative but must be finite)
    // For constant rate, check finiteness; for YieldCurve, skip validation
    if (std::holds_alternative<double>(spec.rate)) {
        double rate = std::get<double>(spec.rate);
        if (!std::isfinite(rate)) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidRate,
                rate));
        }
    }
    // YieldCurve validation is implicitly done during construction

    // Validate dividend yield (must be non-negative and finite)
    if (spec.dividend_yield < 0.0 || !std::isfinite(spec.dividend_yield)) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidDividend,
            spec.dividend_yield));
    }

    return {};
}

std::expected<void, ValidationError> validate_iv_query(const IVQuery& query) {
    // Validate base option spec first (using slicing)
    auto spec_validation = validate_option_spec(static_cast<const OptionSpec&>(query));
    if (!spec_validation) {
        return spec_validation;
    }

    // Validate market price: must be finite
    if (!std::isfinite(query.market_price)) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidMarketPrice,
            query.market_price));
    }

    // Validate market price: must be positive
    if (query.market_price <= 0.0) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidMarketPrice,
            query.market_price));
    }

    // Check for arbitrage violations
    double intrinsic = intrinsic_value(query.spot, query.strike, query.option_type);
    double upper_bound = (query.option_type == OptionType::CALL) ? query.spot : query.strike;
    if (query.option_type == OptionType::PUT) {
        // Exercise pays at most K, at any date in [0,T]. Negative rates
        // can make its present value exceed K. Log-linear discount curves
        // attain their maximum at an endpoint or an interior tenor knot.
        double max_discount = 1.0;
        if (const auto* rate = std::get_if<double>(&query.rate)) {
            max_discount = std::max(max_discount, std::exp(-*rate * query.maturity));
        } else {
            const auto& curve = std::get<YieldCurve>(query.rate);
            max_discount = std::max(max_discount, curve.discount(query.maturity));
            for (const auto& point : curve.points()) {
                if (point.tenor > 0.0 && point.tenor < query.maturity) {
                    max_discount = std::max(max_discount, curve.discount(point.tenor));
                }
            }
        }
        upper_bound *= max_discount;
    }

    if (query.market_price < intrinsic) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidMarketPrice,
            query.market_price));
    }

    if (query.market_price > upper_bound) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidMarketPrice,
            query.market_price));
    }

    // Validate discrete dividends (mirrors validate_pricing_params): each
    // ex-dividend instant must fall within (0, maturity] and the amount be
    // non-negative and finite. The FDM IVSolver honors these dividends.
    for (size_t i = 0; i < query.discrete_dividends.size(); ++i) {
        const auto& div = query.discrete_dividends[i];
        if (div.calendar_time < 0.0 || div.calendar_time > query.maturity) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidDividend,
                div.calendar_time,
                i));
        }
        if (div.amount < 0.0) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidDividend,
                div.amount,
                i));
        }
        if (!std::isfinite(div.calendar_time) || !std::isfinite(div.amount)) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidDividend,
                div.calendar_time,
                i));
        }
    }

    return {};
}

std::expected<void, ValidationError> validate_pricing_params(const PricingParams& params) {
    // Validate base option spec first (using slicing)
    auto spec_validation = validate_option_spec(static_cast<const OptionSpec&>(params));
    if (!spec_validation) {
        return spec_validation;
    }

    // Check volatility
    if (params.volatility <= 0.0 || !std::isfinite(params.volatility)) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidVolatility,
            params.volatility));
    }

    // Validate discrete dividends
    for (size_t i = 0; i < params.discrete_dividends.size(); ++i) {
        const auto& div = params.discrete_dividends[i];
        if (div.calendar_time < 0.0 || div.calendar_time > params.maturity) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidDividend,
                div.calendar_time,
                i));
        }
        if (div.amount < 0.0) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidDividend,
                div.amount,
                i));
        }
        if (!std::isfinite(div.calendar_time) || !std::isfinite(div.amount)) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidDividend,
                div.calendar_time,
                i));
        }
    }

    return {};
}

// Brennan-Schwartz is a one-pass LCP solve.  The Il'in-fitted spatial
// operator guarantees the required off-diagonal signs, but two assumptions
// still have to be enforced at the American-option boundary:
//
//  * 1 + w*r(t) > 0 for every implicit stage (diagonal dominance); and
//  * the exercise set is one-sided.  With q >= 0 (enforced by
//    validate_pricing_params), a rate term structure of one sign has the
//    standard one-sided/empty exercise topology.  A curve which crosses zero
//    is not covered: time-inhomogeneous negative-rate models can develop a
//    floating exercise interval with two boundaries.
//
// Every TR-BDF2/Rannacher implicit coefficient is at most dt/2, and dt <= T,
// so r > -2/T is a conservative grid/config-independent dominance bound.
std::expected<void, ValidationError> validate_pde_rate(
    const RateSpec& rate_spec, double maturity)
{
    const double min_admissible_rate = -2.0 / maturity;
    // Do not admit a value merely because it rounded one ULP above the
    // theoretical open boundary.  At the worst permitted stage weight that
    // can still round 1 + w*r to zero.  Keep a small, scale-aware FP margin.
    const double dominance_margin = 64.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::abs(min_admissible_rate));
    const double min_rate_with_margin = min_admissible_rate + dominance_margin;

    if (const auto* scalar_rate = std::get_if<double>(&rate_spec)) {
        if (*scalar_rate <= min_rate_with_margin) {
            return std::unexpected(ValidationError{ValidationErrorCode::InvalidRate, *scalar_rate});
        }
        return {};
    }

    const auto& curve = std::get<YieldCurve>(rate_spec);
    const auto points = curve.points();
    if (points.size() < 2) {
        // The default-constructed empty curve evaluates to the safe zero rate.
        return {};
    }

    double min_rate = std::numeric_limits<double>::infinity();
    double max_rate = -std::numeric_limits<double>::infinity();
    size_t min_rate_segment = 0;
    for (size_t i = 0; i + 1 < points.size(); ++i) {
        if (points[i].tenor >= maturity) break;
        const double dt = points[i + 1].tenor - points[i].tenor;
        const double rate = -(points[i + 1].log_discount - points[i].log_discount) / dt;
        if (!std::isfinite(rate)) {
            return std::unexpected(ValidationError{ValidationErrorCode::InvalidRate, rate, i});
        }
        if (rate < min_rate) {
            min_rate = rate;
            min_rate_segment = i;
        }
        max_rate = std::max(max_rate, rate);
    }

    if (min_rate == std::numeric_limits<double>::infinity()) {
        return {};
    }
    if (min_rate <= min_rate_with_margin) {
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidRate, min_rate, min_rate_segment});
    }
    // Log-discount interpolation can manufacture a few ULPs of rate around a
    // mathematically flat zero segment.  Classify those as zero; rejecting a
    // curve requires a sign change large enough to be numerically meaningful.
    const double zero_rate_scale = std::max(
        {1.0 / maturity, std::abs(min_rate), std::abs(max_rate)});
    const double zero_rate_tol = 64.0 * std::numeric_limits<double>::epsilon() *
        zero_rate_scale;
    if (min_rate < -zero_rate_tol && max_rate > zero_rate_tol) {
        // Sign-changing forward rates are the reachable vanilla-input route
        // to a floating/non-edge-touching exercise interval.  The oriented
        // one-pass solver has no exact sweep for that topology.
        return std::unexpected(ValidationError{
            ValidationErrorCode::InvalidRate, min_rate, min_rate_segment});
    }
    return {};
}

} // namespace mango
