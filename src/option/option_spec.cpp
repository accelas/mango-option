// SPDX-License-Identifier: MIT
#include "src/option/option_spec.hpp"
#include <cmath>
#include <algorithm>

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
    double intrinsic;
    double upper_bound;

    if (query.type == OptionType::CALL) {
        intrinsic = std::max(query.spot - query.strike, 0.0);
        upper_bound = query.spot;
    } else {  // PUT
        intrinsic = std::max(query.strike - query.spot, 0.0);
        upper_bound = query.strike;
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

} // namespace mango
