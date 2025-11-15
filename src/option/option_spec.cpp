#include "src/option/option_spec.hpp"
#include <cmath>
#include <algorithm>

namespace mango {

std::expected<void, std::string> validate_option_spec(const OptionSpec& spec) {
    // Validate spot price
    if (spec.spot <= 0.0 || !std::isfinite(spec.spot)) {
        return std::unexpected(std::string("Spot price must be positive and finite"));
    }

    // Validate strike price
    if (spec.strike <= 0.0 || !std::isfinite(spec.strike)) {
        return std::unexpected(std::string("Strike price must be positive and finite"));
    }

    // Validate maturity
    if (spec.maturity <= 0.0 || !std::isfinite(spec.maturity)) {
        return std::unexpected(std::string("Time to maturity must be positive and finite"));
    }

    // Validate rate (allow negative but must be finite)
    if (!std::isfinite(spec.rate)) {
        return std::unexpected(std::string("Risk-free rate must be finite"));
    }

    // Validate dividend yield (allow negative but must be finite)
    if (!std::isfinite(spec.dividend_yield)) {
        return std::unexpected(std::string("Dividend yield must be finite"));
    }

    return {};
}

std::expected<void, std::string> validate_iv_query(const IVQuery& query) {
    // Validate option spec first
    auto spec_validation = validate_option_spec(query.option);
    if (!spec_validation) {
        return spec_validation;
    }

    // Validate market price: must be finite
    if (!std::isfinite(query.market_price)) {
        return std::unexpected(std::string("Market price must be finite"));
    }

    // Validate market price: must be positive
    if (query.market_price <= 0.0) {
        return std::unexpected(std::string("Market price must be positive"));
    }

    // Check for arbitrage violations
    double intrinsic;
    double upper_bound;

    if (query.option.type == OptionType::CALL) {
        intrinsic = std::max(query.option.spot - query.option.strike, 0.0);
        upper_bound = query.option.spot;
    } else {  // PUT
        intrinsic = std::max(query.option.strike - query.option.spot, 0.0);
        upper_bound = query.option.strike;
    }

    if (query.market_price < intrinsic) {
        return std::unexpected(std::string("Market price below intrinsic value (arbitrage)"));
    }

    if (query.market_price > upper_bound) {
        const char* opt_type = (query.option.type == OptionType::CALL) ? "Call" : "Put";
        const char* bound_type = (query.option.type == OptionType::CALL) ? "spot" : "strike";
        return std::unexpected(std::string(opt_type) + " price above " + bound_type + " (arbitrage)");
    }

    return {};
}

std::expected<void, std::string> validate_pricing_params(const PricingParams& params) {
    // Check strike
    if (params.strike <= 0.0 || !std::isfinite(params.strike)) {
        return std::unexpected("Strike must be positive and finite");
    }

    // Check spot
    if (params.spot <= 0.0 || !std::isfinite(params.spot)) {
        return std::unexpected("Spot must be positive and finite");
    }

    // Check maturity
    if (params.maturity <= 0.0 || !std::isfinite(params.maturity)) {
        return std::unexpected("Maturity must be positive and finite");
    }

    // Check volatility
    if (params.volatility <= 0.0 || !std::isfinite(params.volatility)) {
        return std::unexpected("Volatility must be positive and finite");
    }

    // Check rate (allow negative but must be finite)
    if (!std::isfinite(params.rate)) {
        return std::unexpected("Rate must be finite");
    }

    // Check continuous dividend yield (must be non-negative and finite)
    if (params.continuous_dividend_yield < 0.0 || !std::isfinite(params.continuous_dividend_yield)) {
        return std::unexpected("Continuous dividend yield must be non-negative and finite");
    }

    // Validate discrete dividends
    for (const auto& [time, amount] : params.discrete_dividends) {
        if (time < 0.0 || time > params.maturity) {
            return std::unexpected("Discrete dividend time must be in [0, maturity]");
        }
        if (amount < 0.0) {
            return std::unexpected("Discrete dividend amount must be non-negative");
        }
        if (!std::isfinite(time) || !std::isfinite(amount)) {
            return std::unexpected("Discrete dividend time and amount must be finite");
        }
    }

    return {};
}

} // namespace mango
