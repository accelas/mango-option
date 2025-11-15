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

    // Validate dividend yield (must be non-negative and finite)
    if (spec.dividend_yield < 0.0 || !std::isfinite(spec.dividend_yield)) {
        return std::unexpected(std::string("Dividend yield must be non-negative and finite"));
    }

    return {};
}

std::expected<void, std::string> validate_iv_query(const IVQuery& query) {
    // Validate base option spec first (using slicing)
    auto spec_validation = validate_option_spec(static_cast<const OptionSpec&>(query));
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

    if (query.type == OptionType::CALL) {
        intrinsic = std::max(query.spot - query.strike, 0.0);
        upper_bound = query.spot;
    } else {  // PUT
        intrinsic = std::max(query.strike - query.spot, 0.0);
        upper_bound = query.strike;
    }

    if (query.market_price < intrinsic) {
        return std::unexpected(std::string("Market price below intrinsic value (arbitrage)"));
    }

    if (query.market_price > upper_bound) {
        const char* opt_type = (query.type == OptionType::CALL) ? "Call" : "Put";
        const char* bound_type = (query.type == OptionType::CALL) ? "spot" : "strike";
        return std::unexpected(std::string(opt_type) + " price above " + bound_type + " (arbitrage)");
    }

    return {};
}

std::expected<void, std::string> validate_pricing_params(const PricingParams& params) {
    // Validate base option spec first (using slicing)
    auto spec_validation = validate_option_spec(static_cast<const OptionSpec&>(params));
    if (!spec_validation) {
        return spec_validation;
    }

    // Check volatility
    if (params.volatility <= 0.0 || !std::isfinite(params.volatility)) {
        return std::unexpected("Volatility must be positive and finite");
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
