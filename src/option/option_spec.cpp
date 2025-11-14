#include "src/option/option_spec.hpp"
#include "src/option/american_option.hpp"
#include <cmath>
#include <algorithm>

namespace mango {

expected<void, std::string> validate_option_spec(const OptionSpec& spec) {
    // Validate spot price
    if (spec.spot <= 0.0 || !std::isfinite(spec.spot)) {
        return unexpected(std::string("Spot price must be positive and finite"));
    }

    // Validate strike price
    if (spec.strike <= 0.0 || !std::isfinite(spec.strike)) {
        return unexpected(std::string("Strike price must be positive and finite"));
    }

    // Validate maturity
    if (spec.maturity <= 0.0 || !std::isfinite(spec.maturity)) {
        return unexpected(std::string("Time to maturity must be positive and finite"));
    }

    // Validate rate (allow negative but must be finite)
    if (!std::isfinite(spec.rate)) {
        return unexpected(std::string("Risk-free rate must be finite"));
    }

    // Validate dividend yield (allow negative but must be finite)
    if (!std::isfinite(spec.dividend_yield)) {
        return unexpected(std::string("Dividend yield must be finite"));
    }

    return {};
}

expected<void, std::string> validate_iv_query(const IVQuery& query) {
    // Validate option spec first
    auto spec_validation = validate_option_spec(query.option);
    if (!spec_validation) {
        return spec_validation;
    }

    // Validate market price: must be finite
    if (!std::isfinite(query.market_price)) {
        return unexpected(std::string("Market price must be finite"));
    }

    // Validate market price: must be positive
    if (query.market_price <= 0.0) {
        return unexpected(std::string("Market price must be positive"));
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
        return unexpected(std::string("Market price below intrinsic value (arbitrage)"));
    }

    if (query.market_price > upper_bound) {
        const char* opt_type = (query.option.type == OptionType::CALL) ? "Call" : "Put";
        const char* bound_type = (query.option.type == OptionType::CALL) ? "spot" : "strike";
        return unexpected(std::string(opt_type) + " price above " + bound_type + " (arbitrage)");
    }

    return {};
}

} // namespace mango
