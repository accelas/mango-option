#include "src/option/option_spec.hpp"
#include <cmath>

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

} // namespace mango
