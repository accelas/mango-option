// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/european_option.hpp"
#include "mango/option/option_spec.hpp"

namespace mango {

/// EEP strategy using closed-form Black-Scholes European pricing.
/// Handles dividend yield. Used for standard (non-segmented) surfaces.
class AnalyticalEEP {
public:
    AnalyticalEEP(OptionType option_type, double dividend_yield)
        : option_type_(option_type), dividend_yield_(dividend_yield) {}

    [[nodiscard]] OptionType option_type() const noexcept { return option_type_; }
    [[nodiscard]] double dividend_yield() const noexcept { return dividend_yield_; }

    [[nodiscard]] double european_price(double spot, double strike, double tau, double sigma,
                                        double rate) const {
        // Preserve the exact payoff subtraction at the construction closure.
        if (tau <= 0.0) return intrinsic_value(spot, strike, option_type_);
        return unit_strike_european(spot, strike, tau, sigma, rate).value() * strike;
    }

    [[nodiscard]] double european_vega(double spot, double strike, double tau, double sigma,
                                       double rate) const {
        return unit_strike_european(spot, strike, tau, sigma, rate).vega() * strike;
    }

    [[nodiscard]] double european_delta(double spot, double strike, double tau, double sigma,
                                        double rate) const {
        return unit_strike_european(spot, strike, tau, sigma, rate).delta();
    }

    [[nodiscard]] double european_gamma(double spot, double strike, double tau, double sigma,
                                        double rate) const {
        return unit_strike_european(spot, strike, tau, sigma, rate).gamma() / strike;
    }

    [[nodiscard]] double european_theta(double spot, double strike, double tau, double sigma,
                                        double rate) const {
        return unit_strike_european(spot, strike, tau, sigma, rate).theta() * strike;
    }

    [[nodiscard]] double european_rho(double spot, double strike, double tau, double sigma,
                                      double rate) const {
        return unit_strike_european(spot, strike, tau, sigma, rate).rho() * strike;
    }

private:
    /// Price homogeneity avoids currency-sized discounted-strike products.
    /// Coordinates/model are unchanged; restore each output's homogeneity
    /// degree only after the dimensionless European calculation.
    [[nodiscard]] EuropeanOptionResult unit_strike_european(double spot, double strike, double tau,
                                                            double sigma, double rate) const {
        return EuropeanOptionSolver(OptionSpec{.spot = spot / strike,
                                               .strike = 1,
                                               .maturity = tau,
                                               .rate = rate,
                                               .dividend_yield = dividend_yield_,
                                               .option_type = option_type_},
                                    sigma)
            .solve()
            .value();
    }

    OptionType option_type_;
    double dividend_yield_;
};

} // namespace mango
