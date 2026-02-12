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

    [[nodiscard]] double european_price(
        double spot, double strike, double tau, double sigma, double rate) const {
        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
                .rate = rate, .dividend_yield = dividend_yield_,
                .option_type = option_type_}, sigma).solve().value();
        return eu.value();
    }

    [[nodiscard]] double european_vega(
        double spot, double strike, double tau, double sigma, double rate) const {
        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
                .rate = rate, .dividend_yield = dividend_yield_,
                .option_type = option_type_}, sigma).solve().value();
        return eu.vega();
    }

    [[nodiscard]] double european_delta(
        double spot, double strike, double tau, double sigma, double rate) const {
        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
                .rate = rate, .dividend_yield = dividend_yield_,
                .option_type = option_type_}, sigma).solve().value();
        return eu.delta();
    }

    [[nodiscard]] double european_gamma(
        double spot, double strike, double tau, double sigma, double rate) const {
        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
                .rate = rate, .dividend_yield = dividend_yield_,
                .option_type = option_type_}, sigma).solve().value();
        return eu.gamma();
    }

    [[nodiscard]] double european_theta(
        double spot, double strike, double tau, double sigma, double rate) const {
        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
                .rate = rate, .dividend_yield = dividend_yield_,
                .option_type = option_type_}, sigma).solve().value();
        return eu.theta();
    }

    [[nodiscard]] double european_rho(
        double spot, double strike, double tau, double sigma, double rate) const {
        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
                .rate = rate, .dividend_yield = dividend_yield_,
                .option_type = option_type_}, sigma).solve().value();
        return eu.rho();
    }

private:
    OptionType option_type_;
    double dividend_yield_;
};

}  // namespace mango
