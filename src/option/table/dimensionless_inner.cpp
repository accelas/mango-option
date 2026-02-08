// SPDX-License-Identifier: MIT
#include "mango/option/table/dimensionless_inner.hpp"
#include "mango/option/european_option.hpp"
#include <cmath>

namespace mango {

DimensionlessEEPInner::DimensionlessEEPInner(
    std::shared_ptr<const PriceTableSurfaceND<3>> surface,
    OptionType type, double K_ref, double dividend_yield)
    : surface_(std::move(surface)), type_(type),
      K_ref_(K_ref), dividend_yield_(dividend_yield) {}

double DimensionlessEEPInner::price(const PriceQuery& q) const {
    double x = std::log(q.spot / q.strike);
    double tau_prime = q.sigma * q.sigma * q.tau / 2.0;
    double ln_kappa = std::log(2.0 * q.rate / (q.sigma * q.sigma));

    double eep = surface_->value({x, tau_prime, ln_kappa});

    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = q.spot, .strike = q.strike, .maturity = q.tau,
                   .rate = q.rate, .dividend_yield = dividend_yield_,
                   .option_type = type_},
        q.sigma).solve().value();

    // Surface stores EEP/K_ref (normalized). Scale by K to get dollar EEP at
    // query strike: dollar_EEP = (EEP/K_ref) * K = EEP_norm * K.
    return eep * q.strike + eu.value();
}

double DimensionlessEEPInner::vega(const PriceQuery& q) const {
    double x = std::log(q.spot / q.strike);
    double tau_prime = q.sigma * q.sigma * q.tau / 2.0;
    double ln_kappa = std::log(2.0 * q.rate / (q.sigma * q.sigma));
    std::array<double, 3> coords = {x, tau_prime, ln_kappa};

    // Chain rule: dEEP/dsigma = sigma*tau * dEEP/dtau' - (2/sigma) * dEEP/d(ln kappa)
    double dEEP_dtau_prime = surface_->partial(1, coords);
    double dEEP_dln_kappa  = surface_->partial(2, coords);
    double eep_vega = q.sigma * q.tau * dEEP_dtau_prime
                    - (2.0 / q.sigma) * dEEP_dln_kappa;

    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = q.spot, .strike = q.strike, .maturity = q.tau,
                   .rate = q.rate, .dividend_yield = dividend_yield_,
                   .option_type = type_},
        q.sigma).solve().value();

    // Same scaling as price: surface stores normalized values, scale by K.
    return q.strike * eep_vega + eu.vega();
}

}  // namespace mango
