// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include "mango/option/table/greek_types.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/support/error_types.hpp"

#include <array>
#include <expected>

namespace mango {

struct ChebyshevTableConfig {
    std::array<size_t, 4> num_pts;   // CGL nodes: (m, tau, sigma, rate)
    Domain<4> domain;                // Axis bounds
    double K_ref;
    OptionType option_type;
    double dividend_yield = 0.0;
};

struct ChebyshevTableResult {
    ChebyshevSurface surface;
    size_t n_pde_solves;
    double build_seconds;

    double price(double spot, double strike, double tau,
                 double sigma, double rate) const {
        return surface.price(spot, strike, tau, sigma, rate);
    }

    double vega(double spot, double strike, double tau,
                double sigma, double rate) const {
        return surface.vega(spot, strike, tau, sigma, rate);
    }

    std::expected<double, GreekError> delta(const PricingParams& params) const {
        return surface.delta(params);
    }
    std::expected<double, GreekError> gamma(const PricingParams& params) const {
        return surface.gamma(params);
    }
    std::expected<double, GreekError> theta(const PricingParams& params) const {
        return surface.theta(params);
    }
    std::expected<double, GreekError> rho(const PricingParams& params) const {
        return surface.rho(params);
    }
};

[[nodiscard]] std::expected<ChebyshevTableResult, PriceTableError>
build_chebyshev_table(const ChebyshevTableConfig& config);

}  // namespace mango
