// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include "mango/support/error_types.hpp"

#include <array>
#include <expected>
#include <variant>

namespace mango {

struct ChebyshevTableConfig {
    std::array<size_t, 4> num_pts;   // CGL nodes: (m, tau, sigma, rate)
    Domain<4> domain;                // Axis bounds
    double K_ref;
    OptionType option_type;
    double dividend_yield = 0.0;
    double tucker_epsilon = 1e-8;    // 0 = use RawTensor
};

struct ChebyshevTableResult {
    std::variant<ChebyshevSurface, ChebyshevRawSurface> surface;
    size_t n_pde_solves;
    double build_seconds;

    double price(double spot, double strike, double tau,
                 double sigma, double rate) const {
        return std::visit([&](const auto& s) {
            return s.price(spot, strike, tau, sigma, rate);
        }, surface);
    }

    double vega(double spot, double strike, double tau,
                double sigma, double rate) const {
        return std::visit([&](const auto& s) {
            return s.vega(spot, strike, tau, sigma, rate);
        }, surface);
    }
};

[[nodiscard]] std::expected<ChebyshevTableResult, PriceTableError>
build_chebyshev_table(const ChebyshevTableConfig& config);

}  // namespace mango
