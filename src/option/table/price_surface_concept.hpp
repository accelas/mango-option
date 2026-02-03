// SPDX-License-Identifier: MIT
#pragma once

#include <concepts>
#include "mango/option/option_spec.hpp"

namespace mango {

template <typename S>
concept PriceSurface = requires(const S& s, double spot, double strike,
                                double tau, double sigma, double rate) {
    { s.price(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { s.vega(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { s.option_type() } -> std::same_as<OptionType>;
    { s.dividend_yield() } -> std::convertible_to<double>;
    { s.m_min() } -> std::convertible_to<double>;
    { s.m_max() } -> std::convertible_to<double>;
    { s.tau_min() } -> std::convertible_to<double>;
    { s.tau_max() } -> std::convertible_to<double>;
    { s.sigma_min() } -> std::convertible_to<double>;
    { s.sigma_max() } -> std::convertible_to<double>;
    { s.rate_min() } -> std::convertible_to<double>;
    { s.rate_max() } -> std::convertible_to<double>;
};

}  // namespace mango
