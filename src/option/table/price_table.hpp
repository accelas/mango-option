// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"

namespace mango {

/// Bounds metadata for a price surface.
struct SurfaceBounds {
    double m_min, m_max;
    double tau_min, tau_max;
    double sigma_min, sigma_max;
    double rate_min, rate_max;
};

/// Top-level queryable price surface with runtime metadata.
/// Used directly by InterpolatedIVSolver.
template <typename Inner>
class PriceTable {
public:
    PriceTable(Inner inner, SurfaceBounds bounds,
                   OptionType option_type, double dividend_yield)
        : inner_(std::move(inner))
        , bounds_(bounds)
        , option_type_(option_type)
        , dividend_yield_(dividend_yield)
    {}

    [[nodiscard]] double price(double spot, double strike,
                                double tau, double sigma, double rate) const {
        return inner_.price(spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] double vega(double spot, double strike,
                               double tau, double sigma, double rate) const {
        return inner_.vega(spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] double m_min() const noexcept { return bounds_.m_min; }
    [[nodiscard]] double m_max() const noexcept { return bounds_.m_max; }
    [[nodiscard]] double tau_min() const noexcept { return bounds_.tau_min; }
    [[nodiscard]] double tau_max() const noexcept { return bounds_.tau_max; }
    [[nodiscard]] double sigma_min() const noexcept { return bounds_.sigma_min; }
    [[nodiscard]] double sigma_max() const noexcept { return bounds_.sigma_max; }
    [[nodiscard]] double rate_min() const noexcept { return bounds_.rate_min; }
    [[nodiscard]] double rate_max() const noexcept { return bounds_.rate_max; }
    [[nodiscard]] OptionType option_type() const noexcept { return option_type_; }
    [[nodiscard]] double dividend_yield() const noexcept { return dividend_yield_; }

    [[nodiscard]] const Inner& inner() const noexcept { return inner_; }

private:
    Inner inner_;
    SurfaceBounds bounds_;
    OptionType option_type_;
    double dividend_yield_;
};

}  // namespace mango
