// SPDX-License-Identifier: MIT
#pragma once

#include <vector>
#include <expected>
#include "mango/option/table/american_price_surface.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/support/error_types.hpp"

namespace mango {

class SegmentedPriceSurface {
public:
    struct Segment {
        AmericanPriceSurface surface;
        double tau_start;  // global τ start (inclusive for all except first)
        double tau_end;    // global τ end (inclusive)
    };

    struct Config {
        std::vector<Segment> segments;  // ordered: last segment first (index 0 has lowest τ)
        std::vector<Dividend> discrete_dividends;
        double K_ref;
        double T;  // expiry in calendar time
    };

    static std::expected<SegmentedPriceSurface, ValidationError> create(Config config);

    [[nodiscard]] double price(double spot, double strike,
                               double tau, double sigma, double rate) const;
    [[nodiscard]] double vega(double spot, double strike,
                              double tau, double sigma, double rate) const;

    [[nodiscard]] double m_min() const noexcept;
    [[nodiscard]] double m_max() const noexcept;
    [[nodiscard]] double tau_min() const noexcept;
    [[nodiscard]] double tau_max() const noexcept;
    [[nodiscard]] double sigma_min() const noexcept;
    [[nodiscard]] double sigma_max() const noexcept;
    [[nodiscard]] double rate_min() const noexcept;
    [[nodiscard]] double rate_max() const noexcept;
    [[nodiscard]] double K_ref() const noexcept { return K_ref_; }
    [[nodiscard]] double T() const noexcept { return T_; }
    [[nodiscard]] OptionType option_type() const noexcept;
    [[nodiscard]] double dividend_yield() const noexcept;

    /// Access segments for migration to new spliced surface types
    [[nodiscard]] std::vector<Segment>& segments() noexcept { return segments_; }
    [[nodiscard]] const std::vector<Segment>& segments() const noexcept { return segments_; }
    [[nodiscard]] const std::vector<Dividend>& dividends() const noexcept { return dividends_; }

private:
    SegmentedPriceSurface() = default;

    const Segment& find_segment(double tau) const;
    double compute_spot_adjustment(double spot, double t_query, double t_boundary) const;

    std::vector<Segment> segments_;
    std::vector<Dividend> dividends_;
    double K_ref_;
    double T_;
    OptionType option_type_ = OptionType::PUT;
    double dividend_yield_ = 0.0;
};

}  // namespace mango
