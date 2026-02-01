// SPDX-License-Identifier: MIT
#pragma once

#include <vector>
#include <expected>
#include "src/option/table/segmented_price_surface.hpp"
#include "src/option/table/price_surface_concept.hpp"
#include "src/support/error_types.hpp"

namespace mango {

class SegmentedMultiKRefSurface {
public:
    struct Entry {
        double K_ref;
        SegmentedPriceSurface surface;
    };

    static std::expected<SegmentedMultiKRefSurface, ValidationError> create(
        std::vector<Entry> entries);

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
    [[nodiscard]] OptionType option_type() const noexcept;
    [[nodiscard]] double dividend_yield() const noexcept;

private:
    SegmentedMultiKRefSurface() = default;

    size_t find_bracket(double strike) const;

    std::vector<Entry> entries_;  // sorted by K_ref

    // Cached bounds (intersection across all entries)
    double m_min_, m_max_;
    double tau_min_, tau_max_;
    double sigma_min_, sigma_max_;
    double rate_min_, rate_max_;
    OptionType option_type_ = OptionType::PUT;
    double dividend_yield_ = 0.0;
};

}  // namespace mango
