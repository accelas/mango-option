// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>
#include <memory>

#include "mango/option/table/price_table_surface.hpp"
#include "mango/option/table/spliced_surface.hpp"

namespace mango {

/// Universal adapter wrapping PriceTableSurface<4> for SplicedInner concept.
/// Replaces AmericanPriceSurfaceAdapter. Used by both standard and segmented paths.
class PriceTableInner {
public:
    explicit PriceTableInner(std::shared_ptr<const PriceTableSurface<4>> surface)
        : surface_(std::move(surface)) {}

    [[nodiscard]] double price(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        return surface_->value({x, q.tau, q.sigma, q.rate});
    }

    [[nodiscard]] double vega(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        return surface_->partial(2, {x, q.tau, q.sigma, q.rate});
    }

    [[nodiscard]] const PriceTableSurface<4>& surface() const { return *surface_; }

private:
    std::shared_ptr<const PriceTableSurface<4>> surface_;
};

static_assert(SplicedInner<PriceTableInner>);

}  // namespace mango
