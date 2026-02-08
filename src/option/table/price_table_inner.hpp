// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>
#include <memory>

#include "mango/option/table/price_table_surface.hpp"
#include "mango/option/table/spliced_surface.hpp"

namespace mango {

/// Universal adapter wrapping PriceTableSurface for SplicedInner concept.
/// Used by segmented paths (NormalizedPrice content; no EEP reconstruction).
class PriceTableInner {
public:
    explicit PriceTableInner(std::shared_ptr<const PriceTableSurface> surface)
        : surface_(std::move(surface)) {}

    [[nodiscard]] double price(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        return surface_->value({x, q.tau, q.sigma, q.rate});
    }

    [[nodiscard]] double vega(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        return surface_->partial(2, {x, q.tau, q.sigma, q.rate});
    }

    [[nodiscard]] const PriceTableSurface& surface() const { return *surface_; }

private:
    std::shared_ptr<const PriceTableSurface> surface_;
};

static_assert(SplicedInner<PriceTableInner>);

}  // namespace mango
