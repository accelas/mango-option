// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/price_table_surface.hpp"
#include <array>
#include <memory>

namespace mango {

/// Adapter that wraps shared_ptr<PriceTableSurfaceND<N>> to satisfy
/// SurfaceInterpolant. Preserves shared ownership semantics.
template <size_t N>
class SharedBSplineInterp {
public:
    explicit SharedBSplineInterp(std::shared_ptr<const PriceTableSurfaceND<N>> surface)
        : surface_(std::move(surface)) {}

    [[nodiscard]] double eval(const std::array<double, N>& coords) const {
        return surface_->value(coords);
    }

    [[nodiscard]] double partial(size_t axis, const std::array<double, N>& coords) const {
        return surface_->partial(axis, coords);
    }

    /// Access underlying surface (for metadata, axes, etc.)
    [[nodiscard]] const PriceTableSurfaceND<N>& surface() const { return *surface_; }

private:
    std::shared_ptr<const PriceTableSurfaceND<N>> surface_;
};

}  // namespace mango
