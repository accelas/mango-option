// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/price_table_axes.hpp"
#include "mango/option/table/price_table_metadata.hpp"
#include "mango/math/bspline_nd.hpp"
#include "mango/support/error_types.hpp"
#include <memory>
#include <expected>

namespace mango {

/// Immutable N-dimensional price surface with B-spline interpolation
///
/// Provides fast interpolation queries and partial derivatives.
/// Thread-safe after construction.
///
/// @tparam N Number of dimensions
template <size_t N>
class PriceTableSurfaceND {
public:
    /// Build surface from axes, coefficients, and metadata
    ///
    /// @param axes Grid points and names for each dimension
    /// @param coeffs Flattened B-spline coefficients (row-major)
    /// @param metadata Reference strike, dividends, etc.
    /// @return Shared pointer to surface or error
    [[nodiscard]] static std::expected<std::shared_ptr<const PriceTableSurfaceND<N>>, PriceTableError>
    build(PriceTableAxesND<N> axes, std::vector<double> coeffs, PriceTableMetadata metadata);

    /// Access axes
    [[nodiscard]] const PriceTableAxesND<N>& axes() const noexcept { return axes_; }

    /// Access metadata
    [[nodiscard]] const PriceTableMetadata& metadata() const noexcept { return meta_; }

    /// Access B-spline coefficients (for serialization)
    [[nodiscard]] const std::vector<double>& coefficients() const noexcept {
        return spline_->coefficients();
    }

    /// Evaluate price at query point
    ///
    /// Queries outside grid bounds are clamped to boundary values.
    /// For accurate results, ensure query points lie within grid bounds.
    ///
    /// @param coords N-dimensional coordinates (axis 0 = log-moneyness)
    /// @return Interpolated value (clamped at boundaries)
    [[nodiscard]] double value(const std::array<double, N>& coords) const;

    /// Partial derivative along specified axis
    ///
    /// @param axis Axis index (0 to N-1)
    /// @param coords N-dimensional coordinates (axis 0 = log-moneyness)
    /// @return Partial derivative estimate
    [[nodiscard]] double partial(size_t axis, const std::array<double, N>& coords) const;

    /// Second partial derivative along specified axis
    ///
    /// @param axis Axis index (0 to N-1)
    /// @param coords N-dimensional coordinates (axis 0 = log-moneyness)
    /// @return Second partial derivative estimate
    [[nodiscard]] double second_partial(size_t axis, const std::array<double, N>& coords) const;

private:
    PriceTableSurfaceND(PriceTableAxesND<N> axes, PriceTableMetadata metadata,
                     std::unique_ptr<BSplineND<double, N>> spline);

    PriceTableAxesND<N> axes_;
    PriceTableMetadata meta_;
    std::unique_ptr<BSplineND<double, N>> spline_;
};

/// Convenience alias for the common 4D case.
using PriceTableSurface = PriceTableSurfaceND<kPriceTableDim>;

/// 3D surface for dimensionless coordinates (x, τ', ln κ).
using DimensionlessPriceSurface = PriceTableSurfaceND<3>;

} // namespace mango
