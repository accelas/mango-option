#pragma once

#include "src/option/price_table_axes.hpp"
#include "src/option/price_table_metadata.hpp"
#include "src/math/bspline_nd.hpp"
#include <memory>
#include <expected>
#include <string>

namespace mango {

/// Immutable N-dimensional price surface with B-spline interpolation
///
/// Provides fast interpolation queries and partial derivatives.
/// Thread-safe after construction.
///
/// @tparam N Number of dimensions
template <size_t N>
class PriceTableSurface {
public:
    /// Build surface from axes, coefficients, and metadata
    ///
    /// @param axes Grid points and names for each dimension
    /// @param coeffs Flattened B-spline coefficients (row-major)
    /// @param metadata Reference strike, dividends, etc.
    /// @return Shared pointer to surface or error message
    [[nodiscard]] static std::expected<std::shared_ptr<const PriceTableSurface<N>>, std::string>
    build(PriceTableAxes<N> axes, std::vector<double> coeffs, PriceTableMetadata metadata);

    /// Access axes
    [[nodiscard]] const PriceTableAxes<N>& axes() const noexcept { return axes_; }

    /// Access metadata
    [[nodiscard]] const PriceTableMetadata& metadata() const noexcept { return meta_; }

    /// Evaluate price at query point
    ///
    /// Queries outside grid bounds are clamped to boundary values.
    /// For accurate results, ensure query points lie within grid bounds.
    ///
    /// @param coords N-dimensional coordinates
    /// @return Interpolated value (clamped at boundaries)
    [[nodiscard]] double value(const std::array<double, N>& coords) const;

    /// Partial derivative along specified axis
    ///
    /// @param axis Axis index (0 to N-1)
    /// @param coords N-dimensional coordinates
    /// @return Partial derivative estimate
    [[nodiscard]] double partial(size_t axis, const std::array<double, N>& coords) const;

private:
    PriceTableSurface(PriceTableAxes<N> axes, PriceTableMetadata metadata,
                     std::unique_ptr<BSplineND<double, N>> spline);

    PriceTableAxes<N> axes_;
    PriceTableMetadata meta_;
    std::unique_ptr<BSplineND<double, N>> spline_;
};

} // namespace mango
