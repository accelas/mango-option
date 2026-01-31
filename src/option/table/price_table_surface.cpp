// SPDX-License-Identifier: MIT
#include "src/option/table/price_table_surface.hpp"
#include "src/math/bspline_nd.hpp"
#include "src/math/bspline_basis.hpp"
#include <cmath>

namespace mango {

template <size_t N>
PriceTableSurface<N>::PriceTableSurface(
    PriceTableAxes<N> axes,
    PriceTableMetadata metadata,
    std::unique_ptr<BSplineND<double, N>> spline)
    : axes_(std::move(axes))
    , meta_(std::move(metadata))
    , spline_(std::move(spline)) {}

template <size_t N>
std::expected<std::shared_ptr<const PriceTableSurface<N>>, PriceTableError>
PriceTableSurface<N>::build(
    PriceTableAxes<N> axes,
    std::vector<double> coeffs,
    PriceTableMetadata metadata)
{
    // Validate axes
    if (auto valid = axes.validate(); !valid.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    // Store original moneyness bounds in metadata before transforming
    // Axis 0 is moneyness (m = S/K), which we transform to log-moneyness
    if constexpr (N >= 1) {
        if (!axes.grids[0].empty()) {
            metadata.m_min = axes.grids[0].front();
            metadata.m_max = axes.grids[0].back();
        }
    }

    // Transform axis 0 from moneyness to log-moneyness for better B-spline interpolation
    // This provides symmetric interpolation around ATM (log(1) = 0) and
    // reduces interpolation error by ~20-40% at the tails
    PriceTableAxes<N> internal_axes = axes;
    if constexpr (N >= 1) {
        for (double& m : internal_axes.grids[0]) {
            m = std::log(m);  // m → ln(m)
        }
    }

    // Check coefficient size matches axes
    size_t expected_size = internal_axes.total_points();
    if (coeffs.size() != expected_size) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::FittingFailed, 0, coeffs.size()});
    }

    // Create knot sequences for clamped cubic B-splines
    typename BSplineND<double, N>::KnotArray knots;
    for (size_t dim = 0; dim < N; ++dim) {
        knots[dim] = clamped_knots_cubic(internal_axes.grids[dim]);
    }

    // Create BSplineND with log-moneyness grid
    typename BSplineND<double, N>::GridArray grids_copy;
    for (size_t dim = 0; dim < N; ++dim) {
        grids_copy[dim] = internal_axes.grids[dim];
    }

    auto spline_result = BSplineND<double, N>::create(
        std::move(grids_copy),
        std::move(knots),
        std::move(coeffs));

    if (!spline_result.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::SurfaceBuildFailed});
    }

    auto spline = std::make_unique<BSplineND<double, N>>(std::move(spline_result.value()));

    // Store internal_axes (with log-moneyness) but keep original axes for external access
    // Note: axes() will return log-moneyness grid; use metadata for original bounds
    auto surface = std::shared_ptr<const PriceTableSurface<N>>(
        new PriceTableSurface<N>(std::move(internal_axes), std::move(metadata), std::move(spline)));

    return surface;
}

template <size_t N>
double PriceTableSurface<N>::value(const std::array<double, N>& coords) const {
    // Transform axis 0 from moneyness to log-moneyness
    // User queries in moneyness (m = S/K), internal storage uses log(m)
    std::array<double, N> internal_coords = coords;
    if constexpr (N >= 1) {
        internal_coords[0] = std::log(coords[0]);
    }
    return spline_->eval(internal_coords);
}

template <size_t N>
double PriceTableSurface<N>::partial(size_t axis, const std::array<double, N>& coords) const {
    // Transform axis 0 from moneyness to log-moneyness
    std::array<double, N> internal_coords = coords;
    if constexpr (N >= 1) {
        internal_coords[0] = std::log(coords[0]);
    }

    // Use analytic B-spline derivative
    double raw_partial = spline_->eval_partial(axis, internal_coords);

    // Chain rule for moneyness axis: ∂f/∂m = (∂f/∂x) * (dx/dm) = (∂f/∂x) / m
    if constexpr (N >= 1) {
        if (axis == 0) {
            return raw_partial / coords[0];
        }
    }
    return raw_partial;
}

template <size_t N>
double PriceTableSurface<N>::second_partial(size_t axis, const std::array<double, N>& coords) const {
    // Transform axis 0 from moneyness to log-moneyness
    std::array<double, N> internal_coords = coords;
    if constexpr (N >= 1) {
        internal_coords[0] = std::log(coords[0]);
    }

    // Chain rule for moneyness axis:
    // ∂²f/∂m² = (g''(x) - g'(x)) / m²  where x = ln(m)
    if constexpr (N >= 1) {
        if (axis == 0) {
            double g_prime = spline_->eval_partial(0, internal_coords);
            double g_double_prime = spline_->eval_second_partial(0, internal_coords);
            double m = coords[0];
            return (g_double_prime - g_prime) / (m * m);
        }
    }

    return spline_->eval_second_partial(axis, internal_coords);
}

// Explicit template instantiations
template class PriceTableSurface<2>;
template class PriceTableSurface<3>;
template class PriceTableSurface<4>;
template class PriceTableSurface<5>;

} // namespace mango
