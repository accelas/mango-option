#include "src/option/table/price_table_surface.hpp"
#include "src/math/bspline_nd.hpp"
#include "src/math/bspline_basis.hpp"

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

    // Check coefficient size matches axes
    size_t expected_size = axes.total_points();
    if (coeffs.size() != expected_size) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::FittingFailed, 0, coeffs.size()});
    }

    // Create knot sequences for clamped cubic B-splines
    typename BSplineND<double, N>::KnotArray knots;
    for (size_t dim = 0; dim < N; ++dim) {
        knots[dim] = clamped_knots_cubic(axes.grids[dim]);
    }

    // Create BSplineND
    typename BSplineND<double, N>::GridArray grids_copy;
    for (size_t dim = 0; dim < N; ++dim) {
        grids_copy[dim] = axes.grids[dim];
    }

    auto spline_result = BSplineND<double, N>::create(
        std::move(grids_copy),
        std::move(knots),
        std::move(coeffs));

    if (!spline_result.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::SurfaceBuildFailed});
    }

    auto spline = std::make_unique<BSplineND<double, N>>(std::move(spline_result.value()));

    auto surface = std::shared_ptr<const PriceTableSurface<N>>(
        new PriceTableSurface<N>(std::move(axes), std::move(metadata), std::move(spline)));

    return surface;
}

template <size_t N>
double PriceTableSurface<N>::value(const std::array<double, N>& coords) const {
    return spline_->eval(coords);
}

template <size_t N>
double PriceTableSurface<N>::partial(size_t axis, const std::array<double, N>& coords) const {
    // Use analytic B-spline derivative (single evaluation, no finite difference noise)
    return spline_->eval_partial(axis, coords);
}

// Explicit template instantiations
template class PriceTableSurface<2>;
template class PriceTableSurface<3>;
template class PriceTableSurface<4>;
template class PriceTableSurface<5>;

} // namespace mango
