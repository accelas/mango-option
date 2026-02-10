// SPDX-License-Identifier: MIT
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/math/bspline_nd.hpp"
#include "mango/math/bspline_basis.hpp"
#include <algorithm>

namespace mango {

template <size_t N>
PriceTableSurfaceND<N>::PriceTableSurfaceND(
    PriceTableAxesND<N> axes,
    PriceTableMetadata metadata,
    std::unique_ptr<BSplineND<double, N>> spline)
    : axes_(std::move(axes))
    , meta_(std::move(metadata))
    , spline_(std::move(spline)) {}

template <size_t N>
std::expected<std::shared_ptr<const PriceTableSurfaceND<N>>, PriceTableError>
PriceTableSurfaceND<N>::build(
    PriceTableAxesND<N> axes,
    std::vector<double> coeffs,
    PriceTableMetadata metadata)
{
    // Validate axes
    if (auto valid = axes.validate(); !valid.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    // Store log-moneyness bounds in metadata
    // Axis 0 is log-moneyness ln(S/K)
    if constexpr (N >= 1) {
        if (!axes.grids[0].empty()) {
            metadata.m_min = axes.grids[0].front();
            metadata.m_max = axes.grids[0].back();
        }
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

    // Create BSplineND with log-moneyness grid
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

    auto surface = std::shared_ptr<const PriceTableSurfaceND<N>>(
        new PriceTableSurfaceND<N>(std::move(axes), std::move(metadata), std::move(spline)));

    return surface;
}

template <size_t N>
double PriceTableSurfaceND<N>::value(const std::array<double, N>& coords) const {
    return spline_->eval(coords);
}

template <size_t N>
double PriceTableSurfaceND<N>::partial(size_t axis, const std::array<double, N>& coords) const {
    return spline_->eval_partial(axis, coords);
}

template <size_t N>
double PriceTableSurfaceND<N>::second_partial(size_t axis, const std::array<double, N>& coords) const {
    return spline_->eval_second_partial(axis, coords);
}

// Explicit template instantiations
template class PriceTableSurfaceND<2>;
template class PriceTableSurfaceND<3>;
template class PriceTableSurfaceND<4>;
template class PriceTableSurfaceND<5>;

std::expected<BSplinePriceTable, std::string>
make_bspline_surface(
    std::shared_ptr<const PriceTableSurface> surface,
    OptionType type)
{
    if (!surface) {
        return std::unexpected(std::string("null surface"));
    }

    const auto& meta = surface->metadata();
    if (meta.content != SurfaceContent::EarlyExercisePremium) {
        return std::unexpected(std::string(
            "make_bspline_surface requires EEP content; got NormalizedPrice. "
            "Build with SurfaceContent::EarlyExercisePremium + EEPDecomposer, "
            "or use make_interpolated_iv_solver() which handles this internally."));
    }

    if (!meta.dividends.discrete_dividends.empty()) {
        return std::unexpected(std::string("discrete dividends not supported; use segmented path"));
    }

    if (meta.K_ref <= 0.0) {
        return std::unexpected(std::string("invalid K_ref"));
    }

    double K_ref = meta.K_ref;
    double dividend_yield = meta.dividends.dividend_yield;
    const auto& axes = surface->axes();

    SharedBSplineInterp<4> interp(surface);
    StandardTransform4D xform;
    AnalyticalEEP eep(type, dividend_yield);
    BSplineLeaf leaf(std::move(interp), xform, eep, K_ref);

    SurfaceBounds bounds{
        .m_min = meta.m_min,
        .m_max = meta.m_max,
        .tau_min = axes.grids[1].front(),
        .tau_max = axes.grids[1].back(),
        .sigma_min = axes.grids[2].front(),
        .sigma_max = axes.grids[2].back(),
        .rate_min = axes.grids[3].front(),
        .rate_max = axes.grids[3].back(),
    };

    return BSplinePriceTable(std::move(leaf), bounds, type, dividend_yield);
}

} // namespace mango
