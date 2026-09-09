// SPDX-License-Identifier: MIT
#pragma once

#include "mango/math/bspline/bspline_basis.hpp"
#include "mango/option/table/bspline/bspline_types.hpp"
#include "mango/option/table/chebyshev/chebyshev_table_builder.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"

namespace mango::test {

// Persistence fixtures use exact constant polynomials on the declared grids.
// These are raw numeric leaves; callers must still use PriceTable::create and
// obtain an actual certificate before exercising financial publication.
template <std::size_t N>
BSplineND<double, N> constant_spline(
    const std::array<std::vector<double>, N>& grids, double value) {
    typename BSplineND<double, N>::KnotArray knots;
    std::size_t size = 1;
    for (std::size_t axis = 0; axis < N; ++axis) {
        knots[axis] = clamped_knots_cubic(grids[axis]);
        size *= grids[axis].size();
    }
    return BSplineND<double, N>::create(grids, std::move(knots),
        std::vector<double>(size, value)).value();
}

template <std::size_t N>
std::shared_ptr<const BSplineND<double, N>> zero_eep(
    const std::shared_ptr<const BSplineND<double, N>>& geometry) {
    typename BSplineND<double, N>::GridArray grids;
    typename BSplineND<double, N>::KnotArray knots;
    for (std::size_t axis = 0; axis < N; ++axis) {
        grids[axis] = geometry->grid(axis);
        knots[axis] = geometry->knots(axis);
    }
    auto spline = BSplineND<double, N>::create(std::move(grids), std::move(knots),
        std::vector<double>(geometry->coefficients().size(), 0.0)).value();
    return std::make_shared<const BSplineND<double, N>>(std::move(spline));
}

inline std::expected<ChebyshevSurface, PriceTableError>
chebyshev_table(const ChebyshevTableConfig& config) {
    std::size_t size = 1;
    for (auto n : config.num_pts) size *= n;
    auto polynomial = ChebyshevModalInterpolant<4>::build_from_coefficients(
        std::vector<double>(size, 0.0), config.domain, config.num_pts).value();
    const auto& lo = config.domain.lo;
    const auto& hi = config.domain.hi;
    SurfaceBounds bounds{lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], lo[3], hi[3]};
    bounds.ratio_bounds = config.ratio_bounds;
    return ChebyshevSurface::create(
        ChebyshevLeaf(ChebyshevTransformLeaf(std::move(polynomial), {}, config.K_ref),
            AnalyticalEEP(config.option_type, config.dividend_yield)),
        bounds, config.option_type, config.dividend_yield);
}

inline std::expected<ChebyshevMultiKRefSurface, PriceTableError>
chebyshev_segments(const SegmentedAdaptiveConfig& config, const IVGrid& grid) {
    auto bounds = requested_segmented_domain(grid, config.maturity, config.ratio_bounds).value();
    bounds.strike_bounds = config.strike_bounds;
    const auto support = expand_segmented_domain(grid, config.maturity, config.dividend_yield,
        config.discrete_dividends, config.kref_config.K_refs.front()).value();
    const auto segments = compute_segment_boundaries(config.discrete_dividends, config.maturity,
        support.tau_min, support.tau_max);
    std::vector<ChebyshevTauSegmented> references;
    for (double strike : config.kref_config.K_refs) {
        auto split = make_tau_split_from_segments(segments.bounds, segments.is_gap, strike);
        std::vector<ChebyshevSegmentedLeaf> leaves;
        for (std::size_t i = 0; i < split.tau_start().size(); ++i) {
            Domain<4> domain{{support.m_min, split.tau_min()[i], support.sigma_min, support.rate_min},
                             {support.m_max, split.tau_max()[i], support.sigma_max, support.rate_max}};
            std::vector<double> coefficients(16, 0.0);
            coefficients[0] = .1;
            auto polynomial = ChebyshevModalInterpolant<4>::build_from_coefficients(
                coefficients, domain, {2, 2, 2, 2}).value();
            leaves.emplace_back(std::move(polynomial), StandardTransform4D{}, strike);
        }
        references.emplace_back(std::move(leaves), std::move(split));
    }
    return ChebyshevMultiKRefSurface::create(
        ChebyshevMultiKRefInner(std::move(references), MultiKRefSplit(config.kref_config.K_refs)),
        bounds, config.option_type, config.dividend_yield,
        make_fixed_expiry_metadata(config.maturity, config.discrete_dividends));
}

// Keep the existing builder's segment routing, grid nodes, knots and reference
// strike while replacing its PDE fit by a known constant price polynomial.
inline BSplineSegmentedSurface constant_segments(const BSplineSegmentedSurface& geometry) {
    std::vector<BSplineSegmentedLeaf> pieces;
    for (const auto& leaf : geometry.pieces()) {
        const auto& original = leaf.interpolant().get();
        BSplineND<double, 4>::GridArray grids;
        BSplineND<double, 4>::KnotArray knots;
        for (std::size_t axis = 0; axis < 4; ++axis) {
            grids[axis] = original.grid(axis);
            knots[axis] = original.knots(axis);
        }
        auto spline = BSplineND<double, 4>::create(std::move(grids), std::move(knots),
            std::vector<double>(original.coefficients().size(), .1)).value();
        pieces.emplace_back(SharedBSplineInterp<4>(
            std::make_shared<const BSplineND<double, 4>>(std::move(spline))),
            StandardTransform4D{}, leaf.K_ref());
    }
    return BSplineSegmentedSurface(std::move(pieces), geometry.split());
}
} // namespace mango::test
