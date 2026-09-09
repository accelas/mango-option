// SPDX-License-Identifier: MIT
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/math/bspline/bspline_nd.hpp"
#include "mango/math/bspline/bspline_basis.hpp"
#include <algorithm>

namespace mango {

std::expected<BSplinePriceTable, PriceTableError>
make_bspline_surface(
    std::shared_ptr<const BSplineND<double, 4>> spline,
    double K_ref,
    double dividend_yield,
    OptionType type)
{
    if (!spline) return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});

    SurfaceBounds bounds{
        .m_min = spline->grid(0).front(),
        .m_max = spline->grid(0).back(),
        .tau_min = spline->grid(1).front(),
        .tau_max = spline->grid(1).back(),
        .sigma_min = spline->grid(2).front(),
        .sigma_max = spline->grid(2).back(),
        .rate_min = spline->grid(3).front(),
        .rate_max = spline->grid(3).back(),
    };

    return make_bspline_surface(
        std::move(spline), K_ref, dividend_yield, type, bounds);
}

std::expected<BSplinePriceTable, PriceTableError>
make_bspline_surface(
    std::shared_ptr<const BSplineND<double, 4>> spline,
    double K_ref,
    double dividend_yield,
    OptionType type,
    const SurfaceBounds& bounds)
{
    if (!spline) return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    if (!std::isfinite(K_ref) || K_ref <= 0.0)
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});

    SharedBSplineInterp<4> interp(spline);
    StandardTransform4D xform;
    AnalyticalEEP eep(type, dividend_yield);
    BSplineTransformLeaf tleaf(std::move(interp), xform, K_ref);
    BSplineLeaf leaf(std::move(tleaf), eep);

    return BSplinePriceTable::create(leaf, bounds, type, dividend_yield);
}

} // namespace mango
