// SPDX-License-Identifier: MIT
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/math/bspline/bspline_nd.hpp"
#include "mango/math/bspline/bspline_basis.hpp"
#include <algorithm>

namespace mango {

std::expected<BSplinePriceTable, std::string>
make_bspline_surface(
    std::shared_ptr<const BSplineND<double, 4>> spline,
    double K_ref,
    double dividend_yield,
    OptionType type)
{
    if (!spline) return std::unexpected(std::string("null spline"));
    if (K_ref <= 0.0) return std::unexpected(std::string("invalid K_ref"));

    SharedBSplineInterp<4> interp(spline);
    StandardTransform4D xform;
    AnalyticalEEP eep(type, dividend_yield);
    BSplineTransformLeaf tleaf(std::move(interp), xform, K_ref);
    BSplineLeaf leaf(std::move(tleaf), eep);

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

    return BSplinePriceTable(std::move(leaf), bounds, type, dividend_yield);
}

} // namespace mango
