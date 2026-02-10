// SPDX-License-Identifier: MIT
#include "mango/option/table/standard_surface.hpp"

namespace mango {

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

}  // namespace mango
