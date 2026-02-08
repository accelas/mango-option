// SPDX-License-Identifier: MIT
#include "mango/option/table/golden_surface.hpp"
#include "mango/option/table/price_table_surface.hpp"

namespace mango {

namespace {
#include "golden_surface_data.inc"

SegmentedDimensionlessSurface::Segment build_segment(
    const double* x, size_t nx,
    const double* tp, size_t ntp,
    const double* lk, size_t nlk,
    const double* coeffs, size_t nc,
    double lk_min, double lk_max)
{
    PriceTableAxesND<3> axes;
    axes.grids[0].assign(x, x + nx);
    axes.grids[1].assign(tp, tp + ntp);
    axes.grids[2].assign(lk, lk + nlk);
    axes.names = {"log_moneyness", "tau_prime", "ln_kappa"};

    PriceTableMetadata meta{
        .K_ref = 100.0,
        .dividends = {},
        .m_min = axes.grids[0].front(),
        .m_max = axes.grids[0].back(),
        .content = SurfaceContent::EarlyExercisePremium,
    };

    auto surface = PriceTableSurfaceND<3>::build(
        std::move(axes),
        std::vector<double>(coeffs, coeffs + nc),
        meta).value();

    return {.surface = std::move(surface), .lk_min = lk_min, .lk_max = lk_max};
}

}  // namespace

std::shared_ptr<const SegmentedDimensionlessSurface>
golden_dimensionless_surface()
{
    static auto surface = [] {
        std::vector<SegmentedDimensionlessSurface::Segment> segments;
        segments.reserve(kNumSegments);

        segments.push_back(build_segment(
            kSeg0X, std::size(kSeg0X),
            kSeg0Tp, std::size(kSeg0Tp),
            kSeg0Lk, std::size(kSeg0Lk),
            kSeg0Coeffs, std::size(kSeg0Coeffs),
            kSeg0LkMin, kSeg0LkMax));

        segments.push_back(build_segment(
            kSeg1X, std::size(kSeg1X),
            kSeg1Tp, std::size(kSeg1Tp),
            kSeg1Lk, std::size(kSeg1Lk),
            kSeg1Coeffs, std::size(kSeg1Coeffs),
            kSeg1LkMin, kSeg1LkMax));

        segments.push_back(build_segment(
            kSeg2X, std::size(kSeg2X),
            kSeg2Tp, std::size(kSeg2Tp),
            kSeg2Lk, std::size(kSeg2Lk),
            kSeg2Coeffs, std::size(kSeg2Coeffs),
            kSeg2LkMin, kSeg2LkMax));

        return std::make_shared<SegmentedDimensionlessSurface>(std::move(segments));
    }();
    return surface;
}

}  // namespace mango
