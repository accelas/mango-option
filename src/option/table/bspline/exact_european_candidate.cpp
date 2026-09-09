// SPDX-License-Identifier: MIT
#include "mango/option/table/bspline/exact_european_candidate.hpp"
#include <cmath>
#include <limits>
#include <memory>

namespace mango::detail {

std::expected<std::optional<BSplineLeaf>, PriceTableError>
make_exact_european_bspline_candidate(
    BSplineND<double, 4>::GridArray grids,
    BSplineND<double, 4>::KnotArray knots,
    const SurfaceBounds& bounds, double reference_strike,
    OptionType option_type, double dividend_yield,
    const std::optional<FixedExpiryMetadata>& model) {
    if (option_type != OptionType::CALL || dividend_yield != 0.0 ||
        bounds.rate_min < 0.0 || (model && !model->discrete_dividends.empty())) {
        return std::optional<BSplineLeaf>{};
    }
    if (!std::isfinite(bounds.rate_min) || !std::isfinite(bounds.rate_max) ||
        bounds.rate_max < bounds.rate_min || !std::isfinite(reference_strike) ||
        reference_strike <= 0.0 || (model && !model->valid(bounds.tau_max))) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }
    size_t count = 1;
    for (const auto& grid : grids) {
        if (grid.empty() || count > std::numeric_limits<size_t>::max() / grid.size()) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
        }
        count *= grid.size();
    }
    auto spline = BSplineND<double, 4>::create(
        std::move(grids), std::move(knots), std::vector<double>(count, 0.0));
    if (!spline) return std::unexpected(convert_to_price_table_error(spline.error()));
    auto shared = std::make_shared<const BSplineND<double, 4>>(std::move(*spline));
    return std::optional<BSplineLeaf>{BSplineLeaf{
        BSplineTransformLeaf{SharedBSplineInterp<4>{std::move(shared)},
                             StandardTransform4D{}, reference_strike},
        AnalyticalEEP{option_type, dividend_yield}}};
}

}  // namespace mango::detail
