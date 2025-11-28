#include "src/option/table/adaptive_grid_builder.hpp"
#include "src/math/black_scholes_analytics.hpp"
#include "src/math/latin_hypercube.hpp"
#include <algorithm>
#include <cmath>

namespace mango {

AdaptiveGridBuilder::AdaptiveGridBuilder(AdaptiveGridParams params)
    : params_(std::move(params))
{}

std::expected<AdaptiveResult, PriceTableError>
AdaptiveGridBuilder::build(const OptionChain& chain,
                           GridSpec<double> grid_spec,
                           size_t n_time,
                           OptionType type)
{
    // TODO: Implement main loop in Task 8
    return std::unexpected(PriceTableError(
        PriceTableErrorCode::InvalidConfig
    ));
}

double AdaptiveGridBuilder::compute_error_metric(
    double interpolated_price, double reference_price,
    double spot, double strike, double tau, double sigma, double rate,
    double dividend_yield) const
{
    double price_error = std::abs(interpolated_price - reference_price);
    double vega = bs_vega(spot, strike, tau, sigma, rate, dividend_yield);

    if (vega >= params_.vega_floor) {
        return price_error / vega;
    } else {
        // Fallback: treat vega_floor as minimum vega
        return price_error / params_.vega_floor;
    }
}

std::vector<double> AdaptiveGridBuilder::refine_dimension(
    const std::vector<double>& current_grid,
    const std::vector<size_t>& problematic_bins,
    size_t dim) const
{
    // TODO: Implement refinement in Task 8
    return current_grid;
}

}  // namespace mango
