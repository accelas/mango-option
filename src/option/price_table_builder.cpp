#include "src/option/price_table_builder.hpp"
#include "src/option/recursion_helpers.hpp"

namespace mango {

template <size_t N>
PriceTableBuilder<N>::PriceTableBuilder(PriceTableConfig config)
    : config_(std::move(config)) {}

template <size_t N>
std::expected<std::shared_ptr<const PriceTableSurface<N>>, std::string>
PriceTableBuilder<N>::build(const PriceTableAxes<N>& axes) {
    // TODO: Implement pipeline
    return std::unexpected("PriceTableBuilder::build() not yet implemented");
}

template <size_t N>
std::vector<AmericanOptionParams>
PriceTableBuilder<N>::make_batch(const PriceTableAxes<N>& axes, double K_ref) const {
    if constexpr (N >= 4) {
        std::vector<AmericanOptionParams> batch;
        batch.reserve(axes.total_points());

        // Iterate over all grid point combinations
        for_each_axis_index<0>(axes, [&](const std::array<size_t, N>& indices) {
            double m = axes.grids[0][indices[0]];       // moneyness
            double tau = axes.grids[1][indices[1]];     // maturity
            double sigma = axes.grids[2][indices[2]];   // volatility
            double r = axes.grids[3][indices[3]];       // rate

            // Convert moneyness to spot: S = m * K_ref
            double spot = m * K_ref;

            // Use constructor: (spot, strike, maturity, rate, dividend_yield, type, volatility, discrete_dividends)
            AmericanOptionParams params(
                spot,
                K_ref,
                tau,
                r,
                config_.dividend_yield,
                config_.option_type,
                sigma,
                config_.discrete_dividends
            );

            batch.push_back(params);
        });

        return batch;
    } else {
        // For N < 4, return empty batch (should not be called in practice)
        return {};
    }
}

// Explicit instantiations
template class PriceTableBuilder<2>;
template class PriceTableBuilder<3>;
template class PriceTableBuilder<4>;
template class PriceTableBuilder<5>;

} // namespace mango
