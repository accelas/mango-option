#include "src/option/price_table_builder.hpp"

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

// Explicit instantiations
template class PriceTableBuilder<2>;
template class PriceTableBuilder<3>;
template class PriceTableBuilder<4>;
template class PriceTableBuilder<5>;

} // namespace mango
