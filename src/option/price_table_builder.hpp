#pragma once

#include "src/option/price_table_config.hpp"
#include "src/option/price_table_axes.hpp"
#include "src/option/price_table_surface.hpp"
#include "src/option/american_option.hpp"
#include <expected>
#include <string>

namespace mango {

/// Builder for N-dimensional price table surfaces
///
/// Orchestrates PDE solves across grid points, fits B-spline coefficients,
/// and constructs immutable PriceTableSurface.
///
/// @tparam N Number of dimensions
template <size_t N>
class PriceTableBuilder {
public:
    /// Construct builder with configuration
    explicit PriceTableBuilder(PriceTableConfig config);

    /// Build price table surface
    ///
    /// @param axes Grid points for each dimension
    /// @return Immutable surface or error message
    [[nodiscard]] std::expected<std::shared_ptr<const PriceTableSurface<N>>, std::string>
    build(const PriceTableAxes<N>& axes);

private:
    PriceTableConfig config_;
};

} // namespace mango
