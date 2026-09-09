// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"
#include "mango/option/table/price_table.hpp"
#include "mango/option/table/bspline/bspline_types.hpp"
#include "mango/option/table/transform_leaf.hpp"
#include "mango/option/table/eep/eep_layer.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/split_surface.hpp"
#include "mango/option/table/splits/tau_segment.hpp"
#include "mango/option/table/splits/multi_kref.hpp"
#include "mango/option/table/transforms/standard_4d.hpp"
#include "mango/option/table/shared_interp.hpp"
#include "mango/math/bspline/bspline_nd.hpp"
#include "mango/support/error_types.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <expected>
#include <memory>
#include <string>
#include <vector>

namespace mango {

/// Number of B-spline interpolation axes.
/// Currently 4: (log-moneyness, tau, sigma, r).
/// See issue #382 for planned 4D->3D reduction.
inline constexpr size_t kPriceTableDim = 4;

/// Metadata for N-dimensional price table axes
///
/// Stores grid points and optional axis names for each dimension.
/// All grids must be strictly monotonic increasing.
///
/// @tparam N Number of dimensions (axes)
template <size_t N>
struct PriceTableAxesND {
    std::array<std::vector<double>, N> grids;  ///< Grid points per axis
    std::array<std::string, N> names;          ///< Optional names (e.g., "moneyness", "maturity")

    /// Calculate total number of grid points (product of all axis sizes)
    [[nodiscard]] size_t total_points() const noexcept {
        size_t total = 1;
        for (size_t i = 0; i < N; ++i) {
            total *= grids[i].size();
        }
        return total;
    }

    /// Validate all grids are non-empty and strictly monotonic
    ///
    /// @return Empty expected on success, ValidationError on failure
    [[nodiscard]] std::expected<void, ValidationError> validate() const {
        for (size_t i = 0; i < N; ++i) {
            if (grids[i].empty()) {
                return std::unexpected(ValidationError(
                    ValidationErrorCode::InvalidGridSize,
                    0.0,
                    i));
            }

            // Check strict monotonicity
            for (size_t j = 1; j < grids[i].size(); ++j) {
                if (grids[i][j] <= grids[i][j-1]) {
                    return std::unexpected(ValidationError(
                        ValidationErrorCode::UnsortedGrid,
                        grids[i][j],
                        i));
                }
            }
        }
        return {};
    }

    /// Get shape (number of points per axis)
    [[nodiscard]] std::array<size_t, N> shape() const noexcept {
        std::array<size_t, N> s;
        for (size_t i = 0; i < N; ++i) {
            s[i] = grids[i].size();
        }
        return s;
    }
};

/// Convenience alias for the common 4D case.
using PriceTableAxes = PriceTableAxesND<kPriceTableDim>;

/// Standard and segmented financial B-spline tables.
using BSplinePriceTable = PriceTable<BSplineLeaf>;
using BSplineMultiKRefSurface = PriceTable<BSplineMultiKRefInner>;

/// Create a BSplinePriceTable from a pre-built EEP B-spline.
/// K_ref and dividend_yield are passed explicitly.
/// The spline must contain EEP data (built with eep_decompose).
///
/// Published bounds are derived from the spline's own knot span.  For a
/// surface built with support headroom beyond its measured/sample domain
/// (adaptive builds, spec D2), use the overload below with explicit bounds
/// instead -- deriving from knots here would publish the unmeasured
/// headroom band as queryable.
[[nodiscard]] std::expected<BSplinePriceTable, std::string>
make_bspline_surface(
    std::shared_ptr<const BSplineND<double, 4>> spline,
    double K_ref,
    double dividend_yield,
    OptionType type);

/// Same as above, but with the published `SurfaceBounds` supplied explicitly
/// instead of derived from the spline's knot span.
[[nodiscard]] std::expected<BSplinePriceTable, std::string>
make_bspline_surface(
    std::shared_ptr<const BSplineND<double, 4>> spline,
    double K_ref,
    double dividend_yield,
    OptionType type,
    const SurfaceBounds& bounds);

} // namespace mango
