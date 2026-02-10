// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/price_table_axes.hpp"
#include "mango/option/table/price_table_metadata.hpp"
#include "mango/option/table/price_table.hpp"
#include "mango/option/table/eep_surface_adapter.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/eep/identity_eep.hpp"
#include "mango/option/table/split_surface.hpp"
#include "mango/option/table/splits/tau_segment.hpp"
#include "mango/option/table/splits/multi_kref.hpp"
#include "mango/option/table/transforms/standard_4d.hpp"
#include "mango/math/bspline_nd.hpp"
#include "mango/support/error_types.hpp"
#include <array>
#include <expected>
#include <memory>
#include <string>

namespace mango {

/// Immutable N-dimensional price surface with B-spline interpolation
///
/// Provides fast interpolation queries and partial derivatives.
/// Thread-safe after construction.
///
/// @tparam N Number of dimensions
template <size_t N>
class PriceTableSurfaceND {
public:
    /// Build surface from axes, coefficients, and metadata
    ///
    /// @param axes Grid points and names for each dimension
    /// @param coeffs Flattened B-spline coefficients (row-major)
    /// @param metadata Reference strike, dividends, etc.
    /// @return Shared pointer to surface or error
    [[nodiscard]] static std::expected<std::shared_ptr<const PriceTableSurfaceND<N>>, PriceTableError>
    build(PriceTableAxesND<N> axes, std::vector<double> coeffs, PriceTableMetadata metadata);

    /// Access axes
    [[nodiscard]] const PriceTableAxesND<N>& axes() const noexcept { return axes_; }

    /// Access metadata
    [[nodiscard]] const PriceTableMetadata& metadata() const noexcept { return meta_; }

    /// Evaluate price at query point
    ///
    /// Queries outside grid bounds are clamped to boundary values.
    /// For accurate results, ensure query points lie within grid bounds.
    ///
    /// @param coords N-dimensional coordinates (axis 0 = log-moneyness)
    /// @return Interpolated value (clamped at boundaries)
    [[nodiscard]] double value(const std::array<double, N>& coords) const;

    /// Partial derivative along specified axis
    ///
    /// @param axis Axis index (0 to N-1)
    /// @param coords N-dimensional coordinates (axis 0 = log-moneyness)
    /// @return Partial derivative estimate
    [[nodiscard]] double partial(size_t axis, const std::array<double, N>& coords) const;

    /// Second partial derivative along specified axis
    ///
    /// @param axis Axis index (0 to N-1)
    /// @param coords N-dimensional coordinates (axis 0 = log-moneyness)
    /// @return Second partial derivative estimate
    [[nodiscard]] double second_partial(size_t axis, const std::array<double, N>& coords) const;

private:
    PriceTableSurfaceND(PriceTableAxesND<N> axes, PriceTableMetadata metadata,
                     std::unique_ptr<BSplineND<double, N>> spline);

    PriceTableAxesND<N> axes_;
    PriceTableMetadata meta_;
    std::unique_ptr<BSplineND<double, N>> spline_;
};

/// Convenience alias for the common 4D case.
using PriceTableSurface = PriceTableSurfaceND<kPriceTableDim>;

/// Adapter that wraps shared_ptr<PriceTableSurfaceND<N>> to satisfy
/// SurfaceInterpolant. Preserves shared ownership semantics.
template <size_t N>
class SharedBSplineInterp {
public:
    explicit SharedBSplineInterp(std::shared_ptr<const PriceTableSurfaceND<N>> surface)
        : surface_(std::move(surface)) {}

    [[nodiscard]] double eval(const std::array<double, N>& coords) const {
        return surface_->value(coords);
    }

    [[nodiscard]] double partial(size_t axis, const std::array<double, N>& coords) const {
        return surface_->partial(axis, coords);
    }

    /// Access underlying surface (for metadata, axes, etc.)
    [[nodiscard]] const PriceTableSurfaceND<N>& surface() const { return *surface_; }

private:
    std::shared_ptr<const PriceTableSurfaceND<N>> surface_;
};

// ===========================================================================
// B-spline type aliases — concept-based layered architecture
// ===========================================================================

/// Leaf adapter for standard (EEP) surfaces
using BSplineLeaf = EEPSurfaceAdapter<SharedBSplineInterp<4>,
                                        StandardTransform4D, AnalyticalEEP>;

/// Standard surface (satisfies PriceSurface concept)
using BSplinePriceTable = PriceTable<BSplineLeaf>;

/// Leaf adapter for segmented surfaces (no EEP decomposition)
using BSplineSegmentedLeaf = EEPSurfaceAdapter<SharedBSplineInterp<4>,
                                         StandardTransform4D, IdentityEEP>;

/// Tau-segmented surface
using BSplineSegmentedSurface = SplitSurface<BSplineSegmentedLeaf, TauSegmentSplit>;

/// Multi-K_ref surface (outer split over K_refs of segmented inner)
using BSplineMultiKRefInner = SplitSurface<BSplineSegmentedSurface, MultiKRefSplit>;

/// Multi-K_ref surface (satisfies PriceSurface concept)
using BSplineMultiKRefSurface = PriceTable<BSplineMultiKRefInner>;


/// Create a BSplinePriceTable from a pre-built EEP surface.
/// Reads K_ref and dividend_yield from surface metadata.
/// Requires SurfaceContent::EarlyExercisePremium; rejects NormalizedPrice.
[[nodiscard]] std::expected<BSplinePriceTable, std::string>
make_bspline_surface(
    std::shared_ptr<const PriceTableSurface> surface,
    OptionType type);

} // namespace mango
