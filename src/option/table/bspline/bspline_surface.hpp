// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"
#include "mango/option/table/price_table.hpp"
#include "mango/option/table/eep/eep_surface_adapter.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/eep/identity_eep.hpp"
#include "mango/option/table/split_surface.hpp"
#include "mango/option/table/splits/tau_segment.hpp"
#include "mango/option/table/splits/multi_kref.hpp"
#include "mango/option/table/transforms/standard_4d.hpp"
#include "mango/math/bspline_nd.hpp"
#include "mango/math/safe_math.hpp"
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
    ///
    /// Uses safe multiplication with overflow detection via __int128.
    /// Returns 0 on overflow (callers should validate grids first).
    [[nodiscard]] size_t total_points() const noexcept {
        size_t total = 1;
        for (size_t i = 0; i < N; ++i) {
            auto result = safe_multiply(total, grids[i].size());
            if (!result.has_value()) {
                return 0;  // Overflow - return 0 to signal error
            }
            total = result.value();
        }
        return total;
    }

    /// Calculate total number of grid points with overflow checking
    ///
    /// @return Total points or OverflowError if product exceeds SIZE_MAX
    [[nodiscard]] std::expected<size_t, OverflowError> total_points_checked() const noexcept {
        const auto s = shape();
        return safe_product(std::span<const size_t, N>(s));
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

/// Immutable N-dimensional price surface with B-spline interpolation
///
/// Provides fast interpolation queries and partial derivatives.
/// Thread-safe after construction.
///
/// @tparam N Number of dimensions
template <size_t N>
class PriceTableSurfaceND {
public:
    /// Build surface from axes, coefficients, and individual fields
    ///
    /// @param axes Grid points and names for each dimension
    /// @param coeffs Flattened B-spline coefficients (row-major)
    /// @param K_ref Reference strike price
    /// @param dividends Continuous yield + discrete dividend schedule
    /// @return Shared pointer to surface or error
    [[nodiscard]] static std::expected<std::shared_ptr<const PriceTableSurfaceND<N>>, PriceTableError>
    build(PriceTableAxesND<N> axes, std::vector<double> coeffs,
          double K_ref, DividendSpec dividends = {});

    /// Access axes
    [[nodiscard]] const PriceTableAxesND<N>& axes() const noexcept { return axes_; }

    /// Reference strike price
    [[nodiscard]] double K_ref() const noexcept { return K_ref_; }

    /// Dividend specification
    [[nodiscard]] const DividendSpec& dividends() const noexcept { return dividends_; }

    /// Minimum log-moneyness (from axes grid)
    [[nodiscard]] double m_min() const noexcept { return axes_.grids[0].front(); }

    /// Maximum log-moneyness (from axes grid)
    [[nodiscard]] double m_max() const noexcept { return axes_.grids[0].back(); }

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
    PriceTableSurfaceND(PriceTableAxesND<N> axes, double K_ref,
                     DividendSpec dividends,
                     std::unique_ptr<BSplineND<double, N>> spline);

    PriceTableAxesND<N> axes_;
    double K_ref_;
    DividendSpec dividends_;
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

/// Standard B-spline price table
using BSplinePriceTable = PriceTable<BSplineLeaf>;

/// Leaf adapter for segmented surfaces (no EEP decomposition)
using BSplineSegmentedLeaf = EEPSurfaceAdapter<SharedBSplineInterp<4>,
                                         StandardTransform4D, IdentityEEP>;

/// Tau-segmented surface
using BSplineSegmentedSurface = SplitSurface<BSplineSegmentedLeaf, TauSegmentSplit>;

/// Multi-K_ref surface (outer split over K_refs of segmented inner)
using BSplineMultiKRefInner = SplitSurface<BSplineSegmentedSurface, MultiKRefSplit>;

/// Multi-K_ref price table
using BSplineMultiKRefSurface = PriceTable<BSplineMultiKRefInner>;


/// Create a BSplinePriceTable from a pre-built EEP surface.
/// Reads K_ref and dividend_yield from surface fields.
/// The surface must contain EEP data (built with eep_decompose).
[[nodiscard]] std::expected<BSplinePriceTable, std::string>
make_bspline_surface(
    std::shared_ptr<const PriceTableSurface> surface,
    OptionType type);

} // namespace mango
