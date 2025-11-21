/**
 * @file bspline_fitter_4d.hpp
 * @brief 4D B-spline coefficient fitting using separable collocation
 *
 * This file provides a 4D-specific interface that wraps the generic
 * BSplineNDSeparable<double, 4> template for backward compatibility.
 *
 * **Implementation:**
 * - Uses BSplineNDSeparable<double, 4> for actual fitting (~100 lines of generic code)
 * - Provides 4D-specific result struct for legacy API compatibility
 * - Replaces previous hardcoded implementation (~400 lines of duplicated code)
 *
 * **Performance:** O(N0² + N1² + N2² + N3²) with banded solver
 *   For 50×30×20×10: ~6ms fitting time (7.8× speedup from banded solver)
 *
 * **Accuracy:** Residuals <1e-6 at all grid points (validated per-axis)
 *
 * **Generic math modules used:**
 * - src/math/bspline_nd_separable.hpp: Generic N-D separable fitter
 * - src/math/bspline_basis.hpp: Cox-de Boor recursion, knot vectors
 *
 * Usage:
 *   auto fitter_result = BSplineFitter4D::create(
 *       axis0_grid, axis1_grid, axis2_grid, axis3_grid);
 *   if (fitter_result.has_value()) {
 *       auto result = fitter_result.value().fit(values_4d);
 *       if (result.success) {
 *           // Use result.coefficients with BSpline4D
 *       }
 *   }
 */

#pragma once

#include "src/math/bspline_basis.hpp"
#include "src/math/bspline_nd_separable.hpp"
#include <expected>
#include "src/support/error_types.hpp"
#include <vector>
#include <algorithm>
#include <string>
#include <array>

namespace mango {

// ============================================================================
// 4D B-spline Fitter (High-level Interface)
// ============================================================================

/// Result of 4D B-spline fitting
struct BSplineFitResult4D {
    std::vector<double> coefficients;  ///< Fitted coefficients (N0 × N1 × N2 × N3)
    bool success;                       ///< Fit succeeded
    std::string error_message;          ///< Error description if failed
    double max_residual;                ///< Maximum absolute residual at grid points

    // Detailed per-axis statistics (populated if success)
    double max_residual_axis0 = 0.0;   ///< Max residual along axis0
    double max_residual_axis1 = 0.0;   ///< Max residual along axis1
    double max_residual_axis2 = 0.0;   ///< Max residual along axis2
    double max_residual_axis3 = 0.0;   ///< Max residual along axis3

    double condition_axis0 = 0.0;      ///< Condition number estimate (axis0)
    double condition_axis1 = 0.0;      ///< Condition number estimate (axis1)
    double condition_axis2 = 0.0;      ///< Condition number estimate (axis2)
    double condition_axis3 = 0.0;      ///< Condition number estimate (axis3)

    size_t failed_slices_axis0 = 0;    ///< Failed fits along axis0
    size_t failed_slices_axis1 = 0;    ///< Failed fits along axis1
    size_t failed_slices_axis2 = 0;    ///< Failed fits along axis2
    size_t failed_slices_axis3 = 0;    ///< Failed fits along axis3
};

/// Separable 4D B-spline coefficient fitter
///
/// Wraps BSplineNDSeparable<double, 4> with 4D-specific interface.
/// Uses tensor-product structure to avoid dense O(n⁴) solve.
/// Performs sequential 1D fits along each dimension.
///
/// Memory: O(N0·N1·N2·N3) for coefficient storage
/// Time: O(N0² + N1² + N2² + N3²) for banded solves
class BSplineFitter4D {
public:
    /// Factory method to create BSplineFitter4D with validation
    ///
    /// @param axis0_grid Grid for axis 0 (sorted, ≥4 points)
    /// @param axis1_grid Grid for axis 1 (sorted, ≥4 points)
    /// @param axis2_grid Grid for axis 2 (sorted, ≥4 points)
    /// @param axis3_grid Grid for axis 3 (sorted, ≥4 points)
    /// @return std::expected<BSplineFitter4D, std::string> - success or error message
    [[nodiscard]] static std::expected<BSplineFitter4D, std::string> create(
        std::vector<double> axis0_grid,
        std::vector<double> axis1_grid,
        std::vector<double> axis2_grid,
        std::vector<double> axis3_grid)
    {
        // Validate grids by attempting to create N-D fitter
        std::array<std::vector<double>, 4> grids = {
            axis0_grid, axis1_grid, axis2_grid, axis3_grid
        };

        auto validation_result = BSplineNDSeparable<double, 4>::create(grids);
        if (!validation_result.has_value()) {
            return std::unexpected(validation_result.error());
        }

        // Grids are valid, create the fitter
        return BSplineFitter4D(std::move(axis0_grid), std::move(axis1_grid),
                               std::move(axis2_grid), std::move(axis3_grid));
    }

    /// Fit B-spline coefficients via separable collocation
    ///
    /// Uses tensor-product structure: sequential 1D fitting along each axis.
    /// Produces numerically accurate coefficients with residuals <1e-6.
    ///
    /// @param values Function values at grid points (size N0 × N1 × N2 × N3)
    ///               Row-major layout: index = ((i*N1 + j)*N2 + k)*N3 + l
    /// @param tolerance Maximum residual per axis (default 1e-6)
    /// @return Fit result with coefficients and diagnostics
    [[nodiscard]] BSplineFitResult4D fit(const std::vector<double>& values, double tolerance = 1e-6) {
        // Create N-D fitter
        std::array<std::vector<double>, 4> grids = {
            axis0_grid_, axis1_grid_, axis2_grid_, axis3_grid_
        };

        auto fitter_result = BSplineNDSeparable<double, 4>::create(std::move(grids));
        if (!fitter_result.has_value()) {
            return {
                .coefficients = std::vector<double>(),
                .success = false,
                .error_message = fitter_result.error(),
                .max_residual = 0.0
            };
        }

        // Perform separable fitting
        BSplineNDSeparableConfig<double> config{.tolerance = tolerance};
        auto result = fitter_result.value().fit(values, config);

        if (!result.success) {
            return {
                .coefficients = std::vector<double>(),
                .success = false,
                .error_message = result.error_message,
                .max_residual = 0.0
            };
        }

        // Aggregate maximum residual across all axes
        double max_residual = *std::max_element(
            result.max_residual_per_axis.begin(),
            result.max_residual_per_axis.end());

        // Return with full statistics (convert array to individual fields for compatibility)
        return {
            .coefficients = std::move(result.coefficients),
            .success = true,
            .error_message = "",
            .max_residual = max_residual,
            .max_residual_axis0 = result.max_residual_per_axis[0],
            .max_residual_axis1 = result.max_residual_per_axis[1],
            .max_residual_axis2 = result.max_residual_per_axis[2],
            .max_residual_axis3 = result.max_residual_per_axis[3],
            .condition_axis0 = result.condition_per_axis[0],
            .condition_axis1 = result.condition_per_axis[1],
            .condition_axis2 = result.condition_per_axis[2],
            .condition_axis3 = result.condition_per_axis[3],
            .failed_slices_axis0 = result.failed_slices[0],
            .failed_slices_axis1 = result.failed_slices[1],
            .failed_slices_axis2 = result.failed_slices[2],
            .failed_slices_axis3 = result.failed_slices[3]
        };
    }

    /// Get grid dimensions
    [[nodiscard]] std::tuple<size_t, size_t, size_t, size_t> dimensions() const noexcept {
        return {N0_, N1_, N2_, N3_};
    }

private:
    /// Private constructor - use factory method create() instead
    BSplineFitter4D(std::vector<double> axis0_grid,
                    std::vector<double> axis1_grid,
                    std::vector<double> axis2_grid,
                    std::vector<double> axis3_grid)
        : axis0_grid_(std::move(axis0_grid)),
          axis1_grid_(std::move(axis1_grid)),
          axis2_grid_(std::move(axis2_grid)),
          axis3_grid_(std::move(axis3_grid)),
          N0_(axis0_grid_.size()),
          N1_(axis1_grid_.size()),
          N2_(axis2_grid_.size()),
          N3_(axis3_grid_.size())
    {
        // Pre-compute knot vectors for compatibility
        t0_ = clamped_knots_cubic<double>(axis0_grid_);
        t1_ = clamped_knots_cubic<double>(axis1_grid_);
        t2_ = clamped_knots_cubic<double>(axis2_grid_);
        t3_ = clamped_knots_cubic<double>(axis3_grid_);
    }

    std::vector<double> axis0_grid_;  ///< Grid for axis 0
    std::vector<double> axis1_grid_;  ///< Grid for axis 1
    std::vector<double> axis2_grid_;  ///< Grid for axis 2
    std::vector<double> axis3_grid_;  ///< Grid for axis 3

    std::vector<double> t0_;  ///< Knot vector for axis 0
    std::vector<double> t1_;  ///< Knot vector for axis 1
    std::vector<double> t2_;  ///< Knot vector for axis 2
    std::vector<double> t3_;  ///< Knot vector for axis 3

    size_t N0_;  ///< Number of points on axis 0
    size_t N1_;  ///< Number of points on axis 1
    size_t N2_;  ///< Number of points on axis 2
    size_t N3_;  ///< Number of points on axis 3
};

}  // namespace mango
