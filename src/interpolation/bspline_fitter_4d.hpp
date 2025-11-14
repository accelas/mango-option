/**
 * @file bspline_fitter_4d.hpp
 * @brief 4D B-spline coefficient fitting using separable collocation
 *
 * Uses tensor-product structure to fit B-spline coefficients efficiently.
 * Instead of a massive O(n⁴) dense system, we solve sequential 1D systems
 * along each axis: axis0 → axis1 → axis2 → axis3.
 *
 * Algorithm:
 *   1. Fit along axis0 for all (axis1, axis2, axis3) slices
 *   2. Fit along axis1 for all (axis0, axis2, axis3) slices
 *   3. Fit along axis2 for all (axis0, axis1, axis3) slices
 *   4. Fit along axis3 for all (axis0, axis1, axis2) slices
 *
 * Each step uses 1D cubic B-spline collocation (bspline_collocation_1d.hpp).
 *
 * Performance: O(N0³ + N1³ + N2³ + N3³)
 *   For 50×30×20×10: ~5ms fitting time
 *
 * Accuracy: Residuals <1e-6 at all grid points (validated per-axis)
 *
 * Usage:
 *   auto fitter_result = BSplineFitter4D::create(axis0_grid, axis1_grid, axis2_grid, axis3_grid);
 *   if (fitter_result.has_value()) {
 *       auto result = fitter_result.value().fit(values_4d);
 *       if (result.success) {
 *           // Use result.coefficients with BSpline4D
 *       }
 *   } else {
 *       // Handle creation error: fitter_result.error()
 *   }
 */

#pragma once

#include "src/interpolation/bspline_fitter_4d_separable.hpp"
#include "src/interpolation/bspline_4d.hpp"
#include "src/support/expected.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace mango {

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
/// Uses tensor-product structure to avoid dense O(n⁴) solve.
/// Performs sequential 1D fits along each dimension.
///
/// Memory: O(N0·N1·N2·N3) for temporary storage
/// Time: O(N0·N1·N2·N3) for all 1D fits
class BSplineFitter4D {
public:
    /// Factory method to create BSplineFitter4D with validation
    ///
    /// @param axis0_grid Grid for axis 0 (sorted, ≥4 points)
    /// @param axis1_grid Grid for axis 1 (sorted, ≥4 points)
    /// @param axis2_grid Grid for axis 2 (sorted, ≥4 points)
    /// @param axis3_grid Grid for axis 3 (sorted, ≥4 points)
    /// @return expected<BSplineFitter4D, std::string> - success or error message
    ///
    /// @note Validation is delegated to BSplineCollocation1D via BSplineFitter4DSeparable.
    ///       We validate at creation time by attempting to create a separable fitter.
    static expected<BSplineFitter4D, std::string> create(
        std::vector<double> axis0_grid,
        std::vector<double> axis1_grid,
        std::vector<double> axis2_grid,
        std::vector<double> axis3_grid) {

        // Validate grids by attempting to create separable fitter
        // This delegates validation to BSplineCollocation1D for each axis
        auto validation_result = BSplineFitter4DSeparable::create(
            axis0_grid, axis1_grid, axis2_grid, axis3_grid);

        if (!validation_result.has_value()) {
            return unexpected(validation_result.error());
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
    BSplineFitResult4D fit(const std::vector<double>& values, double tolerance = 1e-6) {
        // Create separable fitter using factory pattern
        auto fitter_result = BSplineFitter4DSeparable::create(axis0_grid_, axis1_grid_,
                                                              axis2_grid_, axis3_grid_);
        if (!fitter_result.has_value()) {
            return {
                .coefficients = std::vector<double>(),
                .success = false,
                .error_message = fitter_result.error(),
                .max_residual = 0.0
            };
        }
        auto& fitter = fitter_result.value();

        // Perform separable fitting
        auto sep_result = fitter.fit(values, tolerance);

        if (!sep_result.success) {
            return {
                .coefficients = std::vector<double>(),
                .success = false,
                .error_message = sep_result.error_message,
                .max_residual = 0.0
            };
        }

        // Aggregate maximum residual across all axes
        double max_residual = std::max({
            sep_result.max_residual_axis0,
            sep_result.max_residual_axis1,
            sep_result.max_residual_axis2,
            sep_result.max_residual_axis3
        });

        // Return with full statistics
        return {
            .coefficients = sep_result.coefficients,
            .success = true,
            .error_message = "",
            .max_residual = max_residual,
            .max_residual_axis0 = sep_result.max_residual_axis0,
            .max_residual_axis1 = sep_result.max_residual_axis1,
            .max_residual_axis2 = sep_result.max_residual_axis2,
            .max_residual_axis3 = sep_result.max_residual_axis3,
            .condition_axis0 = sep_result.condition_axis0,
            .condition_axis1 = sep_result.condition_axis1,
            .condition_axis2 = sep_result.condition_axis2,
            .condition_axis3 = sep_result.condition_axis3,
            .failed_slices_axis0 = sep_result.failed_slices_axis0,
            .failed_slices_axis1 = sep_result.failed_slices_axis1,
            .failed_slices_axis2 = sep_result.failed_slices_axis2,
            .failed_slices_axis3 = sep_result.failed_slices_axis3
        };
    }

    /// Get detailed diagnostics from last fit (requires storing separable result)
    ///
    /// For now, use fit() which provides basic success/failure.
    /// Future: Add method to get per-axis condition numbers and residuals.

    /// Get grid dimensions
    [[nodiscard]] std::tuple<size_t, size_t, size_t, size_t> dimensions() const noexcept {
        return {N0_, N1_, N2_, N3_};
    }

private:
    /// Private constructor - use factory method create() instead
    ///
    /// @param axis0_grid Grid for axis 0 (validation delegated to separable fitter)
    /// @param axis1_grid Grid for axis 1 (validation delegated to separable fitter)
    /// @param axis2_grid Grid for axis 2 (validation delegated to separable fitter)
    /// @param axis3_grid Grid for axis 3 (validation delegated to separable fitter)
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
        // Pre-compute knot vectors (no validation needed - delegated to separable fitter)
        t0_ = clamped_knots_cubic(axis0_grid_);
        t1_ = clamped_knots_cubic(axis1_grid_);
        t2_ = clamped_knots_cubic(axis2_grid_);
        t3_ = clamped_knots_cubic(axis3_grid_);
    }

    // Friend declaration for factory method to access private constructor
    friend expected<BSplineFitter4D, std::string> create(
        std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>);
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
