/**
 * @file bspline_fitter_4d_separable.hpp
 * @brief Separable 4D B-spline coefficient fitting via sequential 1D solves
 *
 * Exploits the tensor-product structure of 4D B-splines to avoid solving
 * a massive dense system. Instead of solving one (N0·N1·N2·N3)² system,
 * we solve many small 1D systems sequentially along each axis.
 *
 * Algorithm (cache-optimized axis order):
 *   Input: f(x0_i, x1_j, x2_k, x3_l) - function values on 4D grid
 *   Memory layout: ((i*N1 + j)*N2 + k)*N3 + l (row-major, axis3 fastest-varying)
 *
 *   Step 1: Fit along axis3 for each fixed (i,j,k) [stride=1, contiguous]
 *     For each (i,j,k): solve B_3 * c_3 = f[i,j,k,:]
 *     → produces c(x0_i, x1_j, x2_k, x3)
 *
 *   Step 2: Fit along axis2 for each fixed (i,j,l) [stride=N3, small jumps]
 *     For each (i,j,l): solve B_2 * c_2 = c[i,j,:,l]
 *     → produces c(x0_i, x1_j, x2, x3_l)
 *
 *   Step 3: Fit along axis1 for each fixed (i,k,l) [stride=N2*N3, medium jumps]
 *     For each (i,k,l): solve B_1 * c_1 = c[i,:,k,l]
 *     → produces c(x0_i, x1, x2_k, x3_l)
 *
 *   Step 4: Fit along axis0 for each fixed (j,k,l) [stride=N1*N2*N3, large jumps]
 *     For each (j,k,l): solve B_0 * c_0 = c[:,j,k,l]
 *     → produces c(x0, x1_j, x2_k, x3_l) - final coefficients
 *
 * Performance: O(N0³ + N1³ + N2³ + N3³) for all tridiagonal solves
 *              Cache-optimized axis order improves memory bandwidth utilization
 *              For 50×30×20×10: ~5ms fitting time (cache-optimized)
 *
 * Memory: Works in-place, modifying the tensor as we go
 *         Peak memory: O(N0·N1·N2·N3) - just the tensor itself
 *
 * Cache Optimization: Processing axes in order of increasing memory stride
 *                     (axis3 → axis2 → axis1 → axis0) minimizes cache misses in early passes
 */

#pragma once

#include "src/interpolation/bspline_collocation_1d.hpp"
#include "src/support/expected.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>

namespace mango {

/// Result of separable 4D fitting with per-axis diagnostics
struct BSplineFit4DSeparableResult {
    std::vector<double> coefficients;  ///< Final coefficients (N0 × N1 × N2 × N3)
    bool success;                       ///< Overall fit succeeded
    std::string error_message;          ///< Error if failed

    // Per-axis diagnostics
    double max_residual_axis0;          ///< Max residual after axis0 fit
    double max_residual_axis1;          ///< Max residual after axis1 fit
    double max_residual_axis2;          ///< Max residual after axis2 fit
    double max_residual_axis3;          ///< Max residual after axis3 fit

    double condition_axis0;             ///< Condition estimate for axis0
    double condition_axis1;             ///< Condition estimate for axis1
    double condition_axis2;             ///< Condition estimate for axis2
    double condition_axis3;             ///< Condition estimate for axis3

    size_t failed_slices_axis0;         ///< Number of failed 1D fits along axis0
    size_t failed_slices_axis1;         ///< Number of failed 1D fits along axis1
    size_t failed_slices_axis2;         ///< Number of failed 1D fits along axis2
    size_t failed_slices_axis3;         ///< Number of failed 1D fits along axis3
};

/// Separable 4D B-spline fitter
///
/// Performs sequential 1D fitting along each axis using collocation.
/// Works in-place to minimize memory usage.
class BSplineFitter4DSeparable {
public:
    /// Create fitter with validation
    ///
    /// @param axis0_grid Grid for axis 0 (sorted, ≥4 points)
    /// @param axis1_grid Grid for axis 1 (sorted, ≥4 points)
    /// @param axis2_grid Grid for axis 2 (sorted, ≥4 points)
    /// @param axis3_grid Grid for axis 3 (sorted, ≥4 points)
    /// @return Fitter instance or error message
    ///
    /// @note Validation is delegated to BSplineCollocation1D for each axis.
    ///       Grids are checked during 1D solver construction.
    static expected<BSplineFitter4DSeparable, std::string> create(
        std::vector<double> axis0_grid,
        std::vector<double> axis1_grid,
        std::vector<double> axis2_grid,
        std::vector<double> axis3_grid)
    {
        try {
            return BSplineFitter4DSeparable(std::move(axis0_grid), std::move(axis1_grid),
                                           std::move(axis2_grid), std::move(axis3_grid));
        } catch (const std::exception& e) {
            return unexpected(std::string(e.what()));
        }
    }

    /// Fit B-spline coefficients via separable collocation
    ///
    /// @param values Function values at grid points (row-major: i*N1*N2*N3 + j*N2*N3 + k*N3 + l)
    /// @param tolerance Max allowed residual per axis (default 1e-6)
    /// @return Fit result with coefficients and diagnostics
    ///
    /// @note Axis order optimized for cache locality (axis3 → axis2 → axis1 → axis0)
    ///       Processing fastest-varying dimensions first minimizes cache misses
    BSplineFit4DSeparableResult fit(const std::vector<double>& values, double tolerance = 1e-6) {
        if (values.size() != N0_ * N1_ * N2_ * N3_) {
            return {std::vector<double>(), false,
                    "Value array size mismatch", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        }

        // Work in-place: copy values to coefficients array
        std::vector<double> coeffs = values;

        BSplineFit4DSeparableResult result;
        result.success = true;
        result.failed_slices_axis0 = 0;
        result.failed_slices_axis1 = 0;
        result.failed_slices_axis2 = 0;
        result.failed_slices_axis3 = 0;

        // CACHE-OPTIMIZED AXIS ORDER (fastest-varying to slowest-varying)
        // Memory layout: ((i*N1 + j)*N2 + k)*N3 + l
        // Strides: axis3=1, axis2=N3, axis1=N2*N3, axis0=N1*N2*N3
        //
        // Step 1: axis3 (stride=1, contiguous access)
        if (!fit_axis3(coeffs, tolerance, result)) {
            result.success = false;
            result.error_message = "Failed fitting along axis3: " +
                                   std::to_string(result.failed_slices_axis3) + " slices failed";
            return result;
        }

        // Step 2: axis2 (stride=N3, small jumps)
        if (!fit_axis2(coeffs, tolerance, result)) {
            result.success = false;
            result.error_message = "Failed fitting along axis2: " +
                                   std::to_string(result.failed_slices_axis2) + " slices failed";
            return result;
        }

        // Step 3: axis1 (stride=N2*N3, medium jumps)
        if (!fit_axis1(coeffs, tolerance, result)) {
            result.success = false;
            result.error_message = "Failed fitting along axis1: " +
                                   std::to_string(result.failed_slices_axis1) + " slices failed";
            return result;
        }

        // Step 4: axis0 (stride=N1*N2*N3, large jumps - done last)
        if (!fit_axis0(coeffs, tolerance, result)) {
            result.success = false;
            result.error_message = "Failed fitting along axis0: " +
                                   std::to_string(result.failed_slices_axis0) + " slices failed";
            return result;
        }

        result.coefficients = std::move(coeffs);
        return result;
    }

private:
    /// Private constructor for factory method
    BSplineFitter4DSeparable(std::vector<double> axis0_grid,
                             std::vector<double> axis1_grid,
                             std::vector<double> axis2_grid,
                             std::vector<double> axis3_grid)
        : axis0_grid_(std::move(axis0_grid))
        , axis1_grid_(std::move(axis1_grid))
        , axis2_grid_(std::move(axis2_grid))
        , axis3_grid_(std::move(axis3_grid))
        , N0_(axis0_grid_.size())
        , N1_(axis1_grid_.size())
        , N2_(axis2_grid_.size())
        , N3_(axis3_grid_.size())
    {
        // Create 1D solvers for each axis using factory method
        auto axis0_result = BSplineCollocation1D::create(axis0_grid_);
        auto axis1_result = BSplineCollocation1D::create(axis1_grid_);
        auto axis2_result = BSplineCollocation1D::create(axis2_grid_);
        auto axis3_result = BSplineCollocation1D::create(axis3_grid_);

        // Collect all error messages if any solver construction fails
        if (!axis0_result.has_value() || !axis1_result.has_value() ||
            !axis2_result.has_value() || !axis3_result.has_value()) {
            throw std::runtime_error("Failed to create BSplineCollocation1D solvers: " +
                                   (axis0_result.has_value() ? "" : "axis0: " + axis0_result.error() + "; ") +
                                   (axis1_result.has_value() ? "" : "axis1: " + axis1_result.error() + "; ") +
                                   (axis2_result.has_value() ? "" : "axis2: " + axis2_result.error() + "; ") +
                                   (axis3_result.has_value() ? "" : "axis3: " + axis3_result.error()));
        }

        solver_axis0_ = std::make_unique<BSplineCollocation1D>(std::move(axis0_result.value()));
        solver_axis1_ = std::make_unique<BSplineCollocation1D>(std::move(axis1_result.value()));
        solver_axis2_ = std::make_unique<BSplineCollocation1D>(std::move(axis2_result.value()));
        solver_axis3_ = std::make_unique<BSplineCollocation1D>(std::move(axis3_result.value()));
    }

    std::vector<double> axis0_grid_, axis1_grid_, axis2_grid_, axis3_grid_;
    size_t N0_, N1_, N2_, N3_;

    std::unique_ptr<BSplineCollocation1D> solver_axis0_;
    std::unique_ptr<BSplineCollocation1D> solver_axis1_;
    std::unique_ptr<BSplineCollocation1D> solver_axis2_;
    std::unique_ptr<BSplineCollocation1D> solver_axis3_;

    /// Fit along axis0 for all (j,k,l) slices
    bool fit_axis0(std::vector<double>& coeffs, double tolerance,
                   BSplineFit4DSeparableResult& result) {
        std::vector<double> slice(N0_);
        std::vector<double> max_residuals;
        std::vector<double> conditions;

        for (size_t j = 0; j < N1_; ++j) {
            for (size_t k = 0; k < N2_; ++k) {
                for (size_t l = 0; l < N3_; ++l) {
                    // Extract 1D slice along axis0: coeffs[:,j,k,l]
                    for (size_t i = 0; i < N0_; ++i) {
                        size_t idx = ((i * N1_ + j) * N2_ + k) * N3_ + l;
                        slice[i] = coeffs[idx];
                    }

                    // Fit 1D B-spline
                    auto fit_result = solver_axis0_->fit(slice, tolerance);

                    if (!fit_result.success) {
                        ++result.failed_slices_axis0;
                        return false;
                    }

                    // Write coefficients back
                    for (size_t i = 0; i < N0_; ++i) {
                        size_t idx = ((i * N1_ + j) * N2_ + k) * N3_ + l;
                        coeffs[idx] = fit_result.coefficients[i];
                    }

                    max_residuals.push_back(fit_result.max_residual);
                    conditions.push_back(fit_result.condition_estimate);
                }
            }
        }

        result.max_residual_axis0 = *std::max_element(max_residuals.begin(), max_residuals.end());
        result.condition_axis0 = *std::max_element(conditions.begin(), conditions.end());
        return true;
    }

    /// Fit along axis1 for all (i,k,l) slices
    bool fit_axis1(std::vector<double>& coeffs, double tolerance,
                   BSplineFit4DSeparableResult& result) {
        std::vector<double> slice(N1_);
        std::vector<double> max_residuals;
        std::vector<double> conditions;

        for (size_t i = 0; i < N0_; ++i) {
            for (size_t k = 0; k < N2_; ++k) {
                for (size_t l = 0; l < N3_; ++l) {
                    // Extract 1D slice along axis1: coeffs[i,:,k,l]
                    for (size_t j = 0; j < N1_; ++j) {
                        size_t idx = ((i * N1_ + j) * N2_ + k) * N3_ + l;
                        slice[j] = coeffs[idx];
                    }

                    // Fit 1D B-spline
                    auto fit_result = solver_axis1_->fit(slice, tolerance);

                    if (!fit_result.success) {
                        ++result.failed_slices_axis1;
                        return false;
                    }

                    // Write coefficients back
                    for (size_t j = 0; j < N1_; ++j) {
                        size_t idx = ((i * N1_ + j) * N2_ + k) * N3_ + l;
                        coeffs[idx] = fit_result.coefficients[j];
                    }

                    max_residuals.push_back(fit_result.max_residual);
                    conditions.push_back(fit_result.condition_estimate);
                }
            }
        }

        result.max_residual_axis1 = *std::max_element(max_residuals.begin(), max_residuals.end());
        result.condition_axis1 = *std::max_element(conditions.begin(), conditions.end());
        return true;
    }

    /// Fit along axis2 for all (i,j,l) slices
    bool fit_axis2(std::vector<double>& coeffs, double tolerance,
                   BSplineFit4DSeparableResult& result) {
        std::vector<double> slice(N2_);
        std::vector<double> max_residuals;
        std::vector<double> conditions;

        for (size_t i = 0; i < N0_; ++i) {
            for (size_t j = 0; j < N1_; ++j) {
                for (size_t l = 0; l < N3_; ++l) {
                    // Extract 1D slice along axis2: coeffs[i,j,:,l]
                    for (size_t k = 0; k < N2_; ++k) {
                        size_t idx = ((i * N1_ + j) * N2_ + k) * N3_ + l;
                        slice[k] = coeffs[idx];
                    }

                    // Fit 1D B-spline
                    auto fit_result = solver_axis2_->fit(slice, tolerance);

                    if (!fit_result.success) {
                        ++result.failed_slices_axis2;
                        return false;
                    }

                    // Write coefficients back
                    for (size_t k = 0; k < N2_; ++k) {
                        size_t idx = ((i * N1_ + j) * N2_ + k) * N3_ + l;
                        coeffs[idx] = fit_result.coefficients[k];
                    }

                    max_residuals.push_back(fit_result.max_residual);
                    conditions.push_back(fit_result.condition_estimate);
                }
            }
        }

        result.max_residual_axis2 = *std::max_element(max_residuals.begin(), max_residuals.end());
        result.condition_axis2 = *std::max_element(conditions.begin(), conditions.end());
        return true;
    }

    /// Fit along axis3 for all (i,j,k) slices
    bool fit_axis3(std::vector<double>& coeffs, double tolerance,
                   BSplineFit4DSeparableResult& result) {
        std::vector<double> slice(N3_);
        std::vector<double> max_residuals;
        std::vector<double> conditions;

        for (size_t i = 0; i < N0_; ++i) {
            for (size_t j = 0; j < N1_; ++j) {
                for (size_t k = 0; k < N2_; ++k) {
                    // Extract 1D slice along axis3: coeffs[i,j,k,:]
                    for (size_t l = 0; l < N3_; ++l) {
                        size_t idx = ((i * N1_ + j) * N2_ + k) * N3_ + l;
                        slice[l] = coeffs[idx];
                    }

                    // Fit 1D B-spline
                    auto fit_result = solver_axis3_->fit(slice, tolerance);

                    if (!fit_result.success) {
                        ++result.failed_slices_axis3;
                        return false;
                    }

                    // Write coefficients back
                    for (size_t l = 0; l < N3_; ++l) {
                        size_t idx = ((i * N1_ + j) * N2_ + k) * N3_ + l;
                        coeffs[idx] = fit_result.coefficients[l];
                    }

                    max_residuals.push_back(fit_result.max_residual);
                    conditions.push_back(fit_result.condition_estimate);
                }
            }
        }

        result.max_residual_axis3 = *std::max_element(max_residuals.begin(), max_residuals.end());
        result.condition_axis3 = *std::max_element(conditions.begin(), conditions.end());
        return true;
    }
};

}  // namespace mango
