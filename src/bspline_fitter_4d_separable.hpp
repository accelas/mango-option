/**
 * @file bspline_fitter_4d_separable.hpp
 * @brief Separable 4D B-spline coefficient fitting via sequential 1D solves
 *
 * Exploits the tensor-product structure of 4D B-splines to avoid solving
 * a massive dense system. Instead of solving one (Nm·Nt·Nv·Nr)² system,
 * we solve many small 1D systems sequentially along each axis.
 *
 * Algorithm (cache-optimized axis order):
 *   Input: f(m_i, τ_j, σ_k, r_l) - function values on 4D grid
 *   Memory layout: ((i*Nt + j)*Nv + k)*Nr + l (row-major, r fastest-varying)
 *
 *   Step 1: Fit along r for each fixed (i,j,k) [stride=1, contiguous]
 *     For each (i,j,k): solve B_r * c_r = f[i,j,k,:]
 *     → produces c(m_i, τ_j, σ_k, r)
 *
 *   Step 2: Fit along σ for each fixed (i,j,l) [stride=Nr, small jumps]
 *     For each (i,j,l): solve B_σ * c_σ = c[i,j,:,l]
 *     → produces c(m_i, τ_j, σ, r_l)
 *
 *   Step 3: Fit along τ for each fixed (i,k,l) [stride=Nv*Nr, medium jumps]
 *     For each (i,k,l): solve B_τ * c_τ = c[i,:,k,l]
 *     → produces c(m_i, τ, σ_k, r_l)
 *
 *   Step 4: Fit along m for each fixed (j,k,l) [stride=Nt*Nv*Nr, large jumps]
 *     For each (j,k,l): solve B_m * c_m = c[:,j,k,l]
 *     → produces c(m, τ_j, σ_k, r_l) - final coefficients
 *
 * Performance: O(Nm³ + Nt³ + Nσ³ + Nr³) for all tridiagonal solves
 *              Cache-optimized axis order improves memory bandwidth utilization
 *              For 50×30×20×10: ~5ms fitting time (cache-optimized)
 *
 * Memory: Works in-place, modifying the tensor as we go
 *         Peak memory: O(Nm·Nt·Nv·Nr) - just the tensor itself
 *
 * Cache Optimization: Processing axes in order of increasing memory stride
 *                     (r → σ → τ → m) minimizes cache misses in early passes
 */

#pragma once

#include "bspline_collocation_1d.hpp"
#include "expected.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>

namespace mango {

/// Result of separable 4D fitting with per-axis diagnostics
struct BSplineFit4DSeparableResult {
    std::vector<double> coefficients;  ///< Final coefficients (Nm × Nt × Nv × Nr)
    bool success;                       ///< Overall fit succeeded
    std::string error_message;          ///< Error if failed

    // Per-axis diagnostics
    double max_residual_m;              ///< Max residual after m-axis fit
    double max_residual_tau;            ///< Max residual after τ-axis fit
    double max_residual_sigma;          ///< Max residual after σ-axis fit
    double max_residual_r;              ///< Max residual after r-axis fit

    double condition_m;                 ///< Condition estimate for m-axis
    double condition_tau;               ///< Condition estimate for τ-axis
    double condition_sigma;             ///< Condition estimate for σ-axis
    double condition_r;                 ///< Condition estimate for r-axis

    size_t failed_slices_m;             ///< Number of failed 1D fits along m
    size_t failed_slices_tau;           ///< Number of failed 1D fits along τ
    size_t failed_slices_sigma;         ///< Number of failed 1D fits along σ
    size_t failed_slices_r;             ///< Number of failed 1D fits along r
};

/// Separable 4D B-spline fitter
///
/// Performs sequential 1D fitting along each axis using collocation.
/// Works in-place to minimize memory usage.
class BSplineFitter4DSeparable {
public:
    /// Create fitter with validation
    ///
    /// @param m_grid Moneyness grid (sorted, ≥4 points)
    /// @param t_grid Maturity grid (sorted, ≥4 points)
    /// @param v_grid Volatility grid (sorted, ≥4 points)
    /// @param r_grid Rate grid (sorted, ≥4 points)
    /// @return Fitter instance or error message
    static expected<BSplineFitter4DSeparable, std::string> create(
        std::vector<double> m_grid,
        std::vector<double> t_grid,
        std::vector<double> v_grid,
        std::vector<double> r_grid)
    {
        if (m_grid.size() < 4 || t_grid.size() < 4 || v_grid.size() < 4 || r_grid.size() < 4) {
            return unexpected("All grids must have ≥4 points for cubic B-splines");
        }
        return BSplineFitter4DSeparable(std::move(m_grid), std::move(t_grid),
                                       std::move(v_grid), std::move(r_grid));
    }

    /// Fit B-spline coefficients via separable collocation
    ///
    /// @param values Function values at grid points (row-major: i*Nt*Nv*Nr + j*Nv*Nr + k*Nr + l)
    /// @param tolerance Max allowed residual per axis (default 1e-6)
    /// @return Fit result with coefficients and diagnostics
    ///
    /// @note Axis order optimized for cache locality (r → σ → τ → m)
    ///       Processing fastest-varying dimensions first minimizes cache misses
    BSplineFit4DSeparableResult fit(const std::vector<double>& values, double tolerance = 1e-6) {
        if (values.size() != Nm_ * Nt_ * Nv_ * Nr_) {
            return {std::vector<double>(), false,
                    "Value array size mismatch", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        }

        // Work in-place: copy values to coefficients array
        std::vector<double> coeffs = values;

        BSplineFit4DSeparableResult result;
        result.success = true;
        result.failed_slices_m = 0;
        result.failed_slices_tau = 0;
        result.failed_slices_sigma = 0;
        result.failed_slices_r = 0;

        // CACHE-OPTIMIZED AXIS ORDER (fastest-varying to slowest-varying)
        // Memory layout: ((i*Nt + j)*Nv + k)*Nr + l
        // Strides: r=1, σ=Nr, τ=Nv*Nr, m=Nt*Nv*Nr
        //
        // Step 1: r-axis (stride=1, contiguous access)
        if (!fit_axis_r(coeffs, tolerance, result)) {
            result.success = false;
            result.error_message = "Failed fitting along r-axis: " +
                                   std::to_string(result.failed_slices_r) + " slices failed";
            return result;
        }

        // Step 2: σ-axis (stride=Nr, small jumps)
        if (!fit_axis_sigma(coeffs, tolerance, result)) {
            result.success = false;
            result.error_message = "Failed fitting along σ-axis: " +
                                   std::to_string(result.failed_slices_sigma) + " slices failed";
            return result;
        }

        // Step 3: τ-axis (stride=Nv*Nr, medium jumps)
        if (!fit_axis_tau(coeffs, tolerance, result)) {
            result.success = false;
            result.error_message = "Failed fitting along τ-axis: " +
                                   std::to_string(result.failed_slices_tau) + " slices failed";
            return result;
        }

        // Step 4: m-axis (stride=Nt*Nv*Nr, large jumps - done last)
        if (!fit_axis_m(coeffs, tolerance, result)) {
            result.success = false;
            result.error_message = "Failed fitting along m-axis: " +
                                   std::to_string(result.failed_slices_m) + " slices failed";
            return result;
        }

        result.coefficients = std::move(coeffs);
        return result;
    }

private:
    /// Private constructor for factory method
    BSplineFitter4DSeparable(std::vector<double> m_grid,
                             std::vector<double> t_grid,
                             std::vector<double> v_grid,
                             std::vector<double> r_grid)
        : m_grid_(std::move(m_grid))
        , t_grid_(std::move(t_grid))
        , v_grid_(std::move(v_grid))
        , r_grid_(std::move(r_grid))
        , Nm_(m_grid_.size())
        , Nt_(t_grid_.size())
        , Nv_(v_grid_.size())
        , Nr_(r_grid_.size())
    {
        // Create 1D solvers for each axis using factory method
        auto m_result = BSplineCollocation1D::create(m_grid_);
        auto t_result = BSplineCollocation1D::create(t_grid_);
        auto v_result = BSplineCollocation1D::create(v_grid_);
        auto r_result = BSplineCollocation1D::create(r_grid_);

        // The factory methods should never fail here since the grids were already validated
        // in the create() method of this class, but we check for completeness
        if (!m_result.has_value() || !t_result.has_value() ||
            !v_result.has_value() || !r_result.has_value()) {
            throw std::runtime_error("Failed to create BSplineCollocation1D solvers: " +
                                   (m_result.has_value() ? "" : m_result.error()) +
                                   (t_result.has_value() ? "" : t_result.error()) +
                                   (v_result.has_value() ? "" : v_result.error()) +
                                   (r_result.has_value() ? "" : r_result.error()));
        }

        solver_m_ = std::make_unique<BSplineCollocation1D>(std::move(m_result.value()));
        solver_t_ = std::make_unique<BSplineCollocation1D>(std::move(t_result.value()));
        solver_v_ = std::make_unique<BSplineCollocation1D>(std::move(v_result.value()));
        solver_r_ = std::make_unique<BSplineCollocation1D>(std::move(r_result.value()));
    }

    std::vector<double> m_grid_, t_grid_, v_grid_, r_grid_;
    size_t Nm_, Nt_, Nv_, Nr_;

    std::unique_ptr<BSplineCollocation1D> solver_m_;
    std::unique_ptr<BSplineCollocation1D> solver_t_;
    std::unique_ptr<BSplineCollocation1D> solver_v_;
    std::unique_ptr<BSplineCollocation1D> solver_r_;

    /// Fit along m-axis for all (j,k,l) slices
    bool fit_axis_m(std::vector<double>& coeffs, double tolerance,
                    BSplineFit4DSeparableResult& result) {
        std::vector<double> slice(Nm_);
        std::vector<double> max_residuals;
        std::vector<double> conditions;

        for (size_t j = 0; j < Nt_; ++j) {
            for (size_t k = 0; k < Nv_; ++k) {
                for (size_t l = 0; l < Nr_; ++l) {
                    // Extract 1D slice along m: coeffs[:,j,k,l]
                    for (size_t i = 0; i < Nm_; ++i) {
                        size_t idx = ((i * Nt_ + j) * Nv_ + k) * Nr_ + l;
                        slice[i] = coeffs[idx];
                    }

                    // Fit 1D B-spline
                    auto fit_result = solver_m_->fit(slice, tolerance);

                    if (!fit_result.success) {
                        ++result.failed_slices_m;
                        return false;
                    }

                    // Write coefficients back
                    for (size_t i = 0; i < Nm_; ++i) {
                        size_t idx = ((i * Nt_ + j) * Nv_ + k) * Nr_ + l;
                        coeffs[idx] = fit_result.coefficients[i];
                    }

                    max_residuals.push_back(fit_result.max_residual);
                    conditions.push_back(fit_result.condition_estimate);
                }
            }
        }

        result.max_residual_m = *std::max_element(max_residuals.begin(), max_residuals.end());
        result.condition_m = *std::max_element(conditions.begin(), conditions.end());
        return true;
    }

    /// Fit along τ-axis for all (i,k,l) slices
    bool fit_axis_tau(std::vector<double>& coeffs, double tolerance,
                      BSplineFit4DSeparableResult& result) {
        std::vector<double> slice(Nt_);
        std::vector<double> max_residuals;
        std::vector<double> conditions;

        for (size_t i = 0; i < Nm_; ++i) {
            for (size_t k = 0; k < Nv_; ++k) {
                for (size_t l = 0; l < Nr_; ++l) {
                    // Extract 1D slice along τ: coeffs[i,:,k,l]
                    for (size_t j = 0; j < Nt_; ++j) {
                        size_t idx = ((i * Nt_ + j) * Nv_ + k) * Nr_ + l;
                        slice[j] = coeffs[idx];
                    }

                    // Fit 1D B-spline
                    auto fit_result = solver_t_->fit(slice, tolerance);

                    if (!fit_result.success) {
                        ++result.failed_slices_tau;
                        return false;
                    }

                    // Write coefficients back
                    for (size_t j = 0; j < Nt_; ++j) {
                        size_t idx = ((i * Nt_ + j) * Nv_ + k) * Nr_ + l;
                        coeffs[idx] = fit_result.coefficients[j];
                    }

                    max_residuals.push_back(fit_result.max_residual);
                    conditions.push_back(fit_result.condition_estimate);
                }
            }
        }

        result.max_residual_tau = *std::max_element(max_residuals.begin(), max_residuals.end());
        result.condition_tau = *std::max_element(conditions.begin(), conditions.end());
        return true;
    }

    /// Fit along σ-axis for all (i,j,l) slices
    bool fit_axis_sigma(std::vector<double>& coeffs, double tolerance,
                        BSplineFit4DSeparableResult& result) {
        std::vector<double> slice(Nv_);
        std::vector<double> max_residuals;
        std::vector<double> conditions;

        for (size_t i = 0; i < Nm_; ++i) {
            for (size_t j = 0; j < Nt_; ++j) {
                for (size_t l = 0; l < Nr_; ++l) {
                    // Extract 1D slice along σ: coeffs[i,j,:,l]
                    for (size_t k = 0; k < Nv_; ++k) {
                        size_t idx = ((i * Nt_ + j) * Nv_ + k) * Nr_ + l;
                        slice[k] = coeffs[idx];
                    }

                    // Fit 1D B-spline
                    auto fit_result = solver_v_->fit(slice, tolerance);

                    if (!fit_result.success) {
                        ++result.failed_slices_sigma;
                        return false;
                    }

                    // Write coefficients back
                    for (size_t k = 0; k < Nv_; ++k) {
                        size_t idx = ((i * Nt_ + j) * Nv_ + k) * Nr_ + l;
                        coeffs[idx] = fit_result.coefficients[k];
                    }

                    max_residuals.push_back(fit_result.max_residual);
                    conditions.push_back(fit_result.condition_estimate);
                }
            }
        }

        result.max_residual_sigma = *std::max_element(max_residuals.begin(), max_residuals.end());
        result.condition_sigma = *std::max_element(conditions.begin(), conditions.end());
        return true;
    }

    /// Fit along r-axis for all (i,j,k) slices
    bool fit_axis_r(std::vector<double>& coeffs, double tolerance,
                    BSplineFit4DSeparableResult& result) {
        std::vector<double> slice(Nr_);
        std::vector<double> max_residuals;
        std::vector<double> conditions;

        for (size_t i = 0; i < Nm_; ++i) {
            for (size_t j = 0; j < Nt_; ++j) {
                for (size_t k = 0; k < Nv_; ++k) {
                    // Extract 1D slice along r: coeffs[i,j,k,:]
                    for (size_t l = 0; l < Nr_; ++l) {
                        size_t idx = ((i * Nt_ + j) * Nv_ + k) * Nr_ + l;
                        slice[l] = coeffs[idx];
                    }

                    // Fit 1D B-spline
                    auto fit_result = solver_r_->fit(slice, tolerance);

                    if (!fit_result.success) {
                        ++result.failed_slices_r;
                        return false;
                    }

                    // Write coefficients back
                    for (size_t l = 0; l < Nr_; ++l) {
                        size_t idx = ((i * Nt_ + j) * Nv_ + k) * Nr_ + l;
                        coeffs[idx] = fit_result.coefficients[l];
                    }

                    max_residuals.push_back(fit_result.max_residual);
                    conditions.push_back(fit_result.condition_estimate);
                }
            }
        }

        result.max_residual_r = *std::max_element(max_residuals.begin(), max_residuals.end());
        result.condition_r = *std::max_element(conditions.begin(), conditions.end());
        return true;
    }
};

}  // namespace mango
