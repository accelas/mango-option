/**
 * @file bspline_fitter_4d.hpp
 * @brief 4D B-spline coefficient fitting using separable collocation
 *
 * Uses tensor-product structure to fit B-spline coefficients efficiently.
 * Instead of a massive O(n⁴) dense system, we solve sequential 1D systems
 * along each axis: m → τ → σ → r.
 *
 * Algorithm:
 *   1. Fit along m-axis for all (τ,σ,r) slices
 *   2. Fit along τ-axis for all (m,σ,r) slices
 *   3. Fit along σ-axis for all (m,τ,r) slices
 *   4. Fit along r-axis for all (m,τ,σ) slices
 *
 * Each step uses 1D cubic B-spline collocation (bspline_collocation_1d.hpp).
 *
 * Performance: O(Nm³ + Nt³ + Nσ³ + Nr³)
 *   For 50×30×20×10: ~5ms fitting time
 *
 * Accuracy: Residuals <1e-6 at all grid points (validated per-axis)
 *
 * Usage:
 *   auto fitter_result = BSplineFitter4D::create(m_grid, t_grid, v_grid, r_grid);
 *   if (fitter_result.has_value()) {
 *       auto result = fitter_result.value().fit(prices_4d);
 *       if (result.success) {
 *           // Use result.coefficients with BSpline4D_FMA
 *       }
 *   } else {
 *       // Handle creation error: fitter_result.error()
 *   }
 */

#pragma once

#include "bspline_fitter_4d_separable.hpp"
#include "bspline_4d.hpp"
#include "expected.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace mango {

/// Result of 4D B-spline fitting
struct BSplineFitResult4D {
    std::vector<double> coefficients;  ///< Fitted coefficients (Nm × Nt × Nv × Nr)
    bool success;                       ///< Fit succeeded
    std::string error_message;          ///< Error description if failed
    double max_residual;                ///< Maximum absolute residual at grid points

    // Detailed per-axis statistics (populated if success)
    double max_residual_m = 0.0;       ///< Max residual along moneyness axis
    double max_residual_tau = 0.0;     ///< Max residual along maturity axis
    double max_residual_sigma = 0.0;   ///< Max residual along volatility axis
    double max_residual_r = 0.0;       ///< Max residual along rate axis

    double condition_m = 0.0;          ///< Condition number estimate (moneyness)
    double condition_tau = 0.0;        ///< Condition number estimate (maturity)
    double condition_sigma = 0.0;      ///< Condition number estimate (volatility)
    double condition_r = 0.0;          ///< Condition number estimate (rate)

    size_t failed_slices_m = 0;        ///< Failed fits along moneyness
    size_t failed_slices_tau = 0;      ///< Failed fits along maturity
    size_t failed_slices_sigma = 0;    ///< Failed fits along volatility
    size_t failed_slices_r = 0;        ///< Failed fits along rate
};

/// Separable 4D B-spline coefficient fitter
///
/// Uses tensor-product structure to avoid dense O(n⁴) solve.
/// Performs sequential 1D fits along each dimension.
///
/// Memory: O(n·m·p·q) for temporary storage
/// Time: O(n·m·p·q) for all 1D fits
class BSplineFitter4D {
public:
    /// Factory method to create BSplineFitter4D with validation
    ///
    /// @param m_grid Moneyness grid (sorted, ≥4 points)
    /// @param t_grid Maturity grid (sorted, ≥4 points)
    /// @param v_grid Volatility grid (sorted, ≥4 points)
    /// @param r_grid Rate grid (sorted, ≥4 points)
    /// @return expected<BSplineFitter4D, std::string> - success or error message
    static expected<BSplineFitter4D, std::string> create(
        std::vector<double> m_grid,
        std::vector<double> t_grid,
        std::vector<double> v_grid,
        std::vector<double> r_grid) {

        // Validate grid sizes
        if (m_grid.size() < 4) {
            return unexpected(std::string("Moneyness grid must have ≥4 points for cubic B-splines"));
        }
        if (t_grid.size() < 4) {
            return unexpected(std::string("Maturity grid must have ≥4 points for cubic B-splines"));
        }
        if (v_grid.size() < 4) {
            return unexpected(std::string("Volatility grid must have ≥4 points for cubic B-splines"));
        }
        if (r_grid.size() < 4) {
            return unexpected(std::string("Rate grid must have ≥4 points for cubic B-splines"));
        }

        // Verify grids are sorted
        auto is_sorted = [](const std::vector<double>& v) {
            return std::is_sorted(v.begin(), v.end());
        };

        if (!is_sorted(m_grid)) {
            return unexpected(std::string("Moneyness grid must be sorted in ascending order"));
        }
        if (!is_sorted(t_grid)) {
            return unexpected(std::string("Maturity grid must be sorted in ascending order"));
        }
        if (!is_sorted(v_grid)) {
            return unexpected(std::string("Volatility grid must be sorted in ascending order"));
        }
        if (!is_sorted(r_grid)) {
            return unexpected(std::string("Rate grid must be sorted in ascending order"));
        }

        // All validations passed, create the fitter
        return BSplineFitter4D(std::move(m_grid), std::move(t_grid), std::move(v_grid), std::move(r_grid));
    }

    /// Fit B-spline coefficients via separable collocation
    ///
    /// Uses tensor-product structure: sequential 1D fitting along each axis.
    /// Produces numerically accurate coefficients with residuals <1e-6.
    ///
    /// @param values Function values at grid points (size Nm × Nt × Nv × Nr)
    ///               Row-major layout: index = ((i*Nt + j)*Nv + k)*Nr + l
    /// @param tolerance Maximum residual per axis (default 1e-6)
    /// @return Fit result with coefficients and diagnostics
    BSplineFitResult4D fit(const std::vector<double>& values, double tolerance = 1e-6) {
        // Create separable fitter using factory pattern
        auto fitter_result = BSplineFitter4DSeparable::create(m_grid_, t_grid_, v_grid_, r_grid_);
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
            sep_result.max_residual_m,
            sep_result.max_residual_tau,
            sep_result.max_residual_sigma,
            sep_result.max_residual_r
        });

        // Return with full statistics
        return {
            .coefficients = sep_result.coefficients,
            .success = true,
            .error_message = "",
            .max_residual = max_residual,
            .max_residual_m = sep_result.max_residual_m,
            .max_residual_tau = sep_result.max_residual_tau,
            .max_residual_sigma = sep_result.max_residual_sigma,
            .max_residual_r = sep_result.max_residual_r,
            .condition_m = sep_result.condition_m,
            .condition_tau = sep_result.condition_tau,
            .condition_sigma = sep_result.condition_sigma,
            .condition_r = sep_result.condition_r,
            .failed_slices_m = sep_result.failed_slices_m,
            .failed_slices_tau = sep_result.failed_slices_tau,
            .failed_slices_sigma = sep_result.failed_slices_sigma,
            .failed_slices_r = sep_result.failed_slices_r
        };
    }

    /// Get detailed diagnostics from last fit (requires storing separable result)
    ///
    /// For now, use fit() which provides basic success/failure.
    /// Future: Add method to get per-axis condition numbers and residuals.

    /// Get grid dimensions
    [[nodiscard]] std::tuple<size_t, size_t, size_t, size_t> dimensions() const noexcept {
        return {Nm_, Nt_, Nv_, Nr_};
    }

private:
    /// Private constructor - use factory method create() instead
    ///
    /// @param m_grid Moneyness grid (already validated by factory)
    /// @param t_grid Maturity grid (already validated by factory)
    /// @param v_grid Volatility grid (already validated by factory)
    /// @param r_grid Rate grid (already validated by factory)
    BSplineFitter4D(std::vector<double> m_grid,
                    std::vector<double> t_grid,
                    std::vector<double> v_grid,
                    std::vector<double> r_grid)
        : m_grid_(std::move(m_grid)),
          t_grid_(std::move(t_grid)),
          v_grid_(std::move(v_grid)),
          r_grid_(std::move(r_grid)),
          Nm_(m_grid_.size()),
          Nt_(t_grid_.size()),
          Nv_(v_grid_.size()),
          Nr_(r_grid_.size())
    {
        // Pre-compute knot vectors (no validation needed - done by factory)
        tm_ = clamped_knots_cubic(m_grid_);
        tt_ = clamped_knots_cubic(t_grid_);
        tv_ = clamped_knots_cubic(v_grid_);
        tr_ = clamped_knots_cubic(r_grid_);
    }

    // Friend declaration for factory method to access private constructor
    friend expected<BSplineFitter4D, std::string> create(
        std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>);
    std::vector<double> m_grid_;  ///< Moneyness grid
    std::vector<double> t_grid_;  ///< Maturity grid
    std::vector<double> v_grid_;  ///< Volatility grid
    std::vector<double> r_grid_;  ///< Rate grid

    std::vector<double> tm_;  ///< Moneyness knot vector
    std::vector<double> tt_;  ///< Maturity knot vector
    std::vector<double> tv_;  ///< Volatility knot vector
    std::vector<double> tr_;  ///< Rate knot vector

    size_t Nm_;  ///< Number of moneyness points
    size_t Nt_;  ///< Number of maturity points
    size_t Nv_;  ///< Number of volatility points
    size_t Nr_;  ///< Number of rate points
};

}  // namespace mango
