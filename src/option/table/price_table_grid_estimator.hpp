/**
 * @file price_table_grid_estimator.hpp
 * @brief Automatic grid estimation for price table B-spline interpolation
 *
 * Estimates optimal grid density for each dimension based on target IV error.
 * Uses curvature-based budget allocation:
 * - Volatility: highest curvature (vega non-linearity)
 * - Moneyness: high near ATM (gamma peak)
 * - Maturity: medium (sqrt-tau behavior)
 * - Rate: lowest (nearly linear discounting)
 *
 * Theoretical basis: cubic B-spline interpolation error is O(h^4 * f''''(x))
 * where h is grid spacing and f'''' is 4th derivative (curvature).
 */

#pragma once

#include <array>
#include <cmath>
#include <vector>
#include <algorithm>

namespace mango {

/**
 * @brief Grid estimation accuracy parameters for N-dimensional price table
 *
 * Controls the tradeoff between accuracy and computation cost.
 * The estimator uses curvature-based weights to allocate grid points
 * where the price surface has highest variation.
 *
 * @tparam N Number of dimensions
 */
template <size_t N>
struct PriceTableGridAccuracyParams {
    /// Target IV error in absolute terms (default: 10 bps = 0.001)
    /// - 0.001 (10 bps): Fast, ~50-100 PDE solves
    /// - 0.0005 (5 bps): Medium, ~100-200 PDE solves
    /// - 0.0001 (1 bps): High accuracy, ~300-600 PDE solves
    double target_iv_error = 0.001;

    /// Minimum points per dimension (B-spline requires >= 4)
    size_t min_points = 4;

    /// Maximum points per dimension (cost control)
    size_t max_points = 50;

    /// Curvature weights for budget allocation (dimension-specific)
    /// Based on analysis of price surface 4th derivatives.
    /// Default weights must be provided by specialization or user.
    std::array<double, N> curvature_weights = {};

    /// Scale factor calibrated from benchmark data
    /// Empirically: 13x18x8 grid gives ~4 bps error
    /// This factor maps target_iv_error to base grid points
    double scale_factor = 1e-6;
};

/**
 * @brief Specialization for 4D price table [m, tau, sigma, r]
 *
 * Default curvature weights based on analysis:
 * - sigma: 1.5 (highest curvature, vega non-linearity)
 * - m: 1.0 (log-transform handles ATM curvature, moderate weight)
 * - tau: 1.0 (baseline, moderate curvature)
 * - r: 0.6 (nearly linear, lowest curvature)
 *
 * Scale factor calibration:
 * - Empirical: 13×18×8 grid achieves ~4.3 bps average error
 * - With weights [1.0, 1.0, 1.5, 0.6], base_points=12 gives n_sigma=18
 * - Formula: base_points = (scale_factor / target_error)^0.25
 * - For target=0.0001 (1 bps) and base=12: scale = 12^4 * 0.0001 = 2.0736
 */
template <>
struct PriceTableGridAccuracyParams<4> {
    double target_iv_error = 0.001;
    size_t min_points = 4;
    size_t max_points = 50;
    std::array<double, 4> curvature_weights = {1.0, 1.0, 1.5, 0.6};  // [m, tau, sigma, r]
    double scale_factor = 2.0;  // Calibrated from benchmark: 12^4 * 0.0001 ≈ 2
};

/**
 * @brief Result of N-dimensional grid estimation
 *
 * Contains recommended grid vectors for each dimension and
 * estimated computational cost.
 *
 * @tparam N Number of dimensions
 */
template <size_t N>
struct PriceTableGridEstimate {
    std::array<std::vector<double>, N> grids;  ///< Grid vectors for each dimension
    size_t estimated_pde_solves = 0;           ///< Estimated computational cost
};

/**
 * @brief Specialization for 4D price table with named accessors
 *
 * Provides convenient named accessors for the standard 4D case.
 */
template <>
struct PriceTableGridEstimate<4> {
    std::array<std::vector<double>, 4> grids;  ///< Grid vectors [m, tau, sigma, r]
    size_t estimated_pde_solves = 0;           ///< n_vol * n_rate (PDE solves per slice)

    /// Named accessors for clarity
    std::vector<double>& moneyness_grid() { return grids[0]; }
    std::vector<double>& maturity_grid() { return grids[1]; }
    std::vector<double>& volatility_grid() { return grids[2]; }
    std::vector<double>& rate_grid() { return grids[3]; }

    const std::vector<double>& moneyness_grid() const { return grids[0]; }
    const std::vector<double>& maturity_grid() const { return grids[1]; }
    const std::vector<double>& volatility_grid() const { return grids[2]; }
    const std::vector<double>& rate_grid() const { return grids[3]; }
};

namespace detail {

/// Generate uniform grid
inline std::vector<double> uniform_grid(double min_val, double max_val, size_t n) {
    std::vector<double> grid(n);
    for (size_t i = 0; i < n; ++i) {
        grid[i] = min_val + (max_val - min_val) * static_cast<double>(i) / static_cast<double>(n - 1);
    }
    return grid;
}

/// Generate log-uniform grid (uniform in log-space)
inline std::vector<double> log_uniform_grid(double min_val, double max_val, size_t n) {
    std::vector<double> grid(n);
    double log_min = std::log(min_val);
    double log_max = std::log(max_val);
    for (size_t i = 0; i < n; ++i) {
        double log_val = log_min + (log_max - log_min) * static_cast<double>(i) / static_cast<double>(n - 1);
        grid[i] = std::exp(log_val);
    }
    return grid;
}

/// Generate sqrt-uniform grid (uniform in sqrt-space, concentrates near min)
inline std::vector<double> sqrt_uniform_grid(double min_val, double max_val, size_t n) {
    std::vector<double> grid(n);
    double sqrt_min = std::sqrt(min_val);
    double sqrt_max = std::sqrt(max_val);
    for (size_t i = 0; i < n; ++i) {
        double sqrt_val = sqrt_min + (sqrt_max - sqrt_min) * static_cast<double>(i) / static_cast<double>(n - 1);
        grid[i] = sqrt_val * sqrt_val;
    }
    return grid;
}

}  // namespace detail

/**
 * @brief Estimate optimal grid for 4D price table based on target accuracy
 *
 * Uses the relationship between B-spline interpolation error and grid spacing:
 * error ~ h^4 * f''''(x)
 *
 * Allocates more points to dimensions with higher curvature (4th derivative).
 *
 * Grid spacing strategies by dimension:
 * - Moneyness: log-uniform (matches internal log-moneyness storage)
 * - Maturity: sqrt-uniform (concentrates near short maturities)
 * - Volatility: uniform (highest curvature dimension)
 * - Rate: uniform (lowest curvature, nearly linear)
 *
 * @param m_min Minimum moneyness (e.g., 0.8)
 * @param m_max Maximum moneyness (e.g., 1.2)
 * @param tau_min Minimum maturity in years (e.g., 0.01)
 * @param tau_max Maximum maturity in years (e.g., 2.0)
 * @param sigma_min Minimum volatility (e.g., 0.05)
 * @param sigma_max Maximum volatility (e.g., 0.50)
 * @param r_min Minimum rate (e.g., 0.01)
 * @param r_max Maximum rate (e.g., 0.06)
 * @param params Accuracy parameters
 * @return Grid estimate with recommended vectors and cost estimate
 */
inline PriceTableGridEstimate<4> estimate_grid_for_price_table(
    double m_min, double m_max,
    double tau_min, double tau_max,
    double sigma_min, double sigma_max,
    double r_min, double r_max,
    const PriceTableGridAccuracyParams<4>& params = {})
{
    // Step 1: Calculate base points from target error
    // Using h^4 error relationship: n ~ (scale / error)^(1/4)
    double base_points = std::pow(params.scale_factor / params.target_iv_error, 0.25);

    // Step 2: Apply curvature weights to allocate budget per dimension
    auto clamp_points = [&](double weighted) -> size_t {
        size_t n = static_cast<size_t>(std::ceil(base_points * weighted));
        return std::clamp(n, params.min_points, params.max_points);
    };

    size_t n_m = clamp_points(params.curvature_weights[0]);
    size_t n_tau = clamp_points(params.curvature_weights[1]);
    size_t n_sigma = clamp_points(params.curvature_weights[2]);
    size_t n_rate = clamp_points(params.curvature_weights[3]);

    // Step 3: Generate grid vectors with dimension-appropriate spacing
    PriceTableGridEstimate<4> estimate;

    // Moneyness: log-uniform (matches internal log-moneyness storage)
    estimate.grids[0] = detail::log_uniform_grid(m_min, m_max, n_m);

    // Maturity: sqrt-uniform (concentrates near short maturities)
    estimate.grids[1] = detail::sqrt_uniform_grid(tau_min, tau_max, n_tau);

    // Volatility: uniform (highest curvature dimension)
    estimate.grids[2] = detail::uniform_grid(sigma_min, sigma_max, n_sigma);

    // Rate: uniform (lowest curvature, nearly linear)
    estimate.grids[3] = detail::uniform_grid(r_min, r_max, n_rate);

    // Estimated PDE solves: one solve per (sigma, rate) pair
    estimate.estimated_pde_solves = n_sigma * n_rate;

    return estimate;
}

/**
 * @brief Estimate grid from domain bounds extracted from option chain
 *
 * Convenience overload that extracts bounds from min/max of input vectors.
 *
 * @param strikes Strike prices (used with spot to compute moneyness bounds)
 * @param spot Current underlying price
 * @param maturities Available maturities
 * @param vols Implied volatility range
 * @param rates Interest rate range
 * @param params Accuracy parameters
 * @return Grid estimate
 */
inline PriceTableGridEstimate<4> estimate_grid_from_chain_bounds(
    const std::vector<double>& strikes,
    double spot,
    const std::vector<double>& maturities,
    const std::vector<double>& vols,
    const std::vector<double>& rates,
    const PriceTableGridAccuracyParams<4>& params = {})
{
    // Compute moneyness bounds from strikes
    double m_min = spot / *std::max_element(strikes.begin(), strikes.end());
    double m_max = spot / *std::min_element(strikes.begin(), strikes.end());

    // Ensure m_min < m_max (handles edge cases)
    if (m_min > m_max) std::swap(m_min, m_max);

    // Add small padding to ensure interpolation doesn't extrapolate
    m_min *= 0.99;
    m_max *= 1.01;

    // Get bounds for other dimensions
    auto [tau_min_it, tau_max_it] = std::minmax_element(maturities.begin(), maturities.end());
    auto [sigma_min_it, sigma_max_it] = std::minmax_element(vols.begin(), vols.end());
    auto [r_min_it, r_max_it] = std::minmax_element(rates.begin(), rates.end());

    double tau_min = *tau_min_it * 0.9;  // Pad to avoid edge effects
    double tau_max = *tau_max_it * 1.1;
    double sigma_min = std::max(0.01, *sigma_min_it * 0.9);
    double sigma_max = *sigma_max_it * 1.1;
    double r_min = *r_min_it - 0.005;
    double r_max = *r_max_it + 0.005;

    return estimate_grid_for_price_table(
        m_min, m_max,
        tau_min, tau_max,
        sigma_min, sigma_max,
        r_min, r_max,
        params);
}

}  // namespace mango
