/**
 * @file american_option_result.cpp
 * @brief Implementation of AmericanOptionResult wrapper class
 */

#include "american_option_result.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace mango {

AmericanOptionResult::AmericanOptionResult(
    std::shared_ptr<Grid<double>> grid,
    const PricingParams& params)
    : grid_(std::move(grid))
    , params_(params)
    , operator_(nullptr)
{
    assert(grid_ && "Grid must not be null");
}

double AmericanOptionResult::value() const {
    return value_at(params_.spot);
}

double AmericanOptionResult::value_at(double spot_price) const {
    // Convert to log-moneyness: x = ln(S/K)
    double x = std::log(spot_price / params_.strike);

    // Find grid indices for interpolation
    auto [i_left, i_right] = find_grid_index(x);

    // Check boundary cases
    auto x_grid = grid_->x();
    if (x <= x_grid[0]) {
        // Below grid: return left boundary value
        return grid_->solution()[0] * params_.strike;
    }
    if (x >= x_grid[x_grid.size() - 1]) {
        // Above grid: return right boundary value
        return grid_->solution()[x_grid.size() - 1] * params_.strike;
    }

    // Linear interpolation
    double value_normalized = interpolate(x, i_left, i_right);

    // Convert from normalized (V/K) to actual price
    return value_normalized * params_.strike;
}

double AmericanOptionResult::delta() const {
    ensure_operator();

    // Get current solution (normalized by K)
    auto solution = grid_->solution();
    auto x_grid = grid_->x();

    // Find index corresponding to current spot
    double x_spot = std::log(params_.spot / params_.strike);
    auto [i_left, i_right] = find_grid_index(x_spot);

    // For simplicity, use the left index (closest point)
    // TODO: Could improve by interpolating derivatives
    size_t idx = i_left;

    // Clamp to interior points (need i-1 and i+1 for derivative)
    idx = std::max<size_t>(1, std::min(idx, x_grid.size() - 2));

    // Compute first derivative: dV/dx
    std::vector<double> dv_dx(x_grid.size());
    operator_->compute_first_derivative(
        solution, std::span(dv_dx), 1, x_grid.size() - 2);

    // Convert to delta: dV/dS = (dV/dx) / S
    // Since V is normalized by K, we need: delta = (dV_normalized/dx) * (K/S)
    double delta_normalized = dv_dx[idx];
    double delta = delta_normalized * (params_.strike / params_.spot);

    return delta;
}

double AmericanOptionResult::gamma() const {
    ensure_operator();

    // Get current solution (normalized by K)
    auto solution = grid_->solution();
    auto x_grid = grid_->x();

    // Find index corresponding to current spot
    double x_spot = std::log(params_.spot / params_.strike);
    auto [i_left, i_right] = find_grid_index(x_spot);

    // Use left index (closest point)
    size_t idx = i_left;

    // Clamp to interior points
    idx = std::max<size_t>(1, std::min(idx, x_grid.size() - 2));

    // Compute first and second derivatives: dV/dx and d²V/dx²
    std::vector<double> dv_dx(x_grid.size());
    std::vector<double> d2v_dx2(x_grid.size());
    operator_->compute_first_derivative(
        solution, std::span(dv_dx), 1, x_grid.size() - 2);
    operator_->compute_second_derivative(
        solution, std::span(d2v_dx2), 1, x_grid.size() - 2);

    // Convert to gamma using correct change-of-variables formula:
    // Gamma = ∂²V/∂S² = (K/S²) * [∂²V/∂x² - ∂V/∂x]
    //
    // Derivation: Given x = ln(S/K), V = K * V_normalized(x)
    // ∂V/∂S = (K/S) * ∂V_normalized/∂x
    // ∂²V/∂S² = ∂/∂S[(K/S) * ∂V_normalized/∂x]
    //         = (K/S²) * [∂²V_normalized/∂x² - ∂V_normalized/∂x]
    //
    // This matches AmericanOptionSolver::compute_gamma() implementation.
    double K_over_S2 = params_.strike / (params_.spot * params_.spot);
    return std::fma(K_over_S2, d2v_dx2[idx], -K_over_S2 * dv_dx[idx]);
}

double AmericanOptionResult::theta() const {
    // Theta computation requires temporal finite differences from successive solution snapshots.
    // This is not yet implemented - would need Grid to store solution at t and t-dt.
    throw std::runtime_error(
        "theta() is not yet implemented. "
        "Requires temporal finite differences from successive solution snapshots.");
}

std::pair<size_t, size_t> AmericanOptionResult::find_grid_index(double x) const {
    auto x_grid = grid_->x();
    size_t n = x_grid.size();

    // Handle boundary cases
    if (x <= x_grid[0]) {
        return {0, 0};
    }
    if (x >= x_grid[n - 1]) {
        return {n - 1, n - 1};
    }

    // Binary search for left index
    auto it = std::lower_bound(x_grid.begin(), x_grid.end(), x);

    // lower_bound returns first element >= x
    // We want the element just before it for left index
    size_t i_right = std::distance(x_grid.begin(), it);
    size_t i_left = (i_right > 0) ? i_right - 1 : 0;

    // If we're exactly on a grid point, return that index for both
    if (std::abs(x_grid[i_right] - x) < 1e-14) {
        return {i_right, i_right};
    }

    return {i_left, i_right};
}

double AmericanOptionResult::interpolate(double x, size_t i_left, size_t i_right) const {
    auto x_grid = grid_->x();
    auto solution = grid_->solution();

    // Handle exact match
    if (i_left == i_right) {
        return solution[i_left];
    }

    // Linear interpolation
    double x_left = x_grid[i_left];
    double x_right = x_grid[i_right];
    double v_left = solution[i_left];
    double v_right = solution[i_right];

    double alpha = (x - x_left) / (x_right - x_left);
    return v_left + alpha * (v_right - v_left);
}

void AmericanOptionResult::ensure_operator() const {
    if (!operator_) {
        // Get GridSpacing from Grid (it already has one)
        const auto& spacing = grid_->spacing();

        // Create CenteredDifference operator with auto backend selection
        operator_ = std::make_unique<operators::CenteredDifference<double>>(spacing);
    }
}

} // namespace mango
