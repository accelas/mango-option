// SPDX-License-Identifier: MIT
/**
 * @file american_option_result.cpp
 * @brief Implementation of AmericanOptionResult wrapper class
 */

#include "mango/option/american_option_result.hpp"
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
    ensure_spline();

    // Convert to log-moneyness: x = ln(S/K)
    double x = std::log(spot_price / params_.strike);

    // Boundary cases: clamp to grid domain
    auto x_grid = grid_->x();
    if (x <= x_grid[0]) {
        return grid_->solution()[0] * params_.strike;
    }
    if (x >= x_grid[x_grid.size() - 1]) {
        return grid_->solution()[x_grid.size() - 1] * params_.strike;
    }

    // Cubic spline interpolation (clamp to non-negative: option values >= 0)
    double value_normalized = std::max(0.0, spline_.eval(x));

    // Convert from normalized (V/K) to actual price
    return value_normalized * params_.strike;
}

double AmericanOptionResult::delta() const {
    ensure_spline();

    // Evaluate spline derivative at spot in log-moneyness
    double x_spot = std::log(params_.spot / params_.strike);
    double dv_norm_dx = spline_.eval_derivative(x_spot);

    // Convert to delta: ∂V/∂S = (K/S) · ∂V_norm/∂x
    return dv_norm_dx * (params_.strike / params_.spot);
}

double AmericanOptionResult::gamma() const {
    ensure_operator();

    auto solution = grid_->solution();
    auto x_grid = grid_->x();
    const size_t n = x_grid.size();

    double x_spot = std::log(params_.spot / params_.strike);

    // Compute stencil derivatives at all interior nodes
    std::vector<double> dv_dx(n);
    std::vector<double> d2v_dx2(n);
    operator_->compute_first_derivative(
        solution, std::span(dv_dx), 1, n - 2);
    operator_->compute_second_derivative(
        solution, std::span(d2v_dx2), 1, n - 2);

    // Linearly interpolate stencil output at x_spot
    auto [i_left, i_right] = find_grid_index(x_spot);

    double dv_dx_at_spot;
    double d2v_dx2_at_spot;

    if (i_left == i_right) {
        dv_dx_at_spot = dv_dx[i_left];
        d2v_dx2_at_spot = d2v_dx2[i_left];
    } else {
        double alpha = (x_spot - x_grid[i_left]) /
                       (x_grid[i_right] - x_grid[i_left]);
        dv_dx_at_spot = dv_dx[i_left] + alpha * (dv_dx[i_right] - dv_dx[i_left]);
        d2v_dx2_at_spot = d2v_dx2[i_left] + alpha * (d2v_dx2[i_right] - d2v_dx2[i_left]);
    }

    // Gamma = (K/S²) · [∂²V_norm/∂x² - ∂V_norm/∂x]
    double K_over_S2 = params_.strike / (params_.spot * params_.spot);
    return std::fma(K_over_S2, d2v_dx2_at_spot, -K_over_S2 * dv_dx_at_spot);
}

double AmericanOptionResult::theta() const {
    ensure_spline();

    // Theta = ∂V/∂t via backward finite difference
    // solution() = V at τ=0 (current), solution_prev() = V at τ=dt
    // θ ≈ (V_prev - V_current) / dt  (negative for time decay)

    double x_spot = std::log(params_.spot / params_.strike);

    // Interpolate current solution via cached spline
    double v_current = spline_.eval(x_spot);

    // Build temporary spline for previous solution
    CubicSpline<double> prev_spline;
    build_spline(prev_spline, grid_->solution_prev());
    double v_prev = prev_spline.eval(x_spot);

    double dt = grid_->dt();
    double theta_normalized = (v_prev - v_current) / dt;

    return theta_normalized * params_.strike;
}

std::pair<size_t, size_t> AmericanOptionResult::find_grid_index(double x) const {
    auto x_grid = grid_->x();
    size_t n = x_grid.size();

    if (x <= x_grid[0]) {
        return {0, 0};
    }
    if (x >= x_grid[n - 1]) {
        return {n - 1, n - 1};
    }

    auto it = std::lower_bound(x_grid.begin(), x_grid.end(), x);
    size_t i_right = std::distance(x_grid.begin(), it);
    size_t i_left = (i_right > 0) ? i_right - 1 : 0;

    if (std::abs(x_grid[i_right] - x) < 1e-14) {
        return {i_right, i_right};
    }

    return {i_left, i_right};
}

void AmericanOptionResult::build_spline(CubicSpline<double>& spline,
                                         std::span<const double> solution) const {
    auto x_grid = grid_->x();
    auto error = spline.build(x_grid, solution);
    assert(!error.has_value() && "Cubic spline build should not fail on valid grid data");
    (void)error;
}

void AmericanOptionResult::ensure_spline() const {
    if (!spline_built_) {
        build_spline(spline_, grid_->solution());
        spline_built_ = true;
    }
}

void AmericanOptionResult::ensure_operator() const {
    if (!operator_) {
        const auto& spacing = grid_->spacing();
        operator_ = std::make_unique<operators::CenteredDifference<double>>(spacing);
    }
}

} // namespace mango
