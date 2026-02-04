// SPDX-License-Identifier: MIT
/**
 * @file american_option_result.hpp
 * @brief AmericanOptionResult wrapper class for Grid + PricingParams
 *
 * This class wraps Grid<double> and PricingParams to provide a unified
 * interface for option pricing results. It owns the grid and pricing
 * parameters, implements value interpolation, and delegates snapshot
 * queries to the underlying Grid.
 */

#pragma once

#include "mango/pde/core/grid.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/option_concepts.hpp"
#include "mango/pde/operators/centered_difference_facade.hpp"
#include "mango/math/cubic_spline_solver.hpp"
#include <memory>
#include <optional>
#include <span>
#include <cmath>

namespace mango {

/**
 * @brief American option pricing result with interpolation and Greeks
 *
 * Wraps Grid<double> and PricingParams to provide:
 * - Value interpolation at arbitrary spot prices
 * - Greeks computation (delta, gamma, theta)
 * - Snapshot query delegation
 * - Direct grid access for advanced users
 *
 * Thread-safety: Const methods are thread-safe (read-only access).
 * Non-const methods should not be called concurrently.
 */
class AmericanOptionResult {
public:
    /**
     * @brief Construct result from grid and pricing params
     *
     * @param grid Shared pointer to Grid (with solution storage)
     * @param params Pricing parameters (spot, strike, maturity, etc.)
     */
    AmericanOptionResult(std::shared_ptr<Grid<double>> grid,
                         const PricingParams& params);

    // Movable but not copyable (owns shared_ptr)
    AmericanOptionResult(const AmericanOptionResult&) = delete;
    AmericanOptionResult& operator=(const AmericanOptionResult&) = delete;
    AmericanOptionResult(AmericanOptionResult&&) = default;
    AmericanOptionResult& operator=(AmericanOptionResult&&) = default;

    // Backward compatibility: converged field (always true if object exists)
    bool converged = true;

    // Pricing parameter accessors
    double spot() const { return params_.spot; }
    double strike() const { return params_.strike; }
    double maturity() const { return params_.maturity; }
    const RateSpec& rate() const { return params_.rate; }
    double dividend_yield() const { return params_.dividend_yield; }
    OptionType option_type() const { return params_.option_type; }
    double volatility() const { return params_.volatility; }

    /**
     * @brief Get option value at current spot price
     *
     * Convenience method that calls value_at(spot()).
     *
     * @return Option value at current spot
     */
    double value() const;

    /**
     * @brief Get option value at arbitrary spot price
     *
     * Uses cubic spline interpolation in log-moneyness space: x = ln(S/K).
     * Converts normalized price V/K to actual price: V = (V/K) * K.
     *
     * @param spot_price Spot price to evaluate at
     * @return Option value at given spot price
     */
    double value_at(double spot_price) const;

    /**
     * @brief Compute delta: ∂V/∂S
     *
     * Uses cubic spline derivative in log-moneyness space.
     * Delta = (∂V_norm/∂x) * (K/S) where x = ln(S/K).
     *
     * @return Delta at current spot price
     */
    double delta() const;

    /**
     * @brief Compute gamma: ∂²V/∂S²
     *
     * Uses stencil-based second derivative with linear interpolation
     * to the exact spot point.
     * Gamma = (K/S²) * [∂²V_norm/∂x² - ∂V_norm/∂x]
     *
     * @return Gamma at current spot price
     */
    double gamma() const;

    /**
     * @brief Compute theta: ∂V/∂t
     *
     * Uses backward finite difference in time from the two solution snapshots
     * stored in the Grid (current and previous timestep).
     *
     * θ ≈ (V(t+dt) - V(t)) / dt
     *
     * This gives negative theta for time decay (option loses value as time passes).
     *
     * @return Theta at current spot price (typically negative for time decay)
     */
    double theta() const;

    // Snapshot query delegation
    bool has_snapshots() const { return grid_->has_snapshots(); }
    size_t num_snapshots() const { return grid_->num_snapshots(); }
    std::span<const double> at_time(size_t snapshot_idx) const {
        return grid_->at(snapshot_idx);
    }
    std::span<const double> snapshot_times() const {
        return grid_->snapshot_times();
    }

    // Direct grid access for advanced users
    std::shared_ptr<Grid<double>> grid() const { return grid_; }

private:
    /// Build cubic spline from solution on first use
    void ensure_spline() const;

    /// Build cubic spline on an arbitrary solution array
    void build_spline(CubicSpline<double>& spline,
                      std::span<const double> solution) const;

    /// Lazy initialize CenteredDifference operator (for gamma stencil)
    void ensure_operator() const;

    /// Find grid index for linear interpolation of stencil output
    std::pair<size_t, size_t> find_grid_index(double x) const;

    std::shared_ptr<Grid<double>> grid_;
    PricingParams params_;

    // Lazy-initialized cubic spline for value/delta interpolation
    mutable CubicSpline<double> spline_;
    mutable bool spline_built_ = false;

    // Lazy-initialized operator for gamma stencil
    mutable std::unique_ptr<operators::CenteredDifference<double>> operator_;
};

static_assert(OptionResult<AmericanOptionResult>);

} // namespace mango
