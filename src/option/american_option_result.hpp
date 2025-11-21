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

#include "src/pde/core/grid.hpp"
#include "src/option/option_spec.hpp"
#include "src/pde/operators/centered_difference_facade.hpp"
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

    // Pricing parameter accessors
    double spot() const { return params_.spot; }
    double strike() const { return params_.strike; }
    double maturity() const { return params_.maturity; }
    double rate() const { return params_.rate; }
    double dividend_yield() const { return params_.dividend_yield; }
    OptionType option_type() const { return params_.type; }
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
     * Uses linear interpolation in log-moneyness space: x = ln(S/K).
     * Converts normalized price V/K to actual price: V = (V/K) * K.
     *
     * @param spot_price Spot price to evaluate at
     * @return Option value at given spot price
     */
    double value_at(double spot_price) const;

    /**
     * @brief Compute delta: ∂V/∂S
     *
     * Uses first derivative operator in log-moneyness space.
     * Delta = (∂V/∂x) / S where x = ln(S/K).
     *
     * Lazy initialization: Creates CenteredDifference operator on first call.
     *
     * @return Delta at current spot price
     */
    double delta() const;

    /**
     * @brief Compute gamma: ∂²V/∂S²
     *
     * Uses second derivative operator in log-moneyness space.
     * Gamma ≈ (∂²V/∂x²) / S²
     *
     * Lazy initialization: Creates CenteredDifference operator on first call.
     *
     * @return Gamma at current spot price
     */
    double gamma() const;

    /**
     * @brief Compute theta: ∂V/∂t
     *
     * STUB: Returns 0.0 for now.
     * Future implementation will use finite difference from solution_prev().
     *
     * @return Theta at current spot price (currently 0.0)
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
    /**
     * @brief Find grid index for log-moneyness x = ln(S/K)
     *
     * Returns the left index for linear interpolation.
     *
     * @param x Log-moneyness
     * @return Pair of (left_index, right_index) for interpolation
     */
    std::pair<size_t, size_t> find_grid_index(double x) const;

    /**
     * @brief Linear interpolation between two grid points
     *
     * @param x Log-moneyness to evaluate at
     * @param i_left Left grid index
     * @param i_right Right grid index
     * @return Interpolated value
     */
    double interpolate(double x, size_t i_left, size_t i_right) const;

    /**
     * @brief Lazy initialize CenteredDifference operator
     *
     * Creates operator on first call to delta() or gamma().
     */
    void ensure_operator() const;

    std::shared_ptr<Grid<double>> grid_;
    PricingParams params_;

    // Lazy-initialized operator for Greeks
    mutable std::unique_ptr<operators::CenteredDifference<double>> operator_;
};

} // namespace mango
