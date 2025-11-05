/**
 * @file american_option.hpp
 * @brief American option pricing solver using finite difference method
 */

#ifndef MANGO_AMERICAN_OPTION_HPP
#define MANGO_AMERICAN_OPTION_HPP

#include "src/cpp/pde_solver.hpp"
#include "src/cpp/spatial_operators.hpp"
#include "src/cpp/american_obstacle.hpp"
#include "src/cpp/dividend_jump.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>

namespace mango {

/**
 * Option type enumeration.
 */
enum class OptionType {
    CALL,
    PUT
};

/**
 * American option pricing parameters.
 */
struct AmericanOptionParams {
    double strike;           ///< Strike price (dollars)
    double spot;             ///< Current stock price (dollars)
    double maturity;         ///< Time to maturity (years)
    double volatility;       ///< Implied volatility (fraction)
    double rate;             ///< Risk-free rate (fraction)
    double dividend_yield;   ///< Continuous dividend yield (fraction)
    OptionType option_type;  ///< Call or Put

    /// Validate parameters
    void validate() const {
        if (strike <= 0.0) throw std::invalid_argument("Strike must be positive");
        if (spot <= 0.0) throw std::invalid_argument("Spot must be positive");
        if (maturity <= 0.0) throw std::invalid_argument("Maturity must be positive");
        if (volatility <= 0.0) throw std::invalid_argument("Volatility must be positive");
        if (rate < 0.0) throw std::invalid_argument("Rate must be non-negative");
        if (dividend_yield < 0.0) throw std::invalid_argument("Dividend yield must be non-negative");
    }
};

/**
 * Numerical grid parameters for PDE solver.
 */
struct AmericanOptionGrid {
    size_t n_space;    ///< Number of spatial grid points
    size_t n_time;     ///< Number of time steps
    double x_min;      ///< Minimum log-moneyness (default: -3.0)
    double x_max;      ///< Maximum log-moneyness (default: +3.0)

    /// Default constructor with sensible defaults
    AmericanOptionGrid()
        : n_space(101)
        , n_time(1000)
        , x_min(-3.0)
        , x_max(3.0) {}

    void validate() const {
        if (n_space < 10) throw std::invalid_argument("n_space must be >= 10");
        if (n_time < 10) throw std::invalid_argument("n_time must be >= 10");
        if (x_min >= x_max) throw std::invalid_argument("x_min must be < x_max");
    }
};

/**
 * Solver result containing option value and Greeks.
 */
struct AmericanOptionResult {
    double value;    ///< Option value (dollars)
    double delta;    ///< V/S (first derivative wrt spot)
    double gamma;    ///< �V/S� (second derivative wrt spot)
    double theta;    ///< V/t (time decay)
    bool converged;  ///< Solver convergence status

    /// Default constructor
    AmericanOptionResult()
        : value(0.0), delta(0.0), gamma(0.0), theta(0.0), converged(false) {}
};

/**
 * American option pricing solver using finite difference method.
 *
 * Solves the Black-Scholes PDE with obstacle constraints in log-moneyness
 * coordinates using TR-BDF2 time stepping and projection method for
 * early exercise boundary.
 */
class AmericanOptionSolver {
public:
    /**
     * Constructor.
     *
     * @param params Option pricing parameters
     * @param grid Numerical grid parameters
     * @param trbdf2_config TR-BDF2 solver configuration
     * @param root_config Root finding configuration for Newton solver
     */
    AmericanOptionSolver(const AmericanOptionParams& params,
                        const AmericanOptionGrid& grid,
                        const TRBDF2Config& trbdf2_config = {},
                        const RootFindingConfig& root_config = {});

    /**
     * Register discrete dividend payment.
     *
     * @param time Time of dividend payment (years from now)
     * @param amount Dividend amount (dollars)
     */
    void register_dividend(double time, double amount);

    /**
     * Solve for option value and Greeks.
     *
     * @return Result containing option value and Greeks
     */
    AmericanOptionResult solve();

    /**
     * Get the full solution surface (for debugging/analysis).
     *
     * @return Vector of option values across the spatial grid
     */
    std::vector<double> get_solution() const;

private:
    // Parameters
    AmericanOptionParams params_;
    AmericanOptionGrid grid_;

    // Dividend schedule
    std::vector<std::pair<double, double>> dividends_;  // (time, amount)

    // Solution state
    std::vector<double> solution_;
    bool solved_ = false;

    // Helper methods (to be implemented in Task 8)
    double compute_delta() const;
    double compute_gamma() const;
    double compute_theta() const;
};

}  // namespace mango

#endif  // MANGO_AMERICAN_OPTION_HPP
