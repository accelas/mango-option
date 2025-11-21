/**
 * @file price_table_solver_factory.hpp
 * @brief Factory for creating price table solvers with automatic strategy selection
 */

#pragma once

#include "src/option/option_spec.hpp"
#include <expected>
#include <string>
#include <vector>
#include <memory>

namespace mango {

/**
 * @brief Price table grid specification
 *
 * Represents a 4D grid of option parameters to be solved.
 * Output will be prices_4d[i, j, k, l] for (moneyness[i], maturity[j], volatility[k], rate[l]).
 */
struct PriceTableGrid {
    std::span<const double> moneyness;   ///< Moneyness grid (M/K_ref)
    std::span<const double> maturity;    ///< Time to maturity grid (years)
    std::span<const double> volatility;  ///< Volatility grid
    std::span<const double> rate;        ///< Interest rate grid
    double K_ref;                         ///< Reference strike price

    /// Get total number of grid points
    size_t size() const {
        return moneyness.size() * maturity.size() * volatility.size() * rate.size();
    }
};

/**
 * @brief Abstract interface for price table solvers
 *
 * Defines the contract that both normalized and batch solvers must implement.
 */
class IPriceTableSolver {
public:
    virtual ~IPriceTableSolver() = default;

    /**
     * @brief Solve for prices on 4D grid
     *
     * @param prices_4d Output array (Nm × Nt × Nv × Nr), must be pre-sized to grid.size()
     * @param grid Grid specification containing all parameter arrays
     * @return void on success, error message on failure
     */
    virtual std::expected<void, std::string> solve(
        std::span<double> prices_4d,
        const PriceTableGrid& grid) = 0;

    /**
     * @brief Get solver strategy name for diagnostics
     */
    virtual const char* strategy_name() const = 0;
};

/**
 * @brief Factory for creating price table solvers
 *
 * Responsibilities:
 * 1. Validate input configuration
 * 2. Check eligibility for normalized solver (fast path)
 * 3. Create appropriate solver implementation
 *
 * Usage:
 * ```cpp
 * OptionSolverGrid config{...};
 * auto factory_result = PriceTableSolverFactory::create(config, moneyness);
 *
 * if (!factory_result) {
 *     // Handle validation error
 *     return factory_result.error();
 * }
 *
 * auto solver = std::move(factory_result.value());
 * solver->solve(prices_4d, moneyness, maturity, volatility, rate, K_ref);
 * ```
 */
class PriceTableSolverFactory {
public:
    /**
     * @brief Create appropriate solver based on configuration
     *
     * This method:
     * 1. Validates the grid configuration
     * 2. Checks if normalized chain solver is eligible
     * 3. Creates either NormalizedChainSolver or BatchSolver
     *
     * @param config Grid configuration (validated)
     * @param moneyness Moneyness grid (used for eligibility check)
     * @return Solver instance or validation error
     */
    static std::expected<std::unique_ptr<IPriceTableSolver>, std::string> create(
        const OptionSolverGrid& config,
        std::span<const double> moneyness);

private:
    /**
     * @brief Validate grid configuration
     */
    static std::expected<void, std::string> validate_config(
        const OptionSolverGrid& config);

    /**
     * @brief Check if normalized solver is eligible
     */
    static bool is_normalized_solver_eligible(
        const OptionSolverGrid& config,
        std::span<const double> moneyness);
};

} // namespace mango
