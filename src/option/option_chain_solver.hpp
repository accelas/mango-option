/**
 * @file option_chain_solver.hpp
 * @brief Batch solver optimized for option chains (same S, T, σ, r, q; different K)
 *
 * An option chain is a set of options sharing all parameters except strike.
 * This solver exploits that structure for better performance:
 * - Workspace reuse: one SliceSolverWorkspace per chain
 * - Cache-friendly: sequential solving keeps workspace hot
 * - 10x less allocation: ~10 KB per chain vs ~10 KB per option
 */

#pragma once

#include "src/option/american_option.hpp"
#include "src/option/slice_solver_workspace.hpp"
#include "src/support/expected.hpp"
#include <vector>
#include <span>

namespace mango {

/**
 * Option chain configuration.
 *
 * Represents multiple options sharing all parameters except strike.
 * Typical use: All puts (or calls) for same underlying and expiration.
 */
struct AmericanOptionChain {
    double spot;                        ///< Current spot price (shared)
    double maturity;                    ///< Time to maturity in years (shared)
    double volatility;                  ///< Implied volatility (shared)
    double rate;                        ///< Risk-free rate (shared)
    double continuous_dividend_yield;   ///< Continuous dividend yield (shared)
    OptionType option_type;             ///< CALL or PUT (shared)
    std::vector<double> strikes;        ///< Strike prices [K₁, K₂, ..., Kₙ] (variable)

    /// Optional: discrete dividends (shared across all strikes)
    std::vector<std::pair<double, double>> discrete_dividends;

    /// Validate chain parameters
    expected<void, std::string> validate() const {
        if (spot <= 0.0) {
            return unexpected("Spot price must be positive");
        }
        if (maturity <= 0.0) {
            return unexpected("Maturity must be positive");
        }
        if (volatility <= 0.0) {
            return unexpected("Volatility must be positive");
        }
        if (continuous_dividend_yield < 0.0) {
            return unexpected("Continuous dividend yield must be non-negative");
        }
        if (strikes.empty()) {
            return unexpected("Chain must have at least one strike");
        }
        for (double k : strikes) {
            if (k <= 0.0) {
                return unexpected("All strikes must be positive");
            }
        }
        // Validate discrete dividends
        for (const auto& [time, amount] : discrete_dividends) {
            if (time < 0.0 || time > maturity) {
                return unexpected("Discrete dividend time must be in [0, maturity]");
            }
            if (amount < 0.0) {
                return unexpected("Discrete dividend amount must be non-negative");
            }
        }
        return {};
    }
};

/**
 * Result for one strike in a chain.
 */
struct ChainStrikeResult {
    double strike;                                          ///< Strike price
    expected<AmericanOptionResult, SolverError> result;     ///< Price + Greeks or error
};

/**
 * Batch solver optimized for option chains.
 *
 * Provides three modes:
 * 1. solve_chain() - Sequential within chain (workspace reuse)
 * 2. solve_chains() - Parallel across chains (default)
 * 3. solve_chains_advanced() - Thread pool with dynamic scheduling (future)
 */
class OptionChainSolver {
public:
    /**
     * Solve option chain sequentially with workspace reuse.
     *
     * Creates one SliceSolverWorkspace for entire chain and solves
     * all strikes sequentially. This keeps the workspace "hot" and
     * minimizes allocation overhead.
     *
     * Use when: Single chain, or when called from parallel context.
     *
     * Performance: ~10x less allocation, cache-friendly.
     *
     * @param chain Chain configuration (shared params, different strikes)
     * @param grid PDE grid configuration
     * @param trbdf2_config TR-BDF2 solver configuration
     * @param root_config Root finding configuration
     * @return Results for each strike (same order as chain.strikes)
     */
    static std::vector<ChainStrikeResult> solve_chain(
        const AmericanOptionChain& chain,
        const AmericanOptionGrid& grid,
        const TRBDF2Config& trbdf2_config = {},
        const RootFindingConfig& root_config = {});

    /**
     * Solve multiple option chains in parallel.
     *
     * Each chain is solved sequentially (workspace reuse), but chains
     * are processed in parallel using OpenMP. This is the recommended
     * mode for typical use cases.
     *
     * Parallelization strategy:
     * - Parallelize ACROSS chains (not within)
     * - Each thread gets one chain at a time
     * - Sequential solve within chain keeps workspace hot
     *
     * Use when: Multiple chains (typical case).
     *
     * Performance: Same per-chain benefit as solve_chain(), scaled across cores.
     *
     * @param chains Vector of option chains
     * @param grid PDE grid configuration (shared across all chains)
     * @param trbdf2_config TR-BDF2 solver configuration
     * @param root_config Root finding configuration
     * @return Results for each chain (same order as input)
     */
    static std::vector<std::vector<ChainStrikeResult>> solve_chains(
        std::span<const AmericanOptionChain> chains,
        const AmericanOptionGrid& grid,
        const TRBDF2Config& trbdf2_config = {},
        const RootFindingConfig& root_config = {});
};

}  // namespace mango
