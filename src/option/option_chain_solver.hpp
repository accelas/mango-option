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

}  // namespace mango
