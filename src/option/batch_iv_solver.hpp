#pragma once

#include "src/option/iv_solver_fdm.hpp"
#include "src/option/iv_types.hpp"
#include "src/option/option_spec.hpp"
#include <span>
#include <vector>

namespace mango {

/// Batch Implied Volatility Solver
///
/// Solves implied volatility for multiple options in parallel using OpenMP.
/// This is significantly faster than solving options sequentially.
///
/// Example usage:
/// ```cpp
/// std::vector<IVQuery> batch = { ... };
/// IVSolverFDMConfig config;  // Shared configuration
///
/// auto results = solve_implied_vol_batch(batch, config);
/// ```
///
/// Performance:
/// - Single-threaded: ~7 IVs/sec (101x1000 grid)
/// - Parallel (32 cores): ~107 IVs/sec (15.3x speedup)
///
/// Use cases:
/// - Volatility surface construction: Calculate IV for entire grid of strikes/maturities
/// - Market data processing: Batch-process option chains
/// - Risk calculations: Compute sensitivities across multiple scenarios
/// - Model calibration: Evaluate objective function for optimization
class BatchIVSolver {
public:
    /// Solve implied volatility for a batch of options in parallel
    ///
    /// @param queries Vector of IV queries (option specs and market prices)
    /// @param config Shared configuration (grid size, tolerances)
    /// @return Vector of IV results (same order as input)
    static std::vector<IVResult> solve_batch(
        std::span<const IVQuery> queries,
        const IVSolverFDMConfig& config)
    {
        std::vector<IVResult> results(queries.size());
        
        IVSolverFDM solver(config);
        auto batch_result = solver.solve_batch(queries, results);
        
        if (!batch_result) {
            // This shouldn't happen since we sized results correctly,
            // but handle gracefully
            throw std::runtime_error(batch_result.error());
        }
        
        return results;
    }

    /// Solve implied volatility for a batch of options (vector overload)
    static std::vector<IVResult> solve_batch(
        const std::vector<IVQuery>& queries,
        const IVSolverFDMConfig& config)
    {
        return solve_batch(std::span{queries}, config);
    }
};

/// Convenience function for batch IV solving
inline std::vector<IVResult> solve_implied_vol_batch(
    std::span<const IVQuery> queries,
    const IVSolverFDMConfig& config)
{
    return BatchIVSolver::solve_batch(queries, config);
}

/// Convenience function for batch IV solving (vector overload)
inline std::vector<IVResult> solve_implied_vol_batch(
    const std::vector<IVQuery>& queries,
    const IVSolverFDMConfig& config)
{
    return BatchIVSolver::solve_batch(queries, config);
}

}  // namespace mango
