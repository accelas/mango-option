#include "src/option/option_chain_solver.hpp"
#include "src/option/american_option.hpp"
#include "src/option/slice_solver_workspace.hpp"

namespace mango {

std::vector<ChainStrikeResult> OptionChainSolver::solve_chain(
    const AmericanOptionChain& chain,
    const AmericanOptionGrid& grid,
    const TRBDF2Config& trbdf2_config,
    const RootFindingConfig& root_config)
{
    // Validate chain
    auto validation = chain.validate();
    if (!validation.has_value()) {
        // Return error for all strikes
        std::vector<ChainStrikeResult> results;
        results.reserve(chain.strikes.size());
        for (double strike : chain.strikes) {
            results.push_back(ChainStrikeResult{
                strike,
                unexpected(SolverError{
                    .code = SolverErrorCode::InvalidConfiguration,
                    .message = validation.error()
                })
            });
        }
        return results;
    }

    // Validate grid
    auto grid_validation = AmericanOptionGrid::validate_expected(grid);
    if (!grid_validation.has_value()) {
        // Return error for all strikes
        std::vector<ChainStrikeResult> results;
        results.reserve(chain.strikes.size());
        for (double strike : chain.strikes) {
            results.push_back(ChainStrikeResult{
                strike,
                unexpected(SolverError{
                    .code = SolverErrorCode::InvalidConfiguration,
                    .message = grid_validation.error()
                })
            });
        }
        return results;
    }

    std::vector<ChainStrikeResult> results;
    results.reserve(chain.strikes.size());

    // Create workspace ONCE for entire chain
    auto workspace = std::make_shared<SliceSolverWorkspace>(
        grid.x_min, grid.x_max, grid.n_space);

    // Solve each strike SEQUENTIALLY (workspace stays hot)
    for (double strike : chain.strikes) {
        AmericanOptionParams params{
            .strike = strike,
            .spot = chain.spot,
            .maturity = chain.maturity,
            .volatility = chain.volatility,
            .rate = chain.rate,
            .continuous_dividend_yield = chain.continuous_dividend_yield,
            .option_type = chain.option_type,
            .discrete_dividends = chain.discrete_dividends
        };

        // Reuse workspace for this strike
        AmericanOptionSolver solver(params, grid, workspace, trbdf2_config, root_config);
        auto result = solver.solve();

        results.push_back(ChainStrikeResult{strike, std::move(result)});
    }

    return results;
}

}  // namespace mango
