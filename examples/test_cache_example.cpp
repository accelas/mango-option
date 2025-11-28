#include "src/option/table/adaptive_grid_builder.hpp"
#include "src/option/option_chain.hpp"
#include "src/pde/core/grid.hpp"
#include <iostream>
#include <vector>

int main() {
    // Create a simple synthetic option chain
    mango::OptionChain chain;
    chain.spot = 100.0;
    chain.dividend_yield = 0.02;

    // 3 strikes, 3 maturities for small grid
    chain.strikes = {90.0, 100.0, 110.0};
    chain.maturities = {0.25, 0.5, 1.0};

    // 2 vols, 2 rates = 4 (σ,r) pairs
    chain.implied_vols = {0.15, 0.25};
    chain.rates = {0.03, 0.05};

    // Build adaptive grid
    mango::AdaptiveGridParams params;
    params.target_iv_error = 0.01;  // Easy target
    params.max_iterations = 3;
    params.validation_samples = 10;  // Small for speed

    mango::AdaptiveGridBuilder builder(params);

    auto grid_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 101).value();
    auto result = builder.build(chain, grid_spec, 1000, mango::OptionType::PUT);

    if (!result.has_value()) {
        std::cerr << "Build failed\n";
        return 1;
    }

    // Print iteration statistics
    std::cout << "Iterations: " << result->iterations.size() << "\n";
    for (const auto& iter : result->iterations) {
        std::cout << "Iteration " << iter.iteration
                  << ": PDE solves = " << iter.pde_solves_table
                  << ", grid = [" << iter.grid_sizes[0] << ","
                  << iter.grid_sizes[1] << ","
                  << iter.grid_sizes[2] << ","
                  << iter.grid_sizes[3] << "]\n";
    }

    // Check that cache is working: iteration 0 should solve all pairs,
    // iterations 1+ should solve fewer (only new σ,r combos from grid refinement)
    size_t iter0_solves = result->iterations[0].pde_solves_table;
    std::cout << "\nIteration 0 solved " << iter0_solves << " PDE problems\n";

    if (result->iterations.size() > 1) {
        size_t iter1_solves = result->iterations[1].pde_solves_table;
        std::cout << "Iteration 1 solved " << iter1_solves << " PDE problems\n";

        if (iter1_solves < iter0_solves) {
            std::cout << "SUCCESS: Cache reduced PDE solves by "
                      << (iter0_solves - iter1_solves) << "\n";
        } else {
            std::cout << "WARNING: Cache did not reduce PDE solves\n";
        }
    }

    return 0;
}
