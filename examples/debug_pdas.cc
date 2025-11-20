#include "src/option/american_pde_solver.hpp"
#include "src/option/american_solver_workspace.hpp"
#include "src/pde/core/trbdf2_config.hpp"
#include "src/option/option_spec.hpp"
#include <iostream>
#include <iomanip>

int main() {
    using namespace mango;

    // Simple ATM put test
    PricingParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.0,    // dividend
        OptionType::PUT,
        0.25    // volatility
    );

    // Create workspace
    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, 2.0);
    if (!grid_spec.has_value()) {
        std::cerr << "Grid creation failed: " << grid_spec.error() << "\n";
        return 1;
    }

    auto workspace_result = AmericanSolverWorkspace::create(
        grid_spec.value(),
        1000,
        std::pmr::get_default_resource()
    );

    if (!workspace_result.has_value()) {
        std::cerr << "Workspace creation failed: " << workspace_result.error() << "\n";
        return 1;
    }
    auto workspace = workspace_result.value();

    // --- PDAS/Heuristic Solver Configuration Testbed ---
    TRBDF2Config config;

    // To switch between PDAS and the stable Heuristic method, change the line below:
    // config.obstacle_method = ObstacleMethod::Heuristic;
    config.obstacle_method = ObstacleMethod::PDAS;

    // The safety band can be disabled by setting the alpha values to 0.0.
    // The current PDAS implementation with the safety band is known to have convergence issues.
    // config.pdas_gap_alpha = 0.0;
    // config.pdas_lambda_alpha = 0.0;

    AmericanPutSolver solver(params, workspace);
    solver.set_config(config);
    solver.initialize(AmericanPutSolver::payoff);

    auto result = solver.solve();

    std::cout << "\n=== PDAS Investigation Testbed ===\n";
    if (result) {
        // To get the value at spot, we need to interpolate from the solution grid
        auto solution_view = solver.solution();
        auto grid = workspace->grid();
        
        // simple linear interpolation
        double x_target = std::log(params.spot / params.strike);
        size_t i = 0;
        while(i < grid.size()-2 && grid[i+1] < x_target) {
            i++;
        }
        double t = (x_target - grid[i]) / (grid[i+1] - grid[i]);
        double normalized_value = (1.0 - t) * solution_view[i] + t * solution_view[i+1];
        double value = normalized_value * params.strike;


        double intrinsic = std::max(params.strike - params.spot, 0.0);
        std::cout << "ATM Put Value: " << std::fixed << std::setprecision(6)
                  << value << "\n";
        std::cout << "Intrinsic: " << intrinsic << "\n";
        std::cout << "Converged: " << std::boolalpha << true << "\n";
        std::cout << "Expected: > " << intrinsic << " (should have time value)\n";
    } else {
        std::cout << "Converged: " << std::boolalpha << false << "\n";
        std::cerr << "FAILED: " << result.error().message << "\n";
    }

    return 0;
}
