#include "src/option/american_option.hpp"
#include "src/pde/core/pde_workspace.hpp"
#include <iostream>
#include <iomanip>
#include <memory_resource>

int main() {
    using namespace mango;

    // Deep ITM put test (from failing test)
    AmericanOptionParams params(
        0.25,   // spot (S/K = 0.0025, very deep ITM)
        100.0,  // strike
        0.75,   // maturity
        0.05,   // rate
        0.0,    // dividend
        OptionType::PUT,
        0.20    // volatility
    );

    // Create workspace (matching test configuration)
    auto grid_spec = GridSpec<double>::sinh_spaced(-7.0, 2.0, 301, 2.0);
    if (!grid_spec.has_value()) {
        std::cerr << "Grid creation failed: " << grid_spec.error() << "\n";
        return 1;
    }

    std::pmr::synchronized_pool_resource pool;
    auto workspace_result = PDEWorkspaceOwned::create(grid_spec.value(), &pool);

    if (!workspace_result.has_value()) {
        std::cerr << "Workspace creation failed: " << workspace_result.error() << "\n";
        return 1;
    }

    // Create solver with PDEWorkspace
    AmericanOptionSolver solver(params, workspace_result.value().workspace);

    std::cout << "=== Deep ITM Put Test (Projected Thomas - Reformulated) ===\n";
    std::cout << "S=" << params.spot << " K=" << params.strike << " T=" << params.maturity << "\n";
    std::cout << "Log-moneyness: ln(S/K) = " << std::log(params.spot / params.strike) << "\n";
    std::cout << "Intrinsic value: K - S = " << (params.strike - params.spot) << "\n\n";

    auto result = solver.solve();

    if (result.has_value()) {
        const auto& option_result = result.value();
        double value = option_result.value_at(params.spot);
        double intrinsic = std::max(params.strike - params.spot, 0.0);
        double error = value - intrinsic;

        std::cout << "Result:\n";
        std::cout << "  Value at spot: " << std::fixed << std::setprecision(10) << value << "\n";
        std::cout << "  Intrinsic:     " << intrinsic << "\n";
        std::cout << "  Error:         " << error << "\n";
        std::cout << "  Error %:       " << (error / intrinsic * 100.0) << "%\n";

        if (std::abs(error) < 0.5) {
            std::cout << "\n✅ TEST PASSED\n";
            return 0;
        } else {
            std::cout << "\n❌ TEST FAILED (error > 0.5)\n";
            return 1;
        }
    } else {
        std::cerr << "Solver failed: " << result.error().message << "\n";
        return 1;
    }
}
