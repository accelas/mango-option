/**
 * @file example_american_option_pricing.cc
 * @brief Basic American option pricing example using current API
 *
 * Demonstrates:
 * - Creating option parameters with PricingParams
 * - Automatic grid estimation via estimate_grid_for_option()
 * - PMR memory management with synchronized_pool_resource
 * - Creating PDEWorkspace and AmericanOptionSolver
 * - Error handling with std::expected
 * - Extracting price and Greeks
 */

#include "src/option/american_option.hpp"
#include <iostream>
#include <iomanip>
#include <memory_resource>

int main() {
    std::cout << "=== American Option Pricing Example ===\n\n";

    // 1. Define option parameters
    mango::PricingParams params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .type = mango::OptionType::PUT
    };

    std::cout << "Option Parameters:\n";
    std::cout << "  Spot:       $" << params.spot << "\n";
    std::cout << "  Strike:     $" << params.strike << "\n";
    std::cout << "  Maturity:   " << params.maturity << " years\n";
    std::cout << "  Volatility: " << (params.volatility * 100) << "%\n";
    std::cout << "  Rate:       " << (params.rate * 100) << "%\n";
    std::cout << "  Dividend:   " << (params.continuous_dividend_yield * 100) << "%\n";
    std::cout << "  Type:       " << (params.type == mango::OptionType::PUT ? "PUT" : "CALL") << "\n\n";

    // 2. Auto-estimate grid (recommended approach)
    std::cout << "Auto-estimating grid parameters...\n";
    auto [grid_spec, n_time] = mango::estimate_grid_for_option(params);

    std::cout << "  Grid generated: " << grid_spec.n() << " spatial points\n";
    std::cout << "  Time steps:     " << n_time << "\n\n";

    // 3. Create PMR memory resource (thread-safe pool)
    std::pmr::synchronized_pool_resource pool;

    // 4. Create PDE workspace
    std::cout << "Creating PDE workspace...\n";
    auto workspace_result = mango::PDEWorkspace::create(grid_spec, &pool);

    if (!workspace_result.has_value()) {
        std::cerr << "ERROR: Failed to create workspace: " << workspace_result.error() << "\n";
        return 1;
    }

    auto workspace = std::move(workspace_result.value());
    std::cout << "  Workspace created successfully\n\n";

    // 5. Create solver and solve
    std::cout << "Solving American put option...\n";
    mango::AmericanOptionSolver solver(params, workspace, n_time);
    auto result = solver.solve();

    // 6. Check result and extract values
    if (!result.has_value()) {
        std::cerr << "ERROR: Solver failed\n";
        return 1;
    }

    std::cout << "  Solution converged!\n\n";

    // 7. Display results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Results:\n";
    std::cout << "  Price:  $" << result->value() << "\n";
    std::cout << "  Delta:  " << result->delta() << "\n";
    std::cout << "  Gamma:  " << result->gamma() << "\n";

    // 8. Show moneyness analysis
    double moneyness = params.spot / params.strike;
    std::cout << "\nMoneyness Analysis:\n";
    std::cout << "  S/K = " << moneyness << " (";
    if (moneyness < 0.95) {
        std::cout << "Deep ITM";
    } else if (moneyness < 1.05) {
        std::cout << "ATM";
    } else {
        std::cout << "OTM";
    }
    std::cout << ")\n";

    // 9. Intrinsic value comparison (American puts)
    double intrinsic = std::max(params.strike - params.spot, 0.0);
    double time_value = result->value() - intrinsic;

    std::cout << "\nValue Decomposition:\n";
    std::cout << "  Intrinsic value: $" << intrinsic << "\n";
    std::cout << "  Time value:      $" << time_value << "\n";
    std::cout << "  Total value:     $" << result->value() << "\n";

    return 0;
}
