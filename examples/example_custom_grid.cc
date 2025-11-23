/**
 * @file example_custom_grid.cc
 * @brief Custom grid configuration example
 *
 * Demonstrates:
 * - Manual grid specification with GridSpec factory methods
 * - Uniform vs sinh-spaced grids
 * - Multi-sinh grids for multiple concentration points
 * - Comparing auto-estimation vs manual configuration
 * - Grid accuracy tradeoffs
 */

#include "src/option/american_option.hpp"
#include "src/pde/core/grid.hpp"
#include <iostream>
#include <iomanip>
#include <memory_resource>

int main() {
    std::cout << "=== Custom Grid Configuration Example ===\n\n";

    // Base option parameters
    mango::PricingParams params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .type = mango::OptionType::PUT
    };

    std::pmr::synchronized_pool_resource pool;

    std::cout << "Option: ATM Put (S=K=$100, T=1Y, σ=20%)\n\n";

    // =========================================================================
    // 1. AUTO-ESTIMATION (Recommended)
    // =========================================================================
    std::cout << "1. AUTO-ESTIMATED GRID (Recommended)\n";
    std::cout << std::string(60, '-') << "\n";

    auto [auto_grid_spec, auto_n_time] = mango::estimate_grid_for_option(params);

    std::cout << "  Grid: " << auto_grid_spec.n() << " spatial points\n";
    std::cout << "  Time: " << auto_n_time << " steps\n";
    std::cout << "  Range: [" << std::fixed << std::setprecision(3)
              << auto_grid_spec.x_min() << ", "
              << auto_grid_spec.x_max() << "]\n";

    auto auto_workspace = mango::PDEWorkspace::create(auto_grid_spec, &pool).value();
    mango::AmericanOptionSolver auto_solver(params, auto_workspace, auto_n_time);
    auto auto_result = auto_solver.solve();

    std::cout << "  Price: $" << std::setprecision(6) << auto_result->value() << "\n";
    std::cout << "  Delta: " << auto_result->delta() << "\n\n";

    // =========================================================================
    // 2. UNIFORM GRID (Not recommended for option pricing)
    // =========================================================================
    std::cout << "2. UNIFORM GRID\n";
    std::cout << std::string(60, '-') << "\n";

    auto uniform_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 201);
    if (!uniform_spec.has_value()) {
        std::cerr << "Error creating uniform grid: " << uniform_spec.error() << "\n";
        return 1;
    }

    std::cout << "  Grid: 201 uniform points\n";
    std::cout << "  Range: [-3.0, 3.0] (log-moneyness)\n";

    auto uniform_workspace = mango::PDEWorkspace::create(uniform_spec.value(), &pool).value();
    mango::AmericanOptionSolver uniform_solver(params, uniform_workspace, 1000);
    auto uniform_result = uniform_solver.solve();

    std::cout << "  Price: $" << uniform_result->value() << "\n";
    std::cout << "  Delta: " << uniform_result->delta() << "\n";
    std::cout << "  Note: Uniform grids waste points far from strike\n\n";

    // =========================================================================
    // 3. SINH-SPACED GRID (Single concentration point)
    // =========================================================================
    std::cout << "3. SINH-SPACED GRID (Recommended for single strike)\n";
    std::cout << std::string(60, '-') << "\n";

    double alpha = 2.0;  // Clustering strength
    auto sinh_spec = mango::GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, alpha);
    if (!sinh_spec.has_value()) {
        std::cerr << "Error creating sinh grid: " << sinh_spec.error() << "\n";
        return 1;
    }

    std::cout << "  Grid: 201 sinh-spaced points (α=" << alpha << ")\n";
    std::cout << "  Range: [-3.0, 3.0] (log-moneyness)\n";
    std::cout << "  Concentration: Around x=0 (ATM, S=K)\n";

    auto sinh_workspace = mango::PDEWorkspace::create(sinh_spec.value(), &pool).value();
    mango::AmericanOptionSolver sinh_solver(params, sinh_workspace, 1000);
    auto sinh_result = sinh_solver.solve();

    std::cout << "  Price: $" << sinh_result->value() << "\n";
    std::cout << "  Delta: " << sinh_result->delta() << "\n";
    std::cout << "  Benefit: Concentrates points near ATM for better accuracy\n\n";

    // =========================================================================
    // 4. MULTI-SINH GRID (Multiple concentration points)
    // =========================================================================
    std::cout << "4. MULTI-SINH GRID (Multiple strikes)\n";
    std::cout << std::string(60, '-') << "\n";

    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = 0.0,  .alpha = 2.5, .weight = 2.0},   // ATM (higher weight)
        {.center_x = -0.2, .alpha = 2.0, .weight = 1.0}    // 20% ITM
    };

    auto multi_sinh_spec = mango::GridSpec<double>::multi_sinh_spaced(
        -3.0, 3.0, 201, clusters);

    if (!multi_sinh_spec.has_value()) {
        std::cerr << "Error creating multi-sinh grid: " << multi_sinh_spec.error() << "\n";
        return 1;
    }

    std::cout << "  Grid: 201 multi-sinh points\n";
    std::cout << "  Clusters: 2 concentration points\n";
    std::cout << "    - x=0.0 (ATM, weight=2.0)\n";
    std::cout << "    - x=-0.2 (20% ITM, weight=1.0)\n";

    auto multi_sinh_workspace = mango::PDEWorkspace::create(
        multi_sinh_spec.value(), &pool).value();
    mango::AmericanOptionSolver multi_sinh_solver(params, multi_sinh_workspace, 1000);
    auto multi_sinh_result = multi_sinh_solver.solve();

    std::cout << "  Price: $" << multi_sinh_result->value() << "\n";
    std::cout << "  Delta: " << multi_sinh_result->delta() << "\n";
    std::cout << "  Use case: Price tables covering multiple strikes\n\n";

    // =========================================================================
    // 5. CUSTOM ACCURACY PARAMETERS
    // =========================================================================
    std::cout << "5. CUSTOM ACCURACY PARAMETERS\n";
    std::cout << std::string(60, '-') << "\n";

    // Fast mode (lower accuracy, faster computation)
    mango::GridAccuracyParams fast_accuracy{
        .n_sigma = 5.0,
        .alpha = 2.0,
        .tol = 1e-2,  // Fast mode
        .c_t = 0.75,
        .min_spatial_points = 100,
        .max_spatial_points = 200,
        .max_time_steps = 500
    };

    auto [fast_grid, fast_n_time] = mango::estimate_grid_for_option(params, fast_accuracy);
    auto fast_workspace = mango::PDEWorkspace::create(fast_grid.value(), &pool).value();
    mango::AmericanOptionSolver fast_solver(params, fast_workspace, fast_n_time);
    auto fast_result = fast_solver.solve();

    std::cout << "  Fast mode (tol=1e-2):\n";
    std::cout << "    Grid: " << fast_grid.value().n() << " points\n";
    std::cout << "    Time: " << fast_n_time << " steps\n";
    std::cout << "    Price: $" << fast_result->value() << "\n\n";

    // High accuracy mode
    mango::GridAccuracyParams high_accuracy{
        .n_sigma = 5.0,
        .alpha = 2.0,
        .tol = 1e-6,  // High accuracy
        .c_t = 0.75,
        .min_spatial_points = 300,
        .max_spatial_points = 1200,
        .max_time_steps = 5000
    };

    auto [high_grid, high_n_time] = mango::estimate_grid_for_option(params, high_accuracy);
    auto high_workspace = mango::PDEWorkspace::create(high_grid.value(), &pool).value();
    mango::AmericanOptionSolver high_solver(params, high_workspace, high_n_time);
    auto high_result = high_solver.solve();

    std::cout << "  High accuracy mode (tol=1e-6):\n";
    std::cout << "    Grid: " << high_grid.value().n() << " points\n";
    std::cout << "    Time: " << high_n_time << " steps\n";
    std::cout << "    Price: $" << high_result->value() << "\n\n";

    // =========================================================================
    // 6. COMPARISON SUMMARY
    // =========================================================================
    std::cout << "COMPARISON SUMMARY\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << std::setw(20) << "Grid Type"
              << std::setw(12) << "Points"
              << std::setw(15) << "Price"
              << std::setw(12) << "Delta\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::setw(20) << "Auto-estimated"
              << std::setw(12) << auto_grid_spec.n()
              << std::setw(15) << std::setprecision(6) << auto_result->value()
              << std::setw(12) << auto_result->delta() << "\n";
    std::cout << std::setw(20) << "Uniform"
              << std::setw(12) << 201
              << std::setw(15) << uniform_result->value()
              << std::setw(12) << uniform_result->delta() << "\n";
    std::cout << std::setw(20) << "Sinh-spaced"
              << std::setw(12) << 201
              << std::setw(15) << sinh_result->value()
              << std::setw(12) << sinh_result->delta() << "\n";
    std::cout << std::setw(20) << "Multi-sinh"
              << std::setw(12) << 201
              << std::setw(15) << multi_sinh_result->value()
              << std::setw(12) << multi_sinh_result->delta() << "\n";
    std::cout << std::setw(20) << "Fast mode"
              << std::setw(12) << fast_grid.value().n()
              << std::setw(15) << fast_result->value()
              << std::setw(12) << fast_result->delta() << "\n";
    std::cout << std::setw(20) << "High accuracy"
              << std::setw(12) << high_grid.value().n()
              << std::setw(15) << high_result->value()
              << std::setw(12) << high_result->delta() << "\n";
    std::cout << std::string(60, '=') << "\n";

    std::cout << "\nRECOMMENDATIONS:\n";
    std::cout << "  1. Use auto-estimation for most cases (estimate_grid_for_option)\n";
    std::cout << "  2. Use sinh-spaced grids for single strike pricing\n";
    std::cout << "  3. Use multi-sinh grids for price tables (multiple strikes)\n";
    std::cout << "  4. Avoid uniform grids (inefficient for option pricing)\n";
    std::cout << "  5. Adjust GridAccuracyParams for speed/accuracy tradeoffs\n";

    return 0;
}
