/**
 * @file example_iv_solver_fdm.cc
 * @brief Implied volatility calculation example using FDM-based solver
 *
 * Demonstrates:
 * - Setting up OptionSpec and IVQuery
 * - Configuring IVSolverFDM with custom parameters
 * - Error handling with std::expected
 * - Interpreting IVSuccess and IVError results
 * - Input validation and arbitrage checks
 */

#include "src/option/iv_solver_fdm.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== Implied Volatility Solver Example (FDM) ===\n\n";

    // 1. Define option specification
    mango::OptionSpec spec{
        .spot = 100.0,
        .strike = 100.0,
        .maturity = 1.0,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = mango::OptionType::PUT
    };

    // Market price of the option
    double market_price = 10.45;

    std::cout << "Option Specification:\n";
    std::cout << "  Spot:        $" << spec.spot << "\n";
    std::cout << "  Strike:      $" << spec.strike << "\n";
    std::cout << "  Maturity:    " << spec.maturity << " years\n";
    std::cout << "  Rate:        " << (spec.rate * 100) << "%\n";
    std::cout << "  Dividend:    " << (spec.dividend_yield * 100) << "%\n";
    std::cout << "  Type:        " << (spec.type == mango::OptionType::PUT ? "PUT" : "CALL") << "\n";
    std::cout << "  Market Price: $" << market_price << "\n\n";

    // 2. Create IV query (using constructor)
    mango::IVQuery query(
        spec.spot,
        spec.strike,
        spec.maturity,
        spec.rate,
        spec.dividend_yield,
        spec.type,
        market_price
    );

    // 3. Configure solver (optional - uses defaults if not specified)
    mango::IVSolverFDMConfig config{
        .root_config = mango::RootFindingConfig{
            .max_iter = 100,
            .tolerance = 1e-6
        }
        // use_manual_grid = false (default: auto-estimation)
    };

    std::cout << "Solver Configuration:\n";
    std::cout << "  Max iterations: " << config.root_config.max_iter << "\n";
    std::cout << "  Tolerance:      " << config.root_config.tolerance << "\n";
    std::cout << "  Grid mode:      " << (config.use_manual_grid ? "Manual" : "Auto-estimate") << "\n\n";

    // 4. Create solver and solve
    std::cout << "Solving for implied volatility...\n";
    mango::IVSolverFDM solver(config);
    auto result = solver.solve_impl(query);

    // 5. Check result
    if (result.has_value()) {
        // Success case
        const auto& success = result.value();

        std::cout << "SUCCESS!\n\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Results:\n";
        std::cout << "  Implied Volatility: " << (success.implied_vol * 100) << "%\n";
        std::cout << "  Iterations:         " << success.iterations << "\n";
        std::cout << "  Final Error:        " << success.final_error << "\n";

        // Display convergence info
        std::cout << "\nConvergence Info:\n";
        if (success.vega.has_value()) {
            std::cout << "  Vega: " << success.vega.value() << "\n";
        }

        // Convergence analysis
        std::cout << "\nConvergence:\n";
        std::cout << "  Status: Converged in " << success.iterations << " iterations\n";
        std::cout << "  Error:  " << (success.final_error * 100) << "% of market price\n";

    } else {
        // Error case
        const auto& error = result.error();

        std::cerr << "FAILED!\n\n";
        std::cerr << "Error Details:\n";
        std::cerr << "  Code:    " << static_cast<int>(error.code) << "\n";
        std::cerr << "  Message: " << error.message << "\n";

        if (error.iterations > 0) {
            std::cerr << "\nAttempted Iterations: " << error.iterations << "\n";
        }

        if (error.last_vol.has_value()) {
            std::cerr << "Last volatility tried: " << (error.last_vol.value() * 100) << "%\n";
        }

        std::cerr << "Final error: " << error.final_error << "\n";

        return 1;
    }

    // 6. Additional examples with different scenarios
    std::cout << "\n=== Additional Test Cases ===\n\n";

    // Test Case 1: Deep ITM put
    std::cout << "Test 1: Deep ITM Put (S=80, K=100)\n";
    mango::OptionSpec spec_itm{
        .spot = 80.0,
        .strike = 100.0,
        .maturity = 1.0,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = mango::OptionType::PUT
    };
    mango::IVQuery query_itm(spec_itm.spot, spec_itm.strike, spec_itm.maturity,
                             spec_itm.rate, spec_itm.dividend_yield, spec_itm.type, 22.0);
    auto result_itm = solver.solve_impl(query_itm);

    if (result_itm.has_value()) {
        std::cout << "  Implied Vol: " << (result_itm->implied_vol * 100) << "%\n";
        std::cout << "  Iterations:  " << result_itm->iterations << "\n\n";
    } else {
        std::cerr << "  Error: " << result_itm."Error code: " << static_cast<int>(result.error().code) << "\n\n";
    }

    // Test Case 2: OTM put
    std::cout << "Test 2: OTM Put (S=110, K=100)\n";
    mango::OptionSpec spec_otm{
        .spot = 110.0,
        .strike = 100.0,
        .maturity = 1.0,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = mango::OptionType::PUT
    };
    mango::IVQuery query_otm(spec_otm.spot, spec_otm.strike, spec_otm.maturity,
                             spec_otm.rate, spec_otm.dividend_yield, spec_otm.type, 3.5);
    auto result_otm = solver.solve_impl(query_otm);

    if (result_otm.has_value()) {
        std::cout << "  Implied Vol: " << (result_otm->implied_vol * 100) << "%\n";
        std::cout << "  Iterations:  " << result_otm->iterations << "\n\n";
    } else {
        std::cerr << "  Error: " << result_otm."Error code: " << static_cast<int>(result.error().code) << "\n\n";
    }

    // Test Case 3: Arbitrage violation (price too high)
    std::cout << "Test 3: Arbitrage Test (price > strike)\n";
    mango::IVQuery query_arb(spec.spot, spec.strike, spec.maturity,
                             spec.rate, spec.dividend_yield, spec.type, 105.0);
    auto result_arb = solver.solve_impl(query_arb);

    if (result_arb.has_value()) {
        std::cout << "  Unexpected success!\n\n";
    } else {
        std::cout << "  Expected failure: " << result_arb."Error code: " << static_cast<int>(result.error().code) << "\n\n";
    }

    return 0;
}
