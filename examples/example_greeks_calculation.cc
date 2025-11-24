/**
 * @file example_greeks_calculation.cc
 * @brief Greeks calculation example for American options
 *
 * Demonstrates:
 * - Computing Delta (∂V/∂S) via unified CenteredDifference operators
 * - Computing Gamma (∂²V/∂S²) via second derivative operator
 * - Greeks on uniform and non-uniform (sinh) grids
 * - Understanding Greeks behavior across moneyness
 */

#include "src/option/american_option.hpp"
#include <iostream>
#include <iomanip>
#include <memory_resource>
#include <vector>

void print_greeks_profile(const std::string& title,
                          const std::vector<double>& spots,
                          const std::vector<double>& prices,
                          const std::vector<double>& deltas,
                          const std::vector<double>& gammas) {
    std::cout << "\n" << title << "\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << std::setw(10) << "Spot"
              << std::setw(12) << "Price"
              << std::setw(12) << "Delta"
              << std::setw(12) << "Gamma"
              << std::setw(12) << "Moneyness\n";
    std::cout << std::string(70, '-') << "\n";

    for (size_t i = 0; i < spots.size(); ++i) {
        double moneyness = spots[i] / 100.0;  // Strike = 100
        std::string money_str;
        if (moneyness < 0.95) money_str = "Deep ITM";
        else if (moneyness < 0.98) money_str = "ITM";
        else if (moneyness < 1.02) money_str = "ATM";
        else if (moneyness < 1.05) money_str = "OTM";
        else money_str = "Deep OTM";

        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(10) << spots[i]
                  << std::setprecision(4)
                  << std::setw(12) << prices[i]
                  << std::setw(12) << deltas[i]
                  << std::setprecision(6)
                  << std::setw(12) << gammas[i]
                  << "  " << money_str << "\n";
    }
    std::cout << std::string(70, '=') << "\n";
}

int main() {
    std::cout << "=== Greeks Calculation Example ===\n\n";

    // Define base option parameters
    double strike = 100.0;
    double maturity = 1.0;
    double volatility = 0.20;
    double rate = 0.05;
    double dividend_yield = 0.02;

    std::cout << "Base Parameters:\n";
    std::cout << "  Strike:     $" << strike << "\n";
    std::cout << "  Maturity:   " << maturity << " years\n";
    std::cout << "  Volatility: " << (volatility * 100) << "%\n";
    std::cout << "  Rate:       " << (rate * 100) << "%\n";
    std::cout << "  Dividend:   " << (dividend_yield * 100) << "%\n\n";

    // Create PMR pool
    std::pmr::synchronized_pool_resource pool;

    // Compute Greeks across different spot prices
    std::vector<double> spots = {
        80.0,  // Deep ITM
        85.0,  // ITM
        90.0,  // ITM
        95.0,  // Near ATM
        100.0, // ATM
        105.0, // Near ATM
        110.0, // OTM
        115.0, // OTM
        120.0  // Deep OTM
    };

    std::vector<double> prices, deltas, gammas;

    std::cout << "Computing Greeks across moneyness spectrum...\n";

    for (double spot : spots) {
        mango::OptionSpec spec;
        spec.spot = spot;
        spec.strike = strike;
        spec.maturity = maturity;
        spec.rate = rate;
        spec.dividend_yield = dividend_yield;
        spec.type = mango::OptionType::PUT;

        mango::PricingParams params(spec, volatility);

        // Auto-estimate grid
        auto [grid_spec, time_domain] = mango::estimate_grid_for_option(params);
        (void)time_domain;  // Not directly used; solver will reconstruct it

        // Create workspace (buffer must stay alive through solve)
        size_t n = grid_spec.n_points();
        std::pmr::vector<double> buffer(mango::PDEWorkspace::required_size(n), &pool);
        auto workspace = mango::PDEWorkspace::from_buffer(buffer, n).value();

        mango::AmericanOptionSolver solver(params, workspace);
        auto result = solver.solve();

        if (result.has_value()) {
            prices.push_back(result->value());
            deltas.push_back(result->delta());
            gammas.push_back(result->gamma());
        }
    }

    // Display Greeks profile
    print_greeks_profile("American Put Greeks Profile", spots, prices, deltas, gammas);

    // Analysis: Delta behavior
    std::cout << "\nDelta Analysis:\n";
    std::cout << "  Delta measures the rate of change of option price with respect to spot\n";
    std::cout << "  For puts: Delta is negative (price decreases as spot increases)\n";
    std::cout << "  Range: -1.0 (deep ITM) to 0.0 (deep OTM)\n\n";

    std::cout << "  Deep ITM (S=$80):  Delta = " << std::fixed << std::setprecision(4)
              << deltas[0] << " (close to -1.0, moves almost 1-to-1 with spot)\n";
    std::cout << "  ATM (S=$100):      Delta = " << deltas[4]
              << " (around -0.5, balanced sensitivity)\n";
    std::cout << "  Deep OTM (S=$120): Delta = " << deltas[8]
              << " (close to 0.0, minimal sensitivity)\n\n";

    // Analysis: Gamma behavior
    std::cout << "Gamma Analysis:\n";
    std::cout << "  Gamma measures the curvature of option price (second derivative)\n";
    std::cout << "  Peak gamma occurs near ATM (greatest convexity)\n";
    std::cout << "  Low gamma for deep ITM and deep OTM (linear behavior)\n\n";

    // Find max gamma
    size_t max_gamma_idx = 0;
    double max_gamma = 0.0;
    for (size_t i = 0; i < gammas.size(); ++i) {
        if (gammas[i] > max_gamma) {
            max_gamma = gammas[i];
            max_gamma_idx = i;
        }
    }

    std::cout << "  Maximum Gamma: " << std::fixed << std::setprecision(6)
              << max_gamma << " at S=$" << std::setprecision(2) << spots[max_gamma_idx] << "\n";
    std::cout << "  ATM Gamma:     " << std::setprecision(6) << gammas[4]
              << " (S=$100)\n";
    std::cout << "  ITM Gamma:     " << gammas[0] << " (S=$80)\n";
    std::cout << "  OTM Gamma:     " << gammas[8] << " (S=$120)\n\n";

    // Hedging implications
    std::cout << "Hedging Implications:\n";
    std::cout << "  1. Delta hedging: Sell " << std::abs(deltas[4])
              << " shares to hedge 1 ATM put\n";
    std::cout << "  2. Gamma risk: ATM options have highest gamma = "
              << std::setprecision(6) << gammas[4] << "\n";
    std::cout << "     → Delta hedge requires frequent rebalancing\n";
    std::cout << "  3. Deep ITM/OTM: Lower gamma = " << std::setprecision(6)
              << gammas[0] << "/" << gammas[8] << "\n";
    std::cout << "     → Delta hedge more stable (less rebalancing needed)\n";

    return 0;
}
