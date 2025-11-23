/**
 * @file example_batch_pricing.cc
 * @brief Batch American option pricing example
 *
 * Demonstrates:
 * - Pricing multiple options in a batch
 * - Workspace reuse for efficiency
 * - PMR memory management for batch operations
 * - Processing different option parameters
 * - Performance measurement
 */

#include "src/option/american_option.hpp"
#include <iostream>
#include <iomanip>
#include <memory_resource>
#include <vector>
#include <chrono>

int main() {
    std::cout << "=== Batch American Option Pricing Example ===\n\n";

    // 1. Define a batch of options to price
    std::vector<mango::PricingParams> option_batch = {
        // ATM puts at different maturities
        {.strike = 100.0, .spot = 100.0, .maturity = 0.25, .volatility = 0.20,
         .rate = 0.05, .continuous_dividend_yield = 0.02, .type = mango::OptionType::PUT},
        {.strike = 100.0, .spot = 100.0, .maturity = 0.5, .volatility = 0.20,
         .rate = 0.05, .continuous_dividend_yield = 0.02, .type = mango::OptionType::PUT},
        {.strike = 100.0, .spot = 100.0, .maturity = 1.0, .volatility = 0.20,
         .rate = 0.05, .continuous_dividend_yield = 0.02, .type = mango::OptionType::PUT},

        // ITM, ATM, OTM puts (1-year maturity)
        {.strike = 100.0, .spot = 90.0, .maturity = 1.0, .volatility = 0.20,
         .rate = 0.05, .continuous_dividend_yield = 0.02, .type = mango::OptionType::PUT},
        {.strike = 100.0, .spot = 100.0, .maturity = 1.0, .volatility = 0.20,
         .rate = 0.05, .continuous_dividend_yield = 0.02, .type = mango::OptionType::PUT},
        {.strike = 100.0, .spot = 110.0, .maturity = 1.0, .volatility = 0.20,
         .rate = 0.05, .continuous_dividend_yield = 0.02, .type = mango::OptionType::PUT},

        // Different volatilities (ATM, 1-year)
        {.strike = 100.0, .spot = 100.0, .maturity = 1.0, .volatility = 0.10,
         .rate = 0.05, .continuous_dividend_yield = 0.02, .type = mango::OptionType::PUT},
        {.strike = 100.0, .spot = 100.0, .maturity = 1.0, .volatility = 0.30,
         .rate = 0.05, .continuous_dividend_yield = 0.02, .type = mango::OptionType::PUT},
        {.strike = 100.0, .spot = 100.0, .maturity = 1.0, .volatility = 0.40,
         .rate = 0.05, .continuous_dividend_yield = 0.02, .type = mango::OptionType::PUT},
    };

    std::cout << "Batch size: " << option_batch.size() << " options\n\n";

    // 2. Create shared PMR pool for all operations
    std::pmr::synchronized_pool_resource pool;

    // 3. Process batch with timing
    std::cout << "Processing batch...\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << std::setw(5) << "#"
              << std::setw(8) << "Spot"
              << std::setw(8) << "Strike"
              << std::setw(8) << "Mat"
              << std::setw(8) << "Vol"
              << std::setw(12) << "Price"
              << std::setw(10) << "Delta"
              << std::setw(10) << "Gamma"
              << std::setw(10) << "Time(ms)\n";
    std::cout << std::string(80, '-') << "\n";

    auto batch_start = std::chrono::high_resolution_clock::now();

    std::vector<double> prices, deltas, gammas;
    prices.reserve(option_batch.size());
    deltas.reserve(option_batch.size());
    gammas.reserve(option_batch.size());

    for (size_t i = 0; i < option_batch.size(); ++i) {
        const auto& params = option_batch[i];

        auto solve_start = std::chrono::high_resolution_clock::now();

        // Auto-estimate grid for this option
        auto [grid_spec, n_time] = mango::estimate_grid_for_option(params);

        // Create workspace (reusing pool)
        auto workspace = mango::PDEWorkspace::create(grid_spec, &pool);

        // Solve
        mango::AmericanOptionSolver solver(params, workspace, n_time);
        auto result = solver.solve();

        auto solve_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            solve_end - solve_start).count() / 1000.0;

        if (result.has_value()) {
            prices.push_back(result->value());
            deltas.push_back(result->delta());
            gammas.push_back(result->gamma());

            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(5) << (i + 1)
                      << std::setw(8) << params.spot
                      << std::setw(8) << params.strike
                      << std::setw(8) << params.maturity
                      << std::setw(8) << (params.volatility * 100) << "%"
                      << std::setprecision(4)
                      << std::setw(12) << result->value()
                      << std::setw(10) << result->delta()
                      << std::setw(10) << result->gamma()
                      << std::setprecision(1)
                      << std::setw(10) << duration << "\n";
        } else {
            std::cerr << "Option " << (i + 1) << " FAILED\n";
        }
    }

    auto batch_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        batch_end - batch_start).count();

    std::cout << std::string(80, '-') << "\n";

    // 4. Summary statistics
    std::cout << "\nBatch Summary:\n";
    std::cout << "  Total options:    " << option_batch.size() << "\n";
    std::cout << "  Successful:       " << prices.size() << "\n";
    std::cout << "  Failed:           " << (option_batch.size() - prices.size()) << "\n";
    std::cout << "  Total time:       " << total_duration << " ms\n";
    std::cout << "  Average per opt:  " << std::fixed << std::setprecision(1)
              << (static_cast<double>(total_duration) / option_batch.size()) << " ms\n";

    // 5. Analysis: Price vs Spot (moneyness effect)
    std::cout << "\nMoneyness Effect (1Y maturity, 20% vol):\n";
    std::cout << "  S=$90  (ITM): Price = $" << std::fixed << std::setprecision(4)
              << prices[3] << ", Delta = " << deltas[3] << "\n";
    std::cout << "  S=$100 (ATM): Price = $" << prices[4]
              << ", Delta = " << deltas[4] << "\n";
    std::cout << "  S=$110 (OTM): Price = $" << prices[5]
              << ", Delta = " << deltas[5] << "\n";

    // 6. Analysis: Price vs Volatility (vega effect)
    std::cout << "\nVolatility Effect (ATM, 1Y maturity):\n";
    std::cout << "  σ=10%: Price = $" << prices[6] << "\n";
    std::cout << "  σ=20%: Price = $" << prices[4] << "\n";
    std::cout << "  σ=30%: Price = $" << prices[7] << "\n";
    std::cout << "  σ=40%: Price = $" << prices[8] << "\n";

    // 7. Analysis: Price vs Maturity (theta effect)
    std::cout << "\nMaturity Effect (ATM, 20% vol):\n";
    std::cout << "  T=3M:  Price = $" << prices[0] << "\n";
    std::cout << "  T=6M:  Price = $" << prices[1] << "\n";
    std::cout << "  T=1Y:  Price = $" << prices[2] << "\n";

    return 0;
}
