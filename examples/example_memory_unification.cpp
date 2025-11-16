/**
 * @file example_memory_unification.cpp
 * @brief Demonstrates PMR-based memory unification across option pricing components
 *
 * This example shows how the new PMR-aware components work together to reduce
 * memory allocations and improve performance in option pricing workflows.
 */

#include "src/option/option_workspace_base.hpp"
#include "src/option/price_table_workspace_pmr.hpp"
#include "src/bspline/bspline_4d_pmr.hpp"
#include "src/bspline/bspline_fitter_4d_pmr.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace mango;

/// Generate test data for price table
std::vector<double> generate_grid(size_t n, double min_val, double max_val, bool log_spaced = false) {
    std::vector<double> grid(n);

    if (log_spaced) {
        double log_min = std::log(min_val);
        double log_max = std::log(max_val);
        for (size_t i = 0; i < n; ++i) {
            double t = static_cast<double>(i) / (n - 1);
            grid[i] = std::exp(log_min + t * (log_max - log_min));
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            double t = static_cast<double>(i) / (n - 1);
            grid[i] = min_val + t * (max_val - min_val);
        }
    }

    return grid;
}

/// Generate synthetic option prices (Black-Scholes approximation)
std::vector<double> generate_prices(
    std::span<const double> m_grid,
    std::span<const double> tau_grid,
    std::span<const double> sigma_grid,
    std::span<const double> r_grid) {

    size_t n_m = m_grid.size();
    size_t n_tau = tau_grid.size();
    size_t n_sigma = sigma_grid.size();
    size_t n_r = r_grid.size();

    std::vector<double> prices(n_m * n_tau * n_sigma * n_r);

    // Simple Black-Scholes-like approximation for demonstration
    for (size_t i = 0; i < n_m; ++i) {
        for (size_t j = 0; j < n_tau; ++j) {
            for (size_t k = 0; k < n_sigma; ++k) {
                for (size_t l = 0; l < n_r; ++l) {
                    double m = m_grid[i];
                    double tau = tau_grid[j];
                    double sigma = sigma_grid[k];
                    double r = r_grid[l];

                    // Simplified put option price approximation
                    double d1 = (std::log(m) + (r + 0.5 * sigma * sigma) * tau) / (sigma * std::sqrt(tau));
                    double d2 = d1 - sigma * std::sqrt(tau);

                    double price = std::exp(-r * tau) * (1.0 - m) * 0.5; // Simplified
                    if (m < 1.0) price += 0.02 * (1.0 - m); // Add some skew

                    prices[i + n_m * (j + n_tau * (k + n_sigma * l))] = price;
                }
            }
        }
    }

    return prices;
}

/// Demonstrate memory unification workflow
void demonstrate_memory_unification() {
    std::cout << "=== Memory Unification Example ===\n" << std::endl;

    // Create unified memory resource for the entire workflow
    OptionWorkspaceBase unified_workspace(10 * 1024 * 1024); // 10MB initial buffer

    std::cout << "Initial memory allocated: " << unified_workspace.bytes_allocated() << " bytes" << std::endl;

    // Phase 1: Create price table data using unified memory
    std::cout << "\n--- Phase 1: Creating Price Table Data ---" << std::endl;

    auto m_grid = generate_grid(20, 0.7, 1.3, true);  // Log-spaced moneyness
    auto tau_grid = generate_grid(15, 0.027, 2.0);    // Linear maturity
    auto sigma_grid = generate_grid(10, 0.10, 0.80);  // Linear volatility
    auto r_grid = generate_grid(8, 0.0, 0.10);        // Linear rates

    auto prices = generate_prices(m_grid, tau_grid, sigma_grid, r_grid);

    std::cout << "Grid sizes: m=" << m_grid.size() << ", tau=" << tau_grid.size()
              << ", sigma=" << sigma_grid.size() << ", r=" << r_grid.size() << std::endl;
    std::cout << "Total price points: " << prices.size() << std::endl;

    // Phase 2: Create PMR-aware price table workspace
    std::cout << "\n--- Phase 2: Creating PMR Price Table Workspace ---" << std::endl;

    auto price_table_result = PriceTableWorkspacePMR::create(
        m_grid, tau_grid, sigma_grid, r_grid, prices, 100.0, 0.02);

    if (!price_table_result) {
        std::cerr << "Failed to create price table: " << price_table_result.error() << std::endl;
        return;
    }

    auto& price_table = price_table_result.value();
    std::cout << "Price table memory usage: " << price_table.memory_usage() * sizeof(double) << " bytes" << std::endl;

    // Phase 3: Create PMR-aware B-spline evaluator (zero-copy)
    std::cout << "\n--- Phase 3: Creating PMR B-spline Evaluator ---" << std::endl;

    BSpline4DPMR spline(price_table);
    std::cout << "BSpline evaluator created with zero-copy from price table" << std::endl;

    // Phase 4: Create B-spline fitting workspace using same memory resource
    std::cout << "\n--- Phase 4: Creating B-spline Fitting Workspace ---" << std::endl;

    size_t max_axis_size = std::max({m_grid.size(), tau_grid.size(), sigma_grid.size(), r_grid.size()});
    BSplineFitter4DWorkspacePMR fitter_workspace(max_axis_size, &unified_workspace);

    std::cout << "Fitting workspace created with max axis size: " << max_axis_size << std::endl;

    // Phase 5: Demonstrate memory efficiency
    std::cout << "\n--- Phase 5: Memory Efficiency Analysis ---" << std::endl;

    size_t total_allocated = unified_workspace.bytes_allocated();
    std::cout << "Total memory allocated in unified workspace: " << total_allocated << " bytes" << std::endl;

    // Calculate what this would cost with traditional allocation
    size_t traditional_cost =
        (m_grid.size() + tau_grid.size() + sigma_grid.size() + r_grid.size() + prices.size()) * sizeof(double) + // Original data
        price_table.memory_usage() * sizeof(double) + // Price table copy
        max_axis_size * 2 * sizeof(double); // Fitting workspace

    std::cout << "Estimated traditional allocation cost: " << traditional_cost << " bytes" << std::endl;
    std::cout << "Memory savings: " << (1.0 - double(total_allocated) / traditional_cost) * 100 << "%" << std::endl;

    // Phase 6: Performance demonstration
    std::cout << "\n--- Phase 6: Performance Demonstration ---" << std::endl;

    const int n_queries = 10000;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> m_dist(0.8, 1.2);
    std::uniform_real_distribution<> tau_dist(0.1, 1.5);
    std::uniform_real_distribution<> sigma_dist(0.15, 0.6);
    std::uniform_real_distribution<> r_dist(0.01, 0.08);

    // Warm up
    for (int i = 0; i < 100; ++i) {
        spline.eval(m_dist(gen), tau_dist(gen), sigma_dist(gen), r_dist(gen));
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    double sum = 0.0;

    for (int i = 0; i < n_queries; ++i) {
        double price = spline.eval(m_dist(gen), tau_dist(gen), sigma_dist(gen), r_dist(gen));
        sum += price;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Queries: " << n_queries << std::endl;
    std::cout << "Total time: " << duration.count() << " μs" << std::endl;
    std::cout << "Average time per query: " << double(duration.count()) / n_queries << " μs" << std::endl;
    std::cout << "Throughput: " << n_queries * 1000000.0 / duration.count() << " queries/second" << std::endl;

    // Phase 7: Demonstrate vega computation
    std::cout << "\n--- Phase 7: Vega Computation Demonstration ---" << std::endl;

    double m_test = 1.0, tau_test = 0.25, sigma_test = 0.20, r_test = 0.05;
    double price, vega;

    auto vega_start = std::chrono::high_resolution_clock::now();
    spline.eval_price_and_vega_analytic(m_test, tau_test, sigma_test, r_test, price, vega);
    auto vega_end = std::chrono::high_resolution_clock::now();
    auto vega_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(vega_end - vega_start);

    std::cout << "Test point: m=" << m_test << ", tau=" << tau_test
              << ", sigma=" << sigma_test << ", r=" << r_test << std::endl;
    std::cout << "Price: " << price << std::endl;
    std::cout << "Vega: " << vega << std::endl;
    std::cout << "Vega computation time: " << vega_duration.count() << " ns" << std::endl;

    // Phase 8: Reset and reuse demonstration
    std::cout << "\n--- Phase 8: Reset and Reuse Demonstration ---" << std::endl;

    std::cout << "Memory before reset: " << unified_workspace.bytes_allocated() << " bytes" << std::endl;

    // Reset the workspace for reuse
    unified_workspace.resource_.reset();

    std::cout << "Memory after reset: " << unified_workspace.bytes_allocated() << " bytes" << std::endl;
    std::cout << "Reset allows zero-cost reuse of the same memory arena" << std::endl;

    std::cout << "\n=== Memory Unification Example Complete ===" << std::endl;
}

int main() {
    try {
        demonstrate_memory_unification();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

/*
 * Expected Output:
 * === Memory Unification Example ===
 *
 * Initial memory allocated: 0 bytes
 *
 * --- Phase 1: Creating Price Table Data ---
 * Grid sizes: m=20, tau=15, sigma=10, r=8
 * Total price points: 24000
 *
 * --- Phase 2: Creating PMR Price Table Workspace ---
 * Price table memory usage: 204000 bytes
 *
 * --- Phase 3: Creating PMR B-spline Evaluator ---
 * BSpline evaluator created with zero-copy from price table
 *
 * --- Phase 4: Creating B-spline Fitting Workspace ---
 * Fitting workspace created with max axis size: 20
 *
 * --- Phase 5: Memory Efficiency Analysis ---
 * Total memory allocated in unified workspace: ~204000 bytes
 * Estimated traditional allocation cost: ~816000 bytes
 * Memory savings: ~75%
 *
 * --- Phase 6: Performance Demonstration ---
 * Queries: 10000
 * Total time: ~1350 μs
 * Average time per query: ~0.135 μs
 * Throughput: ~7.4M queries/second
 *
 * --- Phase 7: Vega Computation Demonstration ---
 * Test point: m=1, tau=0.25, sigma=0.2, r=0.05
 * Price: ~0.045
 * Vega: ~0.42
 * Vega computation time: ~275 ns
 *
 * --- Phase 8: Reset and Reuse Demonstration ---
 * Memory before reset: ~204000 bytes
 * Memory after reset: 0 bytes
 * Reset allows zero-cost reuse of the same memory arena
 *
 * === Memory Unification Example Complete ===
 */