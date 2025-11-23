/**
 * @file example_multi_sinh_grid.cc
 * @brief Example demonstrating multi-sinh grid generation with multiple concentration regions
 */

#include "src/pde/core/grid.hpp"
#include <iostream>
#include <format>
#include <limits>

using namespace mango;

int main() {
    // Example 1: Single cluster (equivalent to regular sinh_spaced)
    {
        std::cout << "=== Example 1: Single Cluster ===\n";
        std::cout << "Concentrates points at log-moneyness x = 0.0 (ATM)\n\n";

        std::vector<MultiSinhCluster<double>> clusters = {
            {.center_x = 0.0, .alpha = 2.0, .weight = 1.0}
        };

        auto result = GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 11, clusters);
        if (!result.has_value()) {
            std::cerr << "Error: " << result.error() << "\n";
            return 1;
        }

        auto grid = result.value().generate();
        std::cout << "Grid points:\n";
        for (size_t i = 0; i < grid.size(); ++i) {
            std::cout << std::format("  x[{:2}] = {:7.4f}\n", i, grid[i]);
        }
        std::cout << "\n";
    }

    // Example 2: Dual clusters for ATM and deep ITM concentration
    {
        std::cout << "=== Example 2: Dual Clusters (ATM + Deep ITM) ===\n";
        std::cout << "Use case: Price table covering both ATM and 20% ITM strikes\n";
        std::cout << "Concentrates at x = 0.0 (ATM) and x = -0.2 (20% ITM)\n\n";

        std::vector<MultiSinhCluster<double>> clusters = {
            {.center_x = 0.0, .alpha = 2.5, .weight = 2.0},   // ATM (higher weight)
            {.center_x = -0.2, .alpha = 2.0, .weight = 1.0}   // 20% ITM
        };

        auto result = GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 21, clusters);
        if (!result.has_value()) {
            std::cerr << "Error: " << result.error() << "\n";
            return 1;
        }

        auto grid = result.value().generate();
        std::cout << "Grid points with spacing:\n";
        for (size_t i = 0; i < grid.size(); ++i) {
            if (i > 0) {
                double dx = grid[i] - grid[i-1];
                std::cout << std::format("  x[{:2}] = {:7.4f}  (dx = {:7.4f})\n",
                                        i, grid[i], dx);
            } else {
                std::cout << std::format("  x[{:2}] = {:7.4f}\n", i, grid[i]);
            }
        }
        std::cout << "\n";
    }

    // Example 3: Three clusters (Deep ITM, ATM, OTM)
    {
        std::cout << "=== Example 3: Triple Clusters (Deep ITM + ATM + OTM) ===\n";
        std::cout << "Use case: Comprehensive price table covering wide moneyness range\n";
        std::cout << "Concentrates at x = -1.5 (deep ITM), x = 0.0 (ATM), x = 1.5 (OTM)\n\n";

        std::vector<MultiSinhCluster<double>> clusters = {
            {.center_x = -1.5, .alpha = 1.8, .weight = 1.0},  // Deep ITM
            {.center_x = 0.0, .alpha = 2.5, .weight = 2.0},   // ATM (highest weight)
            {.center_x = 1.5, .alpha = 1.8, .weight = 1.0}    // Deep OTM
        };

        auto result = GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 31, clusters);
        if (!result.has_value()) {
            std::cerr << "Error: " << result.error() << "\n";
            return 1;
        }

        auto grid = result.value().generate();

        // Show summary statistics
        double min_dx = std::numeric_limits<double>::max();
        double max_dx = 0.0;
        for (size_t i = 1; i < grid.size(); ++i) {
            double dx = grid[i] - grid[i-1];
            min_dx = std::min(min_dx, dx);
            max_dx = std::max(max_dx, dx);
        }

        std::cout << std::format("Grid range: [{:.2f}, {:.2f}]\n", grid[0], grid[30]);
        std::cout << std::format("Points: {}\n", grid.size());
        std::cout << std::format("Min spacing: {:.6f}\n", min_dx);
        std::cout << std::format("Max spacing: {:.6f}\n", max_dx);
        std::cout << std::format("Spacing ratio: {:.2f}x\n\n", max_dx / min_dx);

        // Show first and last 5 points to see concentration
        std::cout << "First 5 points (near x_min = -3.0):\n";
        for (size_t i = 0; i < 5; ++i) {
            double dx = (i > 0) ? (grid[i] - grid[i-1]) : 0.0;
            if (i > 0) {
                std::cout << std::format("  x[{:2}] = {:7.4f}  (dx = {:7.4f})\n", i, grid[i], dx);
            } else {
                std::cout << std::format("  x[{:2}] = {:7.4f}\n", i, grid[i]);
            }
        }

        std::cout << "\nLast 5 points (near x_max = 3.0):\n";
        for (size_t i = 26; i < 31; ++i) {
            double dx = grid[i] - grid[i-1];
            std::cout << std::format("  x[{:2}] = {:7.4f}  (dx = {:7.4f})\n", i, grid[i], dx);
        }
        std::cout << "\n";
    }

    // Example 4: When NOT to use multi-sinh
    {
        std::cout << "=== Example 4: When NOT to Use Multi-Sinh ===\n";
        std::cout << "If strikes differ by only a few percent (Δx < 0.3/α), use single sinh.\n";
        std::cout << "Example: S=100, K1=100, K2=102 → Δx ≈ ln(102/100) ≈ 0.02\n";
        std::cout << "With α=2.0, threshold is ~0.15, so 0.02 << 0.15\n";
        std::cout << "Recommendation: Use single sinh centered at midpoint instead.\n\n";

        std::vector<MultiSinhCluster<double>> clusters = {
            {.center_x = 0.0, .alpha = 2.0, .weight = 1.0}  // Single cluster suffices
        };

        auto result = GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 21, clusters);
        if (!result.has_value()) {
            std::cerr << "Error: " << result.error() << "\n";
            return 1;
        }

        auto grid = result.value().generate();
        std::cout << "Single cluster grid provides adequate resolution for nearby strikes.\n";
    }

    std::cout << "\n=== Summary ===\n";
    std::cout << "Multi-sinh grids are useful when:\n";
    std::cout << "• Price tables require accuracy at multiple, widely-separated strikes\n";
    std::cout << "• Log-moneyness distance between targets exceeds ~0.3/α (~0.15 for α=2)\n";
    std::cout << "• Batch solving needs a single grid covering diverse moneyness\n\n";
    std::cout << "When to use single-sinh instead:\n";
    std::cout << "• Strikes differ by only a few percent (Δx < 0.3/α)\n";
    std::cout << "• Single concentration region provides adequate accuracy\n";
    std::cout << "• Simplicity preferred over marginal accuracy gains\n\n";

    return 0;
}
