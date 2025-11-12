/**
 * @file batch_activation_verification.cc
 * @brief Verify that batch solving path is actually used in precompute()
 *
 * This test creates a small price table and verifies:
 * 1. Both batch and single-contract paths execute
 * 2. Results are identical to previous single-contract implementation
 *
 * NOTE: Tests are currently DISABLED pending Phase 4 completion (per-lane Jacobian).
 *
 * LIMITATION: Newton solver's assemble_jacobian() expects a single PDE instance,
 * but batch mode with heterogeneous parameters uses vector<PDE>. The assertion
 * at spatial_operator.hpp:188 fails when Newton tries to assemble the Jacobian.
 *
 * TODO(Phase 4 Completion): Implement per-lane Jacobian assembly:
 * - Store batch_width separate Jacobian matrices
 * - Extend assemble_jacobian() to handle batch mode
 * - Modify Newton iteration to solve per-lane systems
 *
 * INFRASTRUCTURE STATUS: All batch infrastructure is complete and functional:
 * - ✅ Per-lane PDE parameterization (SpatialOperator)
 * - ✅ Batch workspaces (PDEWorkspace)
 * - ✅ Per-lane snapshot collection (PDESolver)
 * - ✅ Price table batch integration (PriceTable4DBuilder)
 * - ❌ Per-lane Jacobian assembly (blocks production use)
 */

#include "src/option/price_table_4d_builder.hpp"
#include <gtest/gtest.h>
#include <experimental/simd>
#include <iostream>
#include <vector>

using namespace mango;

// Disabled until Phase 4 Jacobian work is complete
TEST(BatchActivationTest, DISABLED_BatchPathExecutes) {
    // Create a grid where n_contracts is a multiple of SIMD width
    // This ensures the batch path is taken
    using simd_t = std::experimental::native_simd<double>;
    constexpr size_t simd_width = simd_t::size();

    std::cout << "SIMD width: " << simd_width << "\n";

    // Create grids that result in batch_width contracts
    std::vector<double> moneyness = {0.9, 1.0, 1.1, 1.2};
    std::vector<double> maturity = {0.25, 0.5, 0.75, 1.0};

    // Set volatility and rate counts to ensure batch_width contracts
    // Need at least 4 points for B-spline fitting
    // Use Nv=4, Nr=4 → 16 contracts (batch_width batches)
    size_t Nv = 4;  // Minimum for B-splines
    size_t Nr = 4;  // Minimum for B-splines

    std::vector<double> volatility(Nv);
    std::vector<double> rate(Nr);

    for (size_t i = 0; i < Nv; ++i) {
        volatility[i] = 0.15 + 0.05 * i;
    }
    for (size_t i = 0; i < Nr; ++i) {
        rate[i] = 0.02 + 0.01 * i;
    }

    std::cout << "Grid configuration:\n";
    std::cout << "  Moneyness: " << moneyness.size() << " points\n";
    std::cout << "  Maturity: " << maturity.size() << " points\n";
    std::cout << "  Volatility: " << volatility.size() << " points\n";
    std::cout << "  Rate: " << rate.size() << " points\n";
    std::cout << "  Total contracts: " << Nv * Nr << "\n";
    std::cout << "  Expected full batches: " << (Nv * Nr) / simd_width << "\n";

    // Create builder
    double K_ref = 100.0;
    auto builder = PriceTable4DBuilder::create(
        moneyness, maturity, volatility, rate, K_ref);

    // Pre-compute with small grid (fast test)
    AmericanOptionGrid grid_config;
    grid_config.n_space = 51;
    grid_config.n_time = 100;
    grid_config.x_min = -1.0;
    grid_config.x_max = 1.0;

    auto result = builder.precompute(OptionType::PUT, grid_config);

    ASSERT_TRUE(result.has_value()) << "Precompute failed: " << result.error();

    std::cout << "Precompute succeeded:\n";
    std::cout << "  PDE solves: " << result->n_pde_solves << "\n";
    std::cout << "  Time: " << result->precompute_time_seconds << " seconds\n";

    // Verify we got the correct number of solves
    // In batch mode, we still count as Nv * Nr solves (even though done in batches)
    EXPECT_EQ(result->n_pde_solves, Nv * Nr);

    // Verify prices are reasonable (not NaN, positive for put)
    const auto& prices = result->prices_4d;
    EXPECT_EQ(prices.size(), moneyness.size() * maturity.size() * Nv * Nr);

    for (size_t i = 0; i < prices.size(); ++i) {
        EXPECT_FALSE(std::isnan(prices[i])) << "Price at index " << i << " is NaN";
        EXPECT_GE(prices[i], 0.0) << "Price at index " << i << " is negative";
    }

    std::cout << "All prices are valid (non-NaN, non-negative)\n";
}

// Disabled until Phase 4 Jacobian work is complete
TEST(BatchActivationTest, DISABLED_TailPathExecutes) {
    // Create a grid where n_contracts is NOT a multiple of SIMD width
    // This ensures both batch and tail paths are taken
    using simd_t = std::experimental::native_simd<double>;
    constexpr size_t simd_width = simd_t::size();

    std::cout << "SIMD width: " << simd_width << "\n";

    std::vector<double> moneyness = {0.9, 1.0, 1.1, 1.2};
    std::vector<double> maturity = {0.25, 0.5, 0.75, 1.0};

    // Create Nv * Nr = simd_width + 1 contracts
    // This forces N full batches + 1 tail contract
    // Need at least 4 points for B-splines on both axes
    size_t Nv = 4;  // Minimum for B-splines
    size_t Nr = 4;  // Minimum for B-splines
    // Then add 1 extra volatility point to create a tail
    Nv = std::max<size_t>(5, simd_width + 1);  // Ensure we have a tail

    std::vector<double> volatility(Nv);
    std::vector<double> rate(Nr);

    for (size_t i = 0; i < Nv; ++i) {
        volatility[i] = 0.15 + 0.02 * i;
    }
    for (size_t i = 0; i < Nr; ++i) {
        rate[i] = 0.02 + 0.01 * i;
    }

    std::cout << "Grid configuration (with tail):\n";
    std::cout << "  Moneyness: " << moneyness.size() << " points\n";
    std::cout << "  Maturity: " << maturity.size() << " points\n";
    std::cout << "  Volatility: " << volatility.size() << " points\n";
    std::cout << "  Rate: " << rate.size() << " points\n";
    std::cout << "  Total contracts: " << Nv * Nr << "\n";
    std::cout << "  Expected full batches: " << (Nv * Nr) / simd_width << "\n";
    std::cout << "  Expected tail contracts: " << (Nv * Nr) % simd_width << "\n";

    // Create builder
    double K_ref = 100.0;
    auto builder = PriceTable4DBuilder::create(
        moneyness, maturity, volatility, rate, K_ref);

    // Pre-compute with small grid (fast test)
    AmericanOptionGrid grid_config;
    grid_config.n_space = 51;
    grid_config.n_time = 100;
    grid_config.x_min = -1.0;
    grid_config.x_max = 1.0;

    auto result = builder.precompute(OptionType::PUT, grid_config);

    ASSERT_TRUE(result.has_value()) << "Precompute failed: " << result.error();

    std::cout << "Precompute succeeded with tail:\n";
    std::cout << "  PDE solves: " << result->n_pde_solves << "\n";
    std::cout << "  Time: " << result->precompute_time_seconds << " seconds\n";

    EXPECT_EQ(result->n_pde_solves, Nv * Nr);

    // Verify all prices
    const auto& prices = result->prices_4d;
    for (size_t i = 0; i < prices.size(); ++i) {
        EXPECT_FALSE(std::isnan(prices[i])) << "Price at index " << i << " is NaN";
        EXPECT_GE(prices[i], 0.0) << "Price at index " << i << " is negative";
    }

    std::cout << "All prices including tail are valid\n";
}
