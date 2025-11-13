/**
 * @file price_table_4d_integration_test.cc
 * @brief Integration tests for PriceTable4DBuilder with routing
 */

#include "src/option/price_table_4d_builder.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <chrono>

using namespace mango;

TEST(PriceTable4DIntegrationTest, FastPathEligible) {
    // Narrow moneyness range → fast path
    auto builder = PriceTable4DBuilder::create(
        {0.9, 0.95, 1.0, 1.05, 1.1},     // Moneyness (5 points)
        {0.25, 0.5, 1.0, 2.0},           // Maturity (4 points)
        {0.15, 0.20, 0.25, 0.30},        // Volatility (4 points)
        {0.0, 0.02, 0.05, 0.08},         // Rate (4 points)
        100.0);                           // K_ref

    auto result = builder.precompute(OptionType::PUT, -3.0, 3.0, 101, 1000, 0.02);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->n_pde_solves, 4 * 4);  // Nv × Nr = 16

    // Spot check: ATM put with 1y maturity, σ=20%, r=5%
    double price = result->evaluator->eval(1.0, 1.0, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
    EXPECT_LT(price, 100.0);  // Put value < strike for ATM

    // Verify B-spline fitting quality
    EXPECT_LT(result->fitting_stats.max_residual_overall, 0.01);  // <1bp
}

TEST(PriceTable4DIntegrationTest, FallbackWideRange) {
    // Wide moneyness range → fallback
    auto builder = PriceTable4DBuilder::create(
        {0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5},  // Wide range (7 points)
        {0.25, 0.5, 1.0, 2.0},                // Maturity (4 points)
        {0.15, 0.20, 0.25, 0.30},             // Volatility (4 points)
        {0.0, 0.02, 0.05, 0.08},              // Rate (4 points)
        100.0);

    auto result = builder.precompute(OptionType::PUT, -3.5, 3.5, 121, 1000, 0.02);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->n_pde_solves, 4 * 4);  // Nv × Nr = 16

    // Verify prices at extremes
    double price_deep_itm = result->evaluator->eval(0.5, 1.0, 0.20, 0.05);
    double price_deep_otm = result->evaluator->eval(1.5, 1.0, 0.20, 0.05);

    EXPECT_GT(price_deep_itm, price_deep_otm);  // ITM > OTM
}

TEST(PriceTable4DIntegrationTest, FastPathVsFallbackConsistency) {
    // Test same parameters using both paths
    std::vector<double> moneyness = {0.9, 0.95, 1.0, 1.05, 1.1};
    std::vector<double> maturity = {0.25, 0.5, 1.0, 1.5};  // 4 points minimum
    std::vector<double> volatility = {0.15, 0.20, 0.25, 0.30};  // 4 points minimum
    std::vector<double> rate = {0.02, 0.04, 0.06, 0.08};  // 4 points minimum

    // Fast path (narrow range)
    auto builder_fast = PriceTable4DBuilder::create(
        moneyness, maturity, volatility, rate, 100.0);
    auto result_fast = builder_fast.precompute(
        OptionType::PUT, -3.0, 3.0, 101, 1000, 0.02);

    // Fallback (force by using wider grid)
    auto builder_fallback = PriceTable4DBuilder::create(
        moneyness, maturity, volatility, rate, 100.0);
    auto result_fallback = builder_fallback.precompute(
        OptionType::PUT, -3.5, 3.5, 121, 1000, 0.02);

    ASSERT_TRUE(result_fast.has_value());
    ASSERT_TRUE(result_fallback.has_value());

    // Compare prices at same query points
    for (double m : {0.9, 1.0, 1.1}) {
        for (double tau : {0.5, 1.0}) {
            for (double sigma : {0.20, 0.25}) {
                for (double r : {0.04, 0.06}) {
                    double price_fast = result_fast->evaluator->eval(m, tau, sigma, r);
                    double price_fallback = result_fallback->evaluator->eval(m, tau, sigma, r);

                    // Expect <2bp difference (relaxed due to interpolation and grid differences)
                    EXPECT_NEAR(price_fast, price_fallback, 0.02)
                        << "Mismatch at m=" << m << " tau=" << tau << " sigma=" << sigma << " r=" << r;
                }
            }
        }
    }
}

TEST(PriceTable4DIntegrationTest, PerformanceFastPath) {
    // Benchmark fast path
    auto builder = PriceTable4DBuilder::create(
        {0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2},  // 9 points
        {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0},              // 7 points
        {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40},         // 7 points
        {0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10},          // 7 points
        100.0);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = builder.precompute(OptionType::PUT, -3.0, 3.0, 101, 1000, 0.02);
    auto end = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(result.has_value());

    double duration_sec = std::chrono::duration<double>(end - start).count();
    std::cout << "Fast path: " << result->n_pde_solves << " PDEs in "
              << duration_sec << " seconds\n";
    std::cout << "Throughput: " << (result->n_pde_solves / duration_sec)
              << " PDEs/sec\n";

    // Should maintain ~848 options/sec on 32 cores (or scale with cores)
    // Relaxed threshold for CI environments
    EXPECT_LT(duration_sec, 60.0);  // Complete within 1 minute
}
