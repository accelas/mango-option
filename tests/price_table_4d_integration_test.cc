/**
 * @file price_table_4d_integration_test.cc
 * @brief Integration tests for PriceTable4DBuilder with routing
 */

#include "src/option/price_table_4d_builder.hpp"
#include "src/option/price_table_workspace.hpp"
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
    // Use relative error tolerance (1%) instead of absolute (catches scaling bugs)
    for (double m : {0.9, 1.0, 1.1}) {
        for (double tau : {0.5, 1.0}) {
            for (double sigma : {0.20, 0.25}) {
                for (double r : {0.04, 0.06}) {
                    double price_fast = result_fast->evaluator->eval(m, tau, sigma, r);
                    double price_fallback = result_fallback->evaluator->eval(m, tau, sigma, r);

                    // Use relative error: |fast - fallback| / |fallback| < 1%
                    // This catches scaling bugs that absolute tolerance misses
                    double rel_error = std::abs(price_fast - price_fallback) / std::abs(price_fallback);
                    EXPECT_LT(rel_error, 0.01)
                        << "Relative error " << (rel_error * 100) << "% exceeds 1% at "
                        << "m=" << m << " tau=" << tau << " sigma=" << sigma << " r=" << r
                        << " (fast=" << price_fast << ", fallback=" << price_fallback << ")";
                }
            }
        }
    }
}

TEST(PriceTable4DIntegrationTest, FastPathVsFallbackRawPriceEquivalence) {
    // REGRESSION TEST: Compare raw precomputed prices (not interpolated)
    // This test ensures fast path and fallback produce identical numerical results
    // at every grid point. If this test fails after code changes, it indicates
    // a bug was introduced in the fast path that breaks scale invariance.

    std::vector<double> moneyness = {0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15};
    std::vector<double> maturity = {0.25, 0.5, 1.0, 1.5};
    std::vector<double> volatility = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate = {0.02, 0.04, 0.06, 0.08};

    const size_t Nm = moneyness.size();
    const size_t Nt = maturity.size();
    const size_t Nv = volatility.size();
    const size_t Nr = rate.size();

    // Fast path (narrow range → normalized solver)
    auto builder_fast = PriceTable4DBuilder::create(
        moneyness, maturity, volatility, rate, 100.0);
    auto result_fast = builder_fast.precompute(
        OptionType::PUT, -3.0, 3.0, 101, 1000, 0.02);

    // Fallback (wider grid → batch API)
    auto builder_fallback = PriceTable4DBuilder::create(
        moneyness, maturity, volatility, rate, 100.0);
    auto result_fallback = builder_fallback.precompute(
        OptionType::PUT, -3.5, 3.5, 121, 1000, 0.02);

    ASSERT_TRUE(result_fast.has_value());
    ASSERT_TRUE(result_fallback.has_value());

    // Compare raw precomputed prices at EVERY grid point
    // This is stricter than interpolation comparison
    const auto& prices_fast = result_fast->prices_4d;
    const auto& prices_fallback = result_fallback->prices_4d;

    ASSERT_EQ(prices_fast.size(), prices_fallback.size());
    ASSERT_EQ(prices_fast.size(), Nm * Nt * Nv * Nr);

    size_t max_errors_to_show = 5;
    size_t error_count = 0;

    for (size_t i = 0; i < Nm; ++i) {
        for (size_t j = 0; j < Nt; ++j) {
            for (size_t k = 0; k < Nv; ++k) {
                for (size_t l = 0; l < Nr; ++l) {
                    size_t idx = ((i * Nt + j) * Nv + k) * Nr + l;

                    double price_fast = prices_fast[idx];
                    double price_fallback = prices_fallback[idx];

                    // Relative error < 2.0% for raw prices
                    // Looser than interpolated test due to:
                    // 1. Different PDE grid resolutions (101 vs 121 points)
                    // 2. Different PDE domain widths ([-3,3] vs [-3.5,3.5])
                    // 3. Accumulated discretization errors (especially near grid edges)
                    // This tolerance catches scaling bugs (10%+ errors) while allowing
                    // reasonable numerical differences from different discretizations.
                    double rel_error = std::abs(price_fast - price_fallback) /
                                      std::abs(price_fallback);

                    if (rel_error >= 0.02 && error_count < max_errors_to_show) {
                        std::cerr << "Raw price mismatch at grid point ["
                                  << i << "," << j << "," << k << "," << l << "]: "
                                  << "m=" << moneyness[i] << " tau=" << maturity[j]
                                  << " sigma=" << volatility[k] << " r=" << rate[l]
                                  << " fast=" << price_fast
                                  << " fallback=" << price_fallback
                                  << " rel_error=" << (rel_error * 100) << "%\n";
                        ++error_count;
                    }

                    EXPECT_LT(rel_error, 0.02)
                        << "Relative error " << (rel_error * 100) << "% exceeds 2.0% at grid ["
                        << i << "," << j << "," << k << "," << l << "]";
                }
            }
        }
    }

    if (error_count > 0) {
        std::cerr << "Total raw price mismatches: " << error_count << " out of "
                  << (Nm * Nt * Nv * Nr) << " grid points\n";
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

TEST(PriceTableSurface, ConstructsFromWorkspace) {
    std::vector<double> m = {0.8, 0.9, 1.0, 1.1};
    std::vector<double> tau = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4, 10.0);

    auto ws = mango::PriceTableWorkspace::create(m, tau, sigma, r, coeffs, 100.0, 0.015);
    ASSERT_TRUE(ws.has_value());

    mango::PriceTableSurface surface(std::make_shared<mango::PriceTableWorkspace>(std::move(ws.value())));

    EXPECT_TRUE(surface.valid());
    EXPECT_DOUBLE_EQ(surface.K_ref(), 100.0);
    EXPECT_DOUBLE_EQ(surface.dividend_yield(), 0.015);

    auto [m_min, m_max] = surface.moneyness_range();
    EXPECT_DOUBLE_EQ(m_min, 0.8);
    EXPECT_DOUBLE_EQ(m_max, 1.1);
}
