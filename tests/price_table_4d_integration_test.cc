// SPDX-License-Identifier: MIT
/**
 * @file price_table_4d_integration_test.cc
 * @brief Integration tests for PriceTableBuilder with routing
 */

#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/math/bspline_basis.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <chrono>

using namespace mango;

TEST(PriceTable4DIntegrationTest, FastPathEligible) {
    // Narrow moneyness range → fast path
    auto grid_spec_result = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, 2.0);
    ASSERT_TRUE(grid_spec_result.has_value());
    auto grid_spec = grid_spec_result.value();

    auto builder_axes_result = PriceTableBuilder::from_vectors(
        {std::log(0.9), std::log(0.95), std::log(1.0), std::log(1.05), std::log(1.1)},  // Log-moneyness (5 points)
        {0.25, 0.5, 1.0, 2.0},           // Maturity (4 points)
        {0.15, 0.20, 0.25, 0.30},        // Volatility (4 points)
        {0.0, 0.02, 0.05, 0.08},         // Rate (4 points)
        100.0,                            // K_ref
        PDEGridConfig{grid_spec, 1000},
        OptionType::PUT,
        0.02, 0.0);                       // dividend_yield, max_failure_rate
    ASSERT_TRUE(builder_axes_result.has_value()) << "Failed to create builder: " << builder_axes_result.error();
    auto [builder, axes] = std::move(builder_axes_result.value());

    auto result = builder.build(axes);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->n_pde_solves, 4 * 4);  // Nv × Nr = 16

    // Spot check: ATM put with 1y maturity, σ=20%, r=5% (ATM: log-moneyness = 0.0)
    double price = result->spline->eval({0.0, 1.0, 0.20, 0.05});
    EXPECT_GT(price, 0.0);
    EXPECT_LT(price, 100.0);  // Put value < strike for ATM

    // Verify B-spline fitting quality
    EXPECT_LT(result->fitting_stats.max_residual_overall, 0.01);  // <1bp
}

TEST(PriceTable4DIntegrationTest, FallbackWideRange) {
    // Wide moneyness range → fallback
    auto grid_spec_result = GridSpec<double>::sinh_spaced(-3.5, 3.5, 121, 2.0);
    ASSERT_TRUE(grid_spec_result.has_value());
    auto grid_spec = grid_spec_result.value();

    auto builder_axes_result = PriceTableBuilder::from_vectors(
        {std::log(0.5), std::log(0.7), std::log(0.9), std::log(1.0), std::log(1.1), std::log(1.3), std::log(1.5)},  // Wide range (7 points)
        {0.25, 0.5, 1.0, 2.0},                // Maturity (4 points)
        {0.15, 0.20, 0.25, 0.30},             // Volatility (4 points)
        {0.0, 0.02, 0.05, 0.08},              // Rate (4 points)
        100.0,                                 // K_ref
        PDEGridConfig{grid_spec, 1000},
        OptionType::PUT,
        0.02, 0.0);                              // dividend_yield, max_failure_rate
    ASSERT_TRUE(builder_axes_result.has_value()) << "Failed to create builder: " << builder_axes_result.error();
    auto [builder, axes] = std::move(builder_axes_result.value());

    auto result = builder.build(axes);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->n_pde_solves, 4 * 4);  // Nv × Nr = 16

    // Verify prices at extremes (log-moneyness)
    double price_deep_itm = result->spline->eval({std::log(0.5), 1.0, 0.20, 0.05});
    double price_deep_otm = result->spline->eval({std::log(1.5), 1.0, 0.20, 0.05});

    EXPECT_GT(price_deep_itm, price_deep_otm);  // ITM > OTM
}

TEST(PriceTable4DIntegrationTest, FastPathVsFallbackConsistency) {
    // Test same parameters using both paths
    std::vector<double> log_moneyness = {std::log(0.9), std::log(0.95), std::log(1.0), std::log(1.05), std::log(1.1)};
    std::vector<double> maturity = {0.25, 0.5, 1.0, 1.5};  // 4 points minimum
    std::vector<double> volatility = {0.15, 0.20, 0.25, 0.30};  // 4 points minimum
    std::vector<double> rate = {0.02, 0.04, 0.06, 0.08};  // 4 points minimum

    // Fast path (narrow range)
    auto grid_spec_fast_result = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, 2.0);
    ASSERT_TRUE(grid_spec_fast_result.has_value());
    auto grid_spec_fast = grid_spec_fast_result.value();

    auto builder_fast_result = PriceTableBuilder::from_vectors(
        log_moneyness, maturity, volatility, rate, 100.0,
        PDEGridConfig{grid_spec_fast, 1000}, OptionType::PUT, 0.02, 0.0);
    ASSERT_TRUE(builder_fast_result.has_value()) << "Failed to create builder: " << builder_fast_result.error();
    auto [builder_fast, axes_fast] = std::move(builder_fast_result.value());
    auto result_fast = builder_fast.build(axes_fast);

    // Fallback (force by using wider grid)
    auto grid_spec_fallback_result = GridSpec<double>::sinh_spaced(-3.5, 3.5, 121, 2.0);
    ASSERT_TRUE(grid_spec_fallback_result.has_value());
    auto grid_spec_fallback = grid_spec_fallback_result.value();

    auto builder_fallback_result = PriceTableBuilder::from_vectors(
        log_moneyness, maturity, volatility, rate, 100.0,
        PDEGridConfig{grid_spec_fallback, 1000}, OptionType::PUT, 0.02, 0.0);
    ASSERT_TRUE(builder_fallback_result.has_value()) << "Failed to create builder: " << builder_fallback_result.error();
    auto [builder_fallback, axes_fallback] = std::move(builder_fallback_result.value());
    auto result_fallback = builder_fallback.build(axes_fallback);

    ASSERT_TRUE(result_fast.has_value());
    ASSERT_TRUE(result_fallback.has_value());

    // Compare prices at same query points (log-moneyness)
    // Use relative error tolerance (1%) instead of absolute (catches scaling bugs)
    for (double m : {std::log(0.9), std::log(1.0), std::log(1.1)}) {
        for (double tau : {0.5, 1.0}) {
            for (double sigma : {0.20, 0.25}) {
                for (double r : {0.04, 0.06}) {
                    double price_fast = result_fast->spline->eval({m, tau, sigma, r});
                    double price_fallback = result_fallback->spline->eval({m, tau, sigma, r});

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

TEST(PriceTable4DIntegrationTest, FastPathVsFallbackNormalizedPriceEquivalence) {
    // REGRESSION TEST: Compare raw precomputed prices (not interpolated)
    // This test ensures fast path and fallback produce identical numerical results
    // at every grid point. If this test fails after code changes, it indicates
    // a bug was introduced in the fast path that breaks scale invariance.

    std::vector<double> log_moneyness = {std::log(0.85), std::log(0.9), std::log(0.95), std::log(1.0), std::log(1.05), std::log(1.1), std::log(1.15)};
    std::vector<double> maturity = {0.25, 0.5, 1.0, 1.5};
    std::vector<double> volatility = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate = {0.02, 0.04, 0.06, 0.08};

    const size_t Nm = log_moneyness.size();
    const size_t Nt = maturity.size();
    const size_t Nv = volatility.size();
    const size_t Nr = rate.size();

    // Fast path (narrow range → normalized solver)
    auto grid_spec_fast_result = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, 2.0);
    ASSERT_TRUE(grid_spec_fast_result.has_value());
    auto grid_spec_fast = grid_spec_fast_result.value();

    auto builder_fast_result = PriceTableBuilder::from_vectors(
        log_moneyness, maturity, volatility, rate, 100.0,
        PDEGridConfig{grid_spec_fast, 1000}, OptionType::PUT, 0.02, 0.0);
    ASSERT_TRUE(builder_fast_result.has_value()) << "Failed to create builder: " << builder_fast_result.error();
    auto [builder_fast, axes_fast] = std::move(builder_fast_result.value());
    auto result_fast = builder_fast.build(axes_fast);

    // Fallback (wider grid → batch API)
    auto grid_spec_fallback_result = GridSpec<double>::sinh_spaced(-3.5, 3.5, 121, 2.0);
    ASSERT_TRUE(grid_spec_fallback_result.has_value());
    auto grid_spec_fallback = grid_spec_fallback_result.value();

    auto builder_fallback_result = PriceTableBuilder::from_vectors(
        log_moneyness, maturity, volatility, rate, 100.0,
        PDEGridConfig{grid_spec_fallback, 1000}, OptionType::PUT, 0.02, 0.0);
    ASSERT_TRUE(builder_fallback_result.has_value()) << "Failed to create builder: " << builder_fallback_result.error();
    auto [builder_fallback, axes_fallback] = std::move(builder_fallback_result.value());
    auto result_fallback = builder_fallback.build(axes_fallback);

    ASSERT_TRUE(result_fast.has_value());
    ASSERT_TRUE(result_fallback.has_value());

    // Compare raw precomputed prices at EVERY grid point
    // This is stricter than interpolation comparison
    const auto& prices_fast = result_fast->spline->coefficients();
    const auto& prices_fallback = result_fallback->spline->coefficients();

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
                                  << "lnm=" << log_moneyness[i] << " tau=" << maturity[j]
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
    auto grid_spec_result = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, 2.0);
    ASSERT_TRUE(grid_spec_result.has_value());
    auto grid_spec = grid_spec_result.value();

    auto builder_axes_result = PriceTableBuilder::from_vectors(
        {std::log(0.8), std::log(0.85), std::log(0.9), std::log(0.95), std::log(1.0), std::log(1.05), std::log(1.1), std::log(1.15), std::log(1.2)},  // 9 points
        {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0},              // 7 points
        {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40},         // 7 points
        {0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10},          // 7 points
        100.0,
        PDEGridConfig{grid_spec, 1000},
        OptionType::PUT,
        0.02);
    ASSERT_TRUE(builder_axes_result.has_value()) << "Failed to create builder: " << builder_axes_result.error();
    auto [builder, axes] = std::move(builder_axes_result.value());

    auto start = std::chrono::high_resolution_clock::now();
    auto result = builder.build(axes);
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

TEST(BSplineND4D, ConstructsFromGrids) {
    std::vector<double> m = {std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1)};
    std::vector<double> tau = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r = {0.02, 0.03, 0.04, 0.05};

    std::array<std::vector<double>, 4> grids = {m, tau, sigma, r};
    std::array<std::vector<double>, 4> knots;
    for (size_t i = 0; i < 4; ++i) {
        knots[i] = clamped_knots_cubic(grids[i]);
    }

    // 4*4*4*4 = 256 coefficients
    std::vector<double> coeffs(256, 10.0);

    auto spline = BSplineND<double, 4>::create(grids, std::move(knots), std::move(coeffs));
    ASSERT_TRUE(spline.has_value());

    EXPECT_DOUBLE_EQ(spline->grid(0).front(), std::log(0.8));
    EXPECT_DOUBLE_EQ(spline->grid(0).back(), std::log(1.1));
}
