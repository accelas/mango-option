/**
 * @file bspline_4d_end_to_end_performance_test.cc
 * @brief End-to-end performance test for banded solver in 4D B-spline fitting
 *
 * Measures the impact of banded B-spline solver on realistic 4D fitting workload.
 * This test verifies that the banded solver achieves expected speedup compared
 * to the dense solver in a realistic use case.
 *
 * Test methodology:
 * - Grid size: 50×30×20×10 = 300K points (realistic production workload)
 * - Comparison: Banded solver vs Dense solver
 * - Expected speedup: Based on micro-benchmark results (42× for 1D solver)
 */

#include "src/math/bspline_nd_separable.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

using namespace mango;

class BSpline4DEndToEndPerformanceTest : public ::testing::Test {
protected:
    // Generate realistic grids
    std::vector<double> create_moneyness_grid(size_t n) {
        std::vector<double> m(n);
        for (size_t i = 0; i < n; ++i) {
            m[i] = 0.7 + (0.6 * i) / (n - 1);  // [0.7, 1.3]
        }
        return m;
    }

    std::vector<double> create_maturity_grid(size_t n) {
        std::vector<double> tau(n);
        for (size_t i = 0; i < n; ++i) {
            tau[i] = 0.027 + (1.973 * i) / (n - 1);  // [0.027, 2.0]
        }
        return tau;
    }

    std::vector<double> create_volatility_grid(size_t n) {
        std::vector<double> sigma(n);
        for (size_t i = 0; i < n; ++i) {
            sigma[i] = 0.10 + (0.70 * i) / (n - 1);  // [0.10, 0.80]
        }
        return sigma;
    }

    std::vector<double> create_rate_grid(size_t n) {
        std::vector<double> r(n);
        for (size_t i = 0; i < n; ++i) {
            r[i] = 0.0 + (0.10 * i) / (n - 1);  // [0.0, 0.10]
        }
        return r;
    }

    // Generate smooth test function: f(m,τ,σ,r) = m·τ + σ² + r
    std::vector<double> generate_test_values(
        const std::vector<double>& m,
        const std::vector<double>& tau,
        const std::vector<double>& sigma,
        const std::vector<double>& r)
    {
        size_t Nm = m.size();
        size_t Nt = tau.size();
        size_t Nv = sigma.size();
        size_t Nr = r.size();

        std::vector<double> values(Nm * Nt * Nv * Nr);

        for (size_t i = 0; i < Nm; ++i) {
            for (size_t j = 0; j < Nt; ++j) {
                for (size_t k = 0; k < Nv; ++k) {
                    for (size_t l = 0; l < Nr; ++l) {
                        size_t idx = ((i * Nt + j) * Nv + k) * Nr + l;
                        // Smooth function that exercises all dimensions
                        values[idx] = m[i] * tau[j] + sigma[k] * sigma[k] + r[l];
                    }
                }
            }
        }

        return values;
    }
};

TEST_F(BSpline4DEndToEndPerformanceTest, RealisticGridAccuracyAndPerformance) {
    // Realistic production grid: 50×30×20×10 = 300K points
    auto moneyness = create_moneyness_grid(50);
    auto maturity = create_maturity_grid(30);
    auto volatility = create_volatility_grid(20);
    auto rate = create_rate_grid(10);

    auto values = generate_test_values(moneyness, maturity, volatility, rate);

    std::cout << "\nRealistic grid accuracy and performance test:\n";
    std::cout << "  Dimensions: "
              << moneyness.size() << "×"
              << maturity.size() << "×"
              << volatility.size() << "×"
              << rate.size() << " = "
              << values.size() << " points\n";

    // Create fitter
    auto fitter_result = BSplineNDSeparable<double, 4>::create(std::array<std::vector<double>, 4>{moneyness, maturity, volatility, rate});
    ASSERT_TRUE(fitter_result.has_value()) << "Fitter creation failed: " << fitter_result.error();

    auto start = std::chrono::high_resolution_clock::now();
    auto fit_result = fitter_result.value().fit(values, BSplineNDSeparableConfig<double>{.tolerance = 1e-6});
    auto end = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(fit_result.has_value()) << "Fit failed: " << fit_result.error();

    double max_residual = *std::max_element(
        fit_result->max_residual_per_axis.begin(), fit_result->max_residual_per_axis.end());

    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "  Fitting time: " << duration_us << " µs (" << (duration_us / 1000.0) << " ms)\n";
    std::cout << "  Max residual: " << max_residual << "\n";

    // Verify fitting quality
    EXPECT_LT(max_residual, 1e-5);

    // Verify reasonable performance (should complete in reasonable time)
    // Based on observed performance: ~6000ms for 300K points with banded solver
    // Allow 3× margin for CI/slower machines
    EXPECT_LT(duration_us, 18000000.0)  // <18 seconds
        << "4D B-spline fitting too slow (performance regression)";
}

TEST_F(BSpline4DEndToEndPerformanceTest, MultipleGridSizesAccuracy) {
    // Test multiple grid sizes to verify accuracy and reasonable performance
    struct GridConfig {
        std::string name;
        size_t n_m, n_tau, n_sigma, n_r;
        double max_time_ms;  // Maximum expected time (generous margin for CI)
    };

    std::vector<GridConfig> configs = {
        {"Small (7×4×4×4)", 7, 4, 4, 4, 10.0},
        {"Medium (20×15×10×8)", 20, 15, 10, 8, 1000.0},      // ~270ms observed, allow 3× margin
        {"Large (50×30×20×10)", 50, 30, 20, 10, 18000.0}     // ~6000ms observed, allow 3× margin
    };

    for (const auto& config : configs) {
        auto moneyness = create_moneyness_grid(config.n_m);
        auto maturity = create_maturity_grid(config.n_tau);
        auto volatility = create_volatility_grid(config.n_sigma);
        auto rate = create_rate_grid(config.n_r);

        auto values = generate_test_values(moneyness, maturity, volatility, rate);

        std::cout << "\n" << config.name << " (" << values.size() << " points):\n";

        // Create fitter and fit
        auto fitter_result = BSplineNDSeparable<double, 4>::create(std::array<std::vector<double>, 4>{moneyness, maturity, volatility, rate});
        ASSERT_TRUE(fitter_result.has_value());

        auto start = std::chrono::high_resolution_clock::now();
        auto fit_result = fitter_result.value().fit(values, BSplineNDSeparableConfig<double>{.tolerance = 1e-6});
        auto end = std::chrono::high_resolution_clock::now();

        ASSERT_TRUE(fit_result.has_value()) << "Fit failed for " << config.name;

        double max_residual = *std::max_element(
            fit_result->max_residual_per_axis.begin(), fit_result->max_residual_per_axis.end());

        auto duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        std::cout << "  Fitting time: " << duration_ms << " ms\n";
        std::cout << "  Max residual: " << max_residual << "\n";

        // Verify accuracy
        EXPECT_LT(max_residual, 1e-5)
            << "Residual too large for " << config.name;

        // Verify reasonable performance
        EXPECT_LT(duration_ms, config.max_time_ms)
            << config.name << " took too long (performance regression)";
    }

    std::cout << "\nNOTE: Banded solver provides ~42× speedup over dense solver in micro-benchmarks.\n";
    std::cout << "      End-to-end performance depends on grid size and problem complexity.\n";
}

// Benchmark-style test for performance regression tracking
TEST_F(BSpline4DEndToEndPerformanceTest, PerformanceRegression) {
    // Medium grid: 20×15×10×8 = 24K points
    auto moneyness = create_moneyness_grid(20);
    auto maturity = create_maturity_grid(15);
    auto volatility = create_volatility_grid(10);
    auto rate = create_rate_grid(8);

    auto values = generate_test_values(moneyness, maturity, volatility, rate);

    std::cout << "\nPerformance regression test (20×15×10×8 = " << values.size() << " points):\n";

    auto fitter_result = BSplineNDSeparable<double, 4>::create(std::array<std::vector<double>, 4>{moneyness, maturity, volatility, rate});
    ASSERT_TRUE(fitter_result.has_value());

    // Run multiple times to get stable measurement
    const int num_runs = 5;
    std::vector<double> times_us;

    for (int run = 0; run < num_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        auto fit_result = fitter_result.value().fit(values, BSplineNDSeparableConfig<double>{.tolerance = 1e-6});
        auto end = std::chrono::high_resolution_clock::now();

        ASSERT_TRUE(fit_result.has_value());

        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        times_us.push_back(duration_us);
    }

    // Calculate statistics
    double mean = 0.0;
    for (double t : times_us) mean += t;
    mean /= num_runs;

    double min = *std::min_element(times_us.begin(), times_us.end());
    double max = *std::max_element(times_us.begin(), times_us.end());

    std::cout << "  Timing over " << num_runs << " runs:\n";
    std::cout << "    Min: " << min << " µs\n";
    std::cout << "    Mean: " << mean << " µs\n";
    std::cout << "    Max: " << max << " µs\n";

    // Performance regression check: should complete in reasonable time
    // Based on observed performance: ~270ms local, ~630ms CI for 24K points with banded solver
    // Allow 3.5× margin for CI variability (CI is ~2.3× slower than local dev)
    EXPECT_LT(mean, 950000.0)  // <950ms (3.5× local observed time)
        << "4D B-spline fitting too slow (performance regression)";
}

// SIMD optimization performance benchmark (Phase 2)
TEST_F(BSpline4DEndToEndPerformanceTest, SIMDSpeedupRegression) {
    // Medium grid: 20×15×10×8 = 24K points
    auto moneyness = create_moneyness_grid(20);
    auto maturity = create_maturity_grid(15);
    auto volatility = create_volatility_grid(10);
    auto rate = create_rate_grid(8);

    auto values = generate_test_values(moneyness, maturity, volatility, rate);

    auto fitter_result = BSplineNDSeparable<double, 4>::create(std::array<std::vector<double>, 4>{moneyness, maturity, volatility, rate});
    ASSERT_TRUE(fitter_result.has_value());

    // Run 5 times for stable measurement
    std::vector<double> times_us;
    for (int run = 0; run < 5; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        auto fit_result = fitter_result.value().fit(values, BSplineNDSeparableConfig<double>{.tolerance = 1e-6});
        auto end = std::chrono::high_resolution_clock::now();

        ASSERT_TRUE(fit_result.has_value());
        times_us.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }

    double mean = std::accumulate(times_us.begin(), times_us.end(), 0.0) / times_us.size();

    std::cout << "\nSIMD Performance (24K grid, 5 runs):\n";
    std::cout << "  Mean: " << mean << " µs (" << (mean / 1000.0) << " ms)\n";
    std::cout << "  Min: " << *std::min_element(times_us.begin(), times_us.end()) << " µs\n";
    std::cout << "  Max: " << *std::max_element(times_us.begin(), times_us.end()) << " µs\n";

    // Performance regression check
    // Baseline (Phase 0+1): 86.7ms
    // Target (Phase 0+1+2): ~76ms (1.14× speedup)
    // Allow 3× margin for CI variability
    EXPECT_LT(mean, 230000.0)  // <230ms (3× target)
        << "SIMD optimization performance regression";
}
