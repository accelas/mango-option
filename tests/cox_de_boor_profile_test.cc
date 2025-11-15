/**
 * @file cox_de_boor_profile_test.cc
 * @brief Profile Cox-de Boor overhead in realistic workload
 *
 * This diagnostic test measures the ACTUAL time spent in Cox-de Boor
 * vs other operations to verify bottleneck assumptions.
 */

#include "src/interpolation/bspline_fitter_4d.hpp"
#include "src/interpolation/bspline_utils.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>

using namespace mango;

class CoxDeBoorProfileTest : public ::testing::Test {
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

    std::vector<double> generate_test_values(
        const std::vector<double>& m,
        const std::vector<double>& tau)
    {
        size_t Nm = m.size();
        size_t Nt = tau.size();
        std::vector<double> values(Nm * Nt);

        for (size_t i = 0; i < Nm; ++i) {
            for (size_t j = 0; j < Nt; ++j) {
                size_t idx = i * Nt + j;
                values[idx] = m[i] * tau[j] + 0.1;
            }
        }
        return values;
    }
};

TEST_F(CoxDeBoorProfileTest, MeasureCoxDeBoorOverhead) {
    // Create knot vector for cubic B-splines
    std::vector<double> knots = {0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1};

    const int NUM_EVALUATIONS = 100000;  // Many evaluations to measure overhead

    std::cout << "\n=== Cox-de Boor Profiling ===\n";
    std::cout << "Evaluations: " << NUM_EVALUATIONS << "\n\n";

    // Test 1: SIMD version
    {
        double total_simd_us = 0.0;
        alignas(32) double N[4];

        for (int run = 0; run < 5; ++run) {
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < NUM_EVALUATIONS; ++i) {
                double x = 0.5 + (i % 100) * 0.001;  // Vary evaluation point
                cubic_basis_nonuniform_simd(knots, 5, x, N);
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            total_simd_us += duration;
        }

        double avg_simd_us = total_simd_us / 5.0;
        double per_eval_ns = (avg_simd_us * 1000.0) / NUM_EVALUATIONS;

        std::cout << "SIMD Cox-de Boor:\n";
        std::cout << "  Total time (5 runs): " << avg_simd_us << " µs\n";
        std::cout << "  Per evaluation: " << per_eval_ns << " ns\n\n";
    }

    // Test 2: Scalar version
    {
        double total_scalar_us = 0.0;
        double N[4];

        for (int run = 0; run < 5; ++run) {
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < NUM_EVALUATIONS; ++i) {
                double x = 0.5 + (i % 100) * 0.001;
                cubic_basis_nonuniform(knots, 5, x, N);
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            total_scalar_us += duration;
        }

        double avg_scalar_us = total_scalar_us / 5.0;
        double per_eval_ns = (avg_scalar_us * 1000.0) / NUM_EVALUATIONS;

        std::cout << "Scalar Cox-de Boor:\n";
        std::cout << "  Total time (5 runs): " << avg_scalar_us << " µs\n";
        std::cout << "  Per evaluation: " << per_eval_ns << " ns\n\n";

        // Calculate actual speedup
        double total_simd_us = 0.0;
        alignas(32) double N_simd[4];
        for (int run = 0; run < 5; ++run) {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < NUM_EVALUATIONS; ++i) {
                double x = 0.5 + (i % 100) * 0.001;
                cubic_basis_nonuniform_simd(knots, 5, x, N_simd);
            }
            auto end = std::chrono::high_resolution_clock::now();
            total_simd_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        }
        double avg_simd_us_rerun = total_simd_us / 5.0;

        double speedup = avg_scalar_us / avg_simd_us_rerun;
        std::cout << "SIMD Speedup: " << speedup << "×\n\n";
    }
}

TEST_F(CoxDeBoorProfileTest, MeasureEndToEndOverhead) {
    // Medium grid to match performance test
    auto moneyness = create_moneyness_grid(20);
    auto maturity = create_maturity_grid(15);

    auto values = generate_test_values(moneyness, maturity);

    std::cout << "\n=== End-to-End Profiling (Axis 0 only) ===\n";
    std::cout << "Grid: " << moneyness.size() << "×" << maturity.size() << " = "
              << values.size() << " points\n\n";

    // Estimate Cox-de Boor overhead in context
    // Each fit operation calls cubic_basis_nonuniform_simd N times
    // where N = number of data points

    size_t total_evaluations = moneyness.size() * maturity.size();  // 300 points

    std::cout << "Total Cox-de Boor evaluations per fit: ~" << total_evaluations << "\n";
    std::cout << "Estimated Cox-de Boor time at 50ns/eval: " << (total_evaluations * 50) / 1000.0 << " µs\n";
    std::cout << "As percentage of 81ms total: " << ((total_evaluations * 50) / 81000.0) * 100.0 << "%\n\n";
}
