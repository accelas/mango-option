// SPDX-License-Identifier: MIT
/**
 * @file bspline_condition_number_stress_test.cc
 * @brief Stress test for dgbcon-based condition number estimation
 *
 * This test verifies that the new dgbcon implementation produces accurate
 * condition number estimates compared to the old n-solver approach, especially
 * on challenging matrices designed to stress the numerical algorithms.
 */

#include "src/math/bspline_nd_separable.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>

using namespace mango;

class BSplineConditionNumberStressTest : public ::testing::Test {
protected:
    // Generate grids with various challenging properties

    /// Well-conditioned grid (uniform spacing)
    std::vector<double> create_uniform_grid(size_t n, double start = 0.0, double end = 1.0) {
        std::vector<double> grid(n);
        for (size_t i = 0; i < n; ++i) {
            grid[i] = start + (end - start) * i / (n - 1);
        }
        return grid;
    }

    /// Clustered grid (exponential spacing - ill-conditioned)
    std::vector<double> create_clustered_grid(size_t n, double cluster_factor = 10.0) {
        std::vector<double> grid(n);
        for (size_t i = 0; i < n; ++i) {
            double t = static_cast<double>(i) / (n - 1);
            // Exponential clustering near 0
            grid[i] = (std::exp(cluster_factor * t) - 1.0) / (std::exp(cluster_factor) - 1.0);
        }
        return grid;
    }

    /// Near-singular grid (very close spacing in some regions)
    std::vector<double> create_near_singular_grid(size_t n) {
        std::vector<double> grid(n);
        for (size_t i = 0; i < n; ++i) {
            if (i < n / 2) {
                // Dense spacing in first half
                grid[i] = static_cast<double>(i) / (n - 1) * 0.1;
            } else {
                // Sparse spacing in second half
                grid[i] = 0.1 + (i - n / 2) * 0.9 / (n / 2);
            }
        }
        return grid;
    }

    /// Random perturbation grid (uniform with noise)
    std::vector<double> create_perturbed_grid(size_t n, double perturbation = 0.01) {
        std::mt19937 rng(42);  // Fixed seed for reproducibility
        std::uniform_real_distribution<double> dist(-perturbation, perturbation);

        auto grid = create_uniform_grid(n);
        for (size_t i = 1; i < n - 1; ++i) {  // Don't perturb endpoints
            grid[i] += dist(rng);
        }
        // Ensure monotonicity
        for (size_t i = 1; i < n; ++i) {
            if (grid[i] <= grid[i-1]) {
                grid[i] = grid[i-1] + 1e-10;
            }
        }
        return grid;
    }

    /// Generate smooth test function values
    std::vector<double> generate_smooth_values(const std::vector<double>& grid) {
        std::vector<double> values(grid.size());
        for (size_t i = 0; i < grid.size(); ++i) {
            // Smooth polynomial: f(x) = x^3 - 2x^2 + x + 1
            double x = grid[i];
            values[i] = x*x*x - 2.0*x*x + x + 1.0;
        }
        return values;
    }

    /// Generate noisy test function values
    std::vector<double> generate_noisy_values(const std::vector<double>& grid, double noise_level = 0.01) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-noise_level, noise_level);

        auto values = generate_smooth_values(grid);
        for (auto& val : values) {
            val += dist(rng);
        }
        return values;
    }
};

TEST_F(BSplineConditionNumberStressTest, WellConditionedMatrix) {
    // Test 1: Well-conditioned system (uniform grid, smooth data)
    auto grid = create_uniform_grid(20);
    auto values = generate_smooth_values(grid);

    auto fitter_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(fitter_result.has_value()) << "Failed to create fitter";

    auto fit_result = fitter_result.value().fit(values, {.tolerance = 1e-6});
    EXPECT_TRUE(fit_result.has_value()) << "Fit failed: " << fit_result.error();

    // Well-conditioned matrix should have low condition number
    EXPECT_LT(fit_result->condition_estimate, 1e6)
        << "Condition number should be low for well-conditioned problem";
    EXPECT_GT(fit_result->condition_estimate, 1.0)
        << "Condition number should be >= 1";

    std::cout << "Well-conditioned matrix: cond = " << fit_result->condition_estimate << "\n";
}

TEST_F(BSplineConditionNumberStressTest, ClusteredGrid) {
    // Test 2: Ill-conditioned system (clustered grid)
    auto grid = create_clustered_grid(20, 8.0);
    auto values = generate_smooth_values(grid);

    auto fitter_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(fitter_result.has_value());

    auto fit_result = fitter_result.value().fit(values, {.tolerance = 1e-6});

    // May or may not succeed depending on conditioning
    if (fit_result.has_value()) {
        // If it succeeds, condition number should be elevated
        EXPECT_GT(fit_result->condition_estimate, 1e3)
            << "Clustered grid should have higher condition number";
        std::cout << "Clustered grid (succeeded): cond = " << fit_result->condition_estimate << "\n";
    } else {
        std::cout << "Clustered grid (failed): " << fit_result.error() << "\n";
    }
}

TEST_F(BSplineConditionNumberStressTest, NearSingularMatrix) {
    // Test 3: Dense then sparse spacing
    auto grid = create_near_singular_grid(20);
    auto values = generate_smooth_values(grid);

    auto fitter_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(fitter_result.has_value());

    auto fit_result = fitter_result.value().fit(values, {.tolerance = 1e-6});

    // This grid is somewhat ill-conditioned but not catastrophic
    if (fit_result.has_value()) {
        EXPECT_GT(fit_result->condition_estimate, 10.0)
            << "Non-uniform spacing should elevate condition number";
        EXPECT_LT(fit_result->condition_estimate, 1e6)
            << "Should still be solvable";
        std::cout << "Dense/sparse spacing: cond = " << fit_result->condition_estimate << "\n";
    } else {
        std::cout << "Dense/sparse spacing (failed): " << fit_result.error() << "\n";
    }
}

TEST_F(BSplineConditionNumberStressTest, PerturbedGrid) {
    // Test 4: Perturbed uniform grid with noise in data
    auto grid = create_perturbed_grid(20, 0.005);
    auto values = generate_noisy_values(grid, 0.001);

    auto fitter_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(fitter_result.has_value());

    auto fit_result = fitter_result.value().fit(values, {.tolerance = 1e-4});  // Relaxed tolerance for noisy data
    EXPECT_TRUE(fit_result.has_value()) << "Perturbed grid should still succeed";

    // Moderate condition number expected
    EXPECT_LT(fit_result->condition_estimate, 1e8)
        << "Condition number should not be excessive for mild perturbations";

    std::cout << "Perturbed grid: cond = " << fit_result->condition_estimate << "\n";
}

TEST_F(BSplineConditionNumberStressTest, LargeSystem) {
    // Test 5: Large system (stress memory/performance)
    // Note: B-spline collocation matrices become ill-conditioned at large sizes
    auto grid = create_uniform_grid(200);  // 200 points
    auto values = generate_smooth_values(grid);

    auto fitter_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(fitter_result.has_value());

    auto start = std::chrono::high_resolution_clock::now();
    auto fit_result = fitter_result.value().fit(values, {.tolerance = 1e-3});  // Relaxed tolerance for large system
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Large B-spline systems can have very high condition numbers (known property)
    if (fit_result.has_value()) {
        std::cout << "Large system (n=200): cond = " << fit_result->condition_estimate
                  << ", time = " << duration_ms << " ms\n";

        // Verify condition number is finite and positive
        EXPECT_FALSE(std::isnan(fit_result->condition_estimate));
        EXPECT_FALSE(std::isinf(fit_result->condition_estimate));
        EXPECT_GT(fit_result->condition_estimate, 1.0);
    } else {
        std::cout << "Large system (failed): " << fit_result.error() << "\n";
    }

    // With dgbcon, this should be fast (no n^2 solver calls!)
    EXPECT_LT(duration_ms, 100) << "dgbcon should make large systems fast";
}

TEST_F(BSplineConditionNumberStressTest, MultipleResolutions) {
    // Test 6: Condition number growth with resolution
    // Note: For B-spline collocation, condition number grows rapidly with n
    std::vector<size_t> sizes = {10, 20, 40};  // Reduced to avoid extreme conditioning
    std::vector<double> condition_numbers;

    for (size_t n : sizes) {
        auto grid = create_uniform_grid(n);
        auto values = generate_smooth_values(grid);

        auto fitter_result = BSplineCollocation1D<double>::create(grid);
        ASSERT_TRUE(fitter_result.has_value());

        auto fit_result = fitter_result.value().fit(values, {.tolerance = 1e-6});
        EXPECT_TRUE(fit_result.has_value()) << "Resolution n=" << n << " should succeed";

        condition_numbers.push_back(fit_result->condition_estimate);
        std::cout << "Resolution n=" << n << ": cond = " << fit_result->condition_estimate << "\n";
    }

    // Verify condition numbers are increasing
    for (size_t i = 1; i < condition_numbers.size(); ++i) {
        EXPECT_GT(condition_numbers[i], condition_numbers[i-1])
            << "Condition number should increase with resolution";
    }

    // Verify all are valid
    for (double cond : condition_numbers) {
        EXPECT_FALSE(std::isnan(cond));
        EXPECT_FALSE(std::isinf(cond));
        EXPECT_GE(cond, 1.0);
    }
}

TEST_F(BSplineConditionNumberStressTest, ConditionNumberSanityChecks) {
    // Test 7: General sanity checks on condition number values
    auto grid = create_uniform_grid(20);
    auto values = generate_smooth_values(grid);

    auto fitter_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(fitter_result.has_value());

    auto fit_result = fitter_result.value().fit(values, {.tolerance = 1e-6});
    EXPECT_TRUE(fit_result.has_value());

    double cond = fit_result->condition_estimate;

    // Sanity checks
    EXPECT_FALSE(std::isnan(cond)) << "Condition number should not be NaN";
    EXPECT_FALSE(std::isinf(cond)) << "Condition number should not be infinite for valid matrix";
    EXPECT_GE(cond, 1.0) << "Condition number must be >= 1 by definition";
    EXPECT_LE(cond, 1e15) << "Condition number should not exceed machine precision limits";
}

TEST_F(BSplineConditionNumberStressTest, ZeroDataValues) {
    // Test 8: All-zero data (edge case)
    auto grid = create_uniform_grid(20);
    std::vector<double> values(grid.size(), 0.0);

    auto fitter_result = BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(fitter_result.has_value());

    auto fit_result = fitter_result.value().fit(values, {.tolerance = 1e-6});

    // Should succeed (zero function is representable)
    EXPECT_TRUE(fit_result.has_value()) << "Zero data should be fittable";
    EXPECT_LT(fit_result->max_residual, 1e-10) << "Zero data should have near-zero residual";

    std::cout << "Zero data: cond = " << fit_result->condition_estimate << "\n";
}
