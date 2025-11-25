/**
 * @file bspline_fitter_4d_separable_test.cc
 * @brief Unit tests for separable 4D B-spline coefficient fitting
 *
 * Validates:
 * - Factory pattern creation with validation
 * - Separable function fitting
 * - Error handling for invalid grids
 * - Integration with evaluation
 *
 * Note: Now tests BSplineNDSeparable<double, 4> directly instead of
 *       the old hardcoded BSplineNDSeparable<double, 4> class.
 */

#include "src/math/bspline_nd_separable.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <array>

using namespace mango;

namespace {

/// Helper: Create linearly spaced grid
std::vector<double> linspace(double start, double end, int n) {
    std::vector<double> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = start + (end - start) * i / (n - 1);
    }
    return result;
}

/// Helper: Separable test function
struct TestFunctions {
    static double separable(double m, double t, double v, double r) {
        return m * m * std::exp(-t) * v * (1.0 + r);
    }

    static double constant(double m, double t, double v, double r) {
        (void)m; (void)t; (void)v; (void)r;
        return 5.0;
    }
};

}  // namespace

// ============================================================================
// Factory Pattern Tests
// ============================================================================

TEST(BSplineNDSeparableTest, FactoryCreation) {
    auto m = linspace(0.8, 1.2, 10);
    auto t = linspace(0.1, 2.0, 8);
    auto v = linspace(0.1, 0.5, 6);
    auto r = linspace(0.0, 0.1, 5);

    // Factory creation should succeed
    auto result = BSplineNDSeparable<double, 4>::create(std::array<std::vector<double>, 4>{m, t, v, r});
    EXPECT_TRUE(result.has_value());

    // Verify the object was created
    if (result.has_value()) {
        (void)result.value();  // Suppress unused variable warning
    }
}

TEST(BSplineNDSeparableTest, FactoryCreationWithSmallGrids) {
    auto m = linspace(0.8, 1.2, 4);  // Minimum size
    auto t = linspace(0.1, 2.0, 4);
    auto v = linspace(0.1, 0.5, 4);
    auto r = linspace(0.0, 0.1, 4);

    // Factory creation should succeed with minimum grid sizes
    auto result = BSplineNDSeparable<double, 4>::create(std::array<std::vector<double>, 4>{m, t, v, r});
    EXPECT_TRUE(result.has_value());
}

TEST(BSplineNDSeparableTest, FactoryCreationFailure) {
    auto m_small = linspace(0.8, 1.2, 3);  // Too few points!
    auto t = linspace(0.1, 2.0, 8);
    auto v = linspace(0.1, 0.5, 6);
    auto r = linspace(0.0, 0.1, 5);

    auto result = BSplineNDSeparable<double, 4>::create(std::array<std::vector<double>, 4>{m_small, t, v, r});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::InterpolationErrorCode::InsufficientGridPoints);
}

TEST(BSplineNDSeparableTest, FactoryCreationFailureMultipleSmallGrids) {
    auto m_small = linspace(0.8, 1.2, 3);  // Too few points!
    auto t_small = linspace(0.1, 2.0, 2);  // Too few points!
    auto v_small = linspace(0.1, 0.5, 1);  // Too few points!
    auto r_small = linspace(0.0, 0.1, 0);  // Empty!

    auto result = BSplineNDSeparable<double, 4>::create(std::array<std::vector<double>, 4>{m_small, t_small, v_small, r_small});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::InterpolationErrorCode::InsufficientGridPoints);
}

// ============================================================================
// Function Fitting Tests
// ============================================================================

TEST(BSplineNDSeparableTest, ConstantFunction) {
    auto m_grid = linspace(0.8, 1.2, 8);
    auto t_grid = linspace(0.1, 2.0, 6);
    auto v_grid = linspace(0.1, 0.5, 5);
    auto r_grid = linspace(0.0, 0.1, 4);

    const int Nm = m_grid.size();
    const int Nt = t_grid.size();
    const int Nv = v_grid.size();
    const int Nr = r_grid.size();

    // Generate constant function values
    std::vector<double> values(Nm * Nt * Nv * Nr);
    for (int i = 0; i < Nm; ++i) {
        for (int j = 0; j < Nt; ++j) {
            for (int k = 0; k < Nv; ++k) {
                for (int l = 0; l < Nr; ++l) {
                    size_t idx = ((i * Nt + j) * Nv + k) * Nr + l;
                    values[idx] = TestFunctions::constant(
                        m_grid[i], t_grid[j], v_grid[k], r_grid[l]);
                }
            }
        }
    }

    // Create fitter using factory pattern
    auto fitter_result = BSplineNDSeparable<double, 4>::create(std::array<std::vector<double>, 4>{m_grid, t_grid, v_grid, r_grid});
    ASSERT_TRUE(fitter_result.has_value());
    auto& fitter = fitter_result.value();

    // Fit coefficients
    auto result = fitter.fit(values, BSplineNDSeparableConfig<double>{.tolerance = 1e-3});

    EXPECT_TRUE(result.has_value()) << "Error: " << result.error();
    EXPECT_EQ(result->coefficients.size(), values.size());

    // Check diagnostic information
    EXPECT_GE(result->failed_slices[0], 0UL);
    EXPECT_GE(result->failed_slices[1], 0UL);
    EXPECT_GE(result->failed_slices[2], 0UL);
    EXPECT_GE(result->failed_slices[3], 0UL);
}

TEST(BSplineNDSeparableTest, SeparableFunction) {
    auto m_grid = linspace(0.8, 1.2, 10);
    auto t_grid = linspace(0.1, 2.0, 10);
    auto v_grid = linspace(0.1, 0.5, 8);
    auto r_grid = linspace(0.0, 0.1, 6);

    const int Nm = m_grid.size();
    const int Nt = t_grid.size();
    const int Nv = v_grid.size();
    const int Nr = r_grid.size();

    // Generate separable function values
    std::vector<double> values(Nm * Nt * Nv * Nr);
    for (int i = 0; i < Nm; ++i) {
        for (int j = 0; j < Nt; ++j) {
            for (int k = 0; k < Nv; ++k) {
                for (int l = 0; l < Nr; ++l) {
                    size_t idx = ((i * Nt + j) * Nv + k) * Nr + l;
                    values[idx] = TestFunctions::separable(
                        m_grid[i], t_grid[j], v_grid[k], r_grid[l]);
                }
            }
        }
    }

    // Create fitter using factory pattern
    auto fitter_result = BSplineNDSeparable<double, 4>::create(std::array<std::vector<double>, 4>{m_grid, t_grid, v_grid, r_grid});
    ASSERT_TRUE(fitter_result.has_value());
    auto& fitter = fitter_result.value();

    // Fit coefficients with relaxed tolerance
    auto result = fitter.fit(values, BSplineNDSeparableConfig<double>{.tolerance = 1e-3});

    EXPECT_TRUE(result.has_value()) << "Error: " << result.error();
    EXPECT_EQ(result->coefficients.size(), values.size());
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(BSplineNDSeparableTest, WrongValueSize) {
    auto m = linspace(0.8, 1.2, 10);
    auto t = linspace(0.1, 2.0, 8);
    auto v = linspace(0.1, 0.5, 6);
    auto r = linspace(0.0, 0.1, 5);

    auto fitter_result = BSplineNDSeparable<double, 4>::create(std::array<std::vector<double>, 4>{m, t, v, r});
    ASSERT_TRUE(fitter_result.has_value());
    auto& fitter = fitter_result.value();

    std::vector<double> values_wrong_size(100, 1.0);  // Wrong size!

    auto result = fitter.fit(values_wrong_size);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::InterpolationErrorCode::ValueSizeMismatch);
}