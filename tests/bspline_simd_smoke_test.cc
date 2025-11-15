/**
 * @file bspline_simd_smoke_test.cc
 * @brief Smoke tests for SIMD Cox-de Boor basis functions
 *
 * Basic verification that SIMD implementation compiles and produces
 * reasonable output. Comprehensive correctness tests will be added in Task 3.
 */

#include "src/interpolation/bspline_utils.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

namespace mango {
namespace {

// Simple smoke test: verify SIMD matches scalar implementation
TEST(BSplineSIMDSmokeTest, BasicExecution) {
    // Create uniform knot vector for cubic B-spline
    std::vector<double> knots = {0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4};

    double N_scalar[4], N_simd[4];
    double x = 2.0;
    int i = 6;  // Knot span containing x=2.0

    // Call both scalar and SIMD functions
    cubic_basis_nonuniform(knots, i, x, N_scalar);
    cubic_basis_nonuniform_simd(knots, i, x, N_simd);

    // Verify SIMD matches scalar to high precision
    for (int j = 0; j < 4; ++j) {
        EXPECT_NEAR(N_simd[j], N_scalar[j], 1e-14)
            << "SIMD result should match scalar at index " << j;
    }

    // Basic sanity checks
    for (int j = 0; j < 4; ++j) {
        EXPECT_TRUE(std::isfinite(N_simd[j])) << "N[" << j << "] should be finite";
        EXPECT_GE(N_simd[j], 0.0) << "N[" << j << "] should be non-negative";
        EXPECT_LE(N_simd[j], 1.0) << "N[" << j << "] should be <= 1.0";
    }
}

// Test that degree-0 initialization produces reasonable values
TEST(BSplineSIMDSmokeTest, Degree0Initialization) {
    std::vector<double> knots = {0, 1, 2, 3, 4, 5, 6, 7};

    // Point inside interval [3, 4)
    double x = 3.5;
    int i = 3;  // Points to interval [3, 4)

    simd4d N0 = cubic_basis_degree0_simd(knots, i, x);

    double values[4];
    N0.copy_to(values, stdx::element_aligned);

    // Only first basis should be 1.0 (x in [3, 4))
    EXPECT_DOUBLE_EQ(values[0], 1.0) << "Basis covering x should be 1.0";

    for (int j = 1; j < 4; ++j) {
        EXPECT_DOUBLE_EQ(values[j], 0.0) << "Other bases should be 0.0";
    }
}

// Test edge case: evaluation at right boundary
TEST(BSplineSIMDSmokeTest, RightBoundary) {
    std::vector<double> knots = {0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4};

    double N[4];
    double x = 4.0;  // Right boundary
    int i = 6;

    cubic_basis_nonuniform_simd(knots, i, x, N);

    // At right boundary, only first basis should be 1.0
    EXPECT_DOUBLE_EQ(N[0], 1.0);
    EXPECT_DOUBLE_EQ(N[1], 0.0);
    EXPECT_DOUBLE_EQ(N[2], 0.0);
    EXPECT_DOUBLE_EQ(N[3], 0.0);
}

// Test with non-uniform knot spacing - verify SIMD matches scalar
TEST(BSplineSIMDSmokeTest, NonUniformKnots) {
    std::vector<double> knots = {0, 0, 0, 0, 0.5, 1.0, 2.5, 3.0, 3.0, 3.0, 3.0};

    double N_scalar[4], N_simd[4];
    double x = 1.5;
    int i = 6;

    cubic_basis_nonuniform(knots, i, x, N_scalar);
    cubic_basis_nonuniform_simd(knots, i, x, N_simd);

    // Verify SIMD matches scalar
    for (int j = 0; j < 4; ++j) {
        EXPECT_NEAR(N_simd[j], N_scalar[j], 1e-14)
            << "SIMD result should match scalar at index " << j;
    }

    // Basic sanity checks
    for (int j = 0; j < 4; ++j) {
        EXPECT_TRUE(std::isfinite(N_simd[j]));
        EXPECT_GE(N_simd[j], 0.0);
    }
}

}  // namespace
}  // namespace mango
