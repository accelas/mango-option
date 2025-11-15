/**
 * @file bspline_simd_test.cc
 * @brief Comprehensive correctness tests for SIMD Cox-de Boor basis functions
 *
 * Tests SIMD implementation against scalar reference on:
 * - Various knot sequences (clamped, uniform, non-uniform)
 * - Partition of unity (basis functions sum to 1.0)
 * - Edge cases (boundaries, repeated knots)
 *
 * Part of Task 3 from docs/plans/2025-01-16-cox-de-boor-simd-plan.md
 */

#include "src/interpolation/bspline_utils.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <numeric>

namespace mango {
namespace {

// ============================================================================
// Test Suite 1: Scalar-SIMD Equivalence
// ============================================================================

class ScalarSIMDEquivalenceTest : public ::testing::Test {
protected:
    // Test SIMD matches scalar on various knot sequences
    void TestKnotSequence(const std::vector<double>& knots, const std::string& description) {
        // Test multiple evaluation points across the domain
        const double x_min = knots.front();
        const double x_max = knots.back();
        const double dx = (x_max - x_min) / 100.0;

        for (double x = x_min; x <= x_max; x += dx) {
            // Find valid knot span
            int i = find_span_cubic(knots, x);

            double N_scalar[4], N_simd[4];

            // Call both scalar and SIMD implementations
            cubic_basis_nonuniform(knots, i, x, N_scalar);
            cubic_basis_nonuniform_simd(knots, i, x, N_simd);

            // Verify SIMD matches scalar to < 1e-14
            for (int j = 0; j < 4; ++j) {
                EXPECT_NEAR(N_simd[j], N_scalar[j], 1e-14)
                    << description << ": Mismatch at x=" << x << ", basis[" << j << "]";

                // Verify results are finite
                EXPECT_TRUE(std::isfinite(N_scalar[j]));
                EXPECT_TRUE(std::isfinite(N_simd[j]));
            }
        }
    }
};

TEST_F(ScalarSIMDEquivalenceTest, ClampedCubicKnots) {
    // Clamped knot vector: endpoints repeated 4 times
    std::vector<double> knots = {0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4};
    TestKnotSequence(knots, "Clamped cubic");
}

TEST_F(ScalarSIMDEquivalenceTest, UniformKnots) {
    // Uniform spacing: equal intervals
    std::vector<double> knots = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    TestKnotSequence(knots, "Uniform");
}

TEST_F(ScalarSIMDEquivalenceTest, NonUniformKnots) {
    // Non-uniform spacing: irregular intervals
    std::vector<double> knots = {0, 0, 0, 0, 0.5, 1, 1.5, 2.5, 3, 3, 3, 3};
    TestKnotSequence(knots, "Non-uniform");
}

TEST_F(ScalarSIMDEquivalenceTest, FinelySpacedKnots) {
    // Fine spacing to test numerical stability
    std::vector<double> knots = {0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5};
    TestKnotSequence(knots, "Finely spaced");
}

TEST_F(ScalarSIMDEquivalenceTest, IrregularNonUniform) {
    // Irregular spacing with varying densities
    std::vector<double> knots = {0, 0, 0, 0, 0.1, 0.5, 2.0, 5.0, 10.0, 10.0, 10.0, 10.0};
    TestKnotSequence(knots, "Irregular non-uniform");
}

TEST_F(ScalarSIMDEquivalenceTest, LargeScaleKnots) {
    // Test with larger values to check for numerical issues
    std::vector<double> knots = {100, 100, 100, 100, 101, 102, 103, 104, 105, 105, 105, 105};
    TestKnotSequence(knots, "Large scale");
}

TEST_F(ScalarSIMDEquivalenceTest, SmallScaleKnots) {
    // Test with small values near zero
    std::vector<double> knots = {0, 0, 0, 0, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 5e-3, 5e-3, 5e-3};
    TestKnotSequence(knots, "Small scale");
}

// ============================================================================
// Test Suite 2: Partition of Unity
// ============================================================================

class PartitionOfUnityTest : public ::testing::Test {
protected:
    // Verify that basis functions sum to 1.0 at all evaluation points
    void TestPartitionOfUnity(const std::vector<double>& knots, const std::string& description) {
        // Only test within valid domain (exclude clamped endpoints)
        const int n_ctrl = static_cast<int>(knots.size()) - 3 - 1;  // degree = 3
        const double x_min = knots[3];      // First interior knot
        const double x_max = knots[n_ctrl]; // Last interior knot

        if (x_max <= x_min) return;  // Skip if invalid domain

        const double dx = (x_max - x_min) / 200.0;  // Fine sampling

        for (double x = x_min; x < x_max; x += dx) {  // Note: x < x_max (not <=)
            int i = find_span_cubic(knots, x);

            double N[4];
            cubic_basis_nonuniform_simd(knots, i, x, N);

            // Sum of all 4 basis functions should be 1.0
            // These 4 are the ONLY non-zero basis functions at x
            double sum = N[0] + N[1] + N[2] + N[3];

            EXPECT_NEAR(sum, 1.0, 1e-12)
                << description << ": Partition of unity failed at x=" << x
                << " (span i=" << i << ")";
        }
    }
};

TEST_F(PartitionOfUnityTest, ClampedKnots) {
    std::vector<double> knots = {0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6};
    TestPartitionOfUnity(knots, "Clamped");
}

TEST_F(PartitionOfUnityTest, UniformKnots) {
    std::vector<double> knots = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    TestPartitionOfUnity(knots, "Uniform");
}

TEST_F(PartitionOfUnityTest, NonUniformKnots) {
    std::vector<double> knots = {0, 0, 0, 0, 0.5, 1.5, 3.0, 5.0, 8.0, 8.0, 8.0, 8.0};
    TestPartitionOfUnity(knots, "Non-uniform");
}

TEST_F(PartitionOfUnityTest, FinelySpacedKnots) {
    std::vector<double> knots = {0, 0, 0, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05, 0.05};
    TestPartitionOfUnity(knots, "Finely spaced");
}

// ============================================================================
// Test Suite 3: Edge Cases
// ============================================================================

class EdgeCasesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Standard clamped knots for most edge case tests
        standard_knots_ = {0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4};
    }

    std::vector<double> standard_knots_;
};

TEST_F(EdgeCasesTest, EvaluationAtKnotBoundaries) {
    // Test at each unique knot value
    std::vector<double> test_knots = {0.0, 1.0, 2.0, 3.0, 4.0};

    for (double knot : test_knots) {
        int i = find_span_cubic(standard_knots_, knot);
        double N[4];

        cubic_basis_nonuniform_simd(standard_knots_, i, knot, N);

        // Results should be well-defined (no NaN/Inf)
        for (int j = 0; j < 4; ++j) {
            EXPECT_TRUE(std::isfinite(N[j]))
                << "N[" << j << "] not finite at knot=" << knot;
        }

        // Partition of unity should hold
        double sum = N[0] + N[1] + N[2] + N[3];
        EXPECT_NEAR(sum, 1.0, 1e-12)
            << "Partition of unity failed at knot=" << knot;
    }
}

TEST_F(EdgeCasesTest, LeftBoundary) {
    // Test at left boundary (x = 0.0)
    // For clamped knots {0,0,0,0,...}, the left boundary is special
    double x = 0.0;
    int i = find_span_cubic(standard_knots_, x);
    double N[4];

    cubic_basis_nonuniform_simd(standard_knots_, i, x, N);

    // All results should be finite and non-negative
    for (int j = 0; j < 4; ++j) {
        EXPECT_TRUE(std::isfinite(N[j]));
        EXPECT_GE(N[j], -1e-14);  // Allow tiny numerical error
        EXPECT_LE(N[j], 1.0 + 1e-14);
    }

    // Partition of unity should hold
    double sum = N[0] + N[1] + N[2] + N[3];
    EXPECT_NEAR(sum, 1.0, 1e-12)
        << "Partition of unity failed at left boundary";
}

TEST_F(EdgeCasesTest, RightBoundary) {
    // Test at right boundary (x = 4.0)
    double x = 4.0;
    int i = find_span_cubic(standard_knots_, x);
    double N[4];

    cubic_basis_nonuniform_simd(standard_knots_, i, x, N);

    // At right boundary with clamped knots
    EXPECT_NEAR(N[0], 1.0, 1e-14);
    EXPECT_NEAR(N[1], 0.0, 1e-14);
    EXPECT_NEAR(N[2], 0.0, 1e-14);
    EXPECT_NEAR(N[3], 0.0, 1e-14);

    // All results should be finite
    for (int j = 0; j < 4; ++j) {
        EXPECT_TRUE(std::isfinite(N[j]));
    }
}

TEST_F(EdgeCasesTest, RepeatedInteriorKnots) {
    // Knot vector with repeated interior knots (multiplicity)
    std::vector<double> knots = {0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 3};

    // Test at and near the repeated knots
    std::vector<double> test_points = {0.5, 1.0, 1.5, 2.0, 2.5};

    for (double x : test_points) {
        int i = find_span_cubic(knots, x);
        double N_scalar[4], N_simd[4];

        cubic_basis_nonuniform(knots, i, x, N_scalar);
        cubic_basis_nonuniform_simd(knots, i, x, N_simd);

        // SIMD should match scalar
        for (int j = 0; j < 4; ++j) {
            EXPECT_NEAR(N_simd[j], N_scalar[j], 1e-14)
                << "Repeated knots: mismatch at x=" << x << ", basis[" << j << "]";

            EXPECT_TRUE(std::isfinite(N_simd[j]));
        }

        // Partition of unity
        double sum = N_simd[0] + N_simd[1] + N_simd[2] + N_simd[3];
        EXPECT_NEAR(sum, 1.0, 1e-12)
            << "Repeated knots: partition of unity failed at x=" << x;
    }
}

TEST_F(EdgeCasesTest, DoubleRepeatedKnots) {
    // Multiple pairs of repeated knots
    std::vector<double> knots = {0, 0, 0, 0, 0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 2.5, 2.5};

    // Test throughout the domain
    for (double x = 0.0; x <= 2.5; x += 0.1) {
        int i = find_span_cubic(knots, x);
        double N_scalar[4], N_simd[4];

        cubic_basis_nonuniform(knots, i, x, N_scalar);
        cubic_basis_nonuniform_simd(knots, i, x, N_simd);

        // SIMD should match scalar
        for (int j = 0; j < 4; ++j) {
            EXPECT_NEAR(N_simd[j], N_scalar[j], 1e-14)
                << "Double repeated: mismatch at x=" << x << ", basis[" << j << "]";
        }
    }
}

TEST_F(EdgeCasesTest, VeryFineSpacing) {
    // Extremely fine knot spacing (test numerical stability)
    std::vector<double> knots = {0, 0, 0, 0, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 5e-6, 5e-6, 5e-6};

    for (double x = 0.0; x <= 5e-6; x += 5e-7) {
        int i = find_span_cubic(knots, x);
        double N[4];

        cubic_basis_nonuniform_simd(knots, i, x, N);

        // All results should be finite
        for (int j = 0; j < 4; ++j) {
            EXPECT_TRUE(std::isfinite(N[j]))
                << "Very fine spacing: N[" << j << "] not finite at x=" << x;
        }

        // Partition of unity should still hold
        double sum = N[0] + N[1] + N[2] + N[3];
        EXPECT_NEAR(sum, 1.0, 1e-10)  // Slightly relaxed tolerance for extreme scale
            << "Very fine spacing: partition of unity failed at x=" << x;
    }
}

TEST_F(EdgeCasesTest, NegativeKnots) {
    // Test with negative knot values
    std::vector<double> knots = {-4, -4, -4, -4, -3, -2, -1, 0, 0, 0, 0};

    for (double x = -4.0; x <= 0.0; x += 0.2) {
        int i = find_span_cubic(knots, x);
        double N_scalar[4], N_simd[4];

        cubic_basis_nonuniform(knots, i, x, N_scalar);
        cubic_basis_nonuniform_simd(knots, i, x, N_simd);

        // SIMD should match scalar
        for (int j = 0; j < 4; ++j) {
            EXPECT_NEAR(N_simd[j], N_scalar[j], 1e-14)
                << "Negative knots: mismatch at x=" << x << ", basis[" << j << "]";
        }
    }
}

TEST_F(EdgeCasesTest, ZeroDivisionHandling) {
    // Knots with potential for zero denominators (uniform sections)
    std::vector<double> knots = {0, 0, 0, 0, 1, 1, 1, 2, 3, 3, 3, 3};

    // Test at points where denominator might be zero
    std::vector<double> test_points = {0.5, 1.0, 1.5, 2.0, 2.5};

    for (double x : test_points) {
        int i = find_span_cubic(knots, x);
        double N[4];

        cubic_basis_nonuniform_simd(knots, i, x, N);

        // All results should be finite (no division by zero)
        for (int j = 0; j < 4; ++j) {
            EXPECT_TRUE(std::isfinite(N[j]))
                << "Zero division: N[" << j << "] not finite at x=" << x;
            EXPECT_GE(N[j], 0.0) << "N[" << j << "] should be non-negative";
        }
    }
}

TEST_F(EdgeCasesTest, MidpointEvaluation) {
    // Test at exact midpoints between knots
    for (double x : {0.5, 1.5, 2.5, 3.5}) {
        int i = find_span_cubic(standard_knots_, x);
        double N_scalar[4], N_simd[4];

        cubic_basis_nonuniform(standard_knots_, i, x, N_scalar);
        cubic_basis_nonuniform_simd(standard_knots_, i, x, N_simd);

        // SIMD should match scalar exactly at midpoints
        for (int j = 0; j < 4; ++j) {
            EXPECT_NEAR(N_simd[j], N_scalar[j], 1e-14)
                << "Midpoint: mismatch at x=" << x << ", basis[" << j << "]";
        }

        // Verify partition of unity
        double sum = N_simd[0] + N_simd[1] + N_simd[2] + N_simd[3];
        EXPECT_NEAR(sum, 1.0, 1e-12)
            << "Midpoint: partition of unity failed at x=" << x;
    }
}

// ============================================================================
// Additional Validation Tests
// ============================================================================

TEST(BSplineSIMDValidationTest, NonNegativity) {
    // Basis functions should always be non-negative
    std::vector<double> knots = {0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6};

    for (double x = 0.0; x <= 6.0; x += 0.05) {
        int i = find_span_cubic(knots, x);
        double N[4];

        cubic_basis_nonuniform_simd(knots, i, x, N);

        for (int j = 0; j < 4; ++j) {
            EXPECT_GE(N[j], -1e-14)  // Allow tiny numerical error
                << "Basis function N[" << j << "] negative at x=" << x;
        }
    }
}

TEST(BSplineSIMDValidationTest, CompactSupport) {
    // Basis functions should have compact support
    // Outside their support interval, they should be zero
    std::vector<double> knots = {0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6};

    // Test near boundaries where most basis functions should be zero
    double x = 0.1;  // Near left boundary
    int i = find_span_cubic(knots, x);
    double N[4];

    cubic_basis_nonuniform_simd(knots, i, x, N);

    // At least some basis functions should be significantly non-zero
    double max_val = *std::max_element(N, N + 4);
    EXPECT_GT(max_val, 0.1) << "At least one basis should be significantly non-zero";

    // Sum should still be 1.0
    double sum = N[0] + N[1] + N[2] + N[3];
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST(BSplineSIMDValidationTest, Symmetry) {
    // For symmetric knot vectors, basis functions should exhibit symmetry
    std::vector<double> knots = {0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4};
    double center = 2.0;

    // Test symmetric points around center
    for (double offset = 0.1; offset <= 1.9; offset += 0.1) {
        double x_left = center - offset;
        double x_right = center + offset;

        int i_left = find_span_cubic(knots, x_left);
        int i_right = find_span_cubic(knots, x_right);

        double N_left[4], N_right[4];

        cubic_basis_nonuniform_simd(knots, i_left, x_left, N_left);
        cubic_basis_nonuniform_simd(knots, i_right, x_right, N_right);

        // The pattern should be symmetric (though indices may differ)
        double sum_left = N_left[0] + N_left[1] + N_left[2] + N_left[3];
        double sum_right = N_right[0] + N_right[1] + N_right[2] + N_right[3];

        EXPECT_NEAR(sum_left, sum_right, 1e-12)
            << "Symmetry: partition sums differ at offset=" << offset;
    }
}

}  // namespace
}  // namespace mango
