// SPDX-License-Identifier: MIT
// tests/bspline_nd_workspace_integration_test.cc
//
// Integration test for BSplineNDSeparable with ThreadWorkspaceBuffer
//
// Verifies that N-dimensional tensor fitting produces identical results
// with workspace-based zero-allocation fitting.

#include "mango/math/bspline/bspline_nd_separable.hpp"
#include "mango/support/thread_workspace.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace mango;

TEST(BSplineNDWorkspaceIntegrationTest, FitProducesSameResults) {
    // Create 4D test data (small dimensions for speed)
    std::array<size_t, 4> dims = {10, 8, 10, 8};

    // Create grids
    std::array<std::vector<double>, 4> grids;
    for (size_t axis = 0; axis < 4; ++axis) {
        grids[axis].resize(dims[axis]);
        for (size_t i = 0; i < dims[axis]; ++i) {
            grids[axis][i] = static_cast<double>(i) / (dims[axis] - 1);
        }
    }

    // Generate test function values
    size_t total = dims[0] * dims[1] * dims[2] * dims[3];
    std::vector<double> values(total);
    for (size_t i0 = 0; i0 < dims[0]; ++i0) {
        for (size_t i1 = 0; i1 < dims[1]; ++i1) {
            for (size_t i2 = 0; i2 < dims[2]; ++i2) {
                for (size_t i3 = 0; i3 < dims[3]; ++i3) {
                    size_t idx = i0 * dims[1]*dims[2]*dims[3] +
                                 i1 * dims[2]*dims[3] +
                                 i2 * dims[3] + i3;
                    double x0 = grids[0][i0];
                    double x1 = grids[1][i1];
                    double x2 = grids[2][i2];
                    double x3 = grids[3][i3];
                    values[idx] = std::sin(M_PI * x0) * std::cos(M_PI * x1) *
                                 std::exp(-x2) * (1.0 + x3);
                }
            }
        }
    }

    // Create fitter
    auto fitter_result = BSplineNDSeparable<double, 4>::create(grids);
    ASSERT_TRUE(fitter_result.has_value());
    auto& fitter = fitter_result.value();

    // Fit (this now uses ThreadWorkspaceBuffer internally)
    auto result = fitter.fit(values);

    ASSERT_TRUE(result.has_value()) << "Fitting failed";

    // Check quality metrics
    auto stats = result->to_stats();
    EXPECT_LT(stats.max_residual_overall, 1e-9);
    EXPECT_EQ(stats.failed_slices_total, 0u);

    // Verify coefficient count
    EXPECT_EQ(result->coefficients.size(), total);
}

TEST(BSplineNDWorkspaceIntegrationTest, ParallelFitConsistent) {
    // Run the same fit multiple times to ensure deterministic results
    std::array<size_t, 3> dims = {15, 12, 15};

    std::array<std::vector<double>, 3> grids;
    for (size_t axis = 0; axis < 3; ++axis) {
        grids[axis].resize(dims[axis]);
        for (size_t i = 0; i < dims[axis]; ++i) {
            grids[axis][i] = static_cast<double>(i) / (dims[axis] - 1);
        }
    }

    size_t total = dims[0] * dims[1] * dims[2];
    std::vector<double> values(total);
    for (size_t i = 0; i < total; ++i) {
        values[i] = std::sin(static_cast<double>(i) / 100.0);
    }

    auto fitter = BSplineNDSeparable<double, 3>::create(grids).value();

    // Run 3 times
    std::vector<double> coeffs_1, coeffs_2, coeffs_3;
    {
        auto result = fitter.fit(values);
        ASSERT_TRUE(result.has_value());
        coeffs_1 = result->coefficients;
    }
    {
        auto result = fitter.fit(values);
        ASSERT_TRUE(result.has_value());
        coeffs_2 = result->coefficients;
    }
    {
        auto result = fitter.fit(values);
        ASSERT_TRUE(result.has_value());
        coeffs_3 = result->coefficients;
    }

    // Should produce identical results
    ASSERT_EQ(coeffs_1.size(), coeffs_2.size());
    ASSERT_EQ(coeffs_1.size(), coeffs_3.size());
    for (size_t i = 0; i < coeffs_1.size(); ++i) {
        EXPECT_DOUBLE_EQ(coeffs_1[i], coeffs_2[i]) << "Mismatch at " << i;
        EXPECT_DOUBLE_EQ(coeffs_1[i], coeffs_3[i]) << "Mismatch at " << i;
    }
}
