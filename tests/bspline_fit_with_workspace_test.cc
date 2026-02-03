// SPDX-License-Identifier: MIT
// tests/bspline_fit_with_workspace_test.cc
#include "mango/math/bspline_collocation.hpp"
#include "mango/math/bspline_collocation_workspace.hpp"
#include "mango/support/thread_workspace.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace mango;

class BSplineFitWithWorkspaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create uniform grid
        const size_t n = 50;
        grid.resize(n);
        values.resize(n);
        for (size_t i = 0; i < n; ++i) {
            double t = static_cast<double>(i) / (n - 1);
            grid[i] = t;
            values[i] = std::sin(2 * M_PI * t);  // Sine wave
        }

        auto solver_result = BSplineCollocation1D<double>::create(grid);
        ASSERT_TRUE(solver_result.has_value());
        solver = std::make_unique<BSplineCollocation1D<double>>(
            std::move(solver_result.value()));
    }

    std::vector<double> grid;
    std::vector<double> values;
    std::unique_ptr<BSplineCollocation1D<double>> solver;
};

TEST_F(BSplineFitWithWorkspaceTest, ProducesSameResultAsFit) {
    // Fit with regular method
    auto result_regular = solver->fit(values);
    ASSERT_TRUE(result_regular.has_value());

    // Fit with workspace method
    ThreadWorkspaceBuffer buffer(
        BSplineCollocationWorkspace<double>::required_bytes(grid.size()));
    auto ws = BSplineCollocationWorkspace<double>::from_bytes(
        buffer.bytes(), grid.size()).value();

    auto result_ws = solver->fit_with_workspace(
        std::span<const double>(values), ws);
    ASSERT_TRUE(result_ws.has_value());

    // Results should match
    EXPECT_NEAR(result_regular->max_residual, result_ws->max_residual, 1e-14);
    EXPECT_NEAR(result_regular->condition_estimate, result_ws->condition_estimate, 1e-10);

    // Coefficients should match (workspace stores in ws.coeffs())
    ASSERT_EQ(result_regular->coefficients.size(), ws.coeffs().size());
    for (size_t i = 0; i < result_regular->coefficients.size(); ++i) {
        EXPECT_NEAR(result_regular->coefficients[i], ws.coeffs()[i], 1e-14)
            << "Coefficient mismatch at index " << i;
    }
}

TEST_F(BSplineFitWithWorkspaceTest, WorkspaceReusable) {
    ThreadWorkspaceBuffer buffer(
        BSplineCollocationWorkspace<double>::required_bytes(grid.size()));
    auto ws = BSplineCollocationWorkspace<double>::from_bytes(
        buffer.bytes(), grid.size()).value();

    // Fit multiple times with same workspace
    for (int iter = 0; iter < 3; ++iter) {
        // Modify values each iteration
        for (size_t i = 0; i < values.size(); ++i) {
            double t = static_cast<double>(i) / (values.size() - 1);
            values[i] = std::sin(2 * M_PI * t * (iter + 1));
        }

        auto result = solver->fit_with_workspace(
            std::span<const double>(values), ws);

        ASSERT_TRUE(result.has_value()) << "Fit failed on iteration " << iter;
        EXPECT_LT(result->max_residual, 1e-9);
    }
}

TEST_F(BSplineFitWithWorkspaceTest, ValueSizeMismatch) {
    ThreadWorkspaceBuffer buffer(
        BSplineCollocationWorkspace<double>::required_bytes(grid.size()));
    auto ws = BSplineCollocationWorkspace<double>::from_bytes(
        buffer.bytes(), grid.size()).value();

    // Wrong size values
    std::vector<double> wrong_size(grid.size() + 10, 1.0);

    auto result = solver->fit_with_workspace(
        std::span<const double>(wrong_size), ws);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, InterpolationErrorCode::ValueSizeMismatch);
}
