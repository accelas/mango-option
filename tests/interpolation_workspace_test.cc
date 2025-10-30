#include <gtest/gtest.h>
#include <cmath>
extern "C" {
#include "../src/interp_cubic.h"
#include "../src/iv_surface.h"
}

TEST(InterpolationWorkspace, CalculateRequiredSize2D) {
    // Test workspace size calculation for 2D surface
    size_t n_m = 50, n_tau = 30;
    size_t required = cubic_interp_workspace_size_2d(n_m, n_tau);

    // Expected: 4*max(50,30) + 6*max(50,30) + 30 + 50 = 4*50 + 6*50 + 30 + 50 = 580
    EXPECT_EQ(required, 580);
}

TEST(InterpolationWorkspace, CalculateRequiredSize4D) {
    // Test workspace size calculation for 4D table
    size_t n_m = 50, n_tau = 30, n_sigma = 20, n_r = 10;
    size_t required = cubic_interp_workspace_size_4d(n_m, n_tau, n_sigma, n_r);

    // Expected: spline workspace + all intermediate arrays + slice buffers
    size_t max_grid = 50; // max(50, 30, 20, 10)
    size_t spline_ws = 10 * max_grid; // 4n + 6n
    size_t intermediate = (30*20*10) + (20*10) + 10; // intermediate1, intermediate2, intermediate3
    size_t slices = max_grid; // moneyness_slice
    EXPECT_EQ(required, spline_ws + intermediate + slices);
}

TEST(InterpolationWorkspace, CalculateRequiredSize5D) {
    // Test workspace size calculation for 5D table
    size_t n_m = 50, n_tau = 30, n_sigma = 20, n_r = 10, n_q = 5;
    size_t required = cubic_interp_workspace_size_5d(n_m, n_tau, n_sigma, n_r, n_q);

    // Expected: spline workspace + all intermediate arrays + slice buffers
    size_t max_grid = 50; // max(50, 30, 20, 10, 5)
    size_t spline_ws = 10 * max_grid; // 4n + 6n
    size_t intermediate = (30*20*10*5) + (20*10*5) + (10*5) + 5; // All 4 intermediate arrays
    size_t slices = max_grid;
    EXPECT_EQ(required, spline_ws + intermediate + slices);
}

TEST(InterpolationWorkspace, InitializeWorkspace2D) {
    size_t n_m = 50, n_tau = 30;
    size_t ws_size = cubic_interp_workspace_size_2d(n_m, n_tau);

    double *buffer = new double[ws_size];
    CubicInterpWorkspace workspace;
    int ret = cubic_interp_workspace_init(&workspace, buffer, n_m, n_tau, 0, 0, 0);

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(workspace.max_grid_size, 50);
    EXPECT_EQ(workspace.total_size, ws_size);
    EXPECT_EQ(workspace.spline_coeff_workspace, buffer);
    EXPECT_EQ(workspace.spline_temp_workspace, buffer + 4*50);

    delete[] buffer;
}

TEST(InterpolationWorkspace, InitializeWorkspaceNullBuffer) {
    CubicInterpWorkspace workspace;
    int ret = cubic_interp_workspace_init(&workspace, NULL, 5, 3, 0, 0, 0);
    EXPECT_EQ(ret, -1);
}

TEST(InterpolationWorkspace, InitializeWorkspaceNullWorkspace) {
    double buffer[100];
    int ret = cubic_interp_workspace_init(NULL, buffer, 5, 3, 0, 0, 0);
    EXPECT_EQ(ret, -1);
}

TEST(InterpolationWorkspace, Interpolate2DWithWorkspace) {
    // Create a simple 2D IV surface
    double moneyness[] = {0.8, 0.9, 1.0, 1.1, 1.2};
    double maturity[] = {0.25, 0.5, 1.0};
    size_t n_m = 5, n_tau = 3;

    IVSurface *surface = iv_surface_create(moneyness, n_m, maturity, n_tau);
    ASSERT_NE(surface, nullptr);

    // Set some IV values (simple linear function for testing)
    for (size_t i = 0; i < n_m; i++) {
        for (size_t j = 0; j < n_tau; j++) {
            double iv = 0.2 + 0.1 * moneyness[i] + 0.05 * maturity[j];
            iv_surface_set_point(surface, i, j, iv);
        }
    }

    // Allocate workspace
    size_t ws_size = cubic_interp_workspace_size_2d(n_m, n_tau);
    double *buffer = new double[ws_size];
    CubicInterpWorkspace workspace;
    int ret = cubic_interp_workspace_init(&workspace, buffer, n_m, n_tau, 0, 0, 0);
    ASSERT_EQ(ret, 0);

    // Query with workspace (should produce same result as malloc version)
    double m_query = 0.95, tau_query = 0.75;
    double result_ws = cubic_interpolate_2d_workspace(surface, m_query, tau_query, workspace);
    double result_malloc = iv_surface_interpolate(surface, m_query, tau_query);

    // Results should match within floating point precision
    EXPECT_NEAR(result_ws, result_malloc, 1e-10);

    delete[] buffer;
    iv_surface_destroy(surface);
}

TEST(InterpolationWorkspace, Interpolate2DWorkspaceNullSurface) {
    double buffer[100];
    CubicInterpWorkspace workspace;
    cubic_interp_workspace_init(&workspace, buffer, 5, 3, 0, 0, 0);

    double result = cubic_interpolate_2d_workspace(NULL, 1.0, 0.5, workspace);
    EXPECT_TRUE(std::isnan(result));
}

TEST(InterpolationWorkspace, Interpolate2DWorkspaceBoundary) {
    // Test interpolation at grid boundaries
    double moneyness[] = {0.8, 0.9, 1.0, 1.1, 1.2};
    double maturity[] = {0.25, 0.5, 1.0};
    IVSurface *surface = iv_surface_create(moneyness, 5, maturity, 3);

    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 3; j++) {
            iv_surface_set_point(surface, i, j, 0.2);
        }
    }

    size_t ws_size = cubic_interp_workspace_size_2d(5, 3);
    double *buffer = new double[ws_size];
    CubicInterpWorkspace workspace;
    cubic_interp_workspace_init(&workspace, buffer, 5, 3, 0, 0, 0);

    // At grid point should return exact value
    double result = cubic_interpolate_2d_workspace(surface, 1.0, 0.5, workspace);
    EXPECT_NEAR(result, 0.2, 1e-10);

    delete[] buffer;
    iv_surface_destroy(surface);
}
