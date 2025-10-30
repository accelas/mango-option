#include <gtest/gtest.h>
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
