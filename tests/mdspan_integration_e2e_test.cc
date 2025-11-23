#include "src/math/banded_matrix_solver.hpp"
#include "src/math/bspline_nd.hpp"
#include "src/pde/core/grid.hpp"
#include <gtest/gtest.h>

namespace mango {
namespace {

TEST(MdspanIntegrationE2E, BandedMatrixWorkflow) {
    // Full workflow: construct, fill, factorize, solve
    BandedMatrix<double> A(10, 3);  // n=10, bandwidth=3

    // Fill tridiagonal system with bandwidth 3
    for (size_t i = 0; i < 10; ++i) {
        A.set_col_start(i, i > 0 ? i - 1 : 0);
    }

    for (size_t i = 0; i < 10; ++i) {
        A(i, i) = 4.0;
        if (i > 0) A(i, i-1) = -1.0;
        if (i < 9) A(i, i+1) = -1.0;
    }

    // Factorize (zero-copy)
    BandedLUWorkspace<double> workspace(10, 3);
    auto factor_result = factorize_banded(A, workspace);
    ASSERT_TRUE(factor_result.success);

    // Solve
    std::vector<double> b(10, 1.0);
    std::vector<double> x(10);
    auto solve_result = solve_banded(workspace, std::span<const double>(b), std::span(x));
    ASSERT_TRUE(solve_result.success);

    EXPECT_EQ(x.size(), 10);
}

TEST(MdspanIntegrationE2E, BSplineNDEvaluation) {
    // Create 4D B-spline and evaluate
    std::vector<double> grid{0.0, 0.5, 1.0, 1.5, 2.0};

    // Create knot vectors (clamped cubic: grid.size() + 4 = 9 knots)
    std::vector<double> knots{0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.0};

    std::vector<double> coeffs(625);  // 5^4
    std::fill(coeffs.begin(), coeffs.end(), 1.0);

    auto bspline_result = BSplineND<double, 4>::create(
        {grid, grid, grid, grid},
        {knots, knots, knots, knots},
        coeffs
    );

    ASSERT_TRUE(bspline_result.has_value()) << bspline_result.error();
    auto bspline = std::move(bspline_result.value());

    // Evaluate multiple points
    for (double x = 0.0; x <= 2.0; x += 0.5) {
        double val = bspline.eval({x, x, x, x});
        EXPECT_TRUE(std::isfinite(val));
    }
}

TEST(MdspanIntegrationE2E, GridSpacingUsage) {
    // Non-uniform grid spacing with mdspan
    std::vector<double> x{0.0, 0.1, 0.3, 0.7, 1.5, 2.0};

    GridView<double> grid_view(x);
    GridSpacing<double> spacing(grid_view);

    EXPECT_FALSE(spacing.is_uniform());

    auto dx_left = spacing.dx_left_inv();
    auto w_left = spacing.w_left();

    EXPECT_EQ(dx_left.size(), 4);  // n - 2
    EXPECT_EQ(w_left.size(), 4);

    // Values should be valid
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_GT(dx_left[i], 0.0);
        EXPECT_GE(w_left[i], 0.0);
        EXPECT_LE(w_left[i], 1.0);
    }
}

}  // namespace
}  // namespace mango
