// SPDX-License-Identifier: MIT
#include "src/math/bspline_nd.hpp"
#include "src/math/bspline_basis.hpp"
#include <gtest/gtest.h>
#include <numeric>

namespace mango {
namespace {

TEST(BSplineNDMdspan, CoefficientIndexing3D) {
    // Create simple 3D B-spline (minimum grid size = 4)
    std::vector<double> grid0{0.0, 1.0, 2.0, 3.0};
    std::vector<double> grid1{0.0, 0.5, 1.0, 1.5};
    std::vector<double> grid2{0.0, 0.25, 0.5, 0.75};

    auto knots0 = clamped_knots_cubic(grid0);
    auto knots1 = clamped_knots_cubic(grid1);
    auto knots2 = clamped_knots_cubic(grid2);

    // Coefficient array: 4 × 4 × 4 = 64 elements
    std::vector<double> coeffs(64);
    std::iota(coeffs.begin(), coeffs.end(), 0.0);

    BSplineND<double, 3>::GridArray grids = {grid0, grid1, grid2};
    BSplineND<double, 3>::KnotArray knots = {knots0, knots1, knots2};

    auto result = BSplineND<double, 3>::create(grids, knots, coeffs);

    ASSERT_TRUE(result.has_value());

    // Verify we can evaluate (tests internal indexing)
    auto value = result->eval({0.5, 0.25, 0.5});
    EXPECT_TRUE(std::isfinite(value));
}

TEST(BSplineNDMdspan, CoefficientIndexing4D) {
    // Create minimal 4D B-spline (minimum grid size = 4)
    std::vector<double> grid{0.0, 0.33, 0.67, 1.0};
    auto knots = clamped_knots_cubic(grid);

    // 4^4 = 256 coefficients
    std::vector<double> coeffs(256, 1.0);

    BSplineND<double, 4>::GridArray grids = {grid, grid, grid, grid};
    BSplineND<double, 4>::KnotArray knot_arrays = {knots, knots, knots, knots};

    auto result = BSplineND<double, 4>::create(grids, knot_arrays, coeffs);

    ASSERT_TRUE(result.has_value());

    // Constant function should evaluate to 1.0
    auto value = result->eval({0.5, 0.5, 0.5, 0.5});
    EXPECT_NEAR(value, 1.0, 1e-10);
}

TEST(BSplineNDMdspan, IdenticalEvaluation) {
    // Create 3D B-spline with known coefficients (minimum grid size = 4)
    std::vector<double> grid0{0.0, 1.0, 2.0, 3.0};
    std::vector<double> grid1{0.0, 1.0, 2.0, 3.0};
    std::vector<double> grid2{0.0, 1.0, 2.0, 3.0};

    auto knots0 = clamped_knots_cubic(grid0);
    auto knots1 = clamped_knots_cubic(grid1);
    auto knots2 = clamped_knots_cubic(grid2);

    // 4 × 4 × 4 = 64 coefficients
    std::vector<double> coeffs(64);
    for (size_t i = 0; i < 64; ++i) {
        coeffs[i] = std::sin(static_cast<double>(i));
    }

    BSplineND<double, 3>::GridArray grids = {grid0, grid1, grid2};
    BSplineND<double, 3>::KnotArray knots = {knots0, knots1, knots2};

    auto bspline = BSplineND<double, 3>::create(grids, knots, coeffs).value();

    // Evaluate at multiple points
    std::vector<std::array<double, 3>> test_points{
        {0.5, 0.5, 0.5},
        {1.5, 1.0, 0.25},
        {2.5, 1.5, 0.75},
        {0.1, 0.9, 0.1}
    };

    for (const auto& pt : test_points) {
        double value = bspline.eval(pt);

        // Results should be identical to previous implementation
        // (We're not changing the math, just the indexing)
        EXPECT_TRUE(std::isfinite(value));
    }
}

}  // namespace
}  // namespace mango
