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

}  // namespace
}  // namespace mango
