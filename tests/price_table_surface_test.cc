// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/math/bspline/bspline_nd.hpp"
#include "mango/math/bspline/bspline_basis.hpp"

namespace mango {
namespace {

/// Helper: build a BSplineND<double, N> from grids and coefficients
template <size_t N>
auto make_bspline(std::array<std::vector<double>, N> grids,
                  std::vector<double> coeffs) {
    std::array<std::vector<double>, N> knots;
    for (size_t i = 0; i < N; ++i) {
        knots[i] = clamped_knots_cubic(grids[i]);
    }
    return BSplineND<double, N>::create(std::move(grids), std::move(knots), std::move(coeffs));
}

TEST(BSplineNDSurfaceTest, Build2DSurface) {
    std::array<std::vector<double>, 2> grids = {{
        {0.8, 0.9, 1.0, 1.1, 1.2},  // Need at least 4 points for cubic B-splines
        {0.1, 0.5, 1.0, 1.5},
    }};

    // 5x4 = 20 coefficients (row-major: m varies fastest)
    std::vector<double> coeffs(20);
    for (size_t i = 0; i < 20; ++i) {
        coeffs[i] = static_cast<double>(i + 1);
    }

    auto result = make_bspline<2>(grids, std::move(coeffs));
    ASSERT_TRUE(result.has_value()) << "Error code: " << static_cast<int>(result.error().code);

    auto& spline = result.value();
    EXPECT_EQ(spline.grid(0).size(), 5);
}

TEST(BSplineNDSurfaceTest, ValueInterpolation) {
    std::array<std::vector<double>, 2> grids = {{
        {0.8, 0.9, 1.0, 1.1},
        {0.1, 0.5, 1.0, 1.5},
    }};

    // Simple linear coefficients for testing
    size_t total = 4 * 4;
    std::vector<double> coeffs(total);
    for (size_t i = 0; i < total; ++i) {
        coeffs[i] = static_cast<double>(i + 1);
    }

    auto spline = make_bspline<2>(grids, std::move(coeffs)).value();

    // Query at grid point: eval() returns raw B-spline value
    // At (m=0.8, tau=0.1), the spline value is 1.0 (first coefficient)
    double val = spline.eval({0.8, 0.1});
    EXPECT_NEAR(val, 1.0, 1e-10);
}

TEST(BSplineNDSurfaceTest, RejectInvalidCoefficients) {
    std::array<std::vector<double>, 2> grids = {{
        {0.8, 0.9, 1.0, 1.1},  // 4 points
        {0.1, 0.5, 1.0, 1.5},  // 4 points
    }};

    std::vector<double> coeffs = {1.0, 2.0};  // Only 2, need 4*4=16

    auto result = make_bspline<2>(grids, std::move(coeffs));

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::InterpolationErrorCode::CoefficientSizeMismatch);
}

// REGRESSION TEST: Verify BSplineND::create returns correct type for 2D
// This test ensures the template parameter is correctly propagated
TEST(BSplineNDSurfaceTest, BuildReturnsCorrectTemplateType) {
    std::array<std::vector<double>, 2> grids = {{
        {0.8, 0.9, 1.0, 1.1},
        {0.1, 0.5, 1.0, 1.5},
    }};

    std::vector<double> coeffs(16, 1.0);

    auto result = make_bspline<2>(grids, std::move(coeffs));
    ASSERT_TRUE(result.has_value());

    // Compile-time type verification
    BSplineND<double, 2> spline = std::move(result.value());

    // Runtime verification: spline should be usable
    EXPECT_EQ(spline.grid(0).size(), 4);
}

// REGRESSION TEST: Verify 3D BSplineND also has correct template type
TEST(BSplineNDSurfaceTest, Build3DReturnsCorrectTemplateType) {
    std::array<std::vector<double>, 3> grids = {{
        {0.8, 0.9, 1.0, 1.1},
        {0.1, 0.5, 1.0, 1.5},
        {0.15, 0.20, 0.25, 0.30},
    }};

    std::vector<double> coeffs(64, 1.0);  // 4*4*4 = 64

    auto result = make_bspline<3>(grids, std::move(coeffs));
    ASSERT_TRUE(result.has_value());

    // Compile-time type verification
    BSplineND<double, 3> spline = std::move(result.value());
    EXPECT_EQ(spline.grid(0).size(), 4);
}


TEST(BSplineNDSurfaceTest, PartialClampsBeyondBounds) {
    std::array<std::vector<double>, 2> grids = {{
        {0.8, 0.9, 1.0, 1.1},  // moneyness
        {0.1, 0.5, 1.0, 1.5},  // maturity
    }};

    std::vector<double> coeffs(16, 1.0);
    auto spline = make_bspline<2>(grids, std::move(coeffs)).value();

    // Query partial at m=0.5 (below m_min=0.8) should produce same result as at m_min
    double partial_oob = spline.partial(0, {0.5, 0.5});
    double partial_boundary = spline.partial(0, {0.8, 0.5});
    EXPECT_DOUBLE_EQ(partial_oob, partial_boundary);

    // Same for second_partial
    double second_oob = spline.eval_second_partial(0, {0.5, 0.5});
    double second_boundary = spline.eval_second_partial(0, {0.8, 0.5});
    EXPECT_DOUBLE_EQ(second_oob, second_boundary);
}

} // namespace
} // namespace mango
