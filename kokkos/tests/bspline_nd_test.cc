/**
 * @file bspline_nd_test.cc
 * @brief Tests for N-dimensional B-spline interpolation with Kokkos
 */

#include "kokkos/src/math/bspline_nd.hpp"
#include "kokkos/src/math/bspline_basis.hpp"
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>

namespace {

// Global Kokkos initialization
struct KokkosEnvironment : public ::testing::Environment {
    void SetUp() override {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
        }
    }
    void TearDown() override {
        if (Kokkos::is_initialized()) {
            Kokkos::finalize();
        }
    }
};

::testing::Environment* const kokkos_env =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

using MemSpace = Kokkos::HostSpace;
using view_type = Kokkos::View<double*, MemSpace>;

/// Helper to create uniform grid View
view_type create_uniform_grid(double xmin, double xmax, size_t n) {
    view_type grid("grid", n);
    auto h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, grid);
    for (size_t i = 0; i < n; ++i) {
        h(i) = xmin + (xmax - xmin) * static_cast<double>(i) / static_cast<double>(n - 1);
    }
    Kokkos::deep_copy(grid, h);
    return grid;
}

}  // anonymous namespace

class BSplineNDTest : public ::testing::Test {
protected:
    /// Helper to create clamped knot vector View
    view_type create_clamped_knots(view_type grid) {
        const size_t n = grid.extent(0);
        view_type knots("knots", n + 4);
        mango::kokkos::create_clamped_knots_cubic(grid, knots);
        return knots;
    }
};

/// Test 1D B-spline (degenerate case)
TEST_F(BSplineNDTest, OneDimensional) {
    // Create 1D grid
    auto grid = create_uniform_grid(0.0, 1.0, 10);
    auto knots = create_clamped_knots(grid);

    // Constant coefficients
    view_type coeffs("coeffs", 10);
    Kokkos::deep_copy(coeffs, 1.5);

    // Create 1D B-spline
    auto spline = mango::kokkos::BSplineND<MemSpace, 1>::create({grid}, {knots}, coeffs);
    ASSERT_TRUE(spline.has_value()) << "Failed to create 1D B-spline";

    // Test evaluation
    double val = spline->eval({0.5});
    EXPECT_TRUE(std::isfinite(val)) << "Evaluation should return finite value";

    // For constant coefficients, should approximate the constant
    EXPECT_NEAR(val, 1.5, 0.5) << "Should approximate constant value";
}

/// Test 2D B-spline
TEST_F(BSplineNDTest, TwoDimensional) {
    // Create 2D grids
    auto grid_x = create_uniform_grid(0.0, 1.0, 5);
    auto grid_y = create_uniform_grid(0.0, 1.0, 4);

    auto knots_x = create_clamped_knots(grid_x);
    auto knots_y = create_clamped_knots(grid_y);

    // Constant coefficients
    const size_t total_size = 5 * 4;
    view_type coeffs("coeffs", total_size);
    Kokkos::deep_copy(coeffs, 2.0);

    // Create 2D B-spline
    auto spline = mango::kokkos::BSplineND<MemSpace, 2>::create(
        {grid_x, grid_y},
        {knots_x, knots_y},
        coeffs
    );
    ASSERT_TRUE(spline.has_value()) << "Failed to create 2D B-spline";

    // Test dimensions
    auto dims = spline->dimensions();
    EXPECT_EQ(dims[0], 5);
    EXPECT_EQ(dims[1], 4);

    // Test evaluation
    double val = spline->eval({0.5, 0.5});
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_NEAR(val, 2.0, 0.5) << "Should approximate constant value";
}

/// Test 3D B-spline
TEST_F(BSplineNDTest, ThreeDimensional) {
    auto grid_x = create_uniform_grid(0.0, 1.0, 6);
    auto grid_y = create_uniform_grid(0.0, 1.0, 5);
    auto grid_z = create_uniform_grid(0.0, 1.0, 4);

    auto knots_x = create_clamped_knots(grid_x);
    auto knots_y = create_clamped_knots(grid_y);
    auto knots_z = create_clamped_knots(grid_z);

    const size_t total_size = 6 * 5 * 4;
    view_type coeffs("coeffs", total_size);
    Kokkos::deep_copy(coeffs, 3.0);

    auto spline = mango::kokkos::BSplineND<MemSpace, 3>::create(
        {grid_x, grid_y, grid_z},
        {knots_x, knots_y, knots_z},
        coeffs
    );
    ASSERT_TRUE(spline.has_value()) << "Failed to create 3D B-spline";

    auto dims = spline->dimensions();
    EXPECT_EQ(dims[0], 6);
    EXPECT_EQ(dims[1], 5);
    EXPECT_EQ(dims[2], 4);

    double val = spline->eval({0.5, 0.5, 0.5});
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_NEAR(val, 3.0, 0.5);
}

/// Test 4D B-spline (typical option pricing case)
TEST_F(BSplineNDTest, FourDimensional) {
    // Create 4D grids (option pricing dimensions)
    auto grid_m = create_uniform_grid(0.8, 1.2, 5);  // moneyness
    auto grid_t = create_uniform_grid(0.1, 2.0, 4);  // maturity
    auto grid_v = create_uniform_grid(0.1, 0.5, 4);  // volatility
    auto grid_r = create_uniform_grid(0.0, 0.1, 4);  // rate

    auto knots_m = create_clamped_knots(grid_m);
    auto knots_t = create_clamped_knots(grid_t);
    auto knots_v = create_clamped_knots(grid_v);
    auto knots_r = create_clamped_knots(grid_r);

    // Create test function: f(m,t,v,r) = m + t + v + r
    const size_t total_size = 5 * 4 * 4 * 4;
    view_type coeffs("coeffs", total_size);

    auto m_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid_m);
    auto t_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid_t);
    auto v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid_v);
    auto r_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid_r);
    auto coeffs_h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, coeffs);

    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                for (size_t l = 0; l < 4; ++l) {
                    const size_t idx = ((i * 4 + j) * 4 + k) * 4 + l;
                    coeffs_h(idx) = m_h(i) + t_h(j) + v_h(k) + r_h(l);
                }
            }
        }
    }
    Kokkos::deep_copy(coeffs, coeffs_h);

    // Create 4D B-spline
    auto spline = mango::kokkos::BSplineND<MemSpace, 4>::create(
        {grid_m, grid_t, grid_v, grid_r},
        {knots_m, knots_t, knots_v, knots_r},
        coeffs
    );
    ASSERT_TRUE(spline.has_value()) << "Failed to create 4D B-spline";

    // Test dimensions
    auto dims = spline->dimensions();
    EXPECT_EQ(dims[0], 5);
    EXPECT_EQ(dims[1], 4);
    EXPECT_EQ(dims[2], 4);
    EXPECT_EQ(dims[3], 4);

    // Test interpolation at corner (exact)
    double val = spline->eval({m_h(0), t_h(0), v_h(0), r_h(0)});
    double expected = m_h(0) + t_h(0) + v_h(0) + r_h(0);
    EXPECT_NEAR(val, expected, 1e-10) << "Interpolation error at corner";

    // Test evaluation at interior point
    val = spline->eval({1.0, 0.5, 0.2, 0.05});
    expected = 1.0 + 0.5 + 0.2 + 0.05;
    EXPECT_NEAR(val, expected, 0.1) << "Approximation error at interior point";
}

/// Test validation errors - insufficient grid points
TEST_F(BSplineNDTest, ValidationInsufficientPoints) {
    auto grid_x = create_uniform_grid(0.0, 1.0, 3);  // Too small (< 4)
    auto grid_y = create_uniform_grid(0.0, 1.0, 4);
    auto knots_x = create_clamped_knots(grid_x);
    auto knots_y = create_clamped_knots(grid_y);

    view_type coeffs("coeffs", 3 * 4);
    Kokkos::deep_copy(coeffs, 0.0);

    auto result = mango::kokkos::BSplineND<MemSpace, 2>::create(
        {grid_x, grid_y},
        {knots_x, knots_y},
        coeffs
    );
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), mango::kokkos::BSplineNDError::InsufficientGridPoints);
}

/// Test validation errors - coefficient size mismatch
TEST_F(BSplineNDTest, ValidationCoefficientMismatch) {
    auto grid_x = create_uniform_grid(0.0, 1.0, 4);
    auto grid_y = create_uniform_grid(0.0, 1.0, 4);
    auto knots_x = create_clamped_knots(grid_x);
    auto knots_y = create_clamped_knots(grid_y);

    view_type coeffs("coeffs", 10);  // Wrong size (should be 4*4 = 16)
    Kokkos::deep_copy(coeffs, 0.0);

    auto result = mango::kokkos::BSplineND<MemSpace, 2>::create(
        {grid_x, grid_y},
        {knots_x, knots_y},
        coeffs
    );
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), mango::kokkos::BSplineNDError::CoefficientSizeMismatch);
}

/// Test boundary clamping
TEST_F(BSplineNDTest, BoundaryClamping) {
    auto grid = create_uniform_grid(0.0, 1.0, 5);
    auto knots = create_clamped_knots(grid);
    view_type coeffs("coeffs", 5);
    Kokkos::deep_copy(coeffs, 1.0);

    auto spline = mango::kokkos::BSplineND<MemSpace, 1>::create({grid}, {knots}, coeffs);
    ASSERT_TRUE(spline.has_value());

    // Query outside bounds should be clamped
    double val_below = spline->eval({-1.0});
    double val_above = spline->eval({2.0});

    EXPECT_TRUE(std::isfinite(val_below));
    EXPECT_TRUE(std::isfinite(val_above));
}

/// Test device evaluator for batched queries
TEST_F(BSplineNDTest, DeviceEvaluator) {
    // Create 2D B-spline
    auto grid_x = create_uniform_grid(0.0, 1.0, 5);
    auto grid_y = create_uniform_grid(0.0, 1.0, 5);
    auto knots_x = create_clamped_knots(grid_x);
    auto knots_y = create_clamped_knots(grid_y);

    view_type coeffs("coeffs", 25);
    Kokkos::deep_copy(coeffs, 2.5);

    auto spline = mango::kokkos::BSplineND<MemSpace, 2>::create(
        {grid_x, grid_y},
        {knots_x, knots_y},
        coeffs
    );
    ASSERT_TRUE(spline.has_value());

    // Create evaluator for parallel use
    auto evaluator = mango::kokkos::make_evaluator(*spline);

    // Test batched evaluation
    const size_t n_queries = 10;
    view_type results("results", n_queries);

    Kokkos::parallel_for("batch_eval", n_queries,
        KOKKOS_LAMBDA(const size_t i) {
            double query[2] = {0.5, 0.5};
            results(i) = evaluator(query);
        });
    Kokkos::fence();

    // Verify all results
    auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    for (size_t i = 0; i < n_queries; ++i) {
        EXPECT_NEAR(results_h(i), 2.5, 0.5) << "Batch evaluation result " << i;
    }
}
