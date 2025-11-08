/**
 * @file thomas_cubic_spline_test.cc
 * @brief Test modern C++20 Thomas solver and cubic spline
 */

#include "src/thomas_solver.hpp"
#include "src/cubic_spline_solver.hpp"
#include "src/snapshot_interpolator.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <numbers>
#include <string_view>

using namespace mango;

// ========== Thomas Solver Tests ==========

TEST(ThomasSolverTest, SimpleSystem) {
    // Test case: Simple 3x3 system
    // [2  1  0][x0]   [1]
    // [1  2  1][x1] = [2]
    // [0  1  2][x2]   [3]
    //
    // Solution: x = [0.5, 0, 1.5]

    std::vector<double> lower = {1.0, 1.0};
    std::vector<double> diag = {2.0, 2.0, 2.0};
    std::vector<double> upper = {1.0, 1.0};
    std::vector<double> rhs = {1.0, 2.0, 3.0};
    std::vector<double> solution(3);

    ThomasWorkspace<double> workspace(3);

    auto result = solve_thomas<double>(
        std::span{lower},
        std::span{diag},
        std::span{upper},
        std::span{rhs},
        std::span{solution},
        workspace.get()
    );

    ASSERT_TRUE(result.ok()) << "Solver failed: " << result.message();

    // Check solution
    EXPECT_NEAR(solution[0], 0.5, 1e-10);
    EXPECT_NEAR(solution[1], 0.0, 1e-10);
    EXPECT_NEAR(solution[2], 1.5, 1e-10);
}

TEST(ThomasSolverTest, DiagonallyDominant) {
    // Test with diagonal dominance checking
    std::vector<double> lower = {1.0, 1.0};
    std::vector<double> diag = {5.0, 5.0, 5.0};  // Diagonally dominant
    std::vector<double> upper = {1.0, 1.0};
    std::vector<double> rhs = {6.0, 7.0, 8.0};
    std::vector<double> solution(3);

    ThomasConfig<double> config{.check_diagonal_dominance = true};
    ThomasWorkspace<double> workspace(3);

    auto result = solve_thomas<double>(
        std::span{lower},
        std::span{diag},
        std::span{upper},
        std::span{rhs},
        std::span{solution},
        workspace.get(),
        config
    );

    EXPECT_TRUE(result.ok());
}

TEST(ThomasSolverTest, SingularMatrix) {
    // Test singular matrix detection
    std::vector<double> lower = {1.0};
    std::vector<double> diag = {0.0, 2.0};  // First diagonal element is zero
    std::vector<double> upper = {1.0};
    std::vector<double> rhs = {1.0, 2.0};
    std::vector<double> solution(2);

    ThomasWorkspace<double> workspace(2);

    auto result = solve_thomas<double>(
        std::span{lower},
        std::span{diag},
        std::span{upper},
        std::span{rhs},
        std::span{solution},
        workspace.get()
    );

    EXPECT_FALSE(result.ok());
    EXPECT_FALSE(result.message().empty());
}

TEST(ThomasSolverTest, AllocWrapper) {
    // Test convenience wrapper with automatic allocation
    std::vector<double> lower = {1.0, 1.0};
    std::vector<double> diag = {3.0, 3.0, 3.0};
    std::vector<double> upper = {1.0, 1.0};
    std::vector<double> rhs = {4.0, 5.0, 6.0};
    std::vector<double> solution(3);

    auto result = solve_thomas_alloc<double>(
        std::span{lower},
        std::span{diag},
        std::span{upper},
        std::span{rhs},
        std::span{solution}
    );

    EXPECT_TRUE(result.ok());
}

TEST(ThomasSolverTest, FloatPrecision) {
    // Test with single precision (template instantiation)
    std::vector<float> lower = {1.0f, 1.0f};
    std::vector<float> diag = {2.0f, 2.0f, 2.0f};
    std::vector<float> upper = {1.0f, 1.0f};
    std::vector<float> rhs = {1.0f, 2.0f, 3.0f};
    std::vector<float> solution(3);

    ThomasWorkspace<float> workspace(3);

    auto result = solve_thomas<float>(
        std::span{lower},
        std::span{diag},
        std::span{upper},
        std::span{rhs},
        std::span{solution},
        workspace.get()
    );

    EXPECT_TRUE(result.ok());
    EXPECT_NEAR(solution[0], 0.5f, 1e-5f);
    EXPECT_NEAR(solution[1], 0.0f, 1e-5f);
    EXPECT_NEAR(solution[2], 1.5f, 1e-5f);
}

// ========== Cubic Spline Tests ==========

TEST(CubicSplineTest, LinearFunction) {
    // Spline through linear function should be exact
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> y = {0.0, 2.0, 4.0, 6.0};  // y = 2x

    CubicSpline<double> spline;
    auto error = spline.build(std::span{x}, std::span{y});

    ASSERT_FALSE(error.has_value()) << "Spline build failed: " << error.value();

    // Test interpolation
    EXPECT_NEAR(spline.eval(0.5), 1.0, 1e-10);
    EXPECT_NEAR(spline.eval(1.5), 3.0, 1e-10);
    EXPECT_NEAR(spline.eval(2.5), 5.0, 1e-10);

    // Test derivative (should be constant = 2)
    EXPECT_NEAR(spline.eval_derivative(0.5), 2.0, 1e-10);
    EXPECT_NEAR(spline.eval_derivative(1.5), 2.0, 1e-10);
    EXPECT_NEAR(spline.eval_derivative(2.5), 2.0, 1e-10);

    // Test second derivative (should be zero for linear)
    EXPECT_NEAR(spline.eval_second_derivative(0.5), 0.0, 1e-10);
    EXPECT_NEAR(spline.eval_second_derivative(1.5), 0.0, 1e-10);
}

TEST(CubicSplineTest, QuadraticFunction) {
    // Spline through quadratic: y = x²
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y(x.size());

    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = x[i] * x[i];
    }

    CubicSpline<double> spline;
    auto error = spline.build(std::span{x}, std::span{y});

    ASSERT_FALSE(error.has_value());

    // Test interpolation (natural splines don't perfectly match quadratics at interior)
    // But should be reasonably close
    EXPECT_NEAR(spline.eval(0.5), 0.25, 0.1);  // ~0.339 with natural BC
    EXPECT_NEAR(spline.eval(1.5), 2.25, 0.05); // ~2.232 with natural BC
    EXPECT_NEAR(spline.eval(2.5), 6.25, 0.05); // ~6.232 with natural BC

    // Test derivative: y' = 2x (approximate with natural BC)
    EXPECT_NEAR(spline.eval_derivative(1.0), 2.0, 0.2);
    EXPECT_NEAR(spline.eval_derivative(2.0), 4.0, 0.2);
    EXPECT_NEAR(spline.eval_derivative(3.0), 6.0, 0.2);
}

TEST(CubicSplineTest, SineFunction) {
    // Test with sine function
    const size_t n = 11;
    std::vector<double> x(n);
    std::vector<double> y(n);

    for (size_t i = 0; i < n; ++i) {
        x[i] = i * std::numbers::pi / 10.0;
        y[i] = std::sin(x[i]);
    }

    CubicSpline<double> spline;
    auto error = spline.build(std::span{x}, std::span{y});

    ASSERT_FALSE(error.has_value());

    // Test interpolation at mid-points
    for (size_t i = 0; i < n - 1; ++i) {
        double x_mid = (x[i] + x[i+1]) / 2.0;
        double y_exact = std::sin(x_mid);
        double y_interp = spline.eval(x_mid);

        // Cubic spline should be accurate to ~4 decimal places for smooth functions
        EXPECT_NEAR(y_interp, y_exact, 1e-4)
            << "At x=" << x_mid << ", expected sin(x)=" << y_exact
            << ", got " << y_interp;
    }

    // Test derivative: d/dx[sin(x)] = cos(x)
    double x_test = std::numbers::pi / 4.0;
    double deriv_exact = std::cos(x_test);
    double deriv_interp = spline.eval_derivative(x_test);

    EXPECT_NEAR(deriv_interp, deriv_exact, 1e-3);
}

TEST(CubicSplineTest, InputValidation) {
    CubicSpline<double> spline;

    // Test size mismatch
    std::vector<double> x = {0.0, 1.0, 2.0};
    std::vector<double> y = {0.0, 1.0};  // Wrong size

    auto error = spline.build(std::span{x}, std::span{y});
    EXPECT_TRUE(error.has_value());

    // Test too few points
    x = {0.0};
    y = {0.0};
    error = spline.build(std::span{x}, std::span{y});
    EXPECT_TRUE(error.has_value());

    // Test non-increasing x
    x = {0.0, 2.0, 1.0};  // Not strictly increasing
    y = {0.0, 2.0, 1.0};
    error = spline.build(std::span{x}, std::span{y});
    EXPECT_TRUE(error.has_value());
}

TEST(CubicSplineTest, BoundaryExtrapolation) {
    // Test behavior outside interpolation domain
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y = {1.0, 4.0, 9.0};  // y = x²

    CubicSpline<double> spline;
    auto error = spline.build(std::span{x}, std::span{y});

    ASSERT_FALSE(error.has_value());

    // Evaluate outside domain (should extrapolate using boundary cubic)
    double y_below = spline.eval(0.5);
    double y_above = spline.eval(3.5);

    // Just check they don't crash and return reasonable values
    EXPECT_TRUE(std::isfinite(y_below));
    EXPECT_TRUE(std::isfinite(y_above));
}

TEST(CubicSplineTest, EmptySplineHandling) {
    CubicSpline<double> spline;

    // Empty spline should return 0
    EXPECT_EQ(spline.eval(1.0), 0.0);
    EXPECT_EQ(spline.eval_derivative(1.0), 0.0);
    EXPECT_TRUE(spline.empty());
    EXPECT_EQ(spline.size(), 0);
}

TEST(CubicSplineTest, DomainQuery) {
    std::vector<double> x = {-1.0, 0.0, 1.0, 2.0};
    std::vector<double> y = {1.0, 0.0, 1.0, 4.0};

    CubicSpline<double> spline;
    auto error = spline.build(std::span{x}, std::span{y});
    ASSERT_FALSE(error.has_value());

    auto [x_min, x_max] = spline.domain();

    EXPECT_EQ(x_min, -1.0);
    EXPECT_EQ(x_max, 2.0);
    EXPECT_EQ(spline.size(), 4);
}

TEST(CubicSplineTest, RebuildSameGrid) {
    // Test rebuild_same_grid functionality
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y1 = {0.0, 1.0, 4.0, 9.0, 16.0};  // y = x²
    std::vector<double> y2 = {0.0, 2.0, 4.0, 6.0, 8.0};   // y = 2x

    CubicSpline<double> spline;

    // Build with first dataset
    auto error1 = spline.build(std::span{x}, std::span{y1});
    ASSERT_FALSE(error1.has_value());

    // Verify first dataset
    double val1 = spline.eval(1.5);
    EXPECT_GT(val1, 0.0);  // Should be some value for x²

    // Rebuild with second dataset on same grid
    auto error2 = spline.rebuild_same_grid(std::span{y2});
    ASSERT_FALSE(error2.has_value());

    // Verify second dataset (linear function)
    EXPECT_NEAR(spline.eval(0.5), 1.0, 1e-10);
    EXPECT_NEAR(spline.eval(1.5), 3.0, 1e-10);
    EXPECT_NEAR(spline.eval(2.5), 5.0, 1e-10);

    // Derivative should be ~2 for linear function
    EXPECT_NEAR(spline.eval_derivative(1.0), 2.0, 1e-10);
    EXPECT_NEAR(spline.eval_derivative(2.0), 2.0, 1e-10);
}

TEST(CubicSplineTest, RebuildSameGridValidation) {
    // Test rebuild_same_grid error handling
    CubicSpline<double> spline;
    std::vector<double> y = {1.0, 2.0, 3.0};

    // Should fail if build() hasn't been called
    auto error = spline.rebuild_same_grid(std::span{y});
    EXPECT_TRUE(error.has_value());
    EXPECT_NE(error.value().find("Must call build()"), std::string_view::npos);

    // Build first
    std::vector<double> x = {0.0, 1.0, 2.0};
    auto build_error = spline.build(std::span{x}, std::span{y});
    ASSERT_FALSE(build_error.has_value());

    // Should fail with wrong size
    std::vector<double> y_wrong = {1.0, 2.0};  // Wrong size
    auto error2 = spline.rebuild_same_grid(std::span{y_wrong});
    EXPECT_TRUE(error2.has_value());
    EXPECT_NE(error2.value().find("size must match"), std::string_view::npos);
}

// ========== Performance/Benchmark Tests ==========

TEST(CubicSplineTest, LargeDataset) {
    // Test with larger dataset (stress test)
    const size_t n = 1000;
    std::vector<double> x(n);
    std::vector<double> y(n);

    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i) / (n - 1);
        y[i] = std::exp(-x[i] * x[i]);  // Gaussian
    }

    CubicSpline<double> spline;
    auto error = spline.build(std::span{x}, std::span{y});

    ASSERT_FALSE(error.has_value());

    // Test random interpolations
    for (double t : {0.123, 0.456, 0.789}) {
        double y_interp = spline.eval(t);
        double y_exact = std::exp(-t * t);

        EXPECT_NEAR(y_interp, y_exact, 1e-6);
    }
}

TEST(CubicSplineTest, NonUniformGrid) {
    // Test with non-uniform grid spacing
    std::vector<double> x = {0.0, 0.1, 0.5, 1.0, 2.0, 5.0};
    std::vector<double> y(x.size());

    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = std::exp(-x[i]);  // Exponential decay
    }

    CubicSpline<double> spline;
    auto error = spline.build(std::span{x}, std::span{y});
    ASSERT_FALSE(error.has_value());

    // Rebuild with different data
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] = 1.0 / (1.0 + x[i]);  // Different function
    }

    error = spline.rebuild_same_grid(std::span{y});
    ASSERT_FALSE(error.has_value());

    // Verify new function is interpolated
    double val = spline.eval(0.75);
    double expected = 1.0 / (1.0 + 0.75);
    EXPECT_NEAR(val, expected, 0.1);  // Natural splines have some error
}

TEST(CubicSplineTest, RepeatedRebuilds) {
    // Test that repeated rebuilds don't degrade
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {0.0, 1.0, 4.0, 9.0, 16.0};

    CubicSpline<double> spline;
    auto error = spline.build(std::span{x}, std::span{y});
    ASSERT_FALSE(error.has_value());

    // Rebuild 100 times with different data
    for (int i = 0; i < 100; ++i) {
        for (size_t j = 0; j < y.size(); ++j) {
            y[j] = static_cast<double>(i + j);
        }
        error = spline.rebuild_same_grid(std::span{y});
        ASSERT_FALSE(error.has_value()) << "Rebuild " << i << " failed";
    }

    // Verify final rebuild produces correct values
    EXPECT_NEAR(spline.eval(0.0), y[0], 1e-10);
}

// ========== Snapshot Interpolator Tests ==========

TEST(SnapshotInterpolatorTest, DerivedSplineInvalidatedOnGridChange) {
    // CRITICAL TEST: Derived spline must be invalidated when grid changes
    SnapshotInterpolator interp;

    // Build with first grid
    std::vector<double> x1 = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y1 = {0.0, 1.0, 4.0, 9.0, 16.0};  // y = x²
    std::vector<double> deriv1 = {0.0, 2.0, 4.0, 6.0, 8.0};  // dy/dx = 2x

    auto error = interp.build(std::span{x1}, std::span{y1});
    ASSERT_FALSE(error.has_value());

    // Evaluate derivative (this caches derived spline)
    double d1 = interp.eval_from_data(1.5, std::span{deriv1});
    EXPECT_NEAR(d1, 3.0, 0.2);  // Should be ~3.0 for linear derivative

    // Change grid (same size but different values)
    std::vector<double> x2 = {0.0, 0.5, 1.0, 1.5, 2.0};  // Different spacing!
    std::vector<double> y2 = {0.0, 0.5, 1.0, 1.5, 2.0};  // y = x (linear)
    std::vector<double> deriv2 = {1.0, 1.0, 1.0, 1.0, 1.0};  // dy/dx = 1 (constant)

    error = interp.build(std::span{x2}, std::span{y2});
    ASSERT_FALSE(error.has_value());

    // Evaluate derivative on new grid
    // If derived spline was NOT invalidated, this would use old grid structure
    // and produce wrong results
    double d2 = interp.eval_from_data(0.75, std::span{deriv2});
    EXPECT_NEAR(d2, 1.0, 0.2);  // Should be ~1.0 for constant derivative

    // Value interpolation should also work correctly
    double v2 = interp.eval(0.75);
    EXPECT_NEAR(v2, 0.75, 0.1);  // Linear function
}

TEST(SnapshotInterpolatorTest, RebuildSameGridPreservesDerivedCache) {
    // Test that rebuild_same_grid correctly updates derived spline
    SnapshotInterpolator interp;

    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y1 = {0.0, 1.0, 4.0, 9.0, 16.0};
    std::vector<double> deriv1 = {0.0, 2.0, 4.0, 6.0, 8.0};

    auto error = interp.build(std::span{x}, std::span{y1});
    ASSERT_FALSE(error.has_value());

    // Evaluate derivative (caches derived spline)
    double d1 = interp.eval_from_data(1.5, std::span{deriv1});
    EXPECT_GT(std::abs(d1), 0.0);

    // Rebuild with same grid but different y-values
    std::vector<double> y2 = {0.0, 2.0, 4.0, 6.0, 8.0};  // y = 2x
    std::vector<double> deriv2 = {2.0, 2.0, 2.0, 2.0, 2.0};  // dy/dx = 2

    error = interp.rebuild_same_grid(std::span{y2});
    ASSERT_FALSE(error.has_value());

    // Evaluate derivative (should use cached grid structure)
    double d2 = interp.eval_from_data(1.5, std::span{deriv2});
    EXPECT_NEAR(d2, 2.0, 0.2);
}

TEST(SnapshotInterpolatorTest, ErrorHandlingChain) {
    SnapshotInterpolator interp;
    std::vector<double> y = {1.0, 2.0, 3.0};

    // rebuild_same_grid before build should fail
    auto error = interp.rebuild_same_grid(std::span{y});
    EXPECT_TRUE(error.has_value());
    EXPECT_NE(error.value().find("Must call build()"), std::string_view::npos);

    // Build successfully
    std::vector<double> x = {0.0, 1.0, 2.0};
    error = interp.build(std::span{x}, std::span{y});
    ASSERT_FALSE(error.has_value());

    // rebuild_same_grid with wrong size should fail and invalidate interpolator
    std::vector<double> y_wrong = {1.0, 2.0};
    error = interp.rebuild_same_grid(std::span{y_wrong});
    EXPECT_TRUE(error.has_value());
    EXPECT_FALSE(interp.is_built());  // Interpolator should be invalidated

    // After error, must rebuild from scratch (not just rebuild_same_grid)
    std::vector<double> y_correct = {4.0, 5.0, 6.0};
    error = interp.build(std::span{x}, std::span{y_correct});
    EXPECT_FALSE(error.has_value());
    EXPECT_TRUE(interp.is_built());
}

TEST(SnapshotInterpolatorTest, DerivedSplineFallbackOnError) {
    // Test that derived spline falls back to linear on build error
    SnapshotInterpolator interp;

    // Build with valid data
    std::vector<double> x = {0.0, 1.0, 2.0};
    std::vector<double> y = {0.0, 1.0, 4.0};

    auto error = interp.build(std::span{x}, std::span{y});
    ASSERT_FALSE(error.has_value());

    // eval_from_data with empty data should fallback to linear (return 0)
    std::vector<double> empty_data;
    double result = interp.eval_from_data(0.5, std::span{empty_data});
    EXPECT_EQ(result, 0.0);

    // eval_from_data with valid data should work
    std::vector<double> valid_data = {1.0, 2.0, 3.0};
    result = interp.eval_from_data(0.5, std::span{valid_data});
    EXPECT_GT(std::abs(result), 0.0);
}
