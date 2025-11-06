/**
 * @file thomas_cubic_spline_test.cc
 * @brief Test modern C++20 Thomas solver and cubic spline
 */

#include "src/thomas_solver.hpp"
#include "src/cubic_spline_solver.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <numbers>

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
