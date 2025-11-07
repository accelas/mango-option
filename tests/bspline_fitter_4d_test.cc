/**
 * @file bspline_fitter_4d_test.cc
 * @brief Unit tests for 4D B-spline coefficient fitting
 *
 * Validates:
 * - Constant function fitting
 * - Separable function fitting
 * - Integration with BSpline4D_FMA evaluator
 * - Grid dimension validation
 * - Residual quality
 */

#include "src/bspline_fitter_4d.hpp"
#include "src/bspline_4d.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <numbers>

using namespace mango;

namespace {

constexpr double kTolerance = 1e-10;

/// Helper: Create linearly spaced grid
std::vector<double> linspace(double start, double end, int n) {
    std::vector<double> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = start + (end - start) * i / (n - 1);
    }
    return result;
}

/// Helper: Separable test functions
struct TestFunctions {
    static double constant(double m, double t, double v, double r) {
        (void)m; (void)t; (void)v; (void)r;
        return 5.0;
    }

    static double separable(double m, double t, double v, double r) {
        return m * m * std::exp(-t) * v * (1.0 + r);
    }

    static double polynomial(double m, double t, double v, double r) {
        return m + 2.0*t + 3.0*v + 4.0*r + m*t + v*r;
    }

    static double smooth(double m, double t, double v, double r) {
        return std::sin(m * std::numbers::pi) *
               std::exp(-t) *
               std::cos(v * 2.0 * std::numbers::pi) *
               (1.0 + 0.5 * r);
    }
};

}  // namespace

// ============================================================================
// Construction and Validation Tests
// ============================================================================

TEST(BSplineFitter4DTest, Construction) {
    auto m = linspace(0.8, 1.2, 10);
    auto t = linspace(0.1, 2.0, 8);
    auto v = linspace(0.1, 0.5, 6);
    auto r = linspace(0.0, 0.1, 5);

    EXPECT_NO_THROW({
        BSplineFitter4D fitter(m, t, v, r);
        auto [Nm, Nt, Nv, Nr] = fitter.dimensions();
        EXPECT_EQ(Nm, 10UL);
        EXPECT_EQ(Nt, 8UL);
        EXPECT_EQ(Nv, 6UL);
        EXPECT_EQ(Nr, 5UL);
    });
}

TEST(BSplineFitter4DTest, InvalidConstruction) {
    auto m_small = linspace(0.8, 1.2, 3);  // Too few points!
    auto t = linspace(0.1, 2.0, 8);
    auto v = linspace(0.1, 0.5, 6);
    auto r = linspace(0.0, 0.1, 5);

    EXPECT_THROW({
        BSplineFitter4D fitter(m_small, t, v, r);
    }, std::invalid_argument);
}

TEST(BSplineFitter4DTest, UnsortedGrid) {
    auto m = linspace(0.8, 1.2, 10);
    auto t = linspace(0.1, 2.0, 8);
    auto v = linspace(0.1, 0.5, 6);
    auto r = linspace(0.0, 0.1, 5);

    // Reverse one grid to make it unsorted
    std::reverse(t.begin(), t.end());

    EXPECT_THROW({
        BSplineFitter4D fitter(m, t, v, r);
    }, std::invalid_argument);
}

// ============================================================================
// Constant Function Fitting
// ============================================================================

TEST(BSplineFitter4DTest, ConstantFunction) {
    auto m_grid = linspace(0.8, 1.2, 8);
    auto t_grid = linspace(0.1, 2.0, 6);
    auto v_grid = linspace(0.1, 0.5, 5);
    auto r_grid = linspace(0.0, 0.1, 4);

    const int Nm = m_grid.size();
    const int Nt = t_grid.size();
    const int Nv = v_grid.size();
    const int Nr = r_grid.size();

    // Generate constant function values
    std::vector<double> values(Nm * Nt * Nv * Nr);
    for (int i = 0; i < Nm; ++i) {
        for (int j = 0; j < Nt; ++j) {
            for (int k = 0; k < Nv; ++k) {
                for (int l = 0; l < Nr; ++l) {
                    size_t idx = ((i * Nt + j) * Nv + k) * Nr + l;
                    values[idx] = TestFunctions::constant(
                        m_grid[i], t_grid[j], v_grid[k], r_grid[l]);
                }
            }
        }
    }

    // Fit coefficients
    BSplineFitter4D fitter(m_grid, t_grid, v_grid, r_grid);
    auto result = fitter.fit(values);

    ASSERT_TRUE(result.success) << "Error: " << result.error_message;
    EXPECT_EQ(result.coefficients.size(), values.size());

    // Create evaluator and test at random points
    BSpline4D_FMA spline(m_grid, t_grid, v_grid, r_grid, result.coefficients);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> m_dist(0.8, 1.2);
    std::uniform_real_distribution<double> t_dist(0.1, 2.0);
    std::uniform_real_distribution<double> v_dist(0.1, 0.5);
    std::uniform_real_distribution<double> r_dist(0.0, 0.1);

    for (int trial = 0; trial < 100; ++trial) {
        double mq = m_dist(rng);
        double tq = t_dist(rng);
        double vq = v_dist(rng);
        double rq = r_dist(rng);

        double eval_value = spline.eval(mq, tq, vq, rq);
        double expected = TestFunctions::constant(mq, tq, vq, rq);

        EXPECT_NEAR(eval_value, expected, 0.01)
            << "Constant function not reproduced at (" << mq << "," << tq << "," << vq << "," << rq << ")";
    }
}

// ============================================================================
// Separable Function Fitting
// ============================================================================

TEST(BSplineFitter4DTest, SeparableFunction) {
    auto m_grid = linspace(0.8, 1.2, 10);
    auto t_grid = linspace(0.1, 2.0, 10);
    auto v_grid = linspace(0.1, 0.5, 8);
    auto r_grid = linspace(0.0, 0.1, 6);

    const int Nm = m_grid.size();
    const int Nt = t_grid.size();
    const int Nv = v_grid.size();
    const int Nr = r_grid.size();

    // Generate separable function values
    std::vector<double> values(Nm * Nt * Nv * Nr);
    for (int i = 0; i < Nm; ++i) {
        for (int j = 0; j < Nt; ++j) {
            for (int k = 0; k < Nv; ++k) {
                for (int l = 0; l < Nr; ++l) {
                    size_t idx = ((i * Nt + j) * Nv + k) * Nr + l;
                    values[idx] = TestFunctions::separable(
                        m_grid[i], t_grid[j], v_grid[k], r_grid[l]);
                }
            }
        }
    }

    // Fit coefficients
    BSplineFitter4D fitter(m_grid, t_grid, v_grid, r_grid);
    auto result = fitter.fit(values);

    ASSERT_TRUE(result.success) << "Error: " << result.error_message;

    // Create evaluator
    BSpline4D_FMA spline(m_grid, t_grid, v_grid, r_grid, result.coefficients);

    // Test at grid points (should be well-approximated)
    int count_good = 0;
    int count_total = 0;

    for (int i = 0; i < Nm; ++i) {
        for (int j = 0; j < Nt; ++j) {
            for (int k = 0; k < Nv; ++k) {
                for (int l = 0; l < Nr; ++l) {
                    double eval_value = spline.eval(
                        m_grid[i], t_grid[j], v_grid[k], r_grid[l]);
                    double expected = TestFunctions::separable(
                        m_grid[i], t_grid[j], v_grid[k], r_grid[l]);

                    count_total++;
                    if (std::abs(eval_value - expected) < 0.1) {
                        count_good++;
                    }
                }
            }
        }
    }

    double pass_rate = static_cast<double>(count_good) / count_total;
    EXPECT_GT(pass_rate, 0.90)
        << "Only " << (pass_rate * 100) << "% of grid points well-approximated";

    // Test at off-grid points
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> m_dist(0.8, 1.2);
    std::uniform_real_distribution<double> t_dist(0.1, 2.0);
    std::uniform_real_distribution<double> v_dist(0.1, 0.5);
    std::uniform_real_distribution<double> r_dist(0.0, 0.1);

    for (int trial = 0; trial < 100; ++trial) {
        double mq = m_dist(rng);
        double tq = t_dist(rng);
        double vq = v_dist(rng);
        double rq = r_dist(rng);

        double eval_value = spline.eval(mq, tq, vq, rq);

        // Just check it's reasonable (no NaN/Inf)
        EXPECT_FALSE(std::isnan(eval_value)) << "NaN at off-grid point";
        EXPECT_FALSE(std::isinf(eval_value)) << "Inf at off-grid point";
    }
}

// ============================================================================
// Polynomial Function Fitting
// ============================================================================

TEST(BSplineFitter4DTest, PolynomialFunction) {
    auto m_grid = linspace(0.8, 1.2, 12);
    auto t_grid = linspace(0.1, 2.0, 10);
    auto v_grid = linspace(0.1, 0.5, 8);
    auto r_grid = linspace(0.0, 0.1, 6);

    const int Nm = m_grid.size();
    const int Nt = t_grid.size();
    const int Nv = v_grid.size();
    const int Nr = r_grid.size();

    // Generate polynomial function values
    std::vector<double> values(Nm * Nt * Nv * Nr);
    for (int i = 0; i < Nm; ++i) {
        for (int j = 0; j < Nt; ++j) {
            for (int k = 0; k < Nv; ++k) {
                for (int l = 0; l < Nr; ++l) {
                    size_t idx = ((i * Nt + j) * Nv + k) * Nr + l;
                    values[idx] = TestFunctions::polynomial(
                        m_grid[i], t_grid[j], v_grid[k], r_grid[l]);
                }
            }
        }
    }

    // Fit coefficients
    BSplineFitter4D fitter(m_grid, t_grid, v_grid, r_grid);
    auto result = fitter.fit(values);

    ASSERT_TRUE(result.success) << "Error: " << result.error_message;

    // Polynomial functions should be well-approximated by cubic B-splines
    EXPECT_LT(result.max_residual, 0.5)
        << "Large residual for polynomial: " << result.max_residual;
}

// ============================================================================
// Smooth Function Fitting
// ============================================================================

TEST(BSplineFitter4DTest, SmoothFunction) {
    auto m_grid = linspace(0.8, 1.2, 15);
    auto t_grid = linspace(0.1, 2.0, 12);
    auto v_grid = linspace(0.1, 0.5, 10);
    auto r_grid = linspace(0.0, 0.1, 8);

    const int Nm = m_grid.size();
    const int Nt = t_grid.size();
    const int Nv = v_grid.size();
    const int Nr = r_grid.size();

    // Generate smooth function values
    std::vector<double> values(Nm * Nt * Nv * Nr);
    for (int i = 0; i < Nm; ++i) {
        for (int j = 0; j < Nt; ++j) {
            for (int k = 0; k < Nv; ++k) {
                for (int l = 0; l < Nr; ++l) {
                    size_t idx = ((i * Nt + j) * Nv + k) * Nr + l;
                    values[idx] = TestFunctions::smooth(
                        m_grid[i], t_grid[j], v_grid[k], r_grid[l]);
                }
            }
        }
    }

    // Fit coefficients
    BSplineFitter4D fitter(m_grid, t_grid, v_grid, r_grid);
    auto result = fitter.fit(values);

    ASSERT_TRUE(result.success) << "Error: " << result.error_message;

    // Create evaluator
    BSpline4D_FMA spline(m_grid, t_grid, v_grid, r_grid, result.coefficients);

    // Test smoothness: evaluate at many off-grid points
    std::mt19937 rng(456);
    std::uniform_real_distribution<double> m_dist(0.8, 1.2);
    std::uniform_real_distribution<double> t_dist(0.1, 2.0);
    std::uniform_real_distribution<double> v_dist(0.1, 0.5);
    std::uniform_real_distribution<double> r_dist(0.0, 0.1);

    for (int trial = 0; trial < 200; ++trial) {
        double mq = m_dist(rng);
        double tq = t_dist(rng);
        double vq = v_dist(rng);
        double rq = r_dist(rng);

        double eval_value = spline.eval(mq, tq, vq, rq);

        EXPECT_FALSE(std::isnan(eval_value));
        EXPECT_FALSE(std::isinf(eval_value));

        // Smooth functions should be bounded
        EXPECT_LT(std::abs(eval_value), 10.0)
            << "Unexpectedly large value at (" << mq << "," << tq << "," << vq << "," << rq << ")";
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(BSplineFitter4DTest, WrongValueSize) {
    auto m = linspace(0.8, 1.2, 10);
    auto t = linspace(0.1, 2.0, 8);
    auto v = linspace(0.1, 0.5, 6);
    auto r = linspace(0.0, 0.1, 5);

    BSplineFitter4D fitter(m, t, v, r);

    std::vector<double> values_wrong_size(100, 1.0);  // Wrong size!

    auto result = fitter.fit(values_wrong_size);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error_message.empty());
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(BSplineFitter4DTest, EndToEndWorkflow) {
    // Simulate complete workflow: generate data -> fit -> evaluate -> verify

    // Step 1: Setup grids
    auto m_grid = linspace(0.7, 1.3, 20);
    auto t_grid = linspace(0.027, 2.0, 15);
    auto v_grid = linspace(0.10, 0.80, 12);
    auto r_grid = linspace(0.0, 0.10, 8);

    const int Nm = m_grid.size();
    const int Nt = t_grid.size();
    const int Nv = v_grid.size();
    const int Nr = r_grid.size();

    // Step 2: Generate synthetic "option prices" (use smooth test function)
    std::vector<double> option_prices(Nm * Nt * Nv * Nr);
    for (int i = 0; i < Nm; ++i) {
        for (int j = 0; j < Nt; ++j) {
            for (int k = 0; k < Nv; ++k) {
                for (int l = 0; l < Nr; ++l) {
                    size_t idx = ((i * Nt + j) * Nv + k) * Nr + l;
                    option_prices[idx] = TestFunctions::smooth(
                        m_grid[i], t_grid[j], v_grid[k], r_grid[l]);
                }
            }
        }
    }

    // Step 3: Fit B-spline coefficients
    BSplineFitter4D fitter(m_grid, t_grid, v_grid, r_grid);
    auto fit_result = fitter.fit(option_prices);

    ASSERT_TRUE(fit_result.success);
    std::cout << "Fit max residual: " << fit_result.max_residual << "\n";

    // Step 4: Create evaluator
    BSpline4D_FMA spline(m_grid, t_grid, v_grid, r_grid, fit_result.coefficients);

    // Step 5: Query at arbitrary points
    struct Query {
        double m, t, v, r;
        double expected_approx;  // Approximate expected value
    };

    std::vector<Query> queries = {
        {1.0, 0.5, 0.3, 0.05, TestFunctions::smooth(1.0, 0.5, 0.3, 0.05)},
        {0.9, 1.0, 0.2, 0.03, TestFunctions::smooth(0.9, 1.0, 0.2, 0.03)},
        {1.1, 0.25, 0.5, 0.08, TestFunctions::smooth(1.1, 0.25, 0.5, 0.08)},
    };

    for (const auto& q : queries) {
        double result = spline.eval(q.m, q.t, q.v, q.r);

        EXPECT_FALSE(std::isnan(result));
        EXPECT_FALSE(std::isinf(result));

        // Should approximate the true function reasonably well
        double error = std::abs(result - q.expected_approx);
        EXPECT_LT(error, 0.5)
            << "Large error at query point (" << q.m << "," << q.t << "," << q.v << "," << q.r << ")";
    }
}
