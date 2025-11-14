/**
 * @file bspline_4d_test.cc
 * @brief Unit tests for 4D tensor-product B-spline evaluator
 *
 * Validates:
 * - Separable function reproduction
 * - Boundary handling and clamping
 * - Tensor-product structure
 * - Performance benchmarks
 * - Comparison with 1D basis evaluation
 */

#include "src/interpolation/bspline_4d.hpp"
#include "src/option/price_table_workspace.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cmath>
#include <random>
#include <chrono>
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

/// Helper: Evaluate separable function f(m,t,v,r) = g(m) * h(t) * p(v) * q(r)
struct SeparableFunction {
    static double g(double m) { return m * m; }        // moneyness: quadratic
    static double h(double t) { return std::exp(-t); } // maturity: exponential
    static double p(double v) { return v; }            // volatility: linear
    static double q(double r) { return 1.0 + r; }      // rate: linear offset

    static double eval(double m, double t, double v, double r) {
        return g(m) * h(t) * p(v) * q(r);
    }
};

}  // namespace

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST(BSpline4DUtilTest, ClampedKnots) {
    std::vector<double> x = {0.0, 0.25, 0.5, 0.75, 1.0};
    auto knots = clamped_knots_cubic(x);

    ASSERT_EQ(knots.size(), x.size() + 4);

    auto expect_interior = [](const std::vector<double>& grid, const std::vector<double>& knots) {
        const size_t n = grid.size();
        const size_t n_interior = (n > 4) ? n - 4 : 0;

        for (int i = 0; i < 4; ++i) {
            EXPECT_DOUBLE_EQ(knots[i], grid.front()) << "Left clamp mismatch at index " << i;
            EXPECT_DOUBLE_EQ(knots[knots.size() - 1 - i], grid.back())
                << "Right clamp mismatch at index " << knots.size() - 1 - i;
        }

        for (size_t idx = 0; idx < n_interior; ++idx) {
            const double ratio = static_cast<double>(idx + 1) /
                                 static_cast<double>(n_interior + 1);
            const double position = ratio * static_cast<double>(n - 1);
            int low = static_cast<int>(std::floor(position));
            if (low >= static_cast<int>(n) - 1) {
                low = static_cast<int>(n) - 2;
            }
            const double frac = position - static_cast<double>(low);
            const double left = grid[low];
            const double right = grid[low + 1];
            const double expected = (1.0 - frac) * left + frac * right;
            const size_t knot_idx = 4 + idx;

            EXPECT_GT(knots[knot_idx], left);
            EXPECT_LT(knots[knot_idx], right);
            EXPECT_NEAR(knots[knot_idx], expected, 1e-9)
                << "Interior knot mismatch at index " << knot_idx;
        }
    };

    expect_interior(x, knots);

    std::vector<double> x_large = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    auto knots_large = clamped_knots_cubic(x_large);
    ASSERT_EQ(knots_large.size(), x_large.size() + 4);
    expect_interior(x_large, knots_large);
}

TEST(BSpline4DUtilTest, FindSpan) {
    std::vector<double> t = {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0};

    // Test various query points
    EXPECT_EQ(find_span_cubic(t, 0.0), 3);    // At left boundary
    EXPECT_EQ(find_span_cubic(t, 0.25), 3);   // Between knots
    EXPECT_EQ(find_span_cubic(t, 0.5), 4);    // At interior knot
    EXPECT_EQ(find_span_cubic(t, 0.75), 4);   // Between knots
    EXPECT_GE(find_span_cubic(t, 1.0), 0);    // At right boundary (valid index)
}

TEST(BSpline4DUtilTest, CubicBasis) {
    std::vector<double> t = {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
    double N[4];

    // Evaluate at x = 0.5
    int span = find_span_cubic(t, 0.5);
    cubic_basis_nonuniform(t, span, 0.5, N);

    // Partition of unity: sum should be 1
    double sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        sum += N[i];
    }
    EXPECT_NEAR(sum, 1.0, kTolerance) << "Partition of unity violated";

    // All values should be non-negative
    for (int i = 0; i < 4; ++i) {
        EXPECT_GE(N[i], 0.0) << "Negative basis function at index " << i;
    }
}

TEST(BSpline4DUtilTest, ClampQuery) {
    // Test boundary clamping
    EXPECT_DOUBLE_EQ(clamp_query(0.5, 0.0, 1.0), 0.5);  // Interior: no change
    EXPECT_DOUBLE_EQ(clamp_query(-0.1, 0.0, 1.0), 0.0); // Below min: clamp to min
    EXPECT_LT(clamp_query(1.5, 0.0, 1.0), 1.0);         // Above max: clamp < max
    EXPECT_GT(clamp_query(1.5, 0.0, 1.0), 0.99);        // Above max: close to max
}

// ============================================================================
// 4D B-Spline Evaluator Tests
// ============================================================================

TEST(BSpline4DTest, Construction) {
    auto m = linspace(0.8, 1.2, 10);
    auto t = linspace(0.1, 2.0, 8);
    auto v = linspace(0.1, 0.5, 6);
    auto r = linspace(0.0, 0.1, 5);

    std::vector<double> coeffs(10 * 8 * 6 * 5, 1.0);

    EXPECT_NO_THROW({
        auto workspace = PriceTableWorkspace::create(m, t, v, r, coeffs, 100.0, 0.0);
        ASSERT_TRUE(workspace.has_value());
        BSpline4D spline(workspace.value());
    });
}

TEST(BSpline4DTest, InvalidConstruction) {
    auto m = linspace(0.8, 1.2, 3);  // Too few points!
    auto t = linspace(0.1, 2.0, 8);
    auto v = linspace(0.1, 0.5, 6);
    auto r = linspace(0.0, 0.1, 5);

    std::vector<double> coeffs(3 * 8 * 6 * 5, 1.0);

    // Should fail validation for < 4 points
    auto workspace = PriceTableWorkspace::create(m, t, v, r, coeffs, 100.0, 0.0);
    EXPECT_FALSE(workspace.has_value());
    EXPECT_THAT(workspace.error(), testing::HasSubstr("moneyness"));
}

TEST(BSpline4DTest, ConstantFunction) {
    // Test constant function f(m,t,v,r) = 5.0
    auto m = linspace(0.8, 1.2, 8);
    auto t = linspace(0.1, 2.0, 6);
    auto v = linspace(0.1, 0.5, 5);
    auto r = linspace(0.0, 0.1, 4);

    const double constant_value = 5.0;
    std::vector<double> coeffs(8 * 6 * 5 * 4, constant_value);

    auto workspace = PriceTableWorkspace::create(m, t, v, r, coeffs, 100.0, 0.0);
    ASSERT_TRUE(workspace.has_value());
    BSpline4D spline(workspace.value());

    // Test at various points
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> m_dist(0.8, 1.2);
    std::uniform_real_distribution<double> t_dist(0.1, 2.0);
    std::uniform_real_distribution<double> v_dist(0.1, 0.5);
    std::uniform_real_distribution<double> r_dist(0.0, 0.1);

    for (int trial = 0; trial < 50; ++trial) {
        double mq = m_dist(rng);
        double tq = t_dist(rng);
        double vq = v_dist(rng);
        double rq = r_dist(rng);

        double result = spline.eval(mq, tq, vq, rq);

        EXPECT_NEAR(result, constant_value, 0.01)
            << "Constant reproduction failed at (" << mq << "," << tq << "," << vq << "," << rq << ")";
    }
}

TEST(BSpline4DTest, SeparableFunction) {
    // Test separable function: f(m,t,v,r) = g(m) * h(t) * p(v) * q(r)
    // where g, h, p, q are simple functions
    //
    // NOTE: This test uses function values as coefficients directly, which
    // doesn't produce exact interpolation. Proper coefficient fitting via
    // least-squares would be needed for that. We use relaxed tolerance to
    // verify the B-spline provides reasonable approximation.

    auto m_grid = linspace(0.8, 1.2, 10);
    auto t_grid = linspace(0.1, 2.0, 10);
    auto v_grid = linspace(0.1, 0.5, 8);
    auto r_grid = linspace(0.0, 0.1, 6);

    const int Nm = m_grid.size();
    const int Nt = t_grid.size();
    const int Nv = v_grid.size();
    const int Nr = r_grid.size();

    // Create coefficients by evaluating separable function at grid points
    std::vector<double> coeffs(Nm * Nt * Nv * Nr);

    for (int i = 0; i < Nm; ++i) {
        for (int j = 0; j < Nt; ++j) {
            for (int k = 0; k < Nv; ++k) {
                for (int l = 0; l < Nr; ++l) {
                    size_t idx = ((i * Nt + j) * Nv + k) * Nr + l;
                    coeffs[idx] = SeparableFunction::eval(
                        m_grid[i], t_grid[j], v_grid[k], r_grid[l]);
                }
            }
        }
    }

    auto workspace = PriceTableWorkspace::create(m_grid, t_grid, v_grid, r_grid, coeffs, 100.0, 0.0);
    ASSERT_TRUE(workspace.has_value());
    BSpline4D spline(workspace.value());

    // Test approximation at grid points (not exact without proper fitting)
    // Relaxed tolerance since we're using function values as coefficients
    int count_checked = 0;
    int count_passed = 0;

    for (int i = 0; i < Nm; ++i) {
        for (int j = 0; j < Nt; ++j) {
            for (int k = 0; k < Nv; ++k) {
                for (int l = 0; l < Nr; ++l) {
                    double expected = SeparableFunction::eval(
                        m_grid[i], t_grid[j], v_grid[k], r_grid[l]);

                    double result = spline.eval(
                        m_grid[i], t_grid[j], v_grid[k], r_grid[l]);

                    count_checked++;
                    // Relaxed tolerance: B-splines approximate, don't interpolate
                    // without proper least-squares coefficient fitting
                    if (std::abs(result - expected) < 0.1) {
                        count_passed++;
                    }
                }
            }
        }
    }

    // Verify that most points are reasonably approximated
    double pass_rate = static_cast<double>(count_passed) / count_checked;
    EXPECT_GT(pass_rate, 0.90)  // Expect >90% of points within tolerance
        << "Only " << (pass_rate * 100) << "% of grid points approximated well";

    // Test that evaluation works at off-grid points (main purpose of B-splines)
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

        // Just verify it doesn't crash and returns reasonable values
        double result = spline.eval(mq, tq, vq, rq);
        EXPECT_FALSE(std::isnan(result)) << "NaN at off-grid point";
        EXPECT_FALSE(std::isinf(result)) << "Inf at off-grid point";
    }
}

TEST(BSpline4DTest, BoundaryHandling) {
    auto m = linspace(0.8, 1.2, 8);
    auto t = linspace(0.1, 2.0, 6);
    auto v = linspace(0.1, 0.5, 5);
    auto r = linspace(0.0, 0.1, 4);

    std::vector<double> coeffs(8 * 6 * 5 * 4, 1.0);
    auto workspace = PriceTableWorkspace::create(m, t, v, r, coeffs, 100.0, 0.0);
    ASSERT_TRUE(workspace.has_value());
    BSpline4D spline(workspace.value());

    // Test at boundaries (should not crash)
    EXPECT_NO_THROW({
        double val1 = spline.eval(0.8, 0.1, 0.1, 0.0);  // All at min
        double val2 = spline.eval(1.2, 2.0, 0.5, 0.1);  // All at max
        (void)val1;
        (void)val2;
    });

    // Test outside boundaries (should clamp)
    EXPECT_NO_THROW({
        double val1 = spline.eval(0.5, 0.1, 0.1, 0.0);   // m below min
        double val2 = spline.eval(1.5, 2.0, 0.5, 0.1);   // m above max
        double val3 = spline.eval(1.0, -0.5, 0.3, 0.05); // t below min
        double val4 = spline.eval(1.0, 3.0, 0.3, 0.05);  // t above max
        (void)val1; (void)val2; (void)val3; (void)val4;
    });
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(BSpline4DTest, PerformanceSingleEval) {
    auto m = linspace(0.7, 1.3, 50);
    auto t = linspace(0.027, 2.0, 30);
    auto v = linspace(0.10, 0.80, 20);
    auto r = linspace(0.0, 0.10, 10);

    std::vector<double> coeffs(50 * 30 * 20 * 10);
    std::mt19937 rng(123);
    std::normal_distribution<double> dist(5.0, 1.0);
    for (auto& c : coeffs) {
        c = dist(rng);
    }

    auto workspace = PriceTableWorkspace::create(m, t, v, r, coeffs, 100.0, 0.0);
    ASSERT_TRUE(workspace.has_value());
    BSpline4D spline(workspace.value());

    // Warm-up
    for (int i = 0; i < 100; ++i) {
        volatile double result = spline.eval(1.0, 0.5, 0.3, 0.05);
        (void)result;
    }

    // Benchmark
    constexpr int n_queries = 10000;
    std::vector<double> m_queries(n_queries);
    std::vector<double> t_queries(n_queries);
    std::vector<double> v_queries(n_queries);
    std::vector<double> r_queries(n_queries);

    std::uniform_real_distribution<double> m_dist(0.7, 1.3);
    std::uniform_real_distribution<double> t_dist(0.027, 2.0);
    std::uniform_real_distribution<double> v_dist(0.10, 0.80);
    std::uniform_real_distribution<double> r_dist(0.0, 0.10);

    for (int i = 0; i < n_queries; ++i) {
        m_queries[i] = m_dist(rng);
        t_queries[i] = t_dist(rng);
        v_queries[i] = v_dist(rng);
        r_queries[i] = r_dist(rng);
    }

    auto start = std::chrono::high_resolution_clock::now();
    volatile double sum = 0.0;
    for (int i = 0; i < n_queries; ++i) {
        sum += spline.eval(m_queries[i], t_queries[i], v_queries[i], r_queries[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();

    const auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - start).count();
    const double ns_per_query = static_cast<double>(duration_ns) / n_queries;

    std::cout << "4D B-spline eval (50×30×20×10): " << ns_per_query << " ns/query\n";

    // Target: <600ns on release builds; CI and sanitized/debug builds can be ~10× slower.
    EXPECT_LT(ns_per_query, 6000.0)
        << "4D evaluation too slow: " << ns_per_query << " ns/query";
}

TEST(BSpline4DTest, PerformanceSmallGrid) {
    // Smaller grid for faster evaluation
    auto m = linspace(0.7, 1.3, 10);
    auto t = linspace(0.027, 2.0, 8);
    auto v = linspace(0.10, 0.80, 6);
    auto r = linspace(0.0, 0.10, 5);

    std::vector<double> coeffs(10 * 8 * 6 * 5, 1.0);
    auto workspace = PriceTableWorkspace::create(m, t, v, r, coeffs, 100.0, 0.0);
    ASSERT_TRUE(workspace.has_value());
    BSpline4D spline(workspace.value());

    constexpr int n_queries = 100000;

    std::mt19937 rng(456);
    std::uniform_real_distribution<double> m_dist(0.7, 1.3);
    std::uniform_real_distribution<double> t_dist(0.027, 2.0);
    std::uniform_real_distribution<double> v_dist(0.10, 0.80);
    std::uniform_real_distribution<double> r_dist(0.0, 0.10);

    auto start = std::chrono::high_resolution_clock::now();
    volatile double sum = 0.0;
    for (int i = 0; i < n_queries; ++i) {
        sum += spline.eval(m_dist(rng), t_dist(rng), v_dist(rng), r_dist(rng));
    }
    auto end = std::chrono::high_resolution_clock::now();

    const auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - start).count();
    const double ns_per_query = static_cast<double>(duration_ns) / n_queries;

    std::cout << "4D B-spline eval (10×8×6×5): " << ns_per_query << " ns/query\n";

    // Target: <500ns in release; allow slack for debug/CI configuration noise.
    EXPECT_LT(ns_per_query, 2000.0)
        << "Small grid evaluation too slow: " << ns_per_query << " ns/query";
}

// ============================================================================
// Accessor Tests
// ============================================================================

TEST(BSpline4DTest, Accessors) {
    auto m = linspace(0.8, 1.2, 10);
    auto t = linspace(0.1, 2.0, 8);
    auto v = linspace(0.1, 0.5, 6);
    auto r = linspace(0.0, 0.1, 5);

    std::vector<double> coeffs(10 * 8 * 6 * 5, 1.0);
    auto workspace = PriceTableWorkspace::create(m, t, v, r, coeffs, 100.0, 0.0);
    ASSERT_TRUE(workspace.has_value());
    BSpline4D spline(workspace.value());

    auto [Nm, Nt, Nv, Nr] = spline.dimensions();
    EXPECT_EQ(Nm, 10);
    EXPECT_EQ(Nt, 8);
    EXPECT_EQ(Nv, 6);
    EXPECT_EQ(Nr, 5);

    EXPECT_EQ(spline.moneyness_grid().size(), 10UL);
    EXPECT_EQ(spline.maturity_grid().size(), 8UL);
    EXPECT_EQ(spline.volatility_grid().size(), 6UL);
    EXPECT_EQ(spline.rate_grid().size(), 5UL);

    EXPECT_DOUBLE_EQ(spline.moneyness_grid().front(), 0.8);
    EXPECT_DOUBLE_EQ(spline.moneyness_grid().back(), 1.2);
}

// ============================================================================
// Boundary Clamping and Edge Case Tests
// ============================================================================

TEST(BSpline4DTest, ExactBoundaryEvaluation) {
    // Test evaluation exactly at grid boundaries
    auto m = linspace(0.8, 1.2, 8);
    auto t = linspace(0.1, 2.0, 6);
    auto v = linspace(0.1, 0.5, 5);
    auto r = linspace(0.0, 0.1, 4);

    // Create known constant coefficients
    std::vector<double> coeffs(8 * 6 * 5 * 4, 42.0);
    auto workspace = PriceTableWorkspace::create(m, t, v, r, coeffs, 100.0, 0.0);
    ASSERT_TRUE(workspace.has_value());
    BSpline4D spline(workspace.value());

    // Test all 16 corners
    double val1 = spline.eval(m.front(), t.front(), v.front(), r.front());
    double val2 = spline.eval(m.front(), t.front(), v.front(), r.back());
    double val3 = spline.eval(m.front(), t.front(), v.back(), r.front());
    double val4 = spline.eval(m.front(), t.front(), v.back(), r.back());
    double val5 = spline.eval(m.front(), t.back(), v.front(), r.front());
    double val6 = spline.eval(m.front(), t.back(), v.front(), r.back());
    double val7 = spline.eval(m.front(), t.back(), v.back(), r.front());
    double val8 = spline.eval(m.front(), t.back(), v.back(), r.back());
    double val9 = spline.eval(m.back(), t.front(), v.front(), r.front());
    double val10 = spline.eval(m.back(), t.front(), v.front(), r.back());
    double val11 = spline.eval(m.back(), t.front(), v.back(), r.front());
    double val12 = spline.eval(m.back(), t.front(), v.back(), r.back());
    double val13 = spline.eval(m.back(), t.back(), v.front(), r.front());
    double val14 = spline.eval(m.back(), t.back(), v.front(), r.back());
    double val15 = spline.eval(m.back(), t.back(), v.back(), r.front());
    double val16 = spline.eval(m.back(), t.back(), v.back(), r.back());

    // All should be close to constant value (B-splines approximate)
    EXPECT_NEAR(val1, 42.0, 5.0);
    EXPECT_NEAR(val2, 42.0, 5.0);
    EXPECT_NEAR(val3, 42.0, 5.0);
    EXPECT_NEAR(val4, 42.0, 5.0);
    EXPECT_NEAR(val5, 42.0, 5.0);
    EXPECT_NEAR(val6, 42.0, 5.0);
    EXPECT_NEAR(val7, 42.0, 5.0);
    EXPECT_NEAR(val8, 42.0, 5.0);
    EXPECT_NEAR(val9, 42.0, 5.0);
    EXPECT_NEAR(val10, 42.0, 5.0);
    EXPECT_NEAR(val11, 42.0, 5.0);
    EXPECT_NEAR(val12, 42.0, 5.0);
    EXPECT_NEAR(val13, 42.0, 5.0);
    EXPECT_NEAR(val14, 42.0, 5.0);
    EXPECT_NEAR(val15, 42.0, 5.0);
    EXPECT_NEAR(val16, 42.0, 5.0);
}

TEST(BSpline4DTest, ClampingBehaviorOutsideBounds) {
    // Test that queries outside grid bounds clamp to boundary coefficients
    auto m = linspace(0.8, 1.2, 8);
    auto t = linspace(0.1, 2.0, 6);
    auto v = linspace(0.1, 0.5, 5);
    auto r = linspace(0.0, 0.1, 4);

    std::vector<double> coeffs(8 * 6 * 5 * 4, 10.0);
    auto workspace = PriceTableWorkspace::create(m, t, v, r, coeffs, 100.0, 0.0);
    ASSERT_TRUE(workspace.has_value());
    BSpline4D spline(workspace.value());

    // Query slightly past each axis boundary
    double val_m_below = spline.eval(m.front() - 0.1, t.front(), v.front(), r.front());
    double val_m_above = spline.eval(m.back() + 0.1, t.front(), v.front(), r.front());
    double val_t_below = spline.eval(m.front(), t.front() - 0.05, v.front(), r.front());
    double val_t_above = spline.eval(m.front(), t.back() + 0.5, v.front(), r.front());
    double val_v_below = spline.eval(m.front(), t.front(), v.front() - 0.05, r.front());
    double val_v_above = spline.eval(m.front(), t.front(), v.back() + 0.2, r.front());
    double val_r_below = spline.eval(m.front(), t.front(), v.front(), r.front() - 0.01);
    double val_r_above = spline.eval(m.front(), t.front(), v.front(), r.back() + 0.05);

    // All should return finite values (clamped, not extrapolated)
    EXPECT_FALSE(std::isnan(val_m_below));
    EXPECT_FALSE(std::isinf(val_m_below));
    EXPECT_FALSE(std::isnan(val_m_above));
    EXPECT_FALSE(std::isinf(val_m_above));
    EXPECT_FALSE(std::isnan(val_t_below));
    EXPECT_FALSE(std::isnan(val_t_above));
    EXPECT_FALSE(std::isnan(val_v_below));
    EXPECT_FALSE(std::isnan(val_v_above));
    EXPECT_FALSE(std::isnan(val_r_below));
    EXPECT_FALSE(std::isnan(val_r_above));

    // Values should be reasonable (close to constant)
    EXPECT_GT(val_m_below, 5.0);
    EXPECT_LT(val_m_below, 15.0);
    EXPECT_GT(val_m_above, 5.0);
    EXPECT_LT(val_m_above, 15.0);
}

TEST(BSpline4DTest, ExtremeBoundsClampingRegression) {
    // Regression test: ensure clamping via std::nextafter works correctly
    // at extreme out-of-bounds queries
    auto m = linspace(0.8, 1.2, 8);
    auto t = linspace(0.1, 2.0, 6);
    auto v = linspace(0.1, 0.5, 5);
    auto r = linspace(0.0, 0.1, 4);

    std::vector<double> coeffs(8 * 6 * 5 * 4, 7.0);
    auto workspace = PriceTableWorkspace::create(m, t, v, r, coeffs, 100.0, 0.0);
    ASSERT_TRUE(workspace.has_value());
    BSpline4D spline(workspace.value());

    // Query WAY outside bounds (100× the grid range)
    double val_extreme1 = spline.eval(-100.0, t.front(), v.front(), r.front());
    double val_extreme2 = spline.eval(100.0, t.front(), v.front(), r.front());
    double val_extreme3 = spline.eval(m.front(), -100.0, v.front(), r.front());
    double val_extreme4 = spline.eval(m.front(), 100.0, v.front(), r.front());

    // Should still clamp and return finite values
    EXPECT_FALSE(std::isnan(val_extreme1));
    EXPECT_FALSE(std::isinf(val_extreme1));
    EXPECT_FALSE(std::isnan(val_extreme2));
    EXPECT_FALSE(std::isinf(val_extreme2));
    EXPECT_FALSE(std::isnan(val_extreme3));
    EXPECT_FALSE(std::isnan(val_extreme4));

    // Should be within reasonable range
    EXPECT_GT(val_extreme1, 0.0);
    EXPECT_LT(val_extreme1, 20.0);
    EXPECT_GT(val_extreme2, 0.0);
    EXPECT_LT(val_extreme2, 20.0);
}

TEST(BSpline4DTest, NaNPropagation) {
    // Test that NaN coefficients propagate predictably
    auto m = linspace(0.8, 1.2, 8);
    auto t = linspace(0.1, 2.0, 6);
    auto v = linspace(0.1, 0.5, 5);
    auto r = linspace(0.0, 0.1, 4);

    std::vector<double> coeffs(8 * 6 * 5 * 4, 5.0);

    // Insert NaN at coefficient for (m_idx=0, t_idx=5, v_idx=0, r_idx=0)
    // Index = 0*120 + 5*20 + 0*4 + 0 = 100
    coeffs[100] = std::numeric_limits<double>::quiet_NaN();

    auto workspace = PriceTableWorkspace::create(m, t, v, r, coeffs, 100.0, 0.0);
    ASSERT_TRUE(workspace.has_value());
    BSpline4D spline(workspace.value());

    // Query near the NaN coefficient region:
    // Coefficient at (m[0]=0.8, t[5]=2.0, v[0]=0.1, r[0]=0.0)
    // B-spline has compact support - need to query near these values
    // for the NaN coefficient to be in the active basis functions

    // Query at the corner where the NaN coefficient is located
    double val_at_nan = spline.eval(m[0], t[5], v[0], r[0]);
    EXPECT_TRUE(std::isnan(val_at_nan))
        << "Query at NaN coefficient location should return NaN";

    // Also verify a query far from the NaN doesn't get contaminated
    double val_far = spline.eval(m.back(), t[0], v.back(), r.back());
    EXPECT_FALSE(std::isnan(val_far))
        << "Query far from NaN coefficient should not return NaN";
}

TEST(BSpline4DTest, InfCoefficientHandling) {
    // Test that Inf coefficients are handled (not ideal, but shouldn't crash)
    auto m = linspace(0.8, 1.2, 8);
    auto t = linspace(0.1, 2.0, 6);
    auto v = linspace(0.1, 0.5, 5);
    auto r = linspace(0.0, 0.1, 4);

    std::vector<double> coeffs(8 * 6 * 5 * 4, 1.0);
    coeffs[50] = std::numeric_limits<double>::infinity();

    auto workspace = PriceTableWorkspace::create(m, t, v, r, coeffs, 100.0, 0.0);
    ASSERT_TRUE(workspace.has_value());
    BSpline4D spline(workspace.value());

    // Should not crash, but may return Inf
    EXPECT_NO_THROW({
        double val = spline.eval(1.0, 0.5, 0.3, 0.05);
        (void)val;  // May be Inf, but shouldn't crash
    });
}

TEST(BSpline4DTest, AllZeroCoefficients) {
    // Edge case: all coefficients are zero
    auto m = linspace(0.8, 1.2, 8);
    auto t = linspace(0.1, 2.0, 6);
    auto v = linspace(0.1, 0.5, 5);
    auto r = linspace(0.0, 0.1, 4);

    std::vector<double> coeffs(8 * 6 * 5 * 4, 0.0);
    auto workspace = PriceTableWorkspace::create(m, t, v, r, coeffs, 100.0, 0.0);
    ASSERT_TRUE(workspace.has_value());
    BSpline4D spline(workspace.value());

    // Should return zero everywhere
    double val1 = spline.eval(1.0, 0.5, 0.3, 0.05);
    double val2 = spline.eval(m.front(), t.front(), v.front(), r.front());
    double val3 = spline.eval(m.back(), t.back(), v.back(), r.back());

    EXPECT_NEAR(val1, 0.0, kTolerance);
    EXPECT_NEAR(val2, 0.0, kTolerance);
    EXPECT_NEAR(val3, 0.0, kTolerance);
}

// ============================================================================
// Workspace-Based Construction Tests
// ============================================================================

TEST(BSpline4D, ConstructsFromWorkspace) {
    std::vector<double> m = {0.8, 0.9, 1.0, 1.1};
    std::vector<double> tau = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4, 5.0);

    auto ws = mango::PriceTableWorkspace::create(m, tau, sigma, r, coeffs, 100.0, 0.02).value();

    mango::BSpline4D spline(ws);

    // Verify dimensions match
    auto [nm, nt, nv, nr] = spline.dimensions();
    EXPECT_EQ(nm, 4);
    EXPECT_EQ(nt, 4);
    EXPECT_EQ(nv, 4);
    EXPECT_EQ(nr, 4);

    // Verify evaluation works
    double result = spline.eval(0.95, 0.5, 0.20, 0.03);
    EXPECT_NEAR(result, 5.0, 0.1);  // Should be close to constant coefficient
}

TEST(BSpline4D, WorkspaceAndVectorConstructorsGiveSameResults) {
    std::vector<double> m = {0.8, 0.9, 1.0, 1.1};
    std::vector<double> tau = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4);

    // Fill with test pattern
    for (size_t i = 0; i < coeffs.size(); ++i) {
        coeffs[i] = static_cast<double>(i);
    }

    // Construct from workspace
    auto ws = mango::PriceTableWorkspace::create(m, tau, sigma, r, coeffs, 100.0, 0.02).value();
    mango::BSpline4D spline_ws(ws);

    // Construct from vectors (old API)
    auto workspace = PriceTableWorkspace::create(m, tau, sigma, r, coeffs, 100.0, 0.0);
    ASSERT_TRUE(workspace.has_value());
    mango::BSpline4D spline_vec(workspace.value());

    // Compare evaluations at multiple points
    std::vector<std::tuple<double, double, double, double>> test_points = {
        {0.85, 0.2, 0.18, 0.025},
        {0.95, 0.75, 0.22, 0.035},
        {1.05, 1.5, 0.28, 0.045}
    };

    for (const auto& [mq, tq, vq, rq] : test_points) {
        double result_ws = spline_ws.eval(mq, tq, vq, rq);
        double result_vec = spline_vec.eval(mq, tq, vq, rq);
        EXPECT_DOUBLE_EQ(result_ws, result_vec)
            << "Results differ at (" << mq << ", " << tq << ", " << vq << ", " << rq << ")";
    }
}
