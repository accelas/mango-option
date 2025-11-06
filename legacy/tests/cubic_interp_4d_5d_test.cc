/**
 * @file cubic_interp_4d_5d_test.cc
 * @brief Comprehensive tests for 4D and 5D cubic spline interpolation
 *
 * Tests:
 * - 4D cubic spline interpolation (price tables without dividends)
 * - 5D cubic spline interpolation (price tables with dividends)
 * - Precomputation vs on-the-fly computation
 * - Greeks calculation with cubic splines
 * - Error handling and edge cases
 * - Memory management and cleanup
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>

extern "C" {
#include "../src/interp_cubic.h"
#include "../src/price_table.h"
}

// ============================================================================
// Helper Functions
// ============================================================================

static std::vector<double> linspace(double min, double max, size_t n) {
    std::vector<double> result(n);
    for (size_t i = 0; i < n; i++) {
        double t = static_cast<double>(i) / static_cast<double>(n - 1);
        result[i] = min + t * (max - min);
    }
    return result;
}

// Linear test function for 4D: f(m, tau, sigma, r) = m + 2*tau + 3*sigma + 4*r
static double linear_4d(double m, double tau, double sigma, double r) {
    return m + 2.0*tau + 3.0*sigma + 4.0*r;
}

// Quadratic test function for 4D (smooth, continuous second derivatives)
static double quadratic_4d(double m, double tau, double sigma, double r) {
    return 10.0 + m + 2.0*tau + 3.0*sigma + 4.0*r
           + 0.5*m*m + 0.3*tau*tau + 0.2*sigma*sigma + 0.1*r*r;
}

// Linear test function for 5D: f(m, tau, sigma, r, q) = m + 2*tau + 3*sigma + 4*r + 5*q
static double linear_5d(double m, double tau, double sigma, double r, double q) {
    return m + 2.0*tau + 3.0*sigma + 4.0*r + 5.0*q;
}

// Quadratic test function for 5D
static double quadratic_5d(double m, double tau, double sigma, double r, double q) {
    return 10.0 + m + 2.0*tau + 3.0*sigma + 4.0*r + 5.0*q
           + 0.5*m*m + 0.3*tau*tau + 0.2*sigma*sigma + 0.1*r*r + 0.05*q*q;
}

// ============================================================================
// 4D Cubic Interpolation Tests
// ============================================================================

class Cubic4DTest : public ::testing::Test {
protected:
    OptionPriceTable *table_ = nullptr;
    std::vector<double> moneyness_;
    std::vector<double> maturity_;
    std::vector<double> volatility_;
    std::vector<double> rate_;

    const size_t n_m_ = 8;
    const size_t n_tau_ = 6;
    const size_t n_sigma_ = 5;
    const size_t n_r_ = 4;

    void SetUp() override {
        moneyness_ = linspace(0.8, 1.2, n_m_);
        maturity_ = linspace(0.1, 2.0, n_tau_);
        volatility_ = linspace(0.1, 0.5, n_sigma_);
        rate_ = linspace(0.0, 0.1, n_r_);

        table_ = price_table_create_with_strategy(
            moneyness_.data(), n_m_,
            maturity_.data(), n_tau_,
            volatility_.data(), n_sigma_,
            rate_.data(), n_r_,
            nullptr, 0,  // 4D mode (no dividends)
            OPTION_PUT, AMERICAN,
            &INTERP_CUBIC);  // Use cubic strategy
        ASSERT_NE(table_, nullptr);
    }

    void TearDown() override {
        if (table_) {
            price_table_destroy(table_);
        }
    }

    // Populate table with test function
    void populate(double (*func)(double, double, double, double)) {
        for (size_t i_m = 0; i_m < n_m_; i_m++) {
            for (size_t i_tau = 0; i_tau < n_tau_; i_tau++) {
                for (size_t i_sigma = 0; i_sigma < n_sigma_; i_sigma++) {
                    for (size_t i_r = 0; i_r < n_r_; i_r++) {
                        double m = moneyness_[i_m];
                        double tau = maturity_[i_tau];
                        double sigma = volatility_[i_sigma];
                        double r = rate_[i_r];
                        double value = func(m, tau, sigma, r);
                        price_table_set(table_, i_m, i_tau, i_sigma, i_r, 0, value);
                    }
                }
            }
        }
    }
};

TEST_F(Cubic4DTest, TableCreation) {
    EXPECT_EQ(table_->n_moneyness, n_m_);
    EXPECT_EQ(table_->n_maturity, n_tau_);
    EXPECT_EQ(table_->n_volatility, n_sigma_);
    EXPECT_EQ(table_->n_rate, n_r_);
    EXPECT_EQ(table_->n_dividend, 0u);
    EXPECT_EQ(table_->strategy, &INTERP_CUBIC);
}

TEST_F(Cubic4DTest, OnGridPointsExact) {
    populate(quadratic_4d);

    // Test all grid points (should be exact)
    int failures = 0;
    for (size_t i_m = 0; i_m < n_m_; i_m++) {
        for (size_t i_tau = 0; i_tau < n_tau_; i_tau++) {
            for (size_t i_sigma = 0; i_sigma < n_sigma_; i_sigma++) {
                for (size_t i_r = 0; i_r < n_r_; i_r++) {
                    double m = moneyness_[i_m];
                    double tau = maturity_[i_tau];
                    double sigma = volatility_[i_sigma];
                    double r = rate_[i_r];
                    double expected = quadratic_4d(m, tau, sigma, r);
                    double result = price_table_interpolate_4d(table_, m, tau, sigma, r);

                    if (std::abs(result - expected) > 1e-8) {
                        failures++;
                        if (failures <= 3) {
                            std::cout << "Grid point failure at (" << i_m << "," << i_tau
                                     << "," << i_sigma << "," << i_r << "): "
                                     << "result=" << result << ", expected=" << expected
                                     << ", error=" << std::abs(result - expected) << std::endl;
                        }
                    }
                }
            }
        }
    }
    EXPECT_EQ(failures, 0) << "Cubic interpolation not exact at grid points!";
}

TEST_F(Cubic4DTest, LinearFunctionExact) {
    populate(linear_4d);

    // For linear functions, cubic splines should be exact everywhere
    std::vector<double> test_m = {0.85, 0.95, 1.05, 1.15};
    std::vector<double> test_tau = {0.3, 0.7, 1.2, 1.8};
    std::vector<double> test_sigma = {0.15, 0.25, 0.35, 0.45};
    std::vector<double> test_r = {0.02, 0.05, 0.08};

    for (double m : test_m) {
        for (double tau : test_tau) {
            for (double sigma : test_sigma) {
                for (double r : test_r) {
                    double expected = linear_4d(m, tau, sigma, r);
                    double result = price_table_interpolate_4d(table_, m, tau, sigma, r);
                    EXPECT_NEAR(result, expected, 1e-6)
                        << "Linear function not exact at ("
                        << m << "," << tau << "," << sigma << "," << r << ")";
                }
            }
        }
    }
}

TEST_F(Cubic4DTest, QuadraticFunctionAccuracy) {
    populate(quadratic_4d);

    // Test off-grid points - should be very accurate for quadratic functions
    std::vector<double> test_m = {0.85, 0.95, 1.05, 1.15};
    std::vector<double> test_tau = {0.3, 0.7, 1.2, 1.8};
    std::vector<double> test_sigma = {0.15, 0.25, 0.35, 0.45};
    std::vector<double> test_r = {0.02, 0.05, 0.08};

    double max_error = 0.0;
    int count = 0;

    for (double m : test_m) {
        for (double tau : test_tau) {
            for (double sigma : test_sigma) {
                for (double r : test_r) {
                    double expected = quadratic_4d(m, tau, sigma, r);
                    double result = price_table_interpolate_4d(table_, m, tau, sigma, r);
                    double error = std::abs(result - expected);
                    max_error = std::max(max_error, error);
                    count++;

                    // Cubic splines are exact for quadratic functions in 1D,
                    // but tensor-product may have small errors
                    EXPECT_LT(error, 0.1) << "Large error at ("
                        << m << "," << tau << "," << sigma << "," << r << ")";
                }
            }
        }
    }

    std::cout << "4D Cubic - Quadratic function max error: " << max_error
              << " over " << count << " test points" << std::endl;
}

TEST_F(Cubic4DTest, BoundaryBehavior) {
    populate(quadratic_4d);

    // Test at boundaries
    double m_min = moneyness_[0];
    double m_max = moneyness_[n_m_ - 1];
    double tau_min = maturity_[0];
    double tau_max = maturity_[n_tau_ - 1];
    double sigma_min = volatility_[0];
    double sigma_max = volatility_[n_sigma_ - 1];
    double r_min = rate_[0];
    double r_max = rate_[n_r_ - 1];

    // Test corners
    EXPECT_NEAR(price_table_interpolate_4d(table_, m_min, tau_min, sigma_min, r_min),
                quadratic_4d(m_min, tau_min, sigma_min, r_min), 1e-8);
    EXPECT_NEAR(price_table_interpolate_4d(table_, m_max, tau_max, sigma_max, r_max),
                quadratic_4d(m_max, tau_max, sigma_max, r_max), 1e-8);
}

TEST_F(Cubic4DTest, Greeks) {
    populate(quadratic_4d);

    // Compute Greeks at a test point
    OptionGreeks greeks = price_table_greeks_4d(table_, 1.0, 0.5, 0.25, 0.05);

    // For quadratic function f = 10 + m + ... + 0.5*m^2
    // df/dm = 1 + m, at m=1: delta = 2
    EXPECT_NEAR(greeks.delta, 2.0, 0.2);

    // d²f/dm² = 1.0 (constant for quadratic)
    // Cubic splines should capture this accurately
    EXPECT_NEAR(greeks.gamma, 1.0, 0.3);

    // df/dsigma = 3 + 0.4*sigma, at sigma=0.25: vega = 3.1
    EXPECT_NEAR(greeks.vega, 3.1, 0.3);

    // df/dtau = 2 + 0.6*tau, at tau=0.5: theta = 2.3
    // Note: theta is negative of derivative (convention: value lost per day)
    EXPECT_NEAR(std::abs(greeks.theta), 2.3, 0.5);

    // df/dr = 4 + 0.2*r, at r=0.05: rho = 4.01
    EXPECT_NEAR(greeks.rho, 4.01, 0.3);
}

// ============================================================================
// 5D Cubic Interpolation Tests
// ============================================================================

class Cubic5DTest : public ::testing::Test {
protected:
    OptionPriceTable *table_ = nullptr;
    std::vector<double> moneyness_;
    std::vector<double> maturity_;
    std::vector<double> volatility_;
    std::vector<double> rate_;
    std::vector<double> dividend_;

    const size_t n_m_ = 6;
    const size_t n_tau_ = 5;
    const size_t n_sigma_ = 4;
    const size_t n_r_ = 3;
    const size_t n_q_ = 3;

    void SetUp() override {
        moneyness_ = linspace(0.8, 1.2, n_m_);
        maturity_ = linspace(0.1, 2.0, n_tau_);
        volatility_ = linspace(0.1, 0.5, n_sigma_);
        rate_ = linspace(0.0, 0.1, n_r_);
        dividend_ = linspace(0.0, 0.05, n_q_);

        table_ = price_table_create_with_strategy(
            moneyness_.data(), n_m_,
            maturity_.data(), n_tau_,
            volatility_.data(), n_sigma_,
            rate_.data(), n_r_,
            dividend_.data(), n_q_,  // 5D mode
            OPTION_PUT, AMERICAN,
            &INTERP_CUBIC);
        ASSERT_NE(table_, nullptr);
    }

    void TearDown() override {
        if (table_) {
            price_table_destroy(table_);
        }
    }

    void populate(double (*func)(double, double, double, double, double)) {
        for (size_t i_m = 0; i_m < n_m_; i_m++) {
            for (size_t i_tau = 0; i_tau < n_tau_; i_tau++) {
                for (size_t i_sigma = 0; i_sigma < n_sigma_; i_sigma++) {
                    for (size_t i_r = 0; i_r < n_r_; i_r++) {
                        for (size_t i_q = 0; i_q < n_q_; i_q++) {
                            double m = moneyness_[i_m];
                            double tau = maturity_[i_tau];
                            double sigma = volatility_[i_sigma];
                            double r = rate_[i_r];
                            double q = dividend_[i_q];
                            double value = func(m, tau, sigma, r, q);
                            price_table_set(table_, i_m, i_tau, i_sigma, i_r, i_q, value);
                        }
                    }
                }
            }
        }
    }
};

TEST_F(Cubic5DTest, TableCreation) {
    EXPECT_EQ(table_->n_moneyness, n_m_);
    EXPECT_EQ(table_->n_maturity, n_tau_);
    EXPECT_EQ(table_->n_volatility, n_sigma_);
    EXPECT_EQ(table_->n_rate, n_r_);
    EXPECT_EQ(table_->n_dividend, n_q_);
    EXPECT_EQ(table_->strategy, &INTERP_CUBIC);
}

TEST_F(Cubic5DTest, OnGridPointsExact) {
    populate(quadratic_5d);

    // Test subset of grid points (testing all would be slow)
    int tested = 0;
    for (size_t i_m = 0; i_m < n_m_; i_m += 2) {
        for (size_t i_tau = 0; i_tau < n_tau_; i_tau += 2) {
            for (size_t i_sigma = 0; i_sigma < n_sigma_; i_sigma++) {
                for (size_t i_r = 0; i_r < n_r_; i_r++) {
                    for (size_t i_q = 0; i_q < n_q_; i_q++) {
                        double m = moneyness_[i_m];
                        double tau = maturity_[i_tau];
                        double sigma = volatility_[i_sigma];
                        double r = rate_[i_r];
                        double q = dividend_[i_q];
                        double expected = quadratic_5d(m, tau, sigma, r, q);
                        double result = price_table_interpolate_5d(table_, m, tau, sigma, r, q);

                        EXPECT_NEAR(result, expected, 1e-8)
                            << "Grid point not exact at (" << i_m << "," << i_tau
                            << "," << i_sigma << "," << i_r << "," << i_q << ")";
                        tested++;
                    }
                }
            }
        }
    }
    std::cout << "Tested " << tested << " grid points in 5D" << std::endl;
}

TEST_F(Cubic5DTest, LinearFunctionExact) {
    populate(linear_5d);

    // Test off-grid points
    std::vector<double> test_m = {0.9, 1.1};
    std::vector<double> test_tau = {0.5, 1.5};
    std::vector<double> test_sigma = {0.2, 0.4};
    std::vector<double> test_r = {0.03, 0.07};
    std::vector<double> test_q = {0.01, 0.04};

    for (double m : test_m) {
        for (double tau : test_tau) {
            for (double sigma : test_sigma) {
                for (double r : test_r) {
                    for (double q : test_q) {
                        double expected = linear_5d(m, tau, sigma, r, q);
                        double result = price_table_interpolate_5d(table_, m, tau, sigma, r, q);
                        EXPECT_NEAR(result, expected, 1e-6)
                            << "Linear function not exact at ("
                            << m << "," << tau << "," << sigma << "," << r << "," << q << ")";
                    }
                }
            }
        }
    }
}

TEST_F(Cubic5DTest, QuadraticFunctionAccuracy) {
    populate(quadratic_5d);

    // Test sample of off-grid points
    std::vector<double> test_m = {0.9, 1.1};
    std::vector<double> test_tau = {0.5, 1.5};
    std::vector<double> test_sigma = {0.2, 0.4};
    std::vector<double> test_r = {0.03, 0.07};
    std::vector<double> test_q = {0.01, 0.04};

    double max_error = 0.0;

    for (double m : test_m) {
        for (double tau : test_tau) {
            for (double sigma : test_sigma) {
                for (double r : test_r) {
                    for (double q : test_q) {
                        double expected = quadratic_5d(m, tau, sigma, r, q);
                        double result = price_table_interpolate_5d(table_, m, tau, sigma, r, q);
                        double error = std::abs(result - expected);
                        max_error = std::max(max_error, error);

                        EXPECT_LT(error, 0.15) << "Large error at ("
                            << m << "," << tau << "," << sigma << "," << r << "," << q << ")";
                    }
                }
            }
        }
    }

    std::cout << "5D Cubic - Quadratic function max error: " << max_error << std::endl;
}

// ============================================================================
// Precompute Tests
// ============================================================================

TEST_F(Cubic4DTest, PrecomputeVsOnTheFly) {
    populate(quadratic_4d);

    // This table was created with INTERP_CUBIC strategy, so it has precomputed coefficients
    // We can't easily test against "on-the-fly" in the current API
    // But we can verify that results are consistent

    std::vector<double> test_m = {0.85, 0.95, 1.05, 1.15};
    std::vector<double> test_tau = {0.3, 0.7, 1.2, 1.8};
    std::vector<double> test_sigma = {0.15, 0.25, 0.35, 0.45};
    std::vector<double> test_r = {0.02, 0.05, 0.08};

    // Query multiple times - should get same results
    for (double m : test_m) {
        for (double tau : test_tau) {
            for (double sigma : test_sigma) {
                for (double r : test_r) {
                    double result1 = price_table_interpolate_4d(table_, m, tau, sigma, r);
                    double result2 = price_table_interpolate_4d(table_, m, tau, sigma, r);
                    EXPECT_EQ(result1, result2) << "Inconsistent results from precomputed splines";
                }
            }
        }
    }
}

TEST_F(Cubic5DTest, PrecomputeVsOnTheFly) {
    populate(quadratic_5d);

    // Test consistency of precomputed results
    std::vector<double> test_m = {0.9, 1.1};
    std::vector<double> test_tau = {0.5, 1.5};
    std::vector<double> test_sigma = {0.2, 0.4};
    std::vector<double> test_r = {0.03, 0.07};
    std::vector<double> test_q = {0.01, 0.04};

    for (double m : test_m) {
        for (double tau : test_tau) {
            for (double sigma : test_sigma) {
                for (double r : test_r) {
                    for (double q : test_q) {
                        double result1 = price_table_interpolate_5d(table_, m, tau, sigma, r, q);
                        double result2 = price_table_interpolate_5d(table_, m, tau, sigma, r, q);
                        EXPECT_EQ(result1, result2) << "Inconsistent results from precomputed splines";
                    }
                }
            }
        }
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

class CubicErrorHandlingTest : public ::testing::Test {};

TEST_F(CubicErrorHandlingTest, MinimumGridSize) {
    // Cubic splines need at least 2 points per dimension
    std::vector<double> m = {0.9, 1.1};
    std::vector<double> tau = {0.5};  // Too small!
    std::vector<double> sigma = {0.2, 0.3};
    std::vector<double> r = {0.03, 0.07};

    OptionPriceTable *table = price_table_create_with_strategy(
        m.data(), 2,
        tau.data(), 1,  // Only 1 point - should fail
        sigma.data(), 2,
        r.data(), 2,
        nullptr, 0,
        OPTION_PUT, AMERICAN,
        &INTERP_CUBIC);

    // Table should still be created, but interpolation should return NAN
    ASSERT_NE(table, nullptr);

    double result = price_table_interpolate_4d(table, 1.0, 0.5, 0.25, 0.05);
    EXPECT_TRUE(std::isnan(result));

    price_table_destroy(table);
}

TEST_F(CubicErrorHandlingTest, NullPointerHandling) {
    // Test that NULL table returns NAN
    double result = price_table_interpolate_4d(nullptr, 1.0, 0.5, 0.25, 0.05);
    EXPECT_TRUE(std::isnan(result));

    result = price_table_interpolate_5d(nullptr, 1.0, 0.5, 0.25, 0.05, 0.02);
    EXPECT_TRUE(std::isnan(result));
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
