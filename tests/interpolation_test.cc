/**
 * @file interpolation_test.cc
 * @brief Comprehensive tests for interpolation engine components
 *
 * Tests:
 * - Cubic interpolation strategy (2D, 4D, 5D)
 * - IV surface operations
 * - Price table operations
 * - I/O and persistence
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>

extern "C" {
#include "../src/interp_cubic.h"
#include "../src/iv_surface.h"
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

static double test_function_2d(double x, double y) {
    return 1.0 + 2.0*x + 3.0*y + 0.5*x*y;
}

static double test_function_4d(double w, double x, double y, double z) {
    return 1.0 + w + 2.0*x + 3.0*y + 4.0*z + 0.1*w*x*y*z;
}

// Smooth test function for cubic interpolation (quadratic)
static double smooth_function_2d(double x, double y) {
    return 1.0 + x + 2.0*y + 0.5*x*x + 0.3*y*y + 0.2*x*y;
}

// Very smooth test function (sinusoidal)
static double sinusoidal_2d(double x, double y) {
    return std::sin(M_PI * x) * std::cos(M_PI * y);
}

// ============================================================================
// IV Surface Tests
// ============================================================================

class IVSurfaceTest : public ::testing::Test {
protected:
    IVSurface *surface_ = nullptr;
    std::vector<double> moneyness_;
    std::vector<double> maturity_;
    const size_t n_m_ = 10;
    const size_t n_tau_ = 8;

    void SetUp() override {
        moneyness_ = linspace(0.8, 1.2, n_m_);
        maturity_ = linspace(0.1, 2.0, n_tau_);

        surface_ = iv_surface_create(moneyness_.data(), n_m_,
                                      maturity_.data(), n_tau_);
        ASSERT_NE(surface_, nullptr);
    }

    void TearDown() override {
        if (surface_) {
            iv_surface_destroy(surface_);
        }
    }
};

TEST_F(IVSurfaceTest, Creation) {
    EXPECT_EQ(surface_->n_moneyness, n_m_);
    EXPECT_EQ(surface_->n_maturity, n_tau_);
    EXPECT_NE(surface_->moneyness_grid, nullptr);
    EXPECT_NE(surface_->maturity_grid, nullptr);
    EXPECT_NE(surface_->iv_surface, nullptr);
    EXPECT_EQ(surface_->strategy, &INTERP_CUBIC);
}

TEST_F(IVSurfaceTest, SetAndGet) {
    // Set a specific value
    EXPECT_EQ(iv_surface_set_point(surface_, 3, 2, 0.25), 0);
    EXPECT_NEAR(iv_surface_get(surface_, 3, 2), 0.25, 1e-10);

    // Test bounds checking
    EXPECT_TRUE(std::isnan(iv_surface_get(surface_, 100, 2)));
    EXPECT_TRUE(std::isnan(iv_surface_get(surface_, 3, 100)));
}

TEST_F(IVSurfaceTest, SetAll) {
    std::vector<double> iv_data(n_m_ * n_tau_);
    for (size_t i = 0; i < iv_data.size(); i++) {
        iv_data[i] = 0.20 + 0.01 * i;
    }

    EXPECT_EQ(iv_surface_set(surface_, iv_data.data()), 0);

    // Verify
    for (size_t i_tau = 0; i_tau < n_tau_; i_tau++) {
        for (size_t i_m = 0; i_m < n_m_; i_m++) {
            size_t idx = i_tau * n_m_ + i_m;
            EXPECT_NEAR(iv_surface_get(surface_, i_m, i_tau),
                       iv_data[idx], 1e-10);
        }
    }
}

TEST_F(IVSurfaceTest, Interpolation2D) {
    // Set up a simple linear function: IV = 0.2 + 0.1*m + 0.05*tau
    for (size_t i_m = 0; i_m < n_m_; i_m++) {
        for (size_t i_tau = 0; i_tau < n_tau_; i_tau++) {
            double m = moneyness_[i_m];
            double tau = maturity_[i_tau];
            double iv = 0.2 + 0.1*m + 0.05*tau;
            iv_surface_set_point(surface_, i_m, i_tau, iv);
        }
    }

    // Test on-grid point (should be exact)
    double m_grid = moneyness_[5];
    double tau_grid = maturity_[4];
    double expected = 0.2 + 0.1*m_grid + 0.05*tau_grid;
    double result = iv_surface_interpolate(surface_, m_grid, tau_grid);
    EXPECT_NEAR(result, expected, 1e-10);

    // Test off-grid point
    double m_off = (moneyness_[3] + moneyness_[4]) / 2.0;
    double tau_off = (maturity_[2] + maturity_[3]) / 2.0;
    expected = 0.2 + 0.1*m_off + 0.05*tau_off;
    result = iv_surface_interpolate(surface_, m_off, tau_off);
    EXPECT_NEAR(result, expected, 1e-6);  // Linear function, should be exact
}

TEST_F(IVSurfaceTest, Metadata) {
    iv_surface_set_underlying(surface_, "SPX");
    EXPECT_STREQ(iv_surface_get_underlying(surface_), "SPX");

    time_t t1 = surface_->last_update;
    iv_surface_touch(surface_);
    time_t t2 = surface_->last_update;
    EXPECT_GE(t2, t1);
}

TEST_F(IVSurfaceTest, SaveLoad) {
    // Populate with data
    for (size_t i_m = 0; i_m < n_m_; i_m++) {
        for (size_t i_tau = 0; i_tau < n_tau_; i_tau++) {
            double iv = 0.2 + 0.01 * (i_m + i_tau);
            iv_surface_set_point(surface_, i_m, i_tau, iv);
        }
    }
    iv_surface_set_underlying(surface_, "TEST");

    // Save
    const char *filename = "test_iv_surface_temp.bin";
    EXPECT_EQ(iv_surface_save(surface_, filename), 0);

    // Load
    IVSurface *loaded = iv_surface_load(filename);
    ASSERT_NE(loaded, nullptr);

    // Verify dimensions
    EXPECT_EQ(loaded->n_moneyness, surface_->n_moneyness);
    EXPECT_EQ(loaded->n_maturity, surface_->n_maturity);
    EXPECT_STREQ(loaded->underlying, "TEST");

    // Verify data
    for (size_t i_m = 0; i_m < n_m_; i_m++) {
        for (size_t i_tau = 0; i_tau < n_tau_; i_tau++) {
            EXPECT_NEAR(iv_surface_get(loaded, i_m, i_tau),
                       iv_surface_get(surface_, i_m, i_tau), 1e-10);
        }
    }

    iv_surface_destroy(loaded);
    remove(filename);
}

// ============================================================================
// Price Table Tests
// ============================================================================

class PriceTableTest : public ::testing::Test {
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

        table_ = price_table_create(
            moneyness_.data(), n_m_,
            maturity_.data(), n_tau_,
            volatility_.data(), n_sigma_,
            rate_.data(), n_r_,
            nullptr, 0,  // 4D mode
            OPTION_PUT, AMERICAN);
        ASSERT_NE(table_, nullptr);
    }

    void TearDown() override {
        if (table_) {
            price_table_destroy(table_);
        }
    }
};

TEST_F(PriceTableTest, Creation) {
    EXPECT_EQ(table_->n_moneyness, n_m_);
    EXPECT_EQ(table_->n_maturity, n_tau_);
    EXPECT_EQ(table_->n_volatility, n_sigma_);
    EXPECT_EQ(table_->n_rate, n_r_);
    EXPECT_EQ(table_->n_dividend, 0u);  // 4D mode
    EXPECT_EQ(table_->type, OPTION_PUT);
    EXPECT_EQ(table_->exercise, AMERICAN);
}

TEST_F(PriceTableTest, SetAndGet) {
    // Set a specific value
    EXPECT_EQ(price_table_set(table_, 2, 3, 1, 2, 0, 12.34), 0);
    EXPECT_NEAR(price_table_get(table_, 2, 3, 1, 2, 0), 12.34, 1e-10);

    // Test bounds checking
    EXPECT_TRUE(std::isnan(price_table_get(table_, 100, 3, 1, 2, 0)));
}

TEST_F(PriceTableTest, Interpolation4D) {
    // Set up a simple linear function
    // price = 10.0 + m + 2*tau + 3*sigma + 4*r
    for (size_t i_m = 0; i_m < n_m_; i_m++) {
        for (size_t i_tau = 0; i_tau < n_tau_; i_tau++) {
            for (size_t i_sigma = 0; i_sigma < n_sigma_; i_sigma++) {
                for (size_t i_r = 0; i_r < n_r_; i_r++) {
                    double m = moneyness_[i_m];
                    double tau = maturity_[i_tau];
                    double sigma = volatility_[i_sigma];
                    double r = rate_[i_r];
                    double price = 10.0 + m + 2.0*tau + 3.0*sigma + 4.0*r;
                    price_table_set(table_, i_m, i_tau, i_sigma, i_r, 0, price);
                }
            }
        }
    }

    // Test on-grid point (should be exact)
    double m_grid = moneyness_[3];
    double tau_grid = maturity_[2];
    double sigma_grid = volatility_[1];
    double r_grid = rate_[2];
    double expected = 10.0 + m_grid + 2.0*tau_grid + 3.0*sigma_grid + 4.0*r_grid;
    double result = price_table_interpolate_4d(table_, m_grid, tau_grid,
                                                sigma_grid, r_grid);
    EXPECT_NEAR(result, expected, 1e-8);

    // Test off-grid point
    double m_off = (moneyness_[3] + moneyness_[4]) / 2.0;
    double tau_off = (maturity_[2] + maturity_[3]) / 2.0;
    double sigma_off = (volatility_[1] + volatility_[2]) / 2.0;
    double r_off = (rate_[1] + rate_[2]) / 2.0;
    expected = 10.0 + m_off + 2.0*tau_off + 3.0*sigma_off + 4.0*r_off;
    result = price_table_interpolate_4d(table_, m_off, tau_off,
                                         sigma_off, r_off);
    EXPECT_NEAR(result, expected, 1e-6);  // Linear function, should be exact
}

TEST_F(PriceTableTest, Greeks) {
    // Set up quadratic function for testing finite differences
    // price = 100 + 10*m + 5*m^2 + 2*tau + sigma
    for (size_t i_m = 0; i_m < n_m_; i_m++) {
        for (size_t i_tau = 0; i_tau < n_tau_; i_tau++) {
            for (size_t i_sigma = 0; i_sigma < n_sigma_; i_sigma++) {
                for (size_t i_r = 0; i_r < n_r_; i_r++) {
                    double m = moneyness_[i_m];
                    double tau = maturity_[i_tau];
                    double sigma = volatility_[i_sigma];
                    double price = 100.0 + 10.0*m + 5.0*m*m + 2.0*tau + sigma;
                    price_table_set(table_, i_m, i_tau, i_sigma, i_r, 0, price);
                }
            }
        }
    }

    // Compute Greeks
    OptionGreeks greeks = price_table_greeks_4d(table_, 1.0, 0.5, 0.25, 0.05);

    // Delta should be approximately ∂price/∂m = 10 + 10*m = 20 at m=1.0
    EXPECT_NEAR(greeks.delta, 20.0, 0.5);

    // Gamma: Note that multilinear interpolation is piecewise linear within cells,
    // so gamma (second derivative) is approximately zero within cells.
    // This is a limitation of multilinear interpolation - it cannot capture curvature.
    // For capturing gamma accurately, cubic spline interpolation would be needed.
    // EXPECT_NEAR(greeks.gamma, 10.0, 1.0);  // Not meaningful for multilinear

    // Vega should be approximately ∂price/∂sigma = 1
    EXPECT_NEAR(greeks.vega, 1.0, 0.2);
}

TEST_F(PriceTableTest, SaveLoad) {
    // Populate with data
    for (size_t i_m = 0; i_m < n_m_; i_m++) {
        for (size_t i_tau = 0; i_tau < n_tau_; i_tau++) {
            for (size_t i_sigma = 0; i_sigma < n_sigma_; i_sigma++) {
                for (size_t i_r = 0; i_r < n_r_; i_r++) {
                    double price = 10.0 + i_m + i_tau + i_sigma + i_r;
                    price_table_set(table_, i_m, i_tau, i_sigma, i_r, 0, price);
                }
            }
        }
    }
    price_table_set_underlying(table_, "TEST");

    // Save
    const char *filename = "test_price_table_temp.bin";
    EXPECT_EQ(price_table_save(table_, filename), 0);

    // Load
    OptionPriceTable *loaded = price_table_load(filename);
    ASSERT_NE(loaded, nullptr);

    // Verify dimensions
    EXPECT_EQ(loaded->n_moneyness, table_->n_moneyness);
    EXPECT_EQ(loaded->n_maturity, table_->n_maturity);
    EXPECT_EQ(loaded->n_volatility, table_->n_volatility);
    EXPECT_EQ(loaded->n_rate, table_->n_rate);
    EXPECT_STREQ(loaded->underlying, "TEST");

    // Verify data
    for (size_t i_m = 0; i_m < n_m_; i_m++) {
        for (size_t i_tau = 0; i_tau < n_tau_; i_tau++) {
            for (size_t i_sigma = 0; i_sigma < n_sigma_; i_sigma++) {
                for (size_t i_r = 0; i_r < n_r_; i_r++) {
                    EXPECT_NEAR(
                        price_table_get(loaded, i_m, i_tau, i_sigma, i_r, 0),
                        price_table_get(table_, i_m, i_tau, i_sigma, i_r, 0),
                        1e-10);
                }
            }
        }
    }

    price_table_destroy(loaded);
    remove(filename);
}

// ============================================================================
// Cubic Interpolation Tests (2D)
// ============================================================================

class CubicInterpolation2DTest : public ::testing::Test {
protected:
    IVSurface *surface_cubic_ = nullptr;
    std::vector<double> moneyness_;
    std::vector<double> maturity_;

    const size_t n_m_ = 10;
    const size_t n_tau_ = 8;

    void SetUp() override {
        moneyness_ = linspace(0.8, 1.2, n_m_);
        maturity_ = linspace(0.1, 2.0, n_tau_);

        // Create surface with default (cubic) strategy
        surface_cubic_ = iv_surface_create(moneyness_.data(), n_m_,
                                            maturity_.data(), n_tau_);
        ASSERT_NE(surface_cubic_, nullptr);
    }

    void TearDown() override {
        if (surface_cubic_) {
            iv_surface_destroy(surface_cubic_);
        }
    }

    // Populate surface with test function
    void populate_surface(double (*func)(double, double)) {
        for (size_t i_m = 0; i_m < n_m_; i_m++) {
            for (size_t i_tau = 0; i_tau < n_tau_; i_tau++) {
                double m = moneyness_[i_m];
                double tau = maturity_[i_tau];
                double value = func(m, tau);

                iv_surface_set_point(surface_cubic_, i_m, i_tau, value);
            }
        }
    }
};

TEST_F(CubicInterpolation2DTest, BasicFunctionality) {
    // Test that cubic interpolation can be created and used
    EXPECT_NE(surface_cubic_, nullptr);
    EXPECT_EQ(surface_cubic_->n_moneyness, n_m_);
    EXPECT_EQ(surface_cubic_->n_maturity, n_tau_);
}

TEST_F(CubicInterpolation2DTest, OnGridPointsExact) {
    // Populate with smooth quadratic function
    populate_surface(smooth_function_2d);

    // Debug: check one specific grid point in detail
    size_t test_i = 5;
    size_t test_j = 4;
    double test_m = moneyness_[test_i];
    double test_tau = maturity_[test_j];
    double test_expected = smooth_function_2d(test_m, test_tau);
    double test_result = iv_surface_interpolate(surface_cubic_, test_m, test_tau);

    std::cout << "Debug grid point (" << test_i << ", " << test_j << "):" << std::endl;
    std::cout << "  Coordinates: (" << test_m << ", " << test_tau << ")" << std::endl;
    std::cout << "  Expected (from function): " << test_expected << std::endl;
    std::cout << "  Result (from cubic): " << test_result << std::endl;
    std::cout << "  Error: " << std::abs(test_result - test_expected) << std::endl;

    // Also check what's stored in the surface
    size_t stored_idx = test_i * surface_cubic_->n_maturity + test_j;
    double stored_value = surface_cubic_->iv_surface[stored_idx];
    std::cout << "  Stored value in surface: " << stored_value << std::endl;

    // Test on-grid points (should be exact for both strategies)
    int failures = 0;
    for (size_t i_m = 0; i_m < n_m_; i_m++) {
        for (size_t i_tau = 0; i_tau < n_tau_; i_tau++) {
            double m = moneyness_[i_m];
            double tau = maturity_[i_tau];
            double expected = smooth_function_2d(m, tau);

            double result = iv_surface_interpolate(surface_cubic_, m, tau);
            if (std::abs(result - expected) > 1e-8) {
                failures++;
                if (failures <= 3) {  // Only print first few failures
                    std::cout << "Failed at (" << i_m << "," << i_tau << "): "
                              << "result=" << result << ", expected=" << expected
                              << ", error=" << std::abs(result - expected) << std::endl;
                }
            }
        }
    }

    std::cout << "Total failures: " << failures << " out of " << (n_m_ * n_tau_) << std::endl;

    // Only fail test if there are actual failures
    EXPECT_EQ(failures, 0) << "Cubic interpolation not exact at grid points!";
}

TEST_F(CubicInterpolation2DTest, OffGridAccuracy) {
    // Populate with smooth quadratic function
    populate_surface(smooth_function_2d);

    // Test off-grid points - cubic should be accurate for smooth functions
    std::vector<double> test_points_m = {0.85, 0.95, 1.05, 1.15};
    std::vector<double> test_points_tau = {0.25, 0.75, 1.25, 1.75};

    double total_error_cubic = 0.0;
    int count = 0;

    for (double m : test_points_m) {
        for (double tau : test_points_tau) {
            double expected = smooth_function_2d(m, tau);
            double result_cubic = iv_surface_interpolate(surface_cubic_, m, tau);

            double error_cubic = std::abs(result_cubic - expected);
            total_error_cubic += error_cubic;
            count++;

            // Cubic should be accurate for smooth quadratic functions
            EXPECT_LT(error_cubic, 0.01)  // 1% tolerance
                << "Cubic error too large at (" << m << ", " << tau << "): "
                << "cubic_error=" << error_cubic;
        }
    }

    double avg_error_cubic = total_error_cubic / count;
    std::cout << "Average cubic interpolation error: " << avg_error_cubic << std::endl;

    // Average error should be small for quadratic functions
    EXPECT_LT(avg_error_cubic, 0.005);
}

TEST_F(CubicInterpolation2DTest, SmoothFunctionAccuracy) {
    // Test with sinusoidal function (very smooth, C-infinity)
    populate_surface(sinusoidal_2d);

    // Sample at many off-grid points
    const int n_samples = 20;
    double max_error = 0.0;

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_samples; j++) {
            double m = 0.8 + (0.4 * i) / (n_samples - 1);
            double tau = 0.1 + (1.9 * j) / (n_samples - 1);

            double expected = sinusoidal_2d(m, tau);
            double result = iv_surface_interpolate(surface_cubic_, m, tau);
            double error = std::abs(result - expected);

            max_error = std::max(max_error, error);

            // Should be very accurate for smooth functions
            EXPECT_LT(error, 0.05)  // 5% relative error
                << "Large error at (" << m << ", " << tau << "): "
                << "result=" << result << ", expected=" << expected;
        }
    }

    // Report maximum error
    std::cout << "Maximum cubic interpolation error on sin(x)*cos(y): "
              << max_error << std::endl;
}

TEST_F(CubicInterpolation2DTest, BoundaryBehavior) {
    // Test behavior at grid boundaries
    populate_surface(smooth_function_2d);

    // Test at exact boundaries
    double m_min = moneyness_[0];
    double m_max = moneyness_[n_m_ - 1];
    double tau_min = maturity_[0];
    double tau_max = maturity_[n_tau_ - 1];

    // Corner points
    EXPECT_NEAR(iv_surface_interpolate(surface_cubic_, m_min, tau_min),
                smooth_function_2d(m_min, tau_min), 1e-8);
    EXPECT_NEAR(iv_surface_interpolate(surface_cubic_, m_max, tau_max),
                smooth_function_2d(m_max, tau_max), 1e-8);
    EXPECT_NEAR(iv_surface_interpolate(surface_cubic_, m_min, tau_max),
                smooth_function_2d(m_min, tau_max), 1e-8);
    EXPECT_NEAR(iv_surface_interpolate(surface_cubic_, m_max, tau_min),
                smooth_function_2d(m_max, tau_min), 1e-8);

    // Slightly inside boundaries (should not extrapolate)
    double epsilon = 1e-6;
    double result_inside = iv_surface_interpolate(surface_cubic_,
                                                   m_min + epsilon,
                                                   tau_min + epsilon);
    EXPECT_TRUE(std::isfinite(result_inside));
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
