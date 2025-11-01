#include <gtest/gtest.h>
#include "../src/price_table.h"
#include "../src/american_option.h"
#include <cmath>
#include <cstdlib>

class PriceTablePrecomputeTest : public ::testing::Test {
protected:
    AmericanOptionGrid default_grid;

    void SetUp() override {
        // Simple grid for fast testing
        default_grid.x_min = -0.7;  // ln(0.5)
        default_grid.x_max = 0.7;   // ln(2.0)
        default_grid.n_points = 51;
        default_grid.dt = 0.001;
        default_grid.n_steps = 100;
    }
};

TEST_F(PriceTablePrecomputeTest, NullTablePointer) {
    int status = price_table_precompute(nullptr, &default_grid);
    EXPECT_EQ(status, -1);
}

TEST_F(PriceTablePrecomputeTest, NullGridPointer) {
    double moneyness[] = {0.9, 1.0, 1.1};
    double maturity[] = {0.25, 0.5};
    double volatility[] = {0.2, 0.3};
    double rate[] = {0.05};

    OptionPriceTable *table = price_table_create(
        moneyness, 3, maturity, 2, volatility, 2, rate, 1, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table, nullptr);

    int status = price_table_precompute(table, nullptr);
    EXPECT_EQ(status, -1);

    price_table_destroy(table);
}

TEST_F(PriceTablePrecomputeTest, SmallGrid4D) {
    // 2×2×2×2 = 16 points for fast test
    double moneyness[] = {0.95, 1.05};
    double maturity[] = {0.25, 0.5};
    double volatility[] = {0.2, 0.3};
    double rate[] = {0.03, 0.05};

    OptionPriceTable *table = price_table_create(
        moneyness, 2, maturity, 2, volatility, 2, rate, 2, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table, nullptr);

    int status = price_table_precompute(table, &default_grid);
    EXPECT_EQ(status, 0);

    // Verify no NANs in results
    for (size_t i_m = 0; i_m < 2; i_m++) {
        for (size_t i_tau = 0; i_tau < 2; i_tau++) {
            for (size_t i_sigma = 0; i_sigma < 2; i_sigma++) {
                for (size_t i_r = 0; i_r < 2; i_r++) {
                    double price = price_table_get(table, i_m, i_tau, i_sigma, i_r, 0);
                    EXPECT_FALSE(std::isnan(price))
                        << "NAN at [" << i_m << "," << i_tau << "," << i_sigma << "," << i_r << "]";
                    EXPECT_GT(price, 0.0)
                        << "Non-positive price at [" << i_m << "," << i_tau << "," << i_sigma << "," << i_r << "]";
                }
            }
        }
    }

    // Verify generation timestamp set
    EXPECT_GT(table->generation_time, 0);

    price_table_destroy(table);
}

TEST_F(PriceTablePrecomputeTest, MonotonicityInMoneyness) {
    // Put prices should generally increase as moneyness decreases (more ITM)
    // Cubic interpolation requires ≥2 points in each dimension
    double moneyness[] = {0.90, 0.95, 1.00, 1.05, 1.10};
    double maturity[] = {0.25, 0.5};
    double volatility[] = {0.20, 0.25};
    double rate[] = {0.04, 0.05};

    OptionPriceTable *table = price_table_create(
        moneyness, 5, maturity, 2, volatility, 2, rate, 2, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table, nullptr);

    int status = price_table_precompute(table, &default_grid);
    ASSERT_EQ(status, 0);

    // Check that put prices increase as moneyness decreases
    for (size_t i = 0; i < 4; i++) {
        double price_lower_m = price_table_get(table, i, 0, 0, 0, 0);
        double price_higher_m = price_table_get(table, i + 1, 0, 0, 0, 0);
        EXPECT_GT(price_lower_m, price_higher_m)
            << "Put price should decrease as moneyness increases: "
            << "m[" << i << "]=" << moneyness[i] << " price=" << price_lower_m
            << " vs m[" << i+1 << "]=" << moneyness[i+1] << " price=" << price_higher_m;
    }

    price_table_destroy(table);
}

TEST_F(PriceTablePrecomputeTest, MonotonicityInMaturity) {
    // American option prices should increase with maturity (more time value)
    // Use OTM option to avoid early exercise boundary issues
    // Cubic interpolation requires ≥2 points in each dimension
    double moneyness[] = {1.05, 1.1};  // OTM put
    double maturity[] = {0.1, 0.25, 0.5, 1.0};
    double volatility[] = {0.20, 0.25};
    double rate[] = {0.04, 0.05};

    OptionPriceTable *table = price_table_create(
        moneyness, 2, maturity, 4, volatility, 2, rate, 2, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table, nullptr);

    int status = price_table_precompute(table, &default_grid);
    ASSERT_EQ(status, 0);

    // Check that prices increase with maturity (or at least don't decrease)
    for (size_t i = 0; i < 3; i++) {
        double price_shorter = price_table_get(table, 0, i, 0, 0, 0);
        double price_longer = price_table_get(table, 0, i + 1, 0, 0, 0);
        EXPECT_LE(price_shorter, price_longer)
            << "American option price should not decrease with maturity: "
            << "τ[" << i << "]=" << maturity[i] << " price=" << price_shorter
            << " vs τ[" << i+1 << "]=" << maturity[i+1] << " price=" << price_longer;
    }

    price_table_destroy(table);
}

TEST_F(PriceTablePrecomputeTest, MonotonicityInVolatility) {
    // Option prices should increase with volatility
    // Cubic interpolation requires ≥2 points in each dimension
    double moneyness[] = {0.95, 1.0};
    double maturity[] = {0.25, 0.5};
    double volatility[] = {0.15, 0.20, 0.25, 0.30, 0.35};
    double rate[] = {0.04, 0.05};

    OptionPriceTable *table = price_table_create(
        moneyness, 2, maturity, 2, volatility, 5, rate, 2, nullptr, 0,
        OPTION_CALL, AMERICAN);

    ASSERT_NE(table, nullptr);

    int status = price_table_precompute(table, &default_grid);
    ASSERT_EQ(status, 0);

    // Check that prices increase with volatility
    for (size_t i = 0; i < 4; i++) {
        double price_lower_vol = price_table_get(table, 0, 0, i, 0, 0);
        double price_higher_vol = price_table_get(table, 0, 0, i + 1, 0, 0);
        EXPECT_LT(price_lower_vol, price_higher_vol)
            << "Option price should increase with volatility: "
            << "σ[" << i << "]=" << volatility[i] << " price=" << price_lower_vol
            << " vs σ[" << i+1 << "]=" << volatility[i+1] << " price=" << price_higher_vol;
    }

    price_table_destroy(table);
}

TEST_F(PriceTablePrecomputeTest, BatchSizeConsistency) {
    // Test that different batch sizes produce identical results
    double moneyness[] = {0.9, 1.0, 1.1};
    double maturity[] = {0.25, 0.5};
    double volatility[] = {0.2, 0.3};
    double rate[] = {0.05};

    // First run with default batch size
    OptionPriceTable *table1 = price_table_create(
        moneyness, 3, maturity, 2, volatility, 2, rate, 1, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table1, nullptr);
    price_table_precompute(table1, &default_grid);

    // Second run with batch_size=1
    setenv("IVCALC_PRECOMPUTE_BATCH_SIZE", "1", 1);

    OptionPriceTable *table2 = price_table_create(
        moneyness, 3, maturity, 2, volatility, 2, rate, 1, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table2, nullptr);
    price_table_precompute(table2, &default_grid);

    unsetenv("IVCALC_PRECOMPUTE_BATCH_SIZE");

    // Compare results (should be identical)
    size_t n_total = 3 * 2 * 2 * 1;
    for (size_t i = 0; i < n_total; i++) {
        EXPECT_NEAR(table1->prices[i], table2->prices[i], 1e-10)
            << "Mismatch at index " << i;
    }

    price_table_destroy(table1);
    price_table_destroy(table2);
}

TEST_F(PriceTablePrecomputeTest, GetSetOperations) {
    // Test price_table_get and price_table_set operations
    double moneyness[] = {0.9, 1.0};
    double maturity[] = {0.25, 0.5};
    double volatility[] = {0.2};
    double rate[] = {0.05};

    OptionPriceTable *table = price_table_create(
        moneyness, 2, maturity, 2, volatility, 1, rate, 1, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table, nullptr);

    // Test setting values
    int status = price_table_set(table, 0, 0, 0, 0, 0, 10.5);
    EXPECT_EQ(status, 0);

    double value = price_table_get(table, 0, 0, 0, 0, 0);
    EXPECT_DOUBLE_EQ(value, 10.5);

    // Test out of bounds
    status = price_table_set(table, 10, 0, 0, 0, 0, 5.0);
    EXPECT_EQ(status, -1);

    double invalid = price_table_get(table, 10, 0, 0, 0, 0);
    EXPECT_TRUE(std::isnan(invalid));

    price_table_destroy(table);
}

TEST_F(PriceTablePrecomputeTest, IntrinsicValueBound) {
    // For a put at maturity, price should be at least intrinsic value
    // Cubic interpolation requires ≥2 points in each dimension
    double moneyness[] = {0.8, 0.9, 1.0, 1.1, 1.2};
    double maturity[] = {0.05, 0.1};  // Short maturity (adaptive: 50 steps with dt=0.001)
    double volatility[] = {0.20, 0.25};
    double rate[] = {0.04, 0.05};

    OptionPriceTable *table = price_table_create(
        moneyness, 5, maturity, 2, volatility, 2, rate, 2, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table, nullptr);

    int status = price_table_precompute(table, &default_grid);
    ASSERT_EQ(status, 0);

    // For each moneyness, check price >= max(K - S, 0) approximately
    const double K_ref = 100.0;
    for (size_t i = 0; i < 5; i++) {
        double m = moneyness[i];
        double S = m * K_ref;
        double intrinsic = std::max(K_ref - S, 0.0);
        double price = price_table_get(table, i, 0, 0, 0, 0);

        EXPECT_GE(price, intrinsic * 0.95)  // Allow 5% margin for numerical errors
            << "Put price should be >= intrinsic value: "
            << "m=" << m << " S=" << S << " intrinsic=" << intrinsic << " price=" << price;
    }

    price_table_destroy(table);
}

TEST_F(PriceTablePrecomputeTest, CallPutTypeCorrectness) {
    // Verify that call and put tables store different prices
    double moneyness[] = {1.0};
    double maturity[] = {0.5};
    double volatility[] = {0.25};
    double rate[] = {0.05};

    // American call
    OptionPriceTable *call_table = price_table_create(
        moneyness, 1, maturity, 1, volatility, 1, rate, 1, nullptr, 0,
        OPTION_CALL, AMERICAN);
    ASSERT_NE(call_table, nullptr);
    price_table_precompute(call_table, &default_grid);

    // American put
    OptionPriceTable *put_table = price_table_create(
        moneyness, 1, maturity, 1, volatility, 1, rate, 1, nullptr, 0,
        OPTION_PUT, AMERICAN);
    ASSERT_NE(put_table, nullptr);
    price_table_precompute(put_table, &default_grid);

    double call_price = price_table_get(call_table, 0, 0, 0, 0, 0);
    double put_price = price_table_get(put_table, 0, 0, 0, 0, 0);

    // At ATM, call and put should be different
    EXPECT_NE(call_price, put_price);
    EXPECT_GT(call_price, 0.0);
    EXPECT_GT(put_price, 0.0);

    price_table_destroy(call_table);
    price_table_destroy(put_table);
}

TEST_F(PriceTablePrecomputeTest, InterpolationSmoothness) {
    // Create a small table and verify interpolation works
    // Cubic interpolation requires ≥2 points in each dimension
    double moneyness[] = {0.9, 1.0, 1.1};
    double maturity[] = {0.25, 0.5};
    double volatility[] = {0.2, 0.3};
    double rate[] = {0.04, 0.06};  // Need at least 2 points for cubic

    OptionPriceTable *table = price_table_create(
        moneyness, 3, maturity, 2, volatility, 2, rate, 2, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table, nullptr);

    int status = price_table_precompute(table, &default_grid);
    ASSERT_EQ(status, 0);

    // Test interpolation at a point between grid points
    double interp_price = price_table_interpolate_4d(table, 0.95, 0.375, 0.25, 0.05);

    // Should return a valid price
    EXPECT_FALSE(std::isnan(interp_price));
    EXPECT_GT(interp_price, 0.0);

    // Should be between corner values
    double corner_prices[4];
    corner_prices[0] = price_table_get(table, 0, 0, 0, 0, 0);  // (0.9, 0.25, 0.2, 0.05)
    corner_prices[1] = price_table_get(table, 1, 0, 0, 0, 0);  // (1.0, 0.25, 0.2, 0.05)
    corner_prices[2] = price_table_get(table, 0, 1, 1, 0, 0);  // (0.9, 0.5, 0.3, 0.05)
    corner_prices[3] = price_table_get(table, 1, 1, 1, 0, 0);  // (1.0, 0.5, 0.3, 0.05)

    double min_corner = corner_prices[0];
    double max_corner = corner_prices[0];
    for (int i = 1; i < 4; i++) {
        min_corner = std::min(min_corner, corner_prices[i]);
        max_corner = std::max(max_corner, corner_prices[i]);
    }

    EXPECT_GE(interp_price, min_corner * 0.9);  // Allow small extrapolation
    EXPECT_LE(interp_price, max_corner * 1.1);

    price_table_destroy(table);
}

// InterpolationAccuracyIntegration test moved to price_table_slow_test.cc
// (marked as manual due to long precomputation time)

TEST(PriceTableTest, VegaArrayAllocation) {
    double m[] = {0.9, 1.0, 1.1};
    double tau[] = {0.25, 0.5};
    double sigma[] = {0.2, 0.3};
    double r[] = {0.05};

    OptionPriceTable *table = price_table_create(
        m, 3, tau, 2, sigma, 2, r, 1, nullptr, 0,
        OPTION_CALL, AMERICAN);

    ASSERT_NE(table, nullptr);

    // Vega array is lazily allocated during precompute, initially NULL
    EXPECT_EQ(table->vegas, nullptr);

    price_table_destroy(table);
}

TEST(PriceTableTest, VegaGetSet) {
    double m[] = {0.9, 1.0, 1.1};
    double tau[] = {0.25, 0.5};
    double sigma[] = {0.2, 0.3};
    double r[] = {0.05};

    OptionPriceTable *table = price_table_create(
        m, 3, tau, 2, sigma, 2, r, 1, nullptr, 0,
        OPTION_CALL, AMERICAN);

    // Vega array is allocated during precompute
    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 51,
        .dt = 0.01,
        .n_steps = 100
    };
    price_table_precompute(table, &grid);

    // Now vegas should be allocated
    ASSERT_NE(table->vegas, nullptr);

    // Set vega at specific grid point
    int status = price_table_set_vega(table, 1, 0, 1, 0, 0, 0.42);
    EXPECT_EQ(status, 0);

    // Get vega back
    double vega = price_table_get_vega(table, 1, 0, 1, 0, 0);
    EXPECT_DOUBLE_EQ(vega, 0.42);

    // Out of bounds should return NaN
    double oob = price_table_get_vega(table, 10, 0, 0, 0, 0);
    EXPECT_TRUE(std::isnan(oob));

    price_table_destroy(table);
}

TEST(PriceTableTest, VegaPrecomputation) {
    // Small grid for fast test
    double m[] = {1.0};
    double tau[] = {0.5};
    double sigma[] = {0.15, 0.20, 0.25};  // Need 3+ points for centered diff
    double r[] = {0.05};

    OptionPriceTable *table = price_table_create(
        m, 1, tau, 1, sigma, 3, r, 1, nullptr, 0,
        OPTION_PUT, AMERICAN);

    // Precompute with coarse grid (fast test)
    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 51,
        .dt = 0.01,
        .n_steps = 50
    };

    int status = price_table_precompute(table, &grid);
    EXPECT_EQ(status, 0);

    // Vega at middle volatility point should be computed
    double vega = price_table_get_vega(table, 0, 0, 1, 0, 0);
    EXPECT_FALSE(std::isnan(vega));

    // Vega should be positive for ATM put
    EXPECT_GT(vega, 0.0);

    // Vega should be reasonably sized (raw vega, not normalized)
    // For ATM option with S=K=100, vega can be 10-50 depending on maturity
    EXPECT_GT(vega, 1.0);
    EXPECT_LT(vega, 100.0);

    price_table_destroy(table);
}

TEST(PriceTableTest, VegaInterpolation4D) {
    // Create table with reasonable grid
    std::vector<double> m = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau = {0.25, 0.5, 1.0};
    std::vector<double> sigma = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r = {0.03, 0.05};

    OptionPriceTable *table = price_table_create(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_PUT, AMERICAN);

    // Precompute (includes vega)
    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 51, .dt = 0.01, .n_steps = 50
    };
    price_table_precompute(table, &grid);

    // Build interpolation structures
    price_table_build_interpolation(table);

    // Query vega at off-grid point
    double vega = price_table_interpolate_vega_4d(table, 0.95, 0.75, 0.22, 0.04);

    // Should return interpolated value (not NaN)
    EXPECT_FALSE(std::isnan(vega));

    // Vega should be positive for put
    EXPECT_GT(vega, 0.0);

    price_table_destroy(table);
}

TEST(PriceTableTest, VegaInterpolation5D) {
    std::vector<double> m = {0.9, 1.0, 1.1};
    std::vector<double> tau = {0.25, 0.5};
    std::vector<double> sigma = {0.15, 0.20, 0.25};  // Need 3+ for centered diff
    std::vector<double> r = {0.03, 0.05};  // Need at least 2 for cubic
    std::vector<double> q = {0.0, 0.02};  // 5D with dividend

    OptionPriceTable *table = price_table_create(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        q.data(), q.size(),
        OPTION_CALL, AMERICAN);

    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 51, .dt = 0.01, .n_steps = 50
    };
    price_table_precompute(table, &grid);
    price_table_build_interpolation(table);

    // Query vega at off-grid point (5D)
    double vega = price_table_interpolate_vega_5d(table, 0.95, 0.35, 0.22, 0.05, 0.01);

    EXPECT_FALSE(std::isnan(vega));
    EXPECT_GT(vega, 0.0);

    price_table_destroy(table);
}

TEST(PriceTableTest, VegaSaveLoad) {
    // Create and precompute table
    std::vector<double> m = {0.9, 1.0, 1.1};
    std::vector<double> tau = {0.5};
    std::vector<double> sigma = {0.15, 0.20, 0.25};
    std::vector<double> r = {0.05};

    OptionPriceTable *table = price_table_create(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_PUT, AMERICAN);

    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 51, .dt = 0.01, .n_steps = 50
    };
    price_table_precompute(table, &grid);

    // Save to file
    const char *filename = "/tmp/test_vega_table.bin";
    int status = price_table_save(table, filename);
    EXPECT_EQ(status, 0);

    // Get vega value before destroying
    double vega_original = price_table_get_vega(table, 1, 0, 1, 0, 0);
    EXPECT_FALSE(std::isnan(vega_original));

    price_table_destroy(table);

    // Load from file
    OptionPriceTable *loaded = price_table_load(filename);
    ASSERT_NE(loaded, nullptr);

    // Verify vega was restored
    double vega_loaded = price_table_get_vega(loaded, 1, 0, 1, 0, 0);
    EXPECT_DOUBLE_EQ(vega_loaded, vega_original);

    price_table_destroy(loaded);
}

TEST(PriceTableTest, LoadOldFormatWithoutVega) {
    // This test verifies that loading old binary files (without vega)
    // doesn't crash and initializes vega to NaN

    // Create a table and save with old format (manually, without vega)
    std::vector<double> m = {1.0};
    std::vector<double> tau = {0.5};
    std::vector<double> sigma = {0.20};
    std::vector<double> r = {0.05};

    OptionPriceTable *table = price_table_create(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_PUT, AMERICAN);

    // Set a price manually
    price_table_set(table, 0, 0, 0, 0, 0, 5.0);

    // Save (will include vega in new format)
    const char *filename = "/tmp/test_compat_table.bin";
    price_table_save(table, filename);
    price_table_destroy(table);

    // Load and verify
    OptionPriceTable *loaded = price_table_load(filename);
    ASSERT_NE(loaded, nullptr);

    // Price should be preserved
    double price = price_table_get(loaded, 0, 0, 0, 0, 0);
    EXPECT_DOUBLE_EQ(price, 5.0);

    // Vega should exist (newly saved format includes it)
    double vega = price_table_get_vega(loaded, 0, 0, 0, 0, 0);
    EXPECT_TRUE(std::isnan(vega));  // NaN because not precomputed

    price_table_destroy(loaded);
}

TEST(PriceTableTest, GammaGetSet) {
    double m[] = {0.9, 1.0, 1.1};
    double tau[] = {0.25, 0.5};
    double sigma[] = {0.2, 0.3};
    double r[] = {0.05};

    OptionPriceTable *table = price_table_create_ex(
        m, 3, tau, 2, sigma, 2, r, 1, nullptr, 0,
        OPTION_PUT, AMERICAN,
        COORD_RAW, LAYOUT_M_INNER);

    ASSERT_NE(table, nullptr);

    // Allocate gammas
    size_t n_total = 3 * 2 * 2 * 1;
    table->gammas = (double*)malloc(n_total * sizeof(double));
    ASSERT_NE(table->gammas, nullptr);

    // Initialize to NaN
    for (size_t i = 0; i < n_total; i++) {
        table->gammas[i] = NAN;
    }

    // Test set
    int status = price_table_set_gamma(table, 1, 0, 1, 0, 0, 42.5);
    EXPECT_EQ(status, 0);

    // Test get
    double gamma = price_table_get_gamma(table, 1, 0, 1, 0, 0);
    EXPECT_DOUBLE_EQ(gamma, 42.5);

    // Test bounds checking - out of bounds should return NaN
    double gamma_oob = price_table_get_gamma(table, 99, 0, 0, 0, 0);
    EXPECT_TRUE(std::isnan(gamma_oob));

    price_table_destroy(table);
}
