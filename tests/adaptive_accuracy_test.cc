#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" {
#include "../src/validation.h"
#include "../src/price_table.h"
#include "../src/american_option.h"
}

// Test fixture for adaptive accuracy tests
class AdaptiveAccuracyTest : public ::testing::Test {
protected:
    AmericanOptionGrid default_grid = {
        .x_min = -0.7,      // ln(0.5) ≈ -0.7 (50% of strike)
        .x_max = 0.7,       // ln(2.0) ≈ 0.7 (200% of strike)
        .n_points = 101,    // Number of spatial grid points
        .dt = 0.001,
        .n_steps = 500
    };

    // Helper to create a coarse moneyness grid
    std::vector<double> create_coarse_grid() {
        std::vector<double> grid;
        // Log-spaced grid from 0.7 to 1.3 with 10 points
        const double m_min = 0.7;
        const double m_max = 1.3;
        const size_t n = 10;

        double log_min = std::log(m_min);
        double log_max = std::log(m_max);
        for (size_t i = 0; i < n; i++) {
            double log_m = log_min + i * (log_max - log_min) / (n - 1);
            grid.push_back(std::exp(log_m));
        }
        return grid;
    }
};

// Test that adaptive refinement improves accuracy
TEST_F(AdaptiveAccuracyTest, AccuracyImprovement) {
    // Create coarse grid
    auto m_grid = create_coarse_grid();

    // Create other grids
    std::vector<double> tau_grid = {0.1, 0.25, 0.5, 1.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate_grid = {0.02, 0.05};

    // Create table with LAYOUT_M_INNER
    OptionPriceTable *table = price_table_create_ex(
        m_grid.data(), m_grid.size(),
        tau_grid.data(), tau_grid.size(),
        sigma_grid.data(), sigma_grid.size(),
        rate_grid.data(), rate_grid.size(),
        nullptr, 0,
        OPTION_PUT, AMERICAN,
        COORD_RAW,  // Use raw coordinates for simplicity
        LAYOUT_M_INNER  // Required for adaptive refinement
    );
    ASSERT_NE(table, nullptr);

    // Measure error before adaptive refinement
    ValidationResult result_before = validate_interpolation_error(
        table, &default_grid, 100, 1.0  // 100 samples, 1bp target
    );

    double p95_before = result_before.p95_iv_error;
    validation_result_free(&result_before);

    // Run adaptive refinement (lenient target to ensure completion)
    int status = price_table_precompute_adaptive(
        table, &default_grid,
        5.0,  // 5bp target (lenient)
        3,    // Max 3 iterations
        100   // 100 validation samples
    );

    EXPECT_EQ(status, 0) << "Adaptive refinement should succeed";

    // Measure error after adaptive refinement
    ValidationResult result_after = validate_interpolation_error(
        table, &default_grid, 100, 1.0
    );

    double p95_after = result_after.p95_iv_error;

    // After refinement, error should be lower or target achieved
    EXPECT_TRUE(p95_after <= 5.0 || p95_after < p95_before)
        << "P95 error should improve or meet target. Before: "
        << p95_before << " bp, After: " << p95_after << " bp";

    validation_result_free(&result_after);
    price_table_destroy(table);
}

// Test grid expansion preserves existing prices
TEST_F(AdaptiveAccuracyTest, GridExpansionPreservesValues) {
    // Create small grid
    std::vector<double> m_grid = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau_grid = {0.25, 0.5};
    std::vector<double> sigma_grid = {0.20, 0.25};
    std::vector<double> rate_grid = {0.05};

    OptionPriceTable *table = price_table_create_ex(
        m_grid.data(), m_grid.size(),
        tau_grid.data(), tau_grid.size(),
        sigma_grid.data(), sigma_grid.size(),
        rate_grid.data(), rate_grid.size(),
        nullptr, 0,
        OPTION_PUT, AMERICAN,
        COORD_RAW,
        LAYOUT_M_INNER
    );
    ASSERT_NE(table, nullptr);

    // Precompute initial prices
    int status = price_table_precompute(table, &default_grid);
    EXPECT_EQ(status, 0);

    // Save initial prices at key points
    double price_before_1 = price_table_interpolate_4d(table, 0.9, 0.25, 0.20, 0.05);
    double price_before_2 = price_table_interpolate_4d(table, 1.0, 0.5, 0.25, 0.05);
    double price_before_3 = price_table_interpolate_4d(table, 1.1, 0.25, 0.20, 0.05);

    EXPECT_FALSE(std::isnan(price_before_1));
    EXPECT_FALSE(std::isnan(price_before_2));
    EXPECT_FALSE(std::isnan(price_before_3));

    // Add refinement points
    double new_points[] = {0.85, 0.95, 1.05, 1.15};
    status = price_table_expand_grid(table, new_points, 4);
    EXPECT_EQ(status, 0) << "Grid expansion should succeed";

    // Verify grid size increased
    EXPECT_EQ(table->n_moneyness, 9) << "Grid should have 5 + 4 = 9 points";

    // Recompute (only fills NaN entries)
    status = price_table_precompute(table, &default_grid);
    EXPECT_EQ(status, 0);

    // Check that original prices are preserved (within numerical tolerance)
    double price_after_1 = price_table_interpolate_4d(table, 0.9, 0.25, 0.20, 0.05);
    double price_after_2 = price_table_interpolate_4d(table, 1.0, 0.5, 0.25, 0.05);
    double price_after_3 = price_table_interpolate_4d(table, 1.1, 0.25, 0.20, 0.05);

    // Prices at grid points should be exactly preserved (no interpolation)
    EXPECT_NEAR(price_after_1, price_before_1, 0.01)
        << "Original grid point prices should be preserved";
    EXPECT_NEAR(price_after_2, price_before_2, 0.01);
    EXPECT_NEAR(price_after_3, price_before_3, 0.01);

    price_table_destroy(table);
}

// Test validation framework statistics
TEST_F(AdaptiveAccuracyTest, ValidationStatistics) {
    // Create moderate grid
    auto m_grid = create_coarse_grid();
    std::vector<double> tau_grid = {0.25, 0.5, 1.0};
    std::vector<double> sigma_grid = {0.20, 0.25, 0.30};
    std::vector<double> rate_grid = {0.03, 0.05};

    OptionPriceTable *table = price_table_create_ex(
        m_grid.data(), m_grid.size(),
        tau_grid.data(), tau_grid.size(),
        sigma_grid.data(), sigma_grid.size(),
        rate_grid.data(), rate_grid.size(),
        nullptr, 0,
        OPTION_PUT, AMERICAN,
        COORD_RAW,
        LAYOUT_M_INNER
    );
    ASSERT_NE(table, nullptr);

    // Precompute
    int status = price_table_precompute(table, &default_grid);
    EXPECT_EQ(status, 0);

    // Validate with 200 samples
    ValidationResult result = validate_interpolation_error(
        table, &default_grid, 200, 1.0
    );

    // Check that statistics are computed
    EXPECT_EQ(result.n_samples, 200);
    EXPECT_GE(result.mean_iv_error, 0.0);
    EXPECT_GE(result.median_iv_error, 0.0);
    EXPECT_GE(result.p95_iv_error, 0.0);
    EXPECT_GE(result.p99_iv_error, 0.0);
    EXPECT_GE(result.max_iv_error, 0.0);

    // Logical constraints
    EXPECT_LE(result.mean_iv_error, result.p95_iv_error);
    EXPECT_LE(result.p95_iv_error, result.p99_iv_error);
    EXPECT_LE(result.p99_iv_error, result.max_iv_error);

    // Coverage fractions should be between 0 and 1
    EXPECT_GE(result.fraction_below_1bp, 0.0);
    EXPECT_LE(result.fraction_below_1bp, 1.0);
    EXPECT_GE(result.fraction_below_5bp, 0.0);
    EXPECT_LE(result.fraction_below_5bp, 1.0);
    EXPECT_GE(result.fraction_below_10bp, 0.0);
    EXPECT_LE(result.fraction_below_10bp, 1.0);

    // More restrictive thresholds should have lower fractions
    EXPECT_LE(result.fraction_below_1bp, result.fraction_below_5bp);
    EXPECT_LE(result.fraction_below_5bp, result.fraction_below_10bp);

    validation_result_free(&result);
    price_table_destroy(table);
}

// Test refinement point identification
TEST_F(AdaptiveAccuracyTest, RefinementPointSelection) {
    // Create coarse grid
    auto m_grid = create_coarse_grid();
    std::vector<double> tau_grid = {0.25, 0.5};
    std::vector<double> sigma_grid = {0.20, 0.25};
    std::vector<double> rate_grid = {0.05};

    OptionPriceTable *table = price_table_create_ex(
        m_grid.data(), m_grid.size(),
        tau_grid.data(), tau_grid.size(),
        sigma_grid.data(), sigma_grid.size(),
        rate_grid.data(), rate_grid.size(),
        nullptr, 0,
        OPTION_PUT, AMERICAN,
        COORD_RAW,
        LAYOUT_M_INNER
    );
    ASSERT_NE(table, nullptr);

    // Precompute
    price_table_precompute(table, &default_grid);

    // Validate to get high-error points
    ValidationResult result = validate_interpolation_error(
        table, &default_grid, 100, 1.0
    );

    // Identify refinement points
    size_t n_new = 0;
    double *new_points = identify_refinement_points(&result, table, &n_new);

    if (new_points != nullptr) {
        // Should identify some refinement points
        EXPECT_GT(n_new, 0) << "Should identify refinement points";
        EXPECT_LE(n_new, table->n_moneyness)
            << "Should not exceed grid size (max 2x growth)";

        // New points should be in valid range
        for (size_t i = 0; i < n_new; i++) {
            EXPECT_GT(new_points[i], 0.0);
            EXPECT_LT(new_points[i], 2.0) << "Moneyness should be reasonable";
        }

        // New points should be sorted
        for (size_t i = 1; i < n_new; i++) {
            EXPECT_GT(new_points[i], new_points[i-1])
                << "Refinement points should be sorted";
        }

        free(new_points);
    }

    validation_result_free(&result);
    price_table_destroy(table);
}

// Test adaptive refinement convergence
TEST_F(AdaptiveAccuracyTest, AdaptiveConvergence) {
    // Create coarse grid
    auto m_grid = create_coarse_grid();
    std::vector<double> tau_grid = {0.25, 0.5};
    std::vector<double> sigma_grid = {0.20, 0.25};
    std::vector<double> rate_grid = {0.05};

    OptionPriceTable *table = price_table_create_ex(
        m_grid.data(), m_grid.size(),
        tau_grid.data(), tau_grid.size(),
        sigma_grid.data(), sigma_grid.size(),
        rate_grid.data(), rate_grid.size(),
        nullptr, 0,
        OPTION_PUT, AMERICAN,
        COORD_RAW,
        LAYOUT_M_INNER
    );
    ASSERT_NE(table, nullptr);

    size_t initial_size = table->n_moneyness;

    // Run adaptive refinement with achievable target
    int status = price_table_precompute_adaptive(
        table, &default_grid,
        10.0,  // 10bp target (achievable)
        5,     // Max 5 iterations
        100    // 100 samples
    );

    // Should succeed (0) or reach partial success (1)
    EXPECT_TRUE(status == 0 || status == 1)
        << "Adaptive refinement should not fail";

    // Grid should have grown
    EXPECT_GT(table->n_moneyness, initial_size)
        << "Grid should expand during refinement";

    // Grid should not explode
    EXPECT_LT(table->n_moneyness, initial_size * 3)
        << "Grid should not grow more than 3x";

    // Final validation
    ValidationResult result = validate_interpolation_error(
        table, &default_grid, 100, 10.0
    );

    if (status == 0) {
        // If converged, should meet target
        EXPECT_LT(result.p95_iv_error, 10.0)
            << "Converged solution should meet target";
        EXPECT_GT(result.fraction_below_10bp, 0.95)
            << "Converged solution should have good coverage";
    }

    validation_result_free(&result);
    price_table_destroy(table);
}
