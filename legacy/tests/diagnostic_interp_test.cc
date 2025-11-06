#include <gtest/gtest.h>
#include <cmath>

extern "C" {
#include "../src/price_table.h"
#include "../src/american_option.h"
}

// Diagnostic test: verify interpolation works with known analytic function
// This isolates interpolation from PDE solver
TEST(DiagnosticInterpTest, AnalyticFunctionInterpolation) {
    // Create coarse grid
    std::vector<double> m_grid = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3};
    std::vector<double> tau_grid = {0.25, 0.5, 1.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25};
    std::vector<double> rate_grid = {0.02, 0.05};

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

    // Fill table with known analytic function: V(m, τ, σ, r) = sin(log(m))
    // This function depends ONLY on moneyness - easy to verify
    for (size_t i_m = 0; i_m < table->n_moneyness; i_m++) {
        double m = table->moneyness_grid[i_m];
        double value = std::sin(std::log(m));

        for (size_t i_tau = 0; i_tau < table->n_maturity; i_tau++) {
            for (size_t i_sigma = 0; i_sigma < table->n_volatility; i_sigma++) {
                for (size_t i_r = 0; i_r < table->n_rate; i_r++) {
                    price_table_set(table, i_m, i_tau, i_sigma, i_r, 0, value);
                }
            }
        }
    }

    // Build interpolation
    int status = price_table_build_interpolation(table);
    ASSERT_EQ(status, 0) << "Interpolation build should succeed";

    // Test 1: Interpolate at grid points (should be exact)
    printf("\n=== Test 1: Grid Point Interpolation ===\n");
    for (size_t i = 0; i < m_grid.size(); i++) {
        double m = m_grid[i];
        double expected = std::sin(std::log(m));
        double actual = price_table_interpolate_4d(table, m, 0.5, 0.20, 0.02);

        printf("  m=%.2f: expected=%.6f, actual=%.6f, error=%.6f\n",
               m, expected, actual, std::abs(expected - actual));

        EXPECT_NEAR(actual, expected, 1e-10)
            << "Grid point interpolation should be exact at m=" << m;
    }

    // Test 2: Interpolate at midpoints (should be smooth)
    printf("\n=== Test 2: Midpoint Interpolation ===\n");
    for (size_t i = 0; i < m_grid.size() - 1; i++) {
        double m_left = m_grid[i];
        double m_right = m_grid[i + 1];
        double m_mid = std::sqrt(m_left * m_right);  // Geometric mean

        double expected = std::sin(std::log(m_mid));
        double actual = price_table_interpolate_4d(table, m_mid, 0.5, 0.20, 0.02);

        double error = std::abs(expected - actual);
        printf("  m=%.3f (between %.2f and %.2f): expected=%.6f, actual=%.6f, error=%.6f\n",
               m_mid, m_left, m_right, expected, actual, error);

        // Cubic spline should be accurate to ~1e-4 for smooth functions
        EXPECT_LT(error, 1e-3)
            << "Midpoint interpolation error too large at m=" << m_mid;
    }

    // Test 3: Grid expansion preserves interpolation quality
    printf("\n=== Test 3: Grid Expansion ===\n");

    // Save midpoint interpolation results before expansion
    std::vector<double> m_test = {0.75, 0.85, 0.95, 1.05, 1.15, 1.25};
    std::vector<double> values_before;

    for (double m : m_test) {
        double v = price_table_interpolate_4d(table, m, 0.5, 0.20, 0.02);
        values_before.push_back(v);
        printf("  Before expansion: m=%.2f, value=%.6f\n", m, v);
    }

    // Expand grid with new points
    std::vector<double> new_points = {0.75, 0.85, 0.95, 1.05, 1.15, 1.25};

    status = price_table_expand_grid(table, new_points.data(), new_points.size());
    ASSERT_EQ(status, 0) << "Grid expansion should succeed";

    printf("  Grid expanded: %zu → %zu points\n", m_grid.size(), table->n_moneyness);

    // Fill new grid points with same analytic function
    for (size_t i_m = 0; i_m < table->n_moneyness; i_m++) {
        double m = table->moneyness_grid[i_m];
        double value = std::sin(std::log(m));

        for (size_t i_tau = 0; i_tau < table->n_maturity; i_tau++) {
            for (size_t i_sigma = 0; i_sigma < table->n_volatility; i_sigma++) {
                for (size_t i_r = 0; i_r < table->n_rate; i_r++) {
                    double current = price_table_get(table, i_m, i_tau, i_sigma, i_r, 0);
                    if (std::isnan(current)) {
                        price_table_set(table, i_m, i_tau, i_sigma, i_r, 0, value);
                    }
                }
            }
        }
    }

    // Rebuild interpolation
    status = price_table_build_interpolation(table);
    ASSERT_EQ(status, 0) << "Interpolation rebuild should succeed";

    // Interpolate again at same test points
    printf("\n  After expansion:\n");
    for (size_t i = 0; i < m_test.size(); i++) {
        double m = m_test[i];
        double expected = std::sin(std::log(m));
        double actual = price_table_interpolate_4d(table, m, 0.5, 0.20, 0.02);
        double error = std::abs(expected - actual);

        printf("  m=%.2f: expected=%.6f, actual=%.6f, error=%.6f\n",
               m, expected, actual, error);

        // After refinement, interpolation should be MORE accurate, not worse
        EXPECT_LT(error, 1e-3) << "Interpolation quality degraded after expansion at m=" << m;
    }

    price_table_destroy(table);
}
