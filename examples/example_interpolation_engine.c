#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/iv_surface.h"
#include "../src/price_table.h"

/**
 * Example: Interpolation Engine Demonstration
 *
 * This example demonstrates the interpolation-based option pricing engine:
 * 1. IV Surface: 2D interpolation for implied volatility
 * 2. Price Table: 4D interpolation for option prices
 *
 * Phase 1 functionality (pre-computation will be added in Phase 2)
 */

// Helper: Generate log-spaced grid
static void generate_log_spaced_grid(double *grid, size_t n, double min, double max) {
    double log_min = log(min);
    double log_max = log(max);
    for (size_t i = 0; i < n; i++) {
        double t = (double)i / (double)(n - 1);
        grid[i] = exp(log_min + t * (log_max - log_min));
    }
}

// Helper: Generate linear-spaced grid
static void generate_linear_grid(double *grid, size_t n, double min, double max) {
    for (size_t i = 0; i < n; i++) {
        double t = (double)i / (double)(n - 1);
        grid[i] = min + t * (max - min);
    }
}

int main(void) {
    printf("========================================\n");
    printf("Interpolation Engine Example\n");
    printf("========================================\n\n");

    // ========================================
    // Part 1: IV Surface (2D Interpolation)
    // ========================================

    printf("Part 1: IV Surface (2D Interpolation)\n");
    printf("--------------------------------------\n");

    // Create grids for IV surface
    const size_t n_moneyness = 10;
    const size_t n_maturity = 8;

    double *moneyness = malloc(n_moneyness * sizeof(double));
    double *maturity = malloc(n_maturity * sizeof(double));

    generate_log_spaced_grid(moneyness, n_moneyness, 0.8, 1.2);
    generate_linear_grid(maturity, n_maturity, 0.1, 2.0);

    // Create IV surface
    IVSurface *surface = iv_surface_create(moneyness, n_moneyness, maturity, n_maturity);
    if (!surface) {
        fprintf(stderr, "Failed to create IV surface\n");
        return 1;
    }

    iv_surface_set_underlying(surface, "SPX");

    // Populate with synthetic IV data (ATM volatility smile)
    // Simple model: IV = 0.20 + 0.1 * (m - 1)^2 - 0.05 * tau
    for (size_t i_m = 0; i_m < n_moneyness; i_m++) {
        for (size_t i_tau = 0; i_tau < n_maturity; i_tau++) {
            double m = moneyness[i_m];
            double tau = maturity[i_tau];
            double iv = 0.20 + 0.1 * pow(m - 1.0, 2.0) - 0.02 * tau;
            iv_surface_set_point(surface, i_m, i_tau, iv);
        }
    }

    printf("Created IV surface: %s\n", iv_surface_get_underlying(surface));
    printf("Grid: %zu moneyness × %zu maturity = %zu points\n",
           n_moneyness, n_maturity, n_moneyness * n_maturity);
    printf("Memory: ~%.2f KB\n\n", (n_moneyness * n_maturity * sizeof(double)) / 1024.0);

    // Query interpolated values
    printf("Sample IV queries:\n");
    double test_queries[][2] = {
        {1.00, 0.25},  // ATM, 3 months
        {1.05, 0.50},  // 5% OTM, 6 months
        {0.95, 1.00},  // 5% ITM, 1 year
        {1.10, 0.10},  // 10% OTM, ~1 month
    };

    for (size_t i = 0; i < sizeof(test_queries) / sizeof(test_queries[0]); i++) {
        double m = test_queries[i][0];
        double tau = test_queries[i][1];
        double iv = iv_surface_interpolate(surface, m, tau);
        printf("  m=%.2f, τ=%.2f years → IV = %.4f (%.2f%%)\n",
               m, tau, iv, iv * 100);
    }

    // Save to file
    if (iv_surface_save(surface, "test_iv_surface.bin") == 0) {
        printf("\nSaved IV surface to test_iv_surface.bin\n");
    }

    iv_surface_destroy(surface);

    // ========================================
    // Part 2: Price Table (4D Interpolation)
    // ========================================

    printf("\n\nPart 2: Price Table (4D Interpolation)\n");
    printf("--------------------------------------\n");

    // Create grids for price table
    const size_t n_m = 10;
    const size_t n_tau = 6;
    const size_t n_sigma = 5;
    const size_t n_r = 3;

    double *m_grid = malloc(n_m * sizeof(double));
    double *tau_grid = malloc(n_tau * sizeof(double));
    double *sigma_grid = malloc(n_sigma * sizeof(double));
    double *r_grid = malloc(n_r * sizeof(double));

    generate_log_spaced_grid(m_grid, n_m, 0.8, 1.2);
    generate_linear_grid(tau_grid, n_tau, 0.1, 2.0);
    generate_linear_grid(sigma_grid, n_sigma, 0.1, 0.5);
    generate_linear_grid(r_grid, n_r, 0.0, 0.1);

    // Create price table
    OptionPriceTable *table = price_table_create(
        m_grid, n_m,
        tau_grid, n_tau,
        sigma_grid, n_sigma,
        r_grid, n_r,
        NULL, 0,  // No dividend dimension (4D mode)
        OPTION_PUT, AMERICAN);

    if (!table) {
        fprintf(stderr, "Failed to create price table\n");
        return 1;
    }

    price_table_set_underlying(table, "SPX");

    size_t total_points = n_m * n_tau * n_sigma * n_r;
    printf("Created price table: %s\n", price_table_get_underlying(table));
    printf("Grid: %zu×%zu×%zu×%zu = %zu points\n",
           n_m, n_tau, n_sigma, n_r, total_points);
    printf("Memory: ~%.2f MB\n", (total_points * sizeof(double)) / (1024.0 * 1024.0));
    printf("Option: American Put\n\n");

    // Populate with synthetic prices (simplified Black-Scholes-like formula)
    // Note: In Phase 2, this will be done via price_table_precompute()
    printf("Populating price table with synthetic data...\n");
    for (size_t i_m = 0; i_m < n_m; i_m++) {
        for (size_t i_tau = 0; i_tau < n_tau; i_tau++) {
            for (size_t i_sigma = 0; i_sigma < n_sigma; i_sigma++) {
                for (size_t i_r = 0; i_r < n_r; i_r++) {
                    double m = m_grid[i_m];
                    double tau = tau_grid[i_tau];
                    double sigma = sigma_grid[i_sigma];
                    double r = r_grid[i_r];

                    // Simplified put price approximation
                    // Real values will come from FDM in Phase 2
                    double K = 100.0;
                    double S = m * K;
                    double intrinsic = fmax(K - S, 0.0);
                    double time_value = K * sigma * sqrt(tau) * 0.4;
                    double price = intrinsic + time_value * exp(-r * tau);

                    price_table_set(table, i_m, i_tau, i_sigma, i_r, 0, price);
                }
            }
        }
    }
    printf("Done!\n\n");

    // Query interpolated prices
    printf("Sample price queries:\n");
    double price_queries[][4] = {
        {1.00, 0.25, 0.20, 0.05},  // ATM, 3 months, 20% IV, 5% rate
        {1.05, 0.50, 0.25, 0.05},  // 5% OTM, 6 months, 25% IV, 5% rate
        {0.95, 1.00, 0.30, 0.03},  // 5% ITM, 1 year, 30% IV, 3% rate
    };

    for (size_t i = 0; i < sizeof(price_queries) / sizeof(price_queries[0]); i++) {
        double m = price_queries[i][0];
        double tau = price_queries[i][1];
        double sigma = price_queries[i][2];
        double r = price_queries[i][3];
        double price = price_table_interpolate_4d(table, m, tau, sigma, r);

        printf("  m=%.2f, τ=%.2f, σ=%.2f, r=%.2f → Price = $%.4f\n",
               m, tau, sigma, r, price);
    }

    // Compute Greeks
    printf("\nGreeks calculation (via finite differences):\n");
    double m_test = 1.00;
    double tau_test = 0.5;
    double sigma_test = 0.25;
    double r_test = 0.05;

    OptionGreeks greeks = price_table_greeks_4d(table, m_test, tau_test, sigma_test, r_test);

    printf("  At m=%.2f, τ=%.2f, σ=%.2f, r=%.2f:\n",
           m_test, tau_test, sigma_test, r_test);
    printf("    Delta:  %.6f\n", greeks.delta);
    printf("    Gamma:  %.6f\n", greeks.gamma);
    printf("    Vega:   %.6f\n", greeks.vega);
    printf("    Theta:  %.6f\n", greeks.theta);
    printf("    Rho:    %.6f\n", greeks.rho);

    // Save to file
    if (price_table_save(table, "test_price_table.bin") == 0) {
        printf("\nSaved price table to test_price_table.bin\n");
    }

    // ========================================
    // Part 3: Load and Verify
    // ========================================

    printf("\n\nPart 3: Load and Verify\n");
    printf("--------------------------------------\n");

    // Load IV surface
    IVSurface *loaded_surface = iv_surface_load("test_iv_surface.bin");
    if (loaded_surface) {
        printf("Successfully loaded IV surface from file\n");
        double iv_check = iv_surface_interpolate(loaded_surface, 1.00, 0.25);
        printf("  Verification: m=1.00, τ=0.25 → IV = %.4f\n", iv_check);
        iv_surface_destroy(loaded_surface);
    }

    // Load price table
    OptionPriceTable *loaded_table = price_table_load("test_price_table.bin");
    if (loaded_table) {
        printf("Successfully loaded price table from file\n");
        double price_check = price_table_interpolate_4d(loaded_table, 1.00, 0.25, 0.20, 0.05);
        printf("  Verification: m=1.00, τ=0.25, σ=0.20, r=0.05 → Price = $%.4f\n", price_check);
        price_table_destroy(loaded_table);
    }

    // Cleanup
    price_table_destroy(table);
    free(m_grid);
    free(tau_grid);
    free(sigma_grid);
    free(r_grid);

    printf("\n========================================\n");
    printf("Example completed successfully!\n");
    printf("========================================\n");
    printf("\nNext steps (Phase 2):\n");
    printf("  - Implement price_table_precompute() with FDM solver\n");
    printf("  - Add OpenMP parallelization for batch computation\n");
    printf("  - Benchmark interpolation vs FDM performance\n");

    return 0;
}
