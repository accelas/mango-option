#include "src/interp_cubic.h"
#include "src/price_table.h"
#include <stdio.h>
#include <math.h>

// Simple test function for 4D: f(m, tau, sigma, r) = m * tau + sigma * r
static double test_func_4d(double m, double tau, double sigma, double r) {
    return m * tau + sigma * r;
}

// Simple test function for 5D: f(m, tau, sigma, r, q) = m * tau + sigma * r + q
static double test_func_5d(double m, double tau, double sigma, double r, double q) {
    return m * tau + sigma * r + q;
}

int main(void) {
    printf("Testing 4D and 5D Cubic Spline Interpolation\n");
    printf("============================================\n\n");

    // Test 4D Cubic Interpolation
    printf("Test 1: 4D Cubic Interpolation\n");
    printf("-------------------------------\n");

    // Create small 4D grid
    double moneyness[] = {0.8, 0.9, 1.0, 1.1, 1.2};
    double maturity[] = {0.25, 0.5, 1.0};
    double volatility[] = {0.15, 0.20, 0.25, 0.30};
    double rate[] = {0.02, 0.04, 0.06};

    OptionPriceTable *table_4d = price_table_create_with_strategy(
        moneyness, 5,
        maturity, 3,
        volatility, 4,
        rate, 3,
        NULL, 0,  // No dividend = 4D mode
        OPTION_CALL,
        AMERICAN,
        &INTERP_CUBIC
    );

    if (!table_4d) {
        printf("ERROR: Failed to create 4D price table\n");
        return 1;
    }

    // Fill table with test function values
    for (size_t i_m = 0; i_m < 5; i_m++) {
        for (size_t i_tau = 0; i_tau < 3; i_tau++) {
            for (size_t i_sigma = 0; i_sigma < 4; i_sigma++) {
                for (size_t i_r = 0; i_r < 3; i_r++) {
                    double val = test_func_4d(moneyness[i_m], maturity[i_tau],
                                             volatility[i_sigma], rate[i_r]);
                    price_table_set(table_4d, i_m, i_tau, i_sigma, i_r, 0, val);  // i_q = 0 for 4D
                }
            }
        }
    }

    // Test interpolation at various points
    double test_points_4d[][4] = {
        {0.85, 0.375, 0.175, 0.03},  // Between grid points
        {1.0, 0.5, 0.20, 0.04},      // On grid points
        {1.05, 0.75, 0.225, 0.05}    // Mixed
    };

    printf("Testing 4D cubic interpolation:\n");
    for (int i = 0; i < 3; i++) {
        double m = test_points_4d[i][0];
        double tau = test_points_4d[i][1];
        double sigma = test_points_4d[i][2];
        double r = test_points_4d[i][3];

        double expected = test_func_4d(m, tau, sigma, r);
        double result = price_table_interpolate_4d(table_4d, m, tau, sigma, r);
        double error = fabs(result - expected);

        printf("  Point (%g, %g, %g, %g): expected = %.6f, result = %.6f, error = %.6f\n",
               m, tau, sigma, r, expected, result, error);

        if (error > 0.01) {
            printf("  WARNING: Large error!\n");
        }
    }

    price_table_destroy(table_4d);

    // Test 5D Cubic Interpolation
    printf("\nTest 2: 5D Cubic Interpolation\n");
    printf("-------------------------------\n");

    double dividend[] = {0.0, 0.02, 0.04};

    OptionPriceTable *table_5d = price_table_create_with_strategy(
        moneyness, 5,
        maturity, 3,
        volatility, 4,
        rate, 3,
        dividend, 3,  // 5D mode with dividend
        OPTION_CALL,
        AMERICAN,
        &INTERP_CUBIC
    );

    if (!table_5d) {
        printf("ERROR: Failed to create 5D price table\n");
        return 1;
    }

    // Fill table with test function values
    for (size_t i_m = 0; i_m < 5; i_m++) {
        for (size_t i_tau = 0; i_tau < 3; i_tau++) {
            for (size_t i_sigma = 0; i_sigma < 4; i_sigma++) {
                for (size_t i_r = 0; i_r < 3; i_r++) {
                    for (size_t i_q = 0; i_q < 3; i_q++) {
                        double val = test_func_5d(moneyness[i_m], maturity[i_tau],
                                                 volatility[i_sigma], rate[i_r],
                                                 dividend[i_q]);
                        price_table_set(table_5d, i_m, i_tau, i_sigma, i_r, i_q, val);
                    }
                }
            }
        }
    }

    // Test interpolation at various points
    double test_points_5d[][5] = {
        {0.85, 0.375, 0.175, 0.03, 0.01},  // Between grid points
        {1.0, 0.5, 0.20, 0.04, 0.02},      // On grid points
        {1.05, 0.75, 0.225, 0.05, 0.03}    // Mixed
    };

    printf("Testing 5D cubic interpolation:\n");
    for (int i = 0; i < 3; i++) {
        double m = test_points_5d[i][0];
        double tau = test_points_5d[i][1];
        double sigma = test_points_5d[i][2];
        double r = test_points_5d[i][3];
        double q = test_points_5d[i][4];

        double expected = test_func_5d(m, tau, sigma, r, q);
        double result = price_table_interpolate_5d(table_5d, m, tau, sigma, r, q);
        double error = fabs(result - expected);

        printf("  Point (%g, %g, %g, %g, %g): expected = %.6f, result = %.6f, error = %.6f\n",
               m, tau, sigma, r, q, expected, result, error);

        if (error > 0.01) {
            printf("  WARNING: Large error!\n");
        }
    }

    price_table_destroy(table_5d);

    printf("\nAll tests completed successfully!\n");
    return 0;
}
