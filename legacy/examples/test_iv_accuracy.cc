#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

extern "C" {
#include "src/implied_volatility.h"
#include "src/american_option.h"
#include "src/price_table.h"
#include "src/grid_presets.h"
#include "src/grid_transform.h"
}

struct TestCase {
    double spot;
    double strike;
    double tau;        // Time to maturity
    double r;          // Risk-free rate
    double true_sigma; // True volatility to recover
    OptionType type;
    const char* label;
};

void run_accuracy_test() {
    std::cout << "IV Calculation Accuracy Test\n";
    std::cout << "============================\n\n";

    // Test cases covering different moneyness and maturity ranges
    std::vector<TestCase> test_cases = {
        // ATM options
        {100.0, 100.0, 0.25, 0.05, 0.20, OPTION_PUT, "ATM Put, 3M, 20% vol"},
        {100.0, 100.0, 1.00, 0.05, 0.20, OPTION_PUT, "ATM Put, 1Y, 20% vol"},
        {100.0, 100.0, 0.25, 0.05, 0.30, OPTION_CALL, "ATM Call, 3M, 30% vol"},

        // ITM options
        {100.0,  90.0, 0.50, 0.05, 0.25, OPTION_PUT, "ITM Put (S=100, K=90), 6M"},
        {100.0, 110.0, 0.50, 0.05, 0.25, OPTION_CALL, "ITM Call (S=100, K=110), 6M"},

        // OTM options
        {100.0, 110.0, 0.50, 0.05, 0.25, OPTION_PUT, "OTM Put (S=100, K=110), 6M"},
        {100.0,  90.0, 0.50, 0.05, 0.25, OPTION_CALL, "OTM Call (S=100, K=90), 6M"},

        // Short maturity
        {100.0, 100.0, 0.08, 0.05, 0.20, OPTION_PUT, "ATM Put, 1M, 20% vol"},

        // High volatility
        {100.0, 100.0, 0.50, 0.05, 0.50, OPTION_PUT, "ATM Put, 6M, 50% vol"},

        // Low volatility
        {100.0, 100.0, 0.50, 0.05, 0.10, OPTION_PUT, "ATM Put, 6M, 10% vol"},
    };

    // FDM grid for pricing
    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 141,
        .dt = 0.001,
        .n_steps = 1000
    };

    std::cout << "Method 1: FDM-based IV (Brent's method + PDE solver)\n";
    std::cout << "-------------------------------------------------------\n\n";

    double fdm_total_error = 0.0;
    double fdm_max_error = 0.0;
    int fdm_success = 0;

    for (const auto& tc : test_cases) {
        // Generate market price using true volatility
        OptionData option = {
            .strike = tc.strike,
            .volatility = tc.true_sigma,
            .risk_free_rate = tc.r,
            .time_to_maturity = tc.tau,
            .option_type = tc.type,
            .n_dividends = 0,
            .dividend_times = nullptr,
            .dividend_amounts = nullptr
        };

        AmericanOptionResult price_result = american_option_price(&option, &grid);
        if (price_result.status != 0) {
            std::cout << "  ✗ " << tc.label << " - Pricing failed\n";
            american_option_free_result(&price_result);
            continue;
        }

        double market_price = american_option_get_value_at_spot(
            price_result.solver, tc.spot, tc.strike);
        american_option_free_result(&price_result);

        // Now invert to recover IV
        IVParams iv_params = {
            .spot_price = tc.spot,
            .strike = tc.strike,
            .time_to_maturity = tc.tau,
            .risk_free_rate = tc.r,
            .dividend_yield = 0.0,
            .market_price = market_price,
            .option_type = tc.type,
            .exercise_type = AMERICAN
        };

        IVResult iv_result = calculate_iv(&iv_params, &grid, nullptr, 1e-6, 100);

        if (iv_result.converged) {
            double error_bp = std::abs(iv_result.implied_vol - tc.true_sigma) * 10000;
            fdm_total_error += error_bp;
            fdm_max_error = std::max(fdm_max_error, error_bp);
            fdm_success++;

            std::cout << "  ✓ " << std::setw(45) << std::left << tc.label
                      << " True: " << std::fixed << std::setprecision(4) << tc.true_sigma
                      << " → IV: " << iv_result.implied_vol
                      << " (err: " << std::setprecision(2) << error_bp << " bp)"
                      << " [" << iv_result.iterations << " iters]\n";
        } else {
            std::cout << "  ✗ " << tc.label << " - " << iv_result.error << "\n";
        }
    }

    std::cout << "\nFDM-based Summary:\n";
    std::cout << "  Success rate: " << fdm_success << "/" << test_cases.size() << "\n";
    if (fdm_success > 0) {
        std::cout << "  Average error: " << std::fixed << std::setprecision(2)
                  << (fdm_total_error / fdm_success) << " bp\n";
        std::cout << "  Max error: " << fdm_max_error << " bp\n";
    }

    // Method 2: Table-based IV (Newton's method with interpolation)
    std::cout << "\n\nMethod 2: Table-based IV (Newton's method + interpolation)\n";
    std::cout << "------------------------------------------------------------\n\n";

    // Create price table using Balanced preset
    GridConfig config = grid_preset_get(
        GRID_PRESET_ADAPTIVE_BALANCED,
        0.7, 1.3,    // moneyness
        0.027, 2.0,  // maturity
        0.10, 0.80,  // volatility
        0.0, 0.10,   // rate
        0.0, 0.0     // no dividend
    );

    GeneratedGrids grids = grid_generate_all(&config);

    // CRITICAL: Transform grids to LOG_SQRT coordinate system
    // Grids are generated in RAW coordinates, but price_table with COORD_LOG_SQRT
    // expects pre-transformed grids (moneyness → log, maturity → sqrt)
    grid_transform_coordinates(&grids, COORD_LOG_SQRT);

    OptionPriceTable* table = price_table_create_ex(
        grids.moneyness, grids.n_moneyness,
        grids.maturity, grids.n_maturity,
        grids.volatility, grids.n_volatility,
        grids.rate, grids.n_rate,
        nullptr, 0,
        OPTION_PUT, AMERICAN,
        COORD_LOG_SQRT,
        LAYOUT_M_INNER
    );

    if (!table) {
        std::cout << "Failed to create price table\n";
        grid_free_all(&grids);
        return;
    }

    std::cout << "Precomputing price table (this may take a minute)...\n";
    std::cout << "  Grid: " << grids.n_moneyness << "×" << grids.n_maturity << "×"
              << grids.n_volatility << "×" << grids.n_rate
              << " = " << grids.total_points << " points\n";

    int precompute_status = price_table_precompute(table, &grid);
    if (precompute_status != 0) {
        std::cout << "Precompute failed\n";
        price_table_destroy(table);
        grid_free_all(&grids);
        return;
    }
    std::cout << "Precompute complete!\n\n";

    double table_total_error = 0.0;
    double table_max_error = 0.0;
    int table_success = 0;
    int table_fallback = 0;

    for (const auto& tc : test_cases) {
        // Skip calls since table is for puts
        if (tc.type == OPTION_CALL) {
            continue;
        }

        // Generate market price
        OptionData option = {
            .strike = tc.strike,
            .volatility = tc.true_sigma,
            .risk_free_rate = tc.r,
            .time_to_maturity = tc.tau,
            .option_type = tc.type,
            .n_dividends = 0,
            .dividend_times = nullptr,
            .dividend_amounts = nullptr
        };

        AmericanOptionResult price_result = american_option_price(&option, &grid);
        if (price_result.status != 0) {
            std::cout << "  ✗ " << tc.label << " - Pricing failed\n";
            american_option_free_result(&price_result);
            continue;
        }

        double market_price = american_option_get_value_at_spot(
            price_result.solver, tc.spot, tc.strike);
        american_option_free_result(&price_result);

        // Invert using table
        IVParams iv_params = {
            .spot_price = tc.spot,
            .strike = tc.strike,
            .time_to_maturity = tc.tau,
            .risk_free_rate = tc.r,
            .dividend_yield = 0.0,
            .market_price = market_price,
            .option_type = tc.type,
            .exercise_type = AMERICAN
        };

        IVResult iv_result = calculate_iv(&iv_params, &grid, table, 1e-6, 100);

        if (iv_result.converged) {
            double error_bp = std::abs(iv_result.implied_vol - tc.true_sigma) * 10000;
            table_total_error += error_bp;
            table_max_error = std::max(table_max_error, error_bp);
            table_success++;

            // Check if it used Newton (few iters) or Brent fallback (many iters)
            const char* method = iv_result.iterations <= 10 ? "Newton" : "Brent";
            if (iv_result.iterations > 10) table_fallback++;

            std::cout << "  ✓ " << std::setw(45) << std::left << tc.label
                      << " True: " << std::fixed << std::setprecision(4) << tc.true_sigma
                      << " → IV: " << iv_result.implied_vol
                      << " (err: " << std::setprecision(2) << error_bp << " bp)"
                      << " [" << method << ", " << iv_result.iterations << " iters]\n";
        } else {
            std::cout << "  ✗ " << tc.label << " - " << iv_result.error << "\n";
        }
    }

    std::cout << "\nTable-based Summary:\n";
    std::cout << "  Success rate: " << table_success << "/" << test_cases.size() - 3 << " (puts only)\n";
    if (table_success > 0) {
        std::cout << "  Average error: " << std::fixed << std::setprecision(2)
                  << (table_total_error / table_success) << " bp\n";
        std::cout << "  Max error: " << table_max_error << " bp\n";
        std::cout << "  Newton method: " << (table_success - table_fallback) << "/" << table_success << "\n";
        std::cout << "  Brent fallback: " << table_fallback << "/" << table_success << "\n";
    }

    price_table_destroy(table);
    grid_free_all(&grids);
}

int main() {
    run_accuracy_test();
    return 0;
}
