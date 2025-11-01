// Vega Interpolation Accuracy Comparison
// Compares FDM-computed vega vs interpolated vega from precomputed table

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>

extern "C" {
#include "src/price_table.h"
#include "src/american_option.h"
}

// Compute vega using finite differences
double compute_vega_fdm(double spot, double strike, double volatility,
                        double rate, double maturity, bool is_put) {
    const double h = 0.01 * volatility;  // 1% perturbation

    OptionData option_up = {
        .strike = strike,
        .volatility = volatility + h,
        .risk_free_rate = rate,
        .time_to_maturity = maturity,
        .option_type = is_put ? OPTION_PUT : OPTION_CALL,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    OptionData option_down = {
        .strike = strike,
        .volatility = volatility - h,
        .risk_free_rate = rate,
        .time_to_maturity = maturity,
        .option_type = is_put ? OPTION_PUT : OPTION_CALL,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 101,
        .dt = 0.001,
        .n_steps = static_cast<size_t>(maturity / 0.001)
    };

    // Compute prices at sigma + h and sigma - h
    AmericanOptionResult result_up = american_option_price(&option_up, &grid);
    AmericanOptionResult result_down = american_option_price(&option_down, &grid);

    if (result_up.status != 0 || result_down.status != 0) {
        return NAN;
    }

    double price_up = american_option_get_value_at_spot(result_up.solver, spot, strike);
    double price_down = american_option_get_value_at_spot(result_down.solver, spot, strike);

    american_option_free_result(&result_up);
    american_option_free_result(&result_down);

    // Centered difference: dV/dσ = (V(σ+h) - V(σ-h)) / (2h)
    return (price_up - price_down) / (2.0 * h);
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           VEGA INTERPOLATION ACCURACY COMPARISON                       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    // Create price table with moderate grid
    std::vector<double> moneyness = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> maturity = {0.25, 0.5, 1.0, 1.5};
    std::vector<double> volatility = {0.15, 0.20, 0.25, 0.30, 0.40};
    std::vector<double> rate = {0.03, 0.05, 0.07};

    std::cout << "Creating price table with:\n";
    std::cout << "  Moneyness points: " << moneyness.size() << "\n";
    std::cout << "  Maturity points: " << maturity.size() << "\n";
    std::cout << "  Volatility points: " << volatility.size() << "\n";
    std::cout << "  Rate points: " << rate.size() << "\n";
    std::cout << "  Total grid points: " << (moneyness.size() * maturity.size() *
                                             volatility.size() * rate.size()) << "\n\n";

    OptionPriceTable *table = price_table_create_ex(
        moneyness.data(), moneyness.size(),
        maturity.data(), maturity.size(),
        volatility.data(), volatility.size(),
        rate.data(), rate.size(),
        nullptr, 0,
        OPTION_PUT, AMERICAN,
        COORD_RAW, LAYOUT_M_INNER);

    if (!table) {
        std::cerr << "Failed to create price table\n";
        return 1;
    }

    // Precompute prices and vegas
    std::cout << "Precomputing prices and vegas...\n";
    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 101,
        .dt = 0.001,
        .n_steps = 1000
    };

    int status = price_table_precompute(table, &grid);
    if (status != 0) {
        std::cerr << "Precomputation failed\n";
        price_table_destroy(table);
        return 1;
    }

    // Build interpolation structures
    price_table_build_interpolation(table);
    std::cout << "Precomputation complete.\n\n";

    // Test cases - sample points BETWEEN grid points for interpolation test
    struct TestCase {
        std::string name;
        double m;
        double tau;
        double sigma;
        double r;
    };

    std::vector<TestCase> test_cases = {
        {"ATM, Mid-term", 1.0, 0.75, 0.225, 0.05},
        {"OTM, Short-term", 1.15, 0.3, 0.175, 0.04},
        {"ITM, Long-term", 0.85, 1.25, 0.275, 0.06},
        {"ATM, Short-term, Low vol", 1.0, 0.4, 0.18, 0.04},
        {"Deep OTM, Mid-term", 1.18, 0.6, 0.22, 0.055},
    };

    // Results
    double sum_abs_error = 0.0;
    double sum_rel_error = 0.0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    int n_tests = 0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "╔════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                         ACCURACY RESULTS                               ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Test Case              │  FDM Vega │ Interp Vega │ Abs Err │ Rel Err  ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════╣\n";

    for (const auto& tc : test_cases) {
        // Compute vega via FDM at query point
        double K_ref = 100.0;
        double spot = tc.m * K_ref;
        double strike = K_ref;

        double vega_fdm = compute_vega_fdm(spot, strike, tc.sigma, tc.r, tc.tau, true);

        // Get interpolated vega
        double vega_interp = price_table_interpolate_vega_4d(table, tc.m, tc.tau, tc.sigma, tc.r);

        if (std::isnan(vega_fdm) || std::isnan(vega_interp)) {
            std::cout << "║ " << std::left << std::setw(22) << tc.name
                      << " │  FAILED   │   FAILED    │  -  │  -       ║\n";
            continue;
        }

        double abs_error = std::abs(vega_interp - vega_fdm);
        double rel_error = abs_error / std::abs(vega_fdm);

        std::cout << "║ " << std::left << std::setw(22) << tc.name
                  << " │ " << std::right << std::setw(9) << vega_fdm
                  << " │ " << std::setw(11) << vega_interp
                  << " │ " << std::setw(7) << abs_error
                  << " │ " << std::setw(7) << std::setprecision(2) << (rel_error * 100.0) << "%  ║\n";

        sum_abs_error += abs_error;
        sum_rel_error += rel_error;
        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);
        n_tests++;
    }

    std::cout << "╚════════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    if (n_tests > 0) {
        std::cout << "╔════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                         SUMMARY STATISTICS                             ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════════════╝\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Average absolute error: " << (sum_abs_error / n_tests) << "\n";
        std::cout << "  Average relative error: " << std::setprecision(2)
                  << (sum_rel_error / n_tests * 100.0) << "%\n";
        std::cout << "  Maximum absolute error: " << std::setprecision(4) << max_abs_error << "\n";
        std::cout << "  Maximum relative error: " << std::setprecision(2)
                  << (max_rel_error * 100.0) << "%\n";
        std::cout << "\n";

        std::cout << "╔════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                            CONCLUSION                                  ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";

        double avg_rel_pct = sum_rel_error / n_tests * 100.0;
        if (avg_rel_pct < 1.0) {
            std::cout << "✓ EXCELLENT: Vega interpolation has < 1% average error\n";
        } else if (avg_rel_pct < 5.0) {
            std::cout << "✓ GOOD: Vega interpolation has < 5% average error\n";
        } else if (avg_rel_pct < 10.0) {
            std::cout << "⚠ ACCEPTABLE: Vega interpolation has < 10% average error\n";
        } else {
            std::cout << "✗ POOR: Vega interpolation has > 10% average error\n";
        }

        std::cout << "\n";
        std::cout << "Note: This compares FDM-computed vega at query points vs\n";
        std::cout << "      interpolated vega from the precomputed table.\n";
        std::cout << "      Both methods use finite differences, so errors arise\n";
        std::cout << "      from interpolation between grid points.\n";
        std::cout << "\n";
    }

    price_table_destroy(table);

    return 0;
}
