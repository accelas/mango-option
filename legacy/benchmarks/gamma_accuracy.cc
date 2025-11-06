// Gamma Interpolation Accuracy Comparison
// Compares FDM-computed gamma vs interpolated gamma from precomputed table

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>

extern "C" {
#include "src/price_table.h"
#include "src/american_option.h"
}

// Compute gamma using finite differences
double compute_gamma_fdm(double spot, double strike, double volatility,
                        double rate, double maturity, bool is_put) {
    const double h = 0.01 * spot;  // 1% of spot

    OptionData option_up = {
        .strike = strike,
        .volatility = volatility,
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

    // Compute prices at S+h, S, and S-h
    AmericanOptionResult result_center = american_option_price(&option_up, &grid);
    if (result_center.status != 0) return NAN;

    double price_center = american_option_get_value_at_spot(result_center.solver, spot, strike);
    double price_up = american_option_get_value_at_spot(result_center.solver, spot + h, strike);
    double price_down = american_option_get_value_at_spot(result_center.solver, spot - h, strike);

    american_option_free_result(&result_center);

    // Centered difference: γ = (V(S+h) - 2V(S) + V(S-h)) / h²
    return (price_up - 2*price_center + price_down) / (h * h);
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           GAMMA INTERPOLATION ACCURACY COMPARISON                      ║\n";
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

    // Precompute prices and gammas
    std::cout << "Precomputing prices and gammas...\n";
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

    price_table_build_interpolation(table);
    std::cout << "Precomputation complete.\n\n";

    // Test cases - sample points BETWEEN grid points
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
    std::cout << "║ Test Case              │ FDM Gamma │ Interp Gamma │ Abs Err │ Rel Err ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════╣\n";

    for (const auto& tc : test_cases) {
        double K_ref = 100.0;
        double spot = tc.m * K_ref;
        double strike = K_ref;

        double gamma_fdm = compute_gamma_fdm(spot, strike, tc.sigma, tc.r, tc.tau, true);
        double gamma_interp = price_table_interpolate_gamma_4d(table, tc.m, tc.tau, tc.sigma, tc.r);

        if (std::isnan(gamma_fdm) || std::isnan(gamma_interp)) {
            std::cout << "║ " << std::left << std::setw(22) << tc.name
                      << " │  FAILED   │   FAILED     │    -    │    -    ║\n";
            continue;
        }

        double abs_error = std::abs(gamma_interp - gamma_fdm);
        double rel_error = abs_error / std::abs(gamma_fdm);

        std::cout << "║ " << std::left << std::setw(22) << tc.name
                  << " │ " << std::right << std::setw(9) << gamma_fdm
                  << " │ " << std::setw(12) << gamma_interp
                  << " │ " << std::setw(7) << abs_error
                  << " │ " << std::setw(6) << std::setprecision(2) << (rel_error * 100.0) << "% ║\n";

        sum_abs_error += abs_error;
        sum_rel_error += rel_error;
        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);
        n_tests++;
    }

    std::cout << "╚════════════════════════════════════════════════════════════════════════╝\n\n";

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
                  << (max_rel_error * 100.0) << "%\n\n";

        std::cout << "╔════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                            CONCLUSION                                  ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════════════╝\n\n";

        double avg_rel_pct = sum_rel_error / n_tests * 100.0;
        if (avg_rel_pct < 1.0) {
            std::cout << "✓ EXCELLENT: Gamma interpolation has < 1% average error\n";
        } else if (avg_rel_pct < 5.0) {
            std::cout << "✓ GOOD: Gamma interpolation has < 5% average error\n";
        } else if (avg_rel_pct < 10.0) {
            std::cout << "⚠ ACCEPTABLE: Gamma interpolation has < 10% average error\n";
        } else {
            std::cout << "✗ POOR: Gamma interpolation has > 10% average error\n";
        }

        std::cout << "\nNote: Gamma is computed via finite differences.\n";
        std::cout << "      Errors arise from grid interpolation.\n\n";
    }

    price_table_destroy(table);
    return 0;
}
