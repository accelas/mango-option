#include <stdio.h>
#include "../src/american_option.h"

int main() {
    printf("=== American Put Option with Discrete Dividends ===\n\n");

    // Option parameters: American put with discrete dividends
    // Stock pays $2 dividend at 0.25 years and $2 at 0.5 years
    double dividend_times[] = {0.25, 0.5};
    double dividend_amounts[] = {2.0, 2.0};

    OptionData option = {
        .strike = 100.0,
        .risk_free_rate = 0.05,
        .volatility = 0.25,
        .time_to_maturity = 1.0,
        .option_type = OPTION_PUT,
        .n_dividends = 2,
        .dividend_times = dividend_times,
        .dividend_amounts = dividend_amounts
    };

    // Grid parameters for PDE solver (use same as working example)
    AmericanOptionGrid grid = {
        .x_min = -0.7,         // ln(0.5) ≈ -0.69
        .x_max = 0.7,          // ln(2.0) ≈ 0.69
        .n_points = 141,       // Fine grid for accuracy
        .dt = 0.001,           // Small time step for stability
        .n_steps = 1000        // Total steps to maturity
    };

    printf("Option Details:\n");
    printf("  Type: American Put\n");
    printf("  Strike: %.2f\n", option.strike);
    printf("  Spot: %.2f\n", option.strike);  // At-the-money
    printf("  Time to maturity: %.2f years\n", option.time_to_maturity);
    printf("  Risk-free rate: %.2f%%\n", option.risk_free_rate * 100);
    printf("  Volatility: %.2f%%\n", option.volatility * 100);
    printf("  Dividends: $%.2f at %.2f years, $%.2f at %.2f years\n",
           dividend_amounts[0], dividend_times[0],
           dividend_amounts[1], dividend_times[1]);
    printf("\n");

    // Price the option with dividends
    printf("Computing option value with dividends...\n");
    AmericanOptionResult result = american_option_price(&option, &grid);

    if (result.status != 0 || result.solver == NULL) {
        printf("Error: Failed to solve PDE (status=%d)\n", result.status);
        return 1;
    }

    // Get option value at current spot price (at-the-money)
    double value_with_div = american_option_get_value_at_spot(
        result.solver, option.strike, option.strike);

    printf("American put value with dividends: $%.4f\n\n", value_with_div);

    // Compare with no-dividend case
    printf("Computing option value without dividends for comparison...\n");
    OptionData option_no_div = option;
    option_no_div.n_dividends = 0;
    option_no_div.dividend_times = NULL;
    option_no_div.dividend_amounts = NULL;

    AmericanOptionResult result_no_div = american_option_price(&option_no_div, &grid);

    if (result_no_div.status != 0 || result_no_div.solver == NULL) {
        printf("Error: Failed to solve no-dividend case\n");
        american_option_free_result(&result);
        return 1;
    }

    double value_no_div = american_option_get_value_at_spot(
        result_no_div.solver, option.strike, option.strike);

    printf("American put value without dividends: $%.4f\n\n", value_no_div);

    // Show the difference
    double diff = value_with_div - value_no_div;
    double pct_diff = (diff / value_no_div) * 100.0;

    printf("Impact of dividends:\n");
    printf("  Absolute difference: $%.4f\n", diff);
    printf("  Percentage difference: %.2f%%\n", pct_diff);
    printf("\n");

    // Explanation
    printf("Analysis:\n");
    printf("  For American puts, dividends decrease the stock price,\n");
    printf("  making the put more valuable. This is reflected in the\n");
    printf("  higher value when dividends are included.\n");
    printf("\n");
    printf("  The dividend feature is particularly important for\n");
    printf("  American options because early exercise is optimal\n");
    printf("  around ex-dividend dates.\n");

    // Cleanup
    american_option_free_result(&result);
    american_option_free_result(&result_no_div);

    return 0;
}
