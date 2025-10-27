#include "src/implied_volatility.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    printf("=== Implied Volatility Calculator Example ===\n\n");
    printf("NOTE: All examples use implied_volatility_calculate_simple() which\n");
    printf("      automatically determines sensible bounds for the volatility search.\n");
    printf("      Lower bound: 0.0001 (0.01%%), Upper bound: heuristic based on market price\n\n");

    // Example 1: ATM Call Option
    printf("Example 1: At-The-Money Call Option\n");
    printf("------------------------------------\n");

    IVParams atm_call = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,      // 1 year
        .risk_free_rate = 0.05,       // 5% risk-free rate
        .market_price = 10.45,        // Market price of the call
        .is_call = true
    };

    IVResult result1 = implied_volatility_calculate_simple(&atm_call);

    if (result1.converged) {
        printf("Market price: $%.2f\n", atm_call.market_price);
        printf("Implied volatility: %.2f%% (%.6f)\n",
               result1.implied_vol * 100.0, result1.implied_vol);
        printf("Vega: %.6f\n", result1.vega);
        printf("Iterations: %d\n", result1.iterations);

        // Verify by pricing back
        double verify_price = black_scholes_price(atm_call.spot_price,
                                                  atm_call.strike,
                                                  atm_call.time_to_maturity,
                                                  atm_call.risk_free_rate,
                                                  result1.implied_vol,
                                                  atm_call.is_call);
        printf("Verification: BS price = $%.6f (diff: %.2e)\n\n",
               verify_price, fabs(verify_price - atm_call.market_price));
    } else {
        printf("Failed to converge: %s\n\n", result1.error);
    }

    // Example 2: OTM Put Option
    printf("Example 2: Out-of-The-Money Put Option\n");
    printf("---------------------------------------\n");

    IVParams otm_put = {
        .spot_price = 100.0,
        .strike = 95.0,               // Strike below spot
        .time_to_maturity = 0.25,     // 3 months
        .risk_free_rate = 0.05,
        .market_price = 2.50,
        .is_call = false
    };

    IVResult result2 = implied_volatility_calculate_simple(&otm_put);

    if (result2.converged) {
        printf("Market price: $%.2f\n", otm_put.market_price);
        printf("Implied volatility: %.2f%% (%.6f)\n",
               result2.implied_vol * 100.0, result2.implied_vol);
        printf("Vega: %.6f\n", result2.vega);
        printf("Iterations: %d\n", result2.iterations);

        double verify_price = black_scholes_price(otm_put.spot_price,
                                                  otm_put.strike,
                                                  otm_put.time_to_maturity,
                                                  otm_put.risk_free_rate,
                                                  result2.implied_vol,
                                                  otm_put.is_call);
        printf("Verification: BS price = $%.6f (diff: %.2e)\n\n",
               verify_price, fabs(verify_price - otm_put.market_price));
    } else {
        printf("Failed to converge: %s\n\n", result2.error);
    }

    // Example 3: ITM Call with high volatility
    printf("Example 3: In-The-Money Call (High Volatility)\n");
    printf("----------------------------------------------\n");

    IVParams itm_call = {
        .spot_price = 100.0,
        .strike = 90.0,               // Strike well below spot
        .time_to_maturity = 0.5,      // 6 months
        .risk_free_rate = 0.05,
        .market_price = 15.0,
        .is_call = true
    };

    IVResult result3 = implied_volatility_calculate_simple(&itm_call);

    if (result3.converged) {
        printf("Market price: $%.2f\n", itm_call.market_price);
        printf("Implied volatility: %.2f%% (%.6f)\n",
               result3.implied_vol * 100.0, result3.implied_vol);
        printf("Vega: %.6f\n", result3.vega);
        printf("Iterations: %d\n", result3.iterations);

        double verify_price = black_scholes_price(itm_call.spot_price,
                                                  itm_call.strike,
                                                  itm_call.time_to_maturity,
                                                  itm_call.risk_free_rate,
                                                  result3.implied_vol,
                                                  itm_call.is_call);
        printf("Verification: BS price = $%.6f (diff: %.2e)\n\n",
               verify_price, fabs(verify_price - itm_call.market_price));
    } else {
        printf("Failed to converge: %s\n\n", result3.error);
    }

    // Example 4: Error case - price below intrinsic value
    printf("Example 4: Error Handling (Price Below Intrinsic)\n");
    printf("-------------------------------------------------\n");

    IVParams invalid = {
        .spot_price = 100.0,
        .strike = 90.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .market_price = 5.0,          // Too low (intrinsic is ~10)
        .is_call = true
    };

    IVResult result4 = implied_volatility_calculate_simple(&invalid);

    if (!result4.converged && result4.error != nullptr) {
        printf("Expected error detected: %s\n", result4.error);
    } else {
        printf("Unexpected: converged to IV = %.2f%%\n", result4.implied_vol * 100.0);
    }

    printf("\nAll examples completed!\n");
    return 0;
}
