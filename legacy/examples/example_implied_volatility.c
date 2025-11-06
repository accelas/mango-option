#include "src/implied_volatility.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    printf("American Option Implied Volatility Example\n");
    printf("===========================================\n\n");

    // Example 1: ATM American put
    printf("Example 1: At-The-Money American Put\n");
    printf("-------------------------------------\n");
    IVParams params1 = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 6.08,
        .option_type = OPTION_PUT,
        .exercise_type = AMERICAN
    };

    printf("  Spot: %.2f\n", params1.spot_price);
    printf("  Strike: %.2f\n", params1.strike);
    printf("  Maturity: %.2f years\n", params1.time_to_maturity);
    printf("  Rate: %.4f\n", params1.risk_free_rate);
    printf("  Market Price: %.4f\n\n", params1.market_price);

    IVResult result1 = calculate_iv_simple(&params1, NULL);

    if (result1.converged) {
        printf("SUCCESS!\n");
        printf("  Implied Volatility: %.4f (%.2f%%)\n",
               result1.implied_vol, result1.implied_vol * 100);
        printf("  Iterations: %d\n\n", result1.iterations);
    } else {
        printf("FAILED: %s\n\n", result1.error);
    }

    // Example 2: OTM American call
    printf("Example 2: Out-of-The-Money American Call\n");
    printf("------------------------------------------\n");
    IVParams params2 = {
        .spot_price = 100.0,
        .strike = 110.0,
        .time_to_maturity = 0.5,
        .risk_free_rate = 0.03,
        .dividend_yield = 0.0,
        .market_price = 3.0,
        .option_type = OPTION_CALL,
        .exercise_type = AMERICAN
    };

    printf("  Spot: %.2f\n", params2.spot_price);
    printf("  Strike: %.2f\n", params2.strike);
    printf("  Maturity: %.2f years\n", params2.time_to_maturity);
    printf("  Rate: %.4f\n", params2.risk_free_rate);
    printf("  Market Price: %.4f\n\n", params2.market_price);

    IVResult result2 = calculate_iv_simple(&params2, NULL);

    if (result2.converged) {
        printf("SUCCESS!\n");
        printf("  Implied Volatility: %.4f (%.2f%%)\n",
               result2.implied_vol, result2.implied_vol * 100);
        printf("  Iterations: %d\n\n", result2.iterations);
    } else {
        printf("FAILED: %s\n\n", result2.error);
    }

    // Example 3: ITM American put
    printf("Example 3: In-The-Money American Put\n");
    printf("-------------------------------------\n");
    IVParams params3 = {
        .spot_price = 100.0,
        .strike = 110.0,
        .time_to_maturity = 0.25,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 11.0,
        .option_type = OPTION_PUT,
        .exercise_type = AMERICAN
    };

    printf("  Spot: %.2f\n", params3.spot_price);
    printf("  Strike: %.2f\n", params3.strike);
    printf("  Maturity: %.2f years\n", params3.time_to_maturity);
    printf("  Rate: %.4f\n", params3.risk_free_rate);
    printf("  Market Price: %.4f\n\n", params3.market_price);

    IVResult result3 = calculate_iv_simple(&params3, NULL);

    if (result3.converged) {
        printf("SUCCESS!\n");
        printf("  Implied Volatility: %.4f (%.2f%%)\n",
               result3.implied_vol, result3.implied_vol * 100);
        printf("  Iterations: %d\n\n", result3.iterations);
    } else {
        printf("FAILED: %s\n\n", result3.error);
    }

    // Example 4: Error case - price below intrinsic
    printf("Example 4: Error Handling (Below Intrinsic)\n");
    printf("--------------------------------------------\n");
    IVParams params4 = {
        .spot_price = 100.0,
        .strike = 110.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 5.0,  // Below intrinsic (10)
        .option_type = OPTION_PUT,
        .exercise_type = AMERICAN
    };

    printf("  Spot: %.2f\n", params4.spot_price);
    printf("  Strike: %.2f\n", params4.strike);
    printf("  Maturity: %.2f years\n", params4.time_to_maturity);
    printf("  Rate: %.4f\n", params4.risk_free_rate);
    printf("  Market Price: %.4f\n\n", params4.market_price);

    IVResult result4 = calculate_iv_simple(&params4, NULL);

    if (result4.converged) {
        printf("UNEXPECTED: Converged to %.2f%%\n\n", result4.implied_vol * 100);
    } else {
        printf("EXPECTED ERROR: %s\n\n", result4.error);
    }

    printf("Note: This uses FDM-based calculation (~250ms per IV).\n");
    printf("For production, use interpolation-based IV (~7.5µs).\n");
    printf("\nAll examples completed!\n");

    return 0;
}
