#include "src/american_option.h"
#include <stdio.h>
#include <math.h>

// Utility: Print option values as a function of spot price
static void print_option_values(const PDESolver *solver, const OptionData *data,
                               const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == nullptr) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return;
    }

    const double *x = pde_solver_get_grid(solver);
    const double *V = pde_solver_get_solution(solver);
    const size_t n = solver->grid.n_points;
    const double K = data->strike;

    fprintf(fp, "# S (spot) | V(S,0) (option value) | Intrinsic Value\n");
    for (size_t i = 0; i < n; i++) {
        double S = K * exp(x[i]);
        double intrinsic;

        if (data->option_type == OPTION_CALL) {
            intrinsic = fmax(S - K, 0.0);
        } else {
            intrinsic = fmax(K - S, 0.0);
        }

        fprintf(fp, "%.6f %.10e %.10e\n", S, V[i], intrinsic);
    }

    fclose(fp);
    printf("Option values written to %s\n", filename);
}

int main(void) {
    printf("=== American Option Pricing with PDE Solver ===\n\n");

    // Grid parameters for pricing
    AmericanOptionGrid grid_params = {
        .x_min = -0.7,         // ln(0.5) ≈ -0.69
        .x_max = 0.7,          // ln(2.0) ≈ 0.69
        .n_points = 141,       // Fine grid for accuracy
        .dt = 0.001,           // Small time step for stability
        .n_steps = 1000        // Total steps to maturity
    };

    // Example 1: American Call Option (no dividends)
    printf("Example 1: American Call Option (no dividends)\n");
    printf("-----------------------------------------------\n");

    OptionData call_data = {
        .strike = 100.0,
        .volatility = 0.2,       // 20% volatility
        .risk_free_rate = 0.05,  // 5% risk-free rate
        .time_to_maturity = 1.0, // 1 year to maturity
        .option_type = OPTION_CALL,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    // Price the option using the library
    AmericanOptionResult call_result = american_option_price(&call_data, &grid_params);

    if (call_result.status == 0) {
        print_option_values(call_result.solver, &call_data, "american_call_values.dat");

        // Calculate option value at spot = strike (ATM)
        double V_atm = american_option_get_value_at_spot(call_result.solver,
                                                         call_data.strike,
                                                         call_data.strike);
        printf("American Call value at S=K=%.2f: %.6f\n", call_data.strike, V_atm);
    } else {
        printf("Solver failed to converge for American Call\n");
    }

    american_option_free_result(&call_result);

    // Example 2: American Put Option (no dividends)
    printf("\nExample 2: American Put Option (no dividends)\n");
    printf("----------------------------------------------\n");

    OptionData put_data = {
        .strike = 100.0,
        .volatility = 0.2,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    // Price the option using the library
    AmericanOptionResult put_result = american_option_price(&put_data, &grid_params);

    if (put_result.status == 0) {
        print_option_values(put_result.solver, &put_data, "american_put_values.dat");

        double V_atm = american_option_get_value_at_spot(put_result.solver,
                                                         put_data.strike,
                                                         put_data.strike);
        printf("American Put value at S=K=%.2f: %.6f\n", put_data.strike, V_atm);
    } else {
        printf("Solver failed to converge for American Put\n");
    }

    american_option_free_result(&put_result);

    printf("\nAll examples completed!\n");
    printf("\nNote: For American options:\n");
    printf("  - Early exercise premium = American value - European value\n");
    printf("  - Call options have minimal early exercise value (no dividends)\n");
    printf("  - Put options have significant early exercise value\n");

    return 0;
}
