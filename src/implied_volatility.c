#include "implied_volatility.h"
#include "american_option.h"
#include "lets_be_rational.h"
#include "ivcalc_trace.h"
#include <math.h>
#include <stdlib.h>

// Objective function for Brent's method - American option pricing
typedef struct {
    double spot;
    double strike;
    double time_to_maturity;
    double risk_free_rate;
    double market_price;
    bool is_call;
    const AmericanOptionGrid *grid;
} AmericanObjectiveData;

static double american_objective(double volatility, void *user_data) {
    AmericanObjectiveData *data = (AmericanObjectiveData *)user_data;

    // Setup American option with guessed volatility
    OptionData option = {
        .strike = data->strike,
        .volatility = volatility,  // This is what we're solving for
        .risk_free_rate = data->risk_free_rate,
        .time_to_maturity = data->time_to_maturity,
        .option_type = data->is_call ? OPTION_CALL : OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = NULL,
        .dividend_amounts = NULL
    };

    // Solve American option PDE (~21ms per call)
    AmericanOptionResult result = american_option_price(&option, data->grid);
    if (result.status != 0) {
        // PDE solve failed
        return NAN;
    }

    double theoretical_price = american_option_get_value_at_spot(
        result.solver, data->spot, data->strike);

    american_option_free_result(&result);

    return theoretical_price - data->market_price;
}

IVResult calculate_iv(const IVParams *params,
                     const AmericanOptionGrid *grid_params,
                     double tolerance, int max_iter) {
    IVResult result = {0.0, 0.0, 0, false, NULL};

    // Trace calculation start
    MANGO_TRACE_IV_START(params->spot_price, params->strike,
                          params->time_to_maturity, params->market_price);

    // Validate inputs
    if (params->spot_price <= 0.0) {
        MANGO_TRACE_IV_VALIDATION_ERROR(1, params->spot_price, 0.0);
        result.error = "Spot price must be positive";
        return result;
    }
    if (params->strike <= 0.0) {
        MANGO_TRACE_IV_VALIDATION_ERROR(2, params->strike, 0.0);
        result.error = "Strike price must be positive";
        return result;
    }
    if (params->time_to_maturity <= 0.0) {
        MANGO_TRACE_IV_VALIDATION_ERROR(3, params->time_to_maturity, 0.0);
        result.error = "Time to maturity must be positive";
        return result;
    }
    if (params->market_price <= 0.0) {
        MANGO_TRACE_IV_VALIDATION_ERROR(4, params->market_price, 0.0);
        result.error = "Market price must be positive";
        return result;
    }

    // Check for arbitrage bounds (American options)
    double intrinsic_value;
    if (params->is_call) {
        intrinsic_value = fmax(params->spot_price - params->strike, 0.0);
        if (params->market_price > params->spot_price) {
            MANGO_TRACE_IV_VALIDATION_ERROR(5, params->market_price, params->spot_price);
            result.error = "Call price exceeds spot price (arbitrage)";
            return result;
        }
    } else {
        intrinsic_value = fmax(params->strike - params->spot_price, 0.0);
        if (params->market_price > params->strike) {
            MANGO_TRACE_IV_VALIDATION_ERROR(5, params->market_price, params->strike);
            result.error = "Put price exceeds strike (arbitrage)";
            return result;
        }
    }

    if (params->market_price < intrinsic_value - tolerance) {
        MANGO_TRACE_IV_VALIDATION_ERROR(5, params->market_price, intrinsic_value);
        result.error = "Market price below intrinsic value (arbitrage)";
        return result;
    }

    // Get European IV estimate for upper bound
    LBRResult lbr = lbr_implied_volatility(params->spot_price, params->strike,
                                           params->time_to_maturity,
                                           params->risk_free_rate,
                                           params->market_price,
                                           params->is_call);

    // Establish Brent bounds
    double lower_bound = 1e-6;
    double upper_bound = lbr.converged ? lbr.implied_vol * 1.5 : 3.0;  // fallback

    // Setup objective function
    AmericanObjectiveData obj_data = {
        .spot = params->spot_price,
        .strike = params->strike,
        .time_to_maturity = params->time_to_maturity,
        .risk_free_rate = params->risk_free_rate,
        .market_price = params->market_price,
        .is_call = params->is_call,
        .grid = grid_params
    };

    // Use Brent's method to find the root
    BrentResult brent_result = brent_find_root(american_objective,
                                              lower_bound, upper_bound,
                                              tolerance, max_iter, &obj_data);

    if (brent_result.converged) {
        result.implied_vol = brent_result.root;
        result.iterations = brent_result.iterations;
        result.converged = true;
        result.vega = 0.0;  // Could compute via finite differences if needed

        MANGO_TRACE_IV_COMPLETE(result.implied_vol, result.iterations, 1);
    } else {
        result.error = "Failed to converge";
        result.iterations = brent_result.iterations;
        MANGO_TRACE_CONVERGENCE_FAILED(MODULE_IMPLIED_VOL, brent_result.iterations, max_iter, 0.0);
    }

    return result;
}

IVResult calculate_iv_simple(const IVParams *params) {
    // Default grid configuration
    AmericanOptionGrid default_grid = {
        .x_min = -0.7,      // ln(0.5)
        .x_max = 0.7,       // ln(2.0)
        .n_points = 141,
        .dt = 0.001,
        .n_steps = 1000
    };

    return calculate_iv(params, &default_grid, 1e-6, 100);
}
