#include "implied_volatility.h"
#include "american_option.h"
#include "lets_be_rational.h"
#include "ivcalc_trace.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Check if query point is within table bounds
static bool is_in_table_bounds(const OptionPriceTable *table, const IVParams *params) {
    if (!table) return false;

    // Calculate moneyness
    double moneyness = params->spot_price / params->strike;

    // Check all dimensions
    if (moneyness < table->moneyness_grid[0] || moneyness > table->moneyness_grid[table->n_moneyness - 1])
        return false;

    if (params->time_to_maturity < table->maturity_grid[0] ||
        params->time_to_maturity > table->maturity_grid[table->n_maturity - 1])
        return false;

    // For IV solving, we don't know sigma yet, so we can't check sigma bounds
    // We'll handle this during Newton iteration

    if (params->risk_free_rate < table->rate_grid[0] ||
        params->risk_free_rate > table->rate_grid[table->n_rate - 1])
        return false;

    // Check dividend if table has dividend dimension
    if (table->n_dividend > 0) {
        if (params->dividend_yield < table->dividend_grid[0] ||
            params->dividend_yield > table->dividend_grid[table->n_dividend - 1])
            return false;
    }

    // Check option type and exercise type match
    if (table->type != params->option_type)
        return false;

    if (table->exercise != params->exercise_type)
        return false;

    return true;
}

// Newton's method IV solver using table interpolation
static IVResult newton_iv_solver(const IVParams *params,
                                  const OptionPriceTable *table,
                                  double tolerance, int max_iter) {
    IVResult result = {.implied_vol = NAN, .vega = 0.0, .iterations = 0, .converged = false, .error = NULL};

    // Initial guess: middle of sigma range
    double sigma = (table->volatility_grid[0] + table->volatility_grid[table->n_volatility - 1]) / 2.0;

    // Calculate moneyness
    double m = params->spot_price / params->strike;
    double tau = params->time_to_maturity;
    double r = params->risk_free_rate;
    double q = params->dividend_yield;

    // Newton iteration
    for (int iter = 0; iter < max_iter; iter++) {
        result.iterations = iter + 1;

        // Check if sigma is in bounds
        if (sigma < table->volatility_grid[0] || sigma > table->volatility_grid[table->n_volatility - 1]) {
            result.error = "Volatility out of table bounds during iteration";
            return result;
        }

        // Interpolate price and vega
        double price, vega;
        if (table->n_dividend > 0) {
            price = price_table_interpolate_5d(table, m, tau, sigma, r, q);
            vega = price_table_interpolate_vega_5d(table, m, tau, sigma, r, q);
        } else {
            price = price_table_interpolate_4d(table, m, tau, sigma, r);
            vega = price_table_interpolate_vega_4d(table, m, tau, sigma, r);
        }

        // Check for invalid interpolation
        if (isnan(price) || isnan(vega)) {
            result.error = "Interpolation returned NaN";
            return result;
        }

        // Check for zero or negative vega (can't proceed)
        if (vega <= 0.0) {
            result.error = "Vega is zero or negative";
            return result;
        }

        // Newton update: σ_{n+1} = σ_n - (V(σ_n) - V_market) / vega(σ_n)
        double f = price - params->market_price;
        double sigma_new = sigma - f / vega;

        // Check convergence
        if (fabs(f) < tolerance) {
            result.converged = true;
            result.implied_vol = sigma;
            result.vega = vega;
            return result;
        }

        // Update sigma with damping for stability
        const double damping = 0.8;
        sigma = damping * sigma_new + (1.0 - damping) * sigma;
    }

    // Max iterations reached
    result.error = "Maximum iterations reached";
    result.implied_vol = sigma;
    return result;
}

// Objective function for Brent's method - American option pricing
typedef struct {
    double spot;
    double strike;
    double time_to_maturity;
    double risk_free_rate;
    double market_price;
    OptionType option_type;
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
        .option_type = data->option_type,
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
                     const OptionPriceTable *table,
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
    bool is_call = (params->option_type == OPTION_CALL);
    if (is_call) {
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

    // Try table interpolation if available
    if (table && is_in_table_bounds(table, params)) {
        // Check if vegas are available
        if (table->vegas) {
            IVResult newton_result = newton_iv_solver(params, table, tolerance, max_iter);

            // If Newton succeeded or failed with a non-bounds error, return result
            // If out of bounds during iteration, fallback to FDM
            if (newton_result.converged || newton_result.error == NULL ||
                strcmp(newton_result.error, "Volatility out of table bounds during iteration") != 0) {
                MANGO_TRACE_IV_COMPLETE(newton_result.implied_vol, newton_result.iterations, 1);
                return newton_result;
            }
            // Otherwise fall through to FDM
        }
    }

    // Fallback to Brent + FDM solver
    // Get European IV estimate for upper bound
    LBRResult lbr = lbr_implied_volatility(params->spot_price, params->strike,
                                           params->time_to_maturity,
                                           params->risk_free_rate,
                                           params->market_price,
                                           is_call);

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
        .option_type = params->option_type,
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

IVResult calculate_iv_simple(const IVParams *params,
                             const OptionPriceTable *table) {
    // Default grid configuration
    AmericanOptionGrid default_grid = {
        .x_min = -0.7,      // ln(0.5)
        .x_max = 0.7,       // ln(2.0)
        .n_points = 141,
        .dt = 0.001,
        .n_steps = 1000
    };

    return calculate_iv(params, &default_grid, table, 1e-6, 100);
}
