#include "implied_volatility.h"
#include "european_option.h"
#include "ivcalc_trace.h"
#include <math.h>
#include <stdlib.h>

// Objective function for Brent's method
// Returns: theoretical_price(σ) - market_price
typedef struct {
    double spot;
    double strike;
    double time_to_maturity;
    double risk_free_rate;
    double market_price;
    bool is_call;
} BSObjectiveData;

static double bs_objective(double volatility, void *user_data) {
    BSObjectiveData *data = (BSObjectiveData *)user_data;

    double theoretical_price = black_scholes_price(data->spot, data->strike,
                                                   data->time_to_maturity,
                                                   data->risk_free_rate,
                                                   volatility, data->is_call);

    return theoretical_price - data->market_price;
}

// Main IV calculation function
IVResult implied_volatility_calculate(const IVParams *params,
                                     double initial_guess_low,
                                     double initial_guess_high,
                                     double tolerance,
                                     int max_iter) {
    IVResult result = {0.0, 0.0, 0, false, nullptr};

    // Trace calculation start
    IVCALC_TRACE_IV_START(params->spot_price, params->strike,
                          params->time_to_maturity, params->market_price);

    // Validate inputs
    if (params->spot_price <= 0.0) {
        IVCALC_TRACE_IV_VALIDATION_ERROR(1, params->spot_price, 0.0);
        result.error = "Spot price must be positive";
        return result;
    }
    if (params->strike <= 0.0) {
        IVCALC_TRACE_IV_VALIDATION_ERROR(2, params->strike, 0.0);
        result.error = "Strike price must be positive";
        return result;
    }
    if (params->time_to_maturity <= 0.0) {
        IVCALC_TRACE_IV_VALIDATION_ERROR(3, params->time_to_maturity, 0.0);
        result.error = "Time to maturity must be positive";
        return result;
    }
    if (params->market_price <= 0.0) {
        IVCALC_TRACE_IV_VALIDATION_ERROR(4, params->market_price, 0.0);
        result.error = "Market price must be positive";
        return result;
    }

    // Check for arbitrage bounds
    double intrinsic_value;
    if (params->is_call) {
        intrinsic_value = fmax(params->spot_price - params->strike * exp(-params->risk_free_rate * params->time_to_maturity), 0.0);
        if (params->market_price > params->spot_price) {
            IVCALC_TRACE_IV_VALIDATION_ERROR(5, params->market_price, params->spot_price);
            result.error = "Call price exceeds spot price (arbitrage)";
            return result;
        }
    } else {
        intrinsic_value = fmax(params->strike * exp(-params->risk_free_rate * params->time_to_maturity) - params->spot_price, 0.0);
        double max_put_price = params->strike * exp(-params->risk_free_rate * params->time_to_maturity);
        if (params->market_price > max_put_price) {
            IVCALC_TRACE_IV_VALIDATION_ERROR(5, params->market_price, max_put_price);
            result.error = "Put price exceeds discounted strike (arbitrage)";
            return result;
        }
    }

    if (params->market_price < intrinsic_value - tolerance) {
        IVCALC_TRACE_IV_VALIDATION_ERROR(5, params->market_price, intrinsic_value);
        result.error = "Market price below intrinsic value (arbitrage)";
        return result;
    }

    // Setup objective function
    BSObjectiveData obj_data = {
        .spot = params->spot_price,
        .strike = params->strike,
        .time_to_maturity = params->time_to_maturity,
        .risk_free_rate = params->risk_free_rate,
        .market_price = params->market_price,
        .is_call = params->is_call
    };

    // Use Brent's method to find the root
    BrentResult brent_result = brent_find_root(bs_objective,
                                              initial_guess_low,
                                              initial_guess_high,
                                              tolerance,
                                              max_iter,
                                              &obj_data);

    if (!brent_result.converged) {
        IVCALC_TRACE_IV_COMPLETE(brent_result.root, brent_result.iterations, 0);
        result.error = "Failed to converge";
        result.implied_vol = brent_result.root;
        result.iterations = brent_result.iterations;
        return result;
    }

    // Calculate vega at solution
    result.implied_vol = brent_result.root;
    result.vega = black_scholes_vega(params->spot_price, params->strike,
                                    params->time_to_maturity,
                                    params->risk_free_rate,
                                    result.implied_vol);
    result.iterations = brent_result.iterations;
    result.converged = true;

    // Trace successful completion
    IVCALC_TRACE_IV_COMPLETE(result.implied_vol, result.iterations, 1);

    return result;
}

// Determine automatic bounds for volatility search
// Uses heuristics based on option parameters and market price
typedef struct {
    double lower_bound;
    double upper_bound;
} VolatilityBounds;

static VolatilityBounds determine_volatility_bounds(const IVParams *params) {
    VolatilityBounds bounds;

    // Lower bound: Use very small positive value (0.01% volatility)
    bounds.lower_bound = 0.0001;

    // Upper bound: Use heuristic based on market price and intrinsic value
    // The idea is that higher market prices relative to intrinsic value
    // suggest higher time value, which requires higher volatility

    double intrinsic_value;
    double forward_price = params->spot_price * exp(params->risk_free_rate * params->time_to_maturity);

    if (params->is_call) {
        intrinsic_value = fmax(forward_price - params->strike, 0.0);
    } else {
        intrinsic_value = fmax(params->strike - forward_price, 0.0);
    }

    // If market price is close to intrinsic (small time value), use moderate upper bound
    // If market price is high relative to intrinsic, use higher upper bound
    double time_value = params->market_price - intrinsic_value * exp(-params->risk_free_rate * params->time_to_maturity);

    if (time_value <= 0.0) {
        // Deep ITM or at expiry - use moderate bound
        bounds.upper_bound = 2.0;
    } else {
        // Estimate implied variance from time value
        // For ATM option: C ≈ 0.4 * S * σ * sqrt(T) (approximation)
        // So σ ≈ C / (0.4 * S * sqrt(T))
        double vol_estimate = params->market_price / (0.4 * params->spot_price * sqrt(params->time_to_maturity));

        // Use 2x the estimate as upper bound, with reasonable min/max
        bounds.upper_bound = fmax(2.0 * vol_estimate, 1.0);  // At least 100%
        bounds.upper_bound = fmin(bounds.upper_bound, 10.0);  // Cap at 1000%
    }

    // Ensure upper bound is always greater than lower bound
    if (bounds.upper_bound <= bounds.lower_bound) {
        bounds.upper_bound = 5.0;  // Fallback to reasonable default
    }

    return bounds;
}

// Convenience function with default parameters
IVResult implied_volatility_calculate_simple(const IVParams *params) {
    // Automatically determine sensible bounds based on option parameters
    VolatilityBounds bounds = determine_volatility_bounds(params);

    // Search volatility with auto-determined bounds and tolerance 1e-6
    return implied_volatility_calculate(params, bounds.lower_bound, bounds.upper_bound, 1e-6, 100);
}
