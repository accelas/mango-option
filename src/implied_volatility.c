#include "implied_volatility.h"
#include "ivcalc_trace.h"
#include <math.h>
#include <stdlib.h>

// Constants
#define M_SQRT2PI 2.5066282746310002  // sqrt(2*pi)
#define M_1_SQRT2 0.7071067811865475   // 1/sqrt(2)

// Cumulative distribution function for standard normal distribution
// Using Abramowitz and Stegun approximation (maximum error: 7.5e-8)
static double normal_cdf(double x) {
    // Constants for the approximation
    static const double a1 =  0.254829592;
    static const double a2 = -0.284496736;
    static const double a3 =  1.421413741;
    static const double a4 = -1.453152027;
    static const double a5 =  1.061405429;
    static const double p  =  0.3275911;

    // Save the sign of x
    int sign = (x < 0.0) ? -1 : 1;
    x = fabs(x) * M_1_SQRT2;

    // A&S formula (approximation)
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

    return 0.5 * (1.0 + sign * y);
}

// Probability density function for standard normal distribution
static double normal_pdf(double x) {
    return exp(-0.5 * x * x) / M_SQRT2PI;
}

// Black-Scholes d1 parameter
static double bs_d1(double spot, double strike, double time_to_maturity,
                   double risk_free_rate, double volatility) {
    double sigma_sqrt_t = volatility * sqrt(time_to_maturity);
    return (log(spot / strike) + (risk_free_rate + 0.5 * volatility * volatility) * time_to_maturity)
           / sigma_sqrt_t;
}

// Black-Scholes d2 parameter
static double bs_d2(double spot, double strike, double time_to_maturity,
                   double risk_free_rate, double volatility) {
    return bs_d1(spot, strike, time_to_maturity, risk_free_rate, volatility)
           - volatility * sqrt(time_to_maturity);
}

// Black-Scholes option pricing formula
double black_scholes_price(double spot, double strike, double time_to_maturity,
                           double risk_free_rate, double volatility, bool is_call) {
    // Handle edge cases
    if (time_to_maturity <= 0.0) {
        if (is_call) {
            return fmax(spot - strike, 0.0);
        } else {
            return fmax(strike - spot, 0.0);
        }
    }

    if (volatility <= 0.0) {
        return 0.0;
    }

    double d1 = bs_d1(spot, strike, time_to_maturity, risk_free_rate, volatility);
    double d2 = bs_d2(spot, strike, time_to_maturity, risk_free_rate, volatility);

    if (is_call) {
        // Call option: C = S*N(d1) - K*exp(-r*T)*N(d2)
        return spot * normal_cdf(d1) - strike * exp(-risk_free_rate * time_to_maturity) * normal_cdf(d2);
    } else {
        // Put option: P = K*exp(-r*T)*N(-d2) - S*N(-d1)
        return strike * exp(-risk_free_rate * time_to_maturity) * normal_cdf(-d2) - spot * normal_cdf(-d1);
    }
}

// Calculate option vega (∂V/∂σ)
// Vega is the same for calls and puts
double black_scholes_vega(double spot, double strike, double time_to_maturity,
                          double risk_free_rate, double volatility) {
    if (time_to_maturity <= 0.0 || volatility <= 0.0) {
        return 0.0;
    }

    double d1 = bs_d1(spot, strike, time_to_maturity, risk_free_rate, volatility);
    double sqrt_t = sqrt(time_to_maturity);

    // Vega = S * φ(d1) * sqrt(T), where φ is standard normal PDF
    return spot * normal_pdf(d1) * sqrt_t;
}

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

// Convenience function with default parameters
IVResult implied_volatility_calculate_simple(const IVParams *params) {
    // Search volatility in range [1%, 500%] with tolerance 1e-6
    return implied_volatility_calculate(params, 0.01, 5.0, 1e-6, 100);
}
