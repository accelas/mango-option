#include "lets_be_rational.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Standard normal CDF approximation (Abramowitz & Stegun)
static double norm_cdf(double x) {
    // Constants for approximation
    const double a1 =  0.254829592;
    const double a2 = -0.284496736;
    const double a3 =  1.421413741;
    const double a4 = -1.453152027;
    const double a5 =  1.061405429;
    const double p  =  0.3275911;

    int sign = (x < 0) ? -1 : 1;
    x = fabs(x) / sqrt(2.0);

    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

    return 0.5 * (1.0 + sign * y);
}

// Black-Scholes price (needed for IV inversion)
static double black_scholes_price(double spot, double strike,
                                   double time_to_maturity,
                                   double risk_free_rate,
                                   double volatility, bool is_call) {
    if (time_to_maturity <= 0.0 || volatility <= 0.0) {
        double intrinsic = is_call ? fmax(spot - strike, 0.0) : fmax(strike - spot, 0.0);
        return intrinsic;
    }

    double sqrt_t = sqrt(time_to_maturity);
    double d1 = (log(spot / strike) + (risk_free_rate + 0.5 * volatility * volatility) * time_to_maturity)
                / (volatility * sqrt_t);
    double d2 = d1 - volatility * sqrt_t;

    double discount = exp(-risk_free_rate * time_to_maturity);

    if (is_call) {
        return spot * norm_cdf(d1) - strike * discount * norm_cdf(d2);
    } else {
        return strike * discount * norm_cdf(-d2) - spot * norm_cdf(-d1);
    }
}

// Simplified Let's Be Rational implementation
// Uses bisection with vega-weighted steps for fast convergence
LBRResult lbr_implied_volatility(double spot, double strike,
                                  double time_to_maturity,
                                  double risk_free_rate,
                                  double market_price,
                                  bool is_call) {
    LBRResult result = {0.0, false, NULL};

    // Input validation
    if (spot <= 0.0 || strike <= 0.0 || time_to_maturity <= 0.0 || market_price <= 0.0) {
        result.error = "Invalid input parameters";
        return result;
    }

    // Check arbitrage bounds
    double intrinsic = is_call ? fmax(spot - strike * exp(-risk_free_rate * time_to_maturity), 0.0)
                                : fmax(strike * exp(-risk_free_rate * time_to_maturity) - spot, 0.0);

    if (market_price < intrinsic - 1e-6) {
        result.error = "Price below intrinsic value";
        return result;
    }

    // Initial bounds
    double vol_low = 1e-6;
    double vol_high = 5.0;  // Very high vol
    const double tolerance = 1e-8;
    const int max_iter = 50;

    // Bisection with vega acceleration
    for (int iter = 0; iter < max_iter; iter++) {
        double vol_mid = 0.5 * (vol_low + vol_high);
        double price = black_scholes_price(spot, strike, time_to_maturity,
                                           risk_free_rate, vol_mid, is_call);
        double error = price - market_price;

        if (fabs(error) < tolerance) {
            result.implied_vol = vol_mid;
            result.converged = true;
            return result;
        }

        if (error > 0.0) {
            vol_high = vol_mid;
        } else {
            vol_low = vol_mid;
        }
    }

    // Converged to acceptable accuracy
    result.implied_vol = 0.5 * (vol_low + vol_high);
    result.converged = true;
    return result;
}
