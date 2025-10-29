#include "european_option.h"
#include <math.h>

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

// Calculate option delta (∂V/∂S)
double black_scholes_delta(double spot, double strike, double time_to_maturity,
                           double risk_free_rate, double volatility, bool is_call) {
    if (time_to_maturity <= 0.0) {
        // At expiration, delta is either 0 or 1 for calls, 0 or -1 for puts
        if (is_call) {
            return spot >= strike ? 1.0 : 0.0;
        } else {
            return spot <= strike ? -1.0 : 0.0;
        }
    }

    if (volatility <= 0.0) {
        return 0.0;
    }

    double d1 = bs_d1(spot, strike, time_to_maturity, risk_free_rate, volatility);

    if (is_call) {
        // Call delta: N(d1)
        return normal_cdf(d1);
    } else {
        // Put delta: N(d1) - 1
        return normal_cdf(d1) - 1.0;
    }
}

// Calculate option gamma (∂²V/∂S²)
// Gamma is the same for calls and puts
double black_scholes_gamma(double spot, double strike, double time_to_maturity,
                           double risk_free_rate, double volatility) {
    if (time_to_maturity <= 0.0 || volatility <= 0.0 || spot <= 0.0) {
        return 0.0;
    }

    double d1 = bs_d1(spot, strike, time_to_maturity, risk_free_rate, volatility);
    double sigma_sqrt_t = volatility * sqrt(time_to_maturity);

    // Gamma = φ(d1) / (S * σ * sqrt(T)), where φ is standard normal PDF
    return normal_pdf(d1) / (spot * sigma_sqrt_t);
}

// Calculate option theta (∂V/∂t)
double black_scholes_theta(double spot, double strike, double time_to_maturity,
                           double risk_free_rate, double volatility, bool is_call) {
    if (time_to_maturity <= 0.0 || volatility <= 0.0) {
        return 0.0;
    }

    double d1 = bs_d1(spot, strike, time_to_maturity, risk_free_rate, volatility);
    double d2 = bs_d2(spot, strike, time_to_maturity, risk_free_rate, volatility);
    double sqrt_t = sqrt(time_to_maturity);
    double discount = exp(-risk_free_rate * time_to_maturity);

    // Common term for both call and put
    double theta_common = -(spot * normal_pdf(d1) * volatility) / (2.0 * sqrt_t);

    if (is_call) {
        // Call theta: -S*φ(d1)*σ/(2*sqrt(T)) - r*K*exp(-r*T)*N(d2)
        return theta_common - risk_free_rate * strike * discount * normal_cdf(d2);
    } else {
        // Put theta: -S*φ(d1)*σ/(2*sqrt(T)) + r*K*exp(-r*T)*N(-d2)
        return theta_common + risk_free_rate * strike * discount * normal_cdf(-d2);
    }
}

// Calculate option rho (∂V/∂r)
double black_scholes_rho(double spot, double strike, double time_to_maturity,
                         double risk_free_rate, double volatility, bool is_call) {
    if (time_to_maturity <= 0.0 || volatility <= 0.0) {
        return 0.0;
    }

    double d2 = bs_d2(spot, strike, time_to_maturity, risk_free_rate, volatility);
    double discount = exp(-risk_free_rate * time_to_maturity);

    if (is_call) {
        // Call rho: K*T*exp(-r*T)*N(d2)
        return strike * time_to_maturity * discount * normal_cdf(d2);
    } else {
        // Put rho: -K*T*exp(-r*T)*N(-d2)
        return -strike * time_to_maturity * discount * normal_cdf(-d2);
    }
}
