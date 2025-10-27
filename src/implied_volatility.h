#ifndef IMPLIED_VOLATILITY_H
#define IMPLIED_VOLATILITY_H

#include "brent.h"
#include <stdbool.h>

// Implied Volatility Calculator using Brent's method
//
// Solves for the volatility σ that makes the theoretical option price
// equal to the market price using Black-Scholes formula

// Option parameters for IV calculation
typedef struct {
    double spot_price;       // S: Current stock price
    double strike;           // K: Strike price
    double time_to_maturity; // T: Time to expiration (years)
    double risk_free_rate;   // r: Risk-free interest rate
    double market_price;     // Market price of option
    bool is_call;            // true for call, false for put
} IVParams;

// Result structure
typedef struct {
    double implied_vol;      // Calculated implied volatility
    double vega;             // Option vega at solution
    int iterations;          // Number of iterations
    bool converged;          // True if converged
    const char *error;       // Error message if failed (nullptr if success)
} IVResult;

// Calculate implied volatility
//
// Parameters:
//   params: Option parameters and market price
//   initial_guess_low: Lower bound for volatility search (e.g., 0.01 = 1%)
//   initial_guess_high: Upper bound for volatility search (e.g., 5.0 = 500%)
//   tolerance: Convergence tolerance (e.g., 1e-6)
//   max_iter: Maximum iterations (e.g., 100)
//
// Returns:
//   IVResult with implied volatility and convergence status
IVResult implied_volatility_calculate(const IVParams *params,
                                     double initial_guess_low,
                                     double initial_guess_high,
                                     double tolerance,
                                     int max_iter);

// Convenience function with default parameters
// Searches volatility in range [0.01, 5.0] with tolerance 1e-6
IVResult implied_volatility_calculate_simple(const IVParams *params);

// Black-Scholes option pricing (used internally and exposed for testing)
double black_scholes_price(double spot, double strike, double time_to_maturity,
                           double risk_free_rate, double volatility, bool is_call);

// Calculate option vega (sensitivity to volatility)
double black_scholes_vega(double spot, double strike, double time_to_maturity,
                          double risk_free_rate, double volatility);

#endif // IMPLIED_VOLATILITY_H
