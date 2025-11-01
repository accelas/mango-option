#ifndef IMPLIED_VOLATILITY_H
#define IMPLIED_VOLATILITY_H

#include "brent.h"
#include "american_option.h"
#include <stdbool.h>

// Implied Volatility Calculator for American Options using Brent's method
//
// Solves for the volatility σ that makes the theoretical American option price
// equal to the market price using FDM-based PDE solver

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

// Calculate American option implied volatility using FDM
//
// Parameters:
//   params: Option parameters and market price
//   grid_params: FDM solver grid configuration
//   tolerance: Convergence tolerance (e.g., 1e-6)
//   max_iter: Maximum iterations (e.g., 100)
//
// Returns:
//   IVResult with implied volatility and convergence status
//
// Performance: ~250ms per calculation (nested Brent + PDE solver)
IVResult calculate_iv(const IVParams *params,
                     const AmericanOptionGrid *grid_params,
                     double tolerance, int max_iter);

// Convenience function with default grid settings
//
// Uses default FDM grid:
//   - x_min: -0.7, x_max: 0.7 (log-moneyness)
//   - n_points: 141
//   - dt: 0.001, n_steps: 1000
//   - tolerance: 1e-6, max_iter: 100
//
// Performance: ~250ms per calculation
IVResult calculate_iv_simple(const IVParams *params);

#endif // IMPLIED_VOLATILITY_H
