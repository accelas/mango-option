#ifndef LETS_BE_RATIONAL_H
#define LETS_BE_RATIONAL_H

#include <stdbool.h>

// Fast European IV estimation using Jäckel's "Let's Be Rational" method
//
// This module provides fast European implied volatility estimation
// for use in establishing upper bounds for American IV calculation.
// NOT intended for direct European option IV queries.

typedef struct {
    double implied_vol;      // Estimated European IV
    bool converged;          // True if estimation succeeded
    const char *error;       // Error message if failed
} LBRResult;

// Calculate European IV using rational approximation (~100ns)
//
// Parameters:
//   spot: Current stock price (S)
//   strike: Strike price (K)
//   time_to_maturity: Time to expiration in years (T)
//   risk_free_rate: Risk-free interest rate (r)
//   market_price: Observed market price of the option
//   is_call: true for call option, false for put option
//
// Returns:
//   LBRResult with implied_vol if successful
//
// Note: This is a fast approximation for bound estimation only
LBRResult lbr_implied_volatility(double spot, double strike,
                                  double time_to_maturity,
                                  double risk_free_rate,
                                  double market_price,
                                  bool is_call);

#endif // LETS_BE_RATIONAL_H
