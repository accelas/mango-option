#ifndef EUROPEAN_OPTION_H
#define EUROPEAN_OPTION_H

#include <stdbool.h>

// European Option Pricing using Black-Scholes Formula
//
// This module provides European option pricing and Greeks calculations
// using the analytical Black-Scholes-Merton model.

// Black-Scholes option pricing formula
//
// Calculates the theoretical price of a European call or put option
// using the Black-Scholes-Merton model.
//
// Parameters:
//   spot: Current stock price (S)
//   strike: Strike price (K)
//   time_to_maturity: Time to expiration in years (T)
//   risk_free_rate: Risk-free interest rate (r)
//   volatility: Volatility of the underlying asset (σ)
//   is_call: true for call option, false for put option
//
// Returns:
//   Theoretical option price
//
// Edge cases:
//   - If time_to_maturity <= 0: returns intrinsic value max(S-K, 0) or max(K-S, 0)
//   - If volatility <= 0: returns 0.0
double black_scholes_price(double spot, double strike, double time_to_maturity,
                           double risk_free_rate, double volatility, bool is_call);

// Calculate option vega (∂V/∂σ)
//
// Vega measures the sensitivity of the option price to changes in volatility.
// Vega is the same for European calls and puts.
//
// Parameters:
//   spot: Current stock price (S)
//   strike: Strike price (K)
//   time_to_maturity: Time to expiration in years (T)
//   risk_free_rate: Risk-free interest rate (r)
//   volatility: Volatility of the underlying asset (σ)
//
// Returns:
//   Option vega (∂V/∂σ)
//
// Edge cases:
//   - If time_to_maturity <= 0 or volatility <= 0: returns 0.0
double black_scholes_vega(double spot, double strike, double time_to_maturity,
                          double risk_free_rate, double volatility);

// Calculate option delta (∂V/∂S)
//
// Delta measures the sensitivity of the option price to changes in the underlying price.
// For calls: 0 < Δ < 1
// For puts: -1 < Δ < 0
//
// Parameters:
//   spot: Current stock price (S)
//   strike: Strike price (K)
//   time_to_maturity: Time to expiration in years (T)
//   risk_free_rate: Risk-free interest rate (r)
//   volatility: Volatility of the underlying asset (σ)
//   is_call: true for call option, false for put option
//
// Returns:
//   Option delta (∂V/∂S)
double black_scholes_delta(double spot, double strike, double time_to_maturity,
                           double risk_free_rate, double volatility, bool is_call);

// Calculate option gamma (∂²V/∂S²)
//
// Gamma measures the rate of change of delta with respect to the underlying price.
// Gamma is the same for European calls and puts.
//
// Parameters:
//   spot: Current stock price (S)
//   strike: Strike price (K)
//   time_to_maturity: Time to expiration in years (T)
//   risk_free_rate: Risk-free interest rate (r)
//   volatility: Volatility of the underlying asset (σ)
//
// Returns:
//   Option gamma (∂²V/∂S²)
double black_scholes_gamma(double spot, double strike, double time_to_maturity,
                           double risk_free_rate, double volatility);

// Calculate option theta (∂V/∂t)
//
// Theta measures the sensitivity of the option price to the passage of time.
// Theta is typically negative for long positions (time decay).
//
// Parameters:
//   spot: Current stock price (S)
//   strike: Strike price (K)
//   time_to_maturity: Time to expiration in years (T)
//   risk_free_rate: Risk-free interest rate (r)
//   volatility: Volatility of the underlying asset (σ)
//   is_call: true for call option, false for put option
//
// Returns:
//   Option theta (∂V/∂t) - note: this is per year, divide by 365 for daily theta
double black_scholes_theta(double spot, double strike, double time_to_maturity,
                           double risk_free_rate, double volatility, bool is_call);

// Calculate option rho (∂V/∂r)
//
// Rho measures the sensitivity of the option price to changes in the risk-free rate.
//
// Parameters:
//   spot: Current stock price (S)
//   strike: Strike price (K)
//   time_to_maturity: Time to expiration in years (T)
//   risk_free_rate: Risk-free interest rate (r)
//   volatility: Volatility of the underlying asset (σ)
//   is_call: true for call option, false for put option
//
// Returns:
//   Option rho (∂V/∂r) - sensitivity to 1 unit change in rate (e.g., 1% = 0.01)
double black_scholes_rho(double spot, double strike, double time_to_maturity,
                         double risk_free_rate, double volatility, bool is_call);

#endif // EUROPEAN_OPTION_H
