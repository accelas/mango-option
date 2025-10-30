#ifndef AMERICAN_OPTION_H
#define AMERICAN_OPTION_H

#include "pde_solver.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Option type enumeration
typedef enum {
    OPTION_CALL,
    OPTION_PUT
} OptionType;

// American option parameters
typedef struct {
    double strike;           // Strike price K
    double volatility;       // σ (volatility)
    double risk_free_rate;   // r (risk-free rate)
    double time_to_maturity; // T (time to maturity in years)
    OptionType option_type;  // Call or Put

    // Discrete dividend information (optional)
    size_t n_dividends;      // Number of dividend payments
    double *dividend_times;  // Times of dividend payments (in years)
    double *dividend_amounts;// Dividend amounts (absolute cash dividends)
} OptionData;

// Grid parameters for pricing
typedef struct {
    double x_min;            // Minimum log-moneyness (e.g., ln(0.5) ≈ -0.7)
    double x_max;            // Maximum log-moneyness (e.g., ln(2.0) ≈ 0.7)
    size_t n_points;         // Number of grid points (e.g., 141)
    double dt;               // Time step (e.g., 0.001)
    size_t n_steps;          // Number of time steps (e.g., 1000)
} AmericanOptionGrid;

// Result structure
typedef struct {
    PDESolver *solver;       // PDE solver (caller must destroy with american_option_free_result)
    int status;              // 0 = success, -1 = failure
    void *internal_data;     // Internal data (do not access directly)
} AmericanOptionResult;

// High-level API to price American options
// Returns a solver with the solution
// Caller must call american_option_free_result() on result to clean up
AmericanOptionResult american_option_price(const OptionData *option_data,
                                          const AmericanOptionGrid *grid_params);

// Free resources associated with AmericanOptionResult
// This frees both the solver and internal data structures
void american_option_free_result(AmericanOptionResult *result);

// Batch API: Price multiple American options in parallel
// option_data: Array of n_options option specifications
// grid_params: Grid parameters (shared across all options)
// results: Output array of n_options results (caller allocated)
// Returns: 0 on success, -1 on failure
// Note: Caller must call pde_solver_destroy() on each result[i].solver
int american_option_price_batch(const OptionData *option_data,
                                 const AmericanOptionGrid *grid_params,
                                 size_t n_options,
                                 AmericanOptionResult *results);

// Callback functions (exposed for advanced usage)
// These match the PDE solver callback signatures

void american_option_terminal_condition(const double *x, size_t n_points,
                                       double *V, void *user_data);

double american_option_left_boundary(double t, void *user_data);

double american_option_right_boundary(double t, void *user_data);

void american_option_spatial_operator(const double *x, double t, const double *V,
                                     size_t n_points, double *LV, void *user_data);

void american_option_obstacle(const double *x, double t, size_t n_points,
                             double *obstacle, void *user_data);

// Utility: Get option value at specific spot price
// x_spot: log-moneyness ln(S/K)
double american_option_get_value_at_spot(const PDESolver *solver, double spot_price,
                                        double strike);

#ifdef __cplusplus
}
#endif

#endif // AMERICAN_OPTION_H
