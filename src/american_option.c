#include "american_option.h"
#include "ivcalc_trace.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Extended user data structure that includes grid boundaries
// This is needed for boundary conditions that depend on spatial location
typedef struct {
    const OptionData *option_data;
    double x_min;
    double x_max;
} ExtendedOptionData;

// American Option Pricing using Black-Scholes PDE
//
// Black-Scholes PDE in backward time τ = T - t (time to maturity):
// ∂V/∂τ = (1/2)σ²S²∂²V/∂S² + rS∂V/∂S - rV
//
// Using log-price transformation x = ln(S/K):
// ∂V/∂τ = (1/2)σ²∂²V/∂x² + (r - σ²/2)∂V/∂x - rV
//
// Our PDE solver uses forward time t, so we map: t = τ (time to maturity)
// At t=0: τ=0, which is maturity (terminal condition = payoff)
// At t=T: τ=T, which is current time (solution we want)
//
// The PDE in solver time t: ∂V/∂t = L(V)
// where L(V) = (1/2)σ²∂²V/∂x² + (r - σ²/2)∂V/∂x - rV

// Terminal condition: Option payoff at maturity (vectorized)
// At maturity, American = European = max(S - K, 0) for call
void american_option_terminal_condition(const double *x, size_t n_points,
                                       double *V, void *user_data) {
    ExtendedOptionData *ext_data = (ExtendedOptionData *)user_data;
    const OptionData *data = ext_data->option_data;
    const double K = data->strike;

    for (size_t i = 0; i < n_points; i++) {
        // x = ln(S/K), so S = K*exp(x)
        double S = K * exp(x[i]);

        if (data->option_type == OPTION_CALL) {
            V[i] = fmax(S - K, 0.0);
        } else {
            V[i] = fmax(K - S, 0.0);
        }
    }
}

// Left boundary condition (S → 0, x → -∞)
double american_option_left_boundary(double t, void *user_data) {
    ExtendedOptionData *ext_data = (ExtendedOptionData *)user_data;
    const OptionData *data = ext_data->option_data;
    // In our time mapping: t represents time-to-maturity τ
    const double tau = t;

    if (data->option_type == OPTION_CALL) {
        // For call: V(0,τ) = 0 (worthless when S=0)
        return 0.0;
    } else {
        // For put: V(0,τ) ≈ K*exp(-r*τ) (discounted strike)
        return data->strike * exp(-data->risk_free_rate * tau);
    }
}

// Right boundary condition (S → ∞, x → ∞)
double american_option_right_boundary(double t, void *user_data) {
    ExtendedOptionData *ext_data = (ExtendedOptionData *)user_data;
    const OptionData *data = ext_data->option_data;
    const double x_max = ext_data->x_max;
    // In our time mapping: t represents time-to-maturity τ
    const double tau = t;

    if (data->option_type == OPTION_CALL) {
        // For call at large S: V(x_max, τ) ≈ S_max - K*exp(-r*τ)
        // where S_max = K*exp(x_max)
        // This is the European call value at the boundary, which equals the
        // American call value for options with no dividends (early exercise never optimal)
        const double K = data->strike;
        const double r = data->risk_free_rate;
        const double S_max = K * exp(x_max);
        return S_max - K * exp(-r * tau);
    } else {
        // For put: V(S→∞, τ) = 0 (worthless when S is very large)
        return 0.0;
    }
}

// Black-Scholes spatial operator in log-price coordinates (vectorized)
// L(V) = (1/2)σ²∂²V/∂x² + (r - σ²/2)∂V/∂x - rV
void american_option_spatial_operator(const double *x, [[maybe_unused]] double t,
                                     const double *V, size_t n_points,
                                     double *LV, void *user_data) {
    ExtendedOptionData *ext_data = (ExtendedOptionData *)user_data;
    const OptionData *data = ext_data->option_data;
    const double sigma = data->volatility;
    const double r = data->risk_free_rate;
    const double dx = (x[n_points - 1] - x[0]) / (n_points - 1);
    const double dx_inv = 1.0 / dx;
    const double dx2_inv = 1.0 / (dx * dx);

    // Black-Scholes PDE coefficients in log-price coordinates
    // The solver time t represents time-to-maturity τ
    const double coeff_2nd = 0.5 * sigma * sigma;         // (1/2)σ²
    const double coeff_1st = r - 0.5 * sigma * sigma;     // r - σ²/2
    const double coeff_0th = -r;                          // -r

    // Boundaries will be overwritten by BC
    LV[0] = 0.0;
    LV[n_points - 1] = 0.0;

    // Interior points: second-order centered differences
    #pragma omp simd
    for (size_t i = 1; i < n_points - 1; i++) {
        // ∂²V/∂x² ≈ (V[i-1] - 2*V[i] + V[i+1]) / dx²
        double d2V_dx2 = (V[i - 1] - 2.0 * V[i] + V[i + 1]) * dx2_inv;

        // ∂V/∂x ≈ (V[i+1] - V[i-1]) / (2*dx) (centered)
        double dV_dx = (V[i + 1] - V[i - 1]) * 0.5 * dx_inv;

        LV[i] = coeff_2nd * d2V_dx2 + coeff_1st * dV_dx + coeff_0th * V[i];
    }
}

// Obstacle condition: American option constraint (vectorized)
// V(S,t) ≥ intrinsic_value(S)
void american_option_obstacle(const double *x, [[maybe_unused]] double t,
                             size_t n_points, double *obstacle, void *user_data) {
    ExtendedOptionData *ext_data = (ExtendedOptionData *)user_data;
    const OptionData *data = ext_data->option_data;
    const double K = data->strike;

    #pragma omp simd
    for (size_t i = 0; i < n_points; i++) {
        // x = ln(S/K), so S = K*exp(x)
        double S = K * exp(x[i]);

        if (data->option_type == OPTION_CALL) {
            obstacle[i] = fmax(S - K, 0.0);
        } else {
            obstacle[i] = fmax(K - S, 0.0);
        }
    }
}

// Forward declarations
static void american_option_dividend_event(double t, const double *x_grid,
                                           size_t n_points, double *V,
                                           const size_t *event_indices,
                                           size_t n_events_triggered,
                                           void *user_data);

// Discrete dividend adjustment: Handle stock price jump when dividend is paid
// When dividend D is paid, stock price jumps from S to S - D
// In log-price coordinates x = ln(S/K):
//   S_old = K*exp(x) => S_new = K*exp(x) - D
//   x_new = ln((K*exp(x) - D)/K) = ln(exp(x) - D/K)
void american_option_apply_dividend(const double *x_grid, size_t n_points,
                                   const double *V_old, double *V_new,
                                   double dividend, double strike) {

    // When crossing a dividend in the solver (backward in calendar time):
    // V_old: computed with post-dividend stock prices (after div was paid in calendar time)
    // V_new: should use pre-dividend stock prices (before div was paid in calendar time)
    //
    // For each grid point representing pre-dividend stock price S_pre:
    // Find the corresponding post-dividend price S_post = S_pre - D
    // and interpolate V_old at that point
    for (size_t i = 0; i < n_points; i++) {
        // Pre-dividend: x = ln(S_pre/K), S_pre = K*exp(x)
        double S_pre = strike * exp(x_grid[i]);

        // Post-dividend: S_post = S_pre - D (stock drops by dividend amount)
        double S_post = S_pre - dividend;

        // Post-dividend log-price: x_post = ln(S_post/K)
        double x_post = log(S_post / strike);

        // Interpolate V from old grid (post-dividend) to find V(x_post)
        // Linear interpolation
        if (x_post <= x_grid[0]) {
            V_new[i] = V_old[0];
        } else if (x_post >= x_grid[n_points - 1]) {
            V_new[i] = V_old[n_points - 1];
        } else {
            // Find bracketing indices
            size_t j = 0;
            while (j < n_points - 1 && x_grid[j + 1] < x_post) {
                j++;
            }

            // Linear interpolation
            double alpha = (x_post - x_grid[j]) / (x_grid[j + 1] - x_grid[j]);
            V_new[i] = (1.0 - alpha) * V_old[j] + alpha * V_old[j + 1];
        }
    }
}

// High-level API to price American options
AmericanOptionResult american_option_price(const OptionData *option_data,
                                          const AmericanOptionGrid *grid_params) {
    AmericanOptionResult result = {nullptr, -1, nullptr};

    // Trace option pricing start
    IVCALC_TRACE_OPTION_START(option_data->option_type, option_data->strike,
                              option_data->volatility, option_data->time_to_maturity);

    // Create spatial grid in log-price coordinates: x = ln(S/K)
    SpatialGrid grid = pde_create_grid(grid_params->x_min,
                                      grid_params->x_max,
                                      grid_params->n_points);

    // Time domain: solve forward in time-to-maturity
    TimeDomain time = {
        .t_start = 0.0,
        .t_end = option_data->time_to_maturity,
        .dt = grid_params->dt,
        .n_steps = grid_params->n_steps
    };

    // Convert dividend times from calendar time to solver time
    // Solver time: time-to-maturity (t=0 is maturity, t=T is now)
    // Calendar time: time from now
    // Conversion: t_solver = T - t_calendar
    double *div_times_solver = nullptr;
    if (option_data->n_dividends > 0 && option_data->dividend_times != nullptr &&
        option_data->dividend_amounts != nullptr) {
        div_times_solver = (double *)malloc(option_data->n_dividends * sizeof(double));
        if (div_times_solver == nullptr) {
            return result;
        }

        const double T = option_data->time_to_maturity;
        for (size_t i = 0; i < option_data->n_dividends; i++) {
            div_times_solver[i] = T - option_data->dividend_times[i];
        }

        // Sort dividend times in ascending order (for solver)
        for (size_t i = 0; i < option_data->n_dividends; i++) {
            for (size_t j = i + 1; j < option_data->n_dividends; j++) {
                if (div_times_solver[j] < div_times_solver[i]) {
                    double temp = div_times_solver[i];
                    div_times_solver[i] = div_times_solver[j];
                    div_times_solver[j] = temp;
                }
            }
        }
    }

    // Create extended user data to pass grid boundaries to callbacks
    // Allocate dynamically to ensure lifetime extends beyond solver creation
    ExtendedOptionData *ext_data = (ExtendedOptionData *)malloc(sizeof(ExtendedOptionData));
    if (ext_data == nullptr) {
        if (div_times_solver != nullptr) {
            free(div_times_solver);
        }
        return result;
    }

    ext_data->option_data = option_data;
    ext_data->x_min = grid_params->x_min;
    ext_data->x_max = grid_params->x_max;

    // Setup callbacks
    PDECallbacks callbacks = {
        .initial_condition = american_option_terminal_condition,
        .left_boundary = american_option_left_boundary,
        .right_boundary = american_option_right_boundary,
        .spatial_operator = american_option_spatial_operator,
        .jump_condition = nullptr,
        .obstacle = american_option_obstacle,
        .temporal_event = nullptr,
        .n_temporal_events = 0,
        .temporal_event_times = nullptr,
        .user_data = (void *)ext_data
    };

    // Enable temporal event callback for discrete dividends
    if (div_times_solver != nullptr) {
        callbacks.temporal_event = american_option_dividend_event;
        callbacks.n_temporal_events = option_data->n_dividends;
        callbacks.temporal_event_times = div_times_solver;
    }

    // Create solver configuration with relaxed tolerance
    BoundaryConfig bc_config = pde_default_boundary_config();
    TRBDF2Config trbdf2_config = pde_default_trbdf2_config();
    trbdf2_config.tolerance = 1e-4;  // Relaxed tolerance for American options
    trbdf2_config.max_iter = 200;    // More iterations allowed

    // Create and run solver
    PDESolver *solver = pde_solver_create(&grid, &time, &bc_config,
                                          &trbdf2_config, &callbacks);
    if (solver == nullptr) {
        return result;
    }

    pde_solver_initialize(solver);

    // Solve PDE (temporal events handled automatically by solver)
    int status = pde_solver_solve(solver);

    result.solver = solver;
    result.status = status;
    result.internal_data = (void *)ext_data;  // Store for cleanup

    // Clean up dividend time array
    if (div_times_solver != nullptr) {
        free(div_times_solver);
    }

    // Trace option pricing completion
    IVCALC_TRACE_OPTION_COMPLETE(status, grid_params->n_steps);

    return result;
}

// Free resources associated with AmericanOptionResult
void american_option_free_result(AmericanOptionResult *result) {
    if (result == nullptr) {
        return;
    }

    // Free the solver
    if (result->solver != nullptr) {
        pde_solver_destroy(result->solver);
        result->solver = nullptr;
    }

    // Free the extended option data
    if (result->internal_data != nullptr) {
        free(result->internal_data);
        result->internal_data = nullptr;
    }

    result->status = -1;
}

// Temporal event callback for discrete dividends
// Called by solver when dividend events are crossed
static void american_option_dividend_event(double t, const double *x_grid,
                                           size_t n_points, double *V,
                                           const size_t *event_indices,
                                           size_t n_events_triggered,
                                           void *user_data) {
    (void)t; // Unused: time is implicit in event indices
    ExtendedOptionData *ext_data = (ExtendedOptionData *)user_data;
    const OptionData *option_data = ext_data->option_data;

    // Allocate workspace for dividend adjustment
    double *V_temp = (double *)malloc(n_points * sizeof(double));
    if (V_temp == nullptr) {
        return; // Allocation failed, skip dividend handling
    }

    // Apply each triggered dividend
    for (size_t i = 0; i < n_events_triggered; i++) {
        size_t div_idx = event_indices[i];

        // Apply dividend jump to current solution
        american_option_apply_dividend(x_grid, n_points, V, V_temp,
                                      option_data->dividend_amounts[div_idx],
                                      option_data->strike);

        // Copy adjusted solution back
        for (size_t j = 0; j < n_points; j++) {
            V[j] = V_temp[j];
        }
    }

    // Clean up workspace
    free(V_temp);
}

// Utility: Get option value at specific spot price
double american_option_get_value_at_spot(const PDESolver *solver,
                                        double spot_price, double strike) {
    // Convert spot price to log-moneyness: x = ln(S/K)
    double x = log(spot_price / strike);
    return pde_solver_interpolate(solver, x);
}

// Batch API: Price multiple American options in parallel
int american_option_price_batch(const OptionData *option_data,
                                 const AmericanOptionGrid *grid_params,
                                 size_t n_options,
                                 AmericanOptionResult *results) {
    if (option_data == nullptr || grid_params == nullptr ||
        results == nullptr || n_options == 0) {
        return -1;
    }

    // Use OpenMP to price options in parallel
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n_options; i++) {
        results[i] = american_option_price(&option_data[i], grid_params);
    }

    // Check if any pricing failed
    for (size_t i = 0; i < n_options; i++) {
        if (results[i].status != 0) {
            return -1;
        }
    }

    return 0;
}
