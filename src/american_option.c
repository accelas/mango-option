#include "american_option.h"
#include "ivcalc_trace.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// Extended user data structure that includes grid boundaries
// This is needed for boundary conditions that depend on spatial location
typedef struct {
    const OptionData *option_data;
    double x_min;
    double x_max;
    bool use_neumann_bc;  // True if using Neumann BCs (return gradients), false for Dirichlet
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

// Helper: Compute intrinsic value (payoff) for American options (vectorized)
// This is used by both terminal condition and obstacle condition
// Intrinsic value = max(S - K, 0) for call, max(K - S, 0) for put
static inline void compute_intrinsic_value(const double *x, size_t n_points,
                                          double *values, double strike,
                                          OptionType option_type) {
    #pragma omp simd
    for (size_t i = 0; i < n_points; i++) {
        // x = ln(S/K), so S = K*exp(x)
        double S = strike * exp(x[i]);

        if (option_type == OPTION_CALL) {
            values[i] = fmax(S - strike, 0.0);
        } else {
            values[i] = fmax(strike - S, 0.0);
        }
    }
}

// Terminal condition: Option payoff at maturity (vectorized)
// At maturity, American = European = intrinsic value
void american_option_terminal_condition(const double *x, size_t n_points,
                                       double *V, void *user_data) {
    ExtendedOptionData *ext_data = (ExtendedOptionData *)user_data;
    const OptionData *data = ext_data->option_data;
    compute_intrinsic_value(x, n_points, V, data->strike, data->option_type);
}

// Helper: Compute Dirichlet boundary condition value for American options
// Returns the theoretical option value at boundary based on Black-Scholes
typedef enum { BOUNDARY_LEFT, BOUNDARY_RIGHT } BoundarySide;

static inline double compute_dirichlet_bc(const OptionData *data, double tau,
                                         double x_boundary, BoundarySide side) {
    const double K = data->strike;
    const double r = data->risk_free_rate;
    const double discount = exp(-r * tau);

    if (side == BOUNDARY_LEFT) {
        // Left boundary: S → 0 (x → -∞)
        // Call: worthless when S=0
        // Put: worth discounted strike when S=0 (certain to exercise)
        return (data->option_type == OPTION_CALL) ? 0.0 : K * discount;
    } else {
        // Right boundary: S → ∞ (x → +∞)
        // Call: worth intrinsic value S_max - K·e^(-rτ) (European value)
        // Put: worthless when S is very large
        if (data->option_type == OPTION_CALL) {
            const double S_max = K * exp(x_boundary);
            return S_max - K * discount;
        } else {
            return 0.0;
        }
    }
}

// Left boundary condition (S → 0, x → -∞)
double american_option_left_boundary(double t, void *user_data) {
    ExtendedOptionData *ext_data = (ExtendedOptionData *)user_data;

    // Neumann BC: zero gradient
    if (ext_data->use_neumann_bc) {
        return 0.0;
    }

    // Dirichlet BC: theoretical option value at S=0
    return compute_dirichlet_bc(ext_data->option_data, t, ext_data->x_min, BOUNDARY_LEFT);
}

// Right boundary condition (S → ∞, x → ∞)
double american_option_right_boundary(double t, void *user_data) {
    ExtendedOptionData *ext_data = (ExtendedOptionData *)user_data;

    // Neumann BC: zero gradient
    if (ext_data->use_neumann_bc) {
        return 0.0;
    }

    // Dirichlet BC: theoretical option value at S=S_max
    return compute_dirichlet_bc(ext_data->option_data, t, ext_data->x_max, BOUNDARY_RIGHT);
}

// Black-Scholes spatial operator in log-price coordinates (vectorized)
// L(V) = (1/2)σ²∂²V/∂x² + (r - σ²/2)∂V/∂x - rV
//
// Supports both uniform and non-uniform grids with optimized vectorization
void american_option_spatial_operator(const double * restrict x, [[maybe_unused]] double t,
                                     const double * restrict V, size_t n_points,
                                     double * restrict LV, void *user_data) {
    ExtendedOptionData *ext_data = (ExtendedOptionData *)user_data;
    const OptionData *data = ext_data->option_data;
    const double sigma = data->volatility;
    const double r = data->risk_free_rate;

    // Black-Scholes PDE coefficients in log-price coordinates
    const double coeff_2nd = 0.5 * sigma * sigma;         // (1/2)σ²
    const double coeff_1st = r - 0.5 * sigma * sigma;     // r - σ²/2
    const double coeff_0th = -r;                          // -r

    // Boundaries will be overwritten by BC
    LV[0] = 0.0;
    LV[n_points - 1] = 0.0;

    // Check if grid is uniform (common case from american_option_create_grid)
    // A grid is uniform if spacing is constant within numerical tolerance
    const double dx_first = x[1] - x[0];
    const double dx_last = x[n_points - 1] - x[n_points - 2];
    const double rel_diff = fabs(dx_last - dx_first) / dx_first;
    const bool is_uniform = (rel_diff < 1e-10);

    if (is_uniform) {
        // FAST PATH: Uniform grid - fully vectorizable with simple stencil
        // ∂V/∂x ≈ (V[i+1] - V[i-1]) / (2·dx)
        // ∂²V/∂x² ≈ (V[i+1] - 2·V[i] + V[i-1]) / dx²
        const double dx = dx_first;
        const double dx_inv = 1.0 / dx;
        const double dx2_inv = dx_inv * dx_inv;
        const double half_dx_inv = 0.5 * dx_inv;

        // Vectorized loop: no dependencies, simple memory access pattern
        #pragma omp simd
        for (size_t i = 1; i < n_points - 1; i++) {
            const double dV_dx = (V[i + 1] - V[i - 1]) * half_dx_inv;
            const double d2V_dx2 = (V[i + 1] - 2.0 * V[i] + V[i - 1]) * dx2_inv;
            LV[i] = coeff_2nd * d2V_dx2 + coeff_1st * dV_dx + coeff_0th * V[i];
        }
    } else {
        // SLOW PATH: Non-uniform grid - second-order accurate formulas
        // For non-uniform grid with local spacings:
        //   h_minus = x[i] - x[i-1]
        //   h_plus = x[i+1] - x[i]
        //
        // First derivative:
        //   ∂V/∂x = [-h+²·V[i-1] + (h+² - h-²)·V[i] + h-²·V[i+1]] / [h-·h+·(h- + h+)]
        //
        // Second derivative:
        //   ∂²V/∂x² = 2·[h+·V[i-1] - (h+ + h-)·V[i] + h-·V[i+1]] / [h-·h+·(h- + h+)]
        //
        // Note: Harder to vectorize due to per-point spacing calculations
        for (size_t i = 1; i < n_points - 1; i++) {
            const double h_minus = x[i] - x[i - 1];
            const double h_plus = x[i + 1] - x[i];
            const double h_sum = h_plus + h_minus;
            const double h_prod = h_minus * h_plus;
            const double denom = h_prod * h_sum;

            // First derivative (second-order accurate on non-uniform grid)
            const double dV_dx = (-h_plus * h_plus * V[i - 1] +
                                  (h_plus * h_plus - h_minus * h_minus) * V[i] +
                                  h_minus * h_minus * V[i + 1]) / denom;

            // Second derivative (second-order accurate on non-uniform grid)
            const double d2V_dx2 = 2.0 * (h_plus * V[i - 1] - h_sum * V[i] +
                                          h_minus * V[i + 1]) / denom;

            LV[i] = coeff_2nd * d2V_dx2 + coeff_1st * dV_dx + coeff_0th * V[i];
        }
    }
}

// Obstacle condition: American option constraint (vectorized)
// V(S,t) ≥ intrinsic_value(S)
void american_option_obstacle(const double *x, [[maybe_unused]] double t,
                             size_t n_points, double *obstacle, void *user_data) {
    ExtendedOptionData *ext_data = (ExtendedOptionData *)user_data;
    const OptionData *data = ext_data->option_data;
    compute_intrinsic_value(x, n_points, obstacle, data->strike, data->option_type);
}

// Forward declarations
static void american_option_dividend_event(double t, const double *x_grid,
                                           size_t n_points, double *V,
                                           const size_t *event_indices,
                                           size_t n_events_triggered,
                                           void *user_data,
                                           double *workspace);

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

        // Handle case where dividend causes stock price to go to zero or negative
        // In this case, we can't compute log(S_post/K), so we need special handling
        if (S_post <= 0.0) {
            // When stock price drops to zero or negative, use the boundary value
            // This is the most extreme case in the grid
            // For puts, this would be close to strike value
            // For calls, this would be close to zero
            V_new[i] = V_old[0];  // Use leftmost boundary value (lowest x)
            continue;
        }

        // Post-dividend log-price: x_post = ln(S_post/K)
        double x_post = log(S_post / strike);

        // Interpolate V from old grid (post-dividend) to find V(x_post)
        if (x_post <= x_grid[0]) {
            V_new[i] = V_old[0];
        } else if (x_post >= x_grid[n_points - 1]) {
            V_new[i] = V_old[n_points - 1];
        } else {
            // Binary search to find bracketing indices (O(log n) instead of O(n))
            // Find largest j such that x_grid[j] <= x_post < x_grid[j+1]
            size_t left = 0;
            size_t right = n_points - 1;
            while (right - left > 1) {
                size_t mid = left + (right - left) / 2;
                if (x_grid[mid] <= x_post) {
                    left = mid;
                } else {
                    right = mid;
                }
            }
            size_t j = left;

            // Linear interpolation between x_grid[j] and x_grid[j+1]
            double alpha = (x_post - x_grid[j]) / (x_grid[j + 1] - x_grid[j]);
            V_new[i] = (1.0 - alpha) * V_old[j] + alpha * V_old[j + 1];
        }
    }
}

// Internal implementation: Solve American option on moneyness grid with auto-detected BCs
// This is the shared implementation used by both public APIs
static AmericanOptionResult american_option_solve_internal(
    const OptionData *option_data,
    const double *m_grid,
    size_t n_m,
    double dt,
    size_t n_steps) {

    AmericanOptionResult result = {nullptr, -1, nullptr};

    if (option_data == nullptr || m_grid == nullptr || n_m < 3) {
        return result;
    }

    // Validate that moneyness grid is sorted in ascending order
    for (size_t i = 1; i < n_m; i++) {
        if (m_grid[i] <= m_grid[i - 1]) {
            // Grid must be strictly increasing
            return result;
        }
    }

    // Trace option pricing start
    MANGO_TRACE_OPTION_START(option_data->option_type, option_data->strike,
                              option_data->volatility, option_data->time_to_maturity);

    // Convert moneyness grid to log-moneyness: x = ln(m) = ln(S/K)
    double *x_grid = (double *)malloc(n_m * sizeof(double));
    if (x_grid == nullptr) {
        return result;
    }

    #pragma omp simd
    for (size_t i = 0; i < n_m; i++) {
        x_grid[i] = log(m_grid[i]);
    }

    // Auto-detect appropriate boundary conditions based on grid extent
    // Wide grids that extend to natural boundaries (S→0, S→∞) use Dirichlet BCs
    // Narrow grids use Neumann BCs (zero gradient) to avoid specifying unknown boundary values
    //
    // Heuristic: Use Dirichlet if grid extends significantly beyond ATM region
    // - Dirichlet: x_min < -0.5 AND x_max > 0.5 (covers m ∈ [0.6, 1.65] approximately)
    // - Neumann: Otherwise (narrow or moderate grids)
    const double x_min = x_grid[0];
    const double x_max = x_grid[n_m - 1];
    const double WIDE_GRID_THRESHOLD = 0.5;  // ln(1.65) ≈ 0.5, ln(0.6) ≈ -0.5
    bool use_neumann_bc = !(x_min < -WIDE_GRID_THRESHOLD && x_max > WIDE_GRID_THRESHOLD);

    // Create spatial grid (takes ownership of x_grid)
    SpatialGrid grid = {
        .x_min = x_grid[0],
        .x_max = x_grid[n_m - 1],
        .n_points = n_m,
        .dx = (x_grid[n_m - 1] - x_grid[0]) / (n_m - 1),  // Average spacing
        .x = x_grid
    };

    // Time domain: solve forward in time-to-maturity
    TimeDomain time = {
        .t_start = 0.0,
        .t_end = option_data->time_to_maturity,
        .dt = dt,
        .n_steps = n_steps
    };

    // Convert dividend times from calendar time to solver time
    double *div_times_solver = nullptr;
    if (option_data->n_dividends > 0 && option_data->dividend_times != nullptr &&
        option_data->dividend_amounts != nullptr) {
        div_times_solver = (double *)malloc(option_data->n_dividends * sizeof(double));
        if (div_times_solver == nullptr) {
            free(x_grid);
            return result;
        }

        const double T = option_data->time_to_maturity;
        #pragma omp simd
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
    ExtendedOptionData *ext_data = (ExtendedOptionData *)malloc(sizeof(ExtendedOptionData));
    if (ext_data == nullptr) {
        if (div_times_solver != nullptr) {
            free(div_times_solver);
        }
        free(x_grid);
        return result;
    }

    ext_data->option_data = option_data;
    ext_data->x_min = grid.x_min;
    ext_data->x_max = grid.x_max;
    ext_data->use_neumann_bc = use_neumann_bc;  // Use BC type from parameter

    // Setup callbacks (reuse existing callbacks)
    PDECallbacks callbacks = {
        .initial_condition = american_option_terminal_condition,
        .left_boundary = american_option_left_boundary,
        .right_boundary = american_option_right_boundary,
        .spatial_operator = american_option_spatial_operator,
        .diffusion_coeff = 0.5 * option_data->volatility * option_data->volatility,  // σ²/2
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
    // Configure BCs based on use_neumann_bc parameter:
    // - Neumann (zero gradient) for arbitrary user-provided grids that may not extend to natural boundaries
    // - Dirichlet for standard grids that extend to S→0, S→∞
    BoundaryConfig bc_config;
    if (use_neumann_bc) {
        bc_config = (BoundaryConfig){
            .left_type = BC_NEUMANN,
            .right_type = BC_NEUMANN,
            .left_robin_a = 1.0,
            .left_robin_b = 0.0,
            .right_robin_a = 1.0,
            .right_robin_b = 0.0
        };
    } else {
        bc_config = pde_default_boundary_config();  // Dirichlet BCs
    }
    TRBDF2Config trbdf2_config = pde_default_trbdf2_config();
    trbdf2_config.tolerance = 1e-4;  // Relaxed tolerance for American options
    trbdf2_config.max_iter = 200;    // More iterations allowed

    // Create and run solver (takes ownership of grid)
    PDESolver *solver = pde_solver_create(&grid, &time, &bc_config,
                                          &trbdf2_config, &callbacks);
    if (solver == nullptr) {
        if (div_times_solver != nullptr) {
            free(div_times_solver);
        }
        free(ext_data);
        // Note: grid.x is nullptr after pde_solver_create, no need to free x_grid
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
    MANGO_TRACE_OPTION_COMPLETE(status, n_steps);

    return result;
}

// Helper: Create uniform moneyness grid for American option pricing
double* american_option_create_grid(const AmericanOptionGrid *grid_params, size_t *n_out) {
    if (grid_params == nullptr || n_out == nullptr || grid_params->n_points < 3) {
        if (n_out != nullptr) {
            *n_out = 0;
        }
        return nullptr;
    }

    //Generate uniform moneyness grid from log-moneyness bounds
    // x = ln(S/K), so S/K = exp(x)
    double *m_grid = (double *)malloc(grid_params->n_points * sizeof(double));
    if (m_grid == nullptr) {
        *n_out = 0;
        return nullptr;
    }

    // Create uniform grid in log-moneyness space, then convert to moneyness
    for (size_t i = 0; i < grid_params->n_points; i++) {
        double x = grid_params->x_min +
                   (grid_params->x_max - grid_params->x_min) * i / (grid_params->n_points - 1);
        m_grid[i] = exp(x);  // Convert to moneyness S/K
    }

    *n_out = grid_params->n_points;
    return m_grid;
}

// Solve American option (PRIMARY AND ONLY API)
AmericanOptionResult american_option_solve(
    const OptionData *option_data,
    const double *m_grid,
    size_t n_m,
    double dt,
    size_t n_steps) {
    // Call unified internal implementation (auto-detects appropriate BCs)
    return american_option_solve_internal(option_data, m_grid, n_m, dt, n_steps);
}

// TEMPORARY COMPATIBILITY WRAPPER - DO NOT USE IN NEW CODE
// This exists only to avoid breaking existing callers during refactoring
// Will be removed once all callers are updated to use american_option_solve()
AmericanOptionResult american_option_price(const OptionData *option_data,
                                          const AmericanOptionGrid *grid_params) {
    AmericanOptionResult result = {nullptr, -1, nullptr};

    size_t n_m;
    double *m_grid = american_option_create_grid(grid_params, &n_m);
    if (m_grid == nullptr) {
        return result;
    }

    result = american_option_solve(option_data, m_grid, n_m,
                                  grid_params->dt, grid_params->n_steps);
    free(m_grid);
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
static void american_option_dividend_event([[maybe_unused]] double t,
                                           const double *x_grid,
                                           size_t n_points, double *V,
                                           const size_t *event_indices,
                                           size_t n_events_triggered,
                                           void *user_data,
                                           double *workspace) {
    ExtendedOptionData *ext_data = (ExtendedOptionData *)user_data;
    const OptionData *option_data = ext_data->option_data;

    // Use workspace instead of malloc
    // workspace already allocated by solver (n_points doubles)
    double *V_temp = workspace;  // No malloc needed!

    // Apply each triggered dividend
    for (size_t i = 0; i < n_events_triggered; i++) {
        size_t div_idx = event_indices[i];

        // Apply dividend jump to current solution
        american_option_apply_dividend(x_grid, n_points, V, V_temp,
                                      option_data->dividend_amounts[div_idx],
                                      option_data->strike);

        // Copy adjusted solution back
        #pragma omp simd
        for (size_t j = 0; j < n_points; j++) {
            V[j] = V_temp[j];
        }
    }

    // No free needed - workspace managed by solver
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

    // Create grid once for all options in batch (shared grid)
    size_t n_m;
    double *m_grid = american_option_create_grid(grid_params, &n_m);
    if (m_grid == nullptr) {
        return -1;
    }

    // Use OpenMP to price options in parallel
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n_options; i++) {
        results[i] = american_option_solve(&option_data[i], m_grid, n_m,
                                          grid_params->dt, grid_params->n_steps);
    }

    // Clean up shared grid
    free(m_grid);

    // Check if any pricing failed
    for (size_t i = 0; i < n_options; i++) {
        if (results[i].status != 0) {
            return -1;
        }
    }

    return 0;
}
