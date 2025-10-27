#include "american_option.h"
#include <math.h>
#include <stdlib.h>

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
    OptionData *data = (OptionData *)user_data;
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
    OptionData *data = (OptionData *)user_data;
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
double american_option_right_boundary([[maybe_unused]] double t, void *user_data) {
    OptionData *data = (OptionData *)user_data;

    if (data->option_type == OPTION_CALL) {
        // For call at large S: V ≈ S - K*exp(-r*τ)
        // But for American options with no dividends, V ≈ S (never exercise early)
        // At the grid boundary, we approximate V = S_max - K
        // This will be enforced by the obstacle condition anyway
        return 0.0; // Will be overridden by obstacle
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
    OptionData *data = (OptionData *)user_data;
    const double sigma = data->volatility;
    const double r = data->risk_free_rate;
    const double dx = (x[n_points - 1] - x[0]) / (n_points - 1);
    const double dx_inv = 1.0 / dx;
    const double dx2_inv = 1.0 / (dx * dx);

    // TODO: Fix time mapping - current implementation has issues
    // The correct formulation requires careful mapping between calendar time
    // and time-to-maturity for backward parabolic PDE
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
    OptionData *data = (OptionData *)user_data;
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

// High-level API to price American options
AmericanOptionResult american_option_price(const OptionData *option_data,
                                          const AmericanOptionGrid *grid_params) {
    AmericanOptionResult result = {nullptr, -1};

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

    // Setup callbacks
    PDECallbacks callbacks = {
        .initial_condition = american_option_terminal_condition,
        .left_boundary = american_option_left_boundary,
        .right_boundary = american_option_right_boundary,
        .spatial_operator = american_option_spatial_operator,
        .jump_condition = nullptr, // TODO: Implement for discrete dividends
        .obstacle = american_option_obstacle,
        .user_data = (void *)option_data
    };

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
    int status = pde_solver_solve(solver);

    result.solver = solver;
    result.status = status;
    return result;
}

// Utility: Get option value at specific spot price
double american_option_get_value_at_spot(const PDESolver *solver,
                                        double spot_price, double strike) {
    // Convert spot price to log-moneyness: x = ln(S/K)
    double x = log(spot_price / strike);
    return pde_solver_interpolate(solver, x);
}
