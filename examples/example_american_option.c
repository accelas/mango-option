#include "src/pde_solver.h"
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

// American Option Pricing using Black-Scholes PDE with discrete dividends
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

typedef enum {
    OPTION_CALL,
    OPTION_PUT
} OptionType;

typedef struct {
    double strike;           // Strike price K
    double volatility;       // σ
    double risk_free_rate;   // r
    double time_to_maturity; // T
    OptionType option_type;  // Call or Put

    // Discrete dividend information
    size_t n_dividends;      // Number of dividend payments
    double *dividend_times;  // Times of dividend payments (in years)
    double *dividend_amounts;// Dividend amounts (absolute cash dividends)
} OptionData;

// Initial condition: European option value at maturity (vectorized)
// At maturity, American = European = max(S - K, 0) for call
static void option_terminal_condition(const double *x, size_t n_points,
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
static double option_left_boundary(double t, void *user_data) {
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
static double option_right_boundary([[maybe_unused]] double t, void *user_data) {
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
static void option_spatial_operator(const double *x, [[maybe_unused]] double t, const double *V,
                                   size_t n_points, double *LV, void *user_data) {
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

// Note: Jump condition for discrete dividends would be implemented here
// When a dividend D is paid, stock price jumps: S → S - D
// In log-space: x = ln(S/K) → ln((S-D)/K)
// This requires tracking dividend payment times during the solve
// For this example, we don't use the jump condition callback

// Obstacle condition: American option constraint (vectorized)
// V(S,t) ≥ intrinsic_value(S)
static void option_obstacle(const double *x, [[maybe_unused]] double t, size_t n_points,
                           double *obstacle, void *user_data) {
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

// Utility: Print option value as a function of spot price
static void print_option_values(const PDESolver *solver, const OptionData *data,
                               const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == nullptr) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return;
    }

    const double *x = pde_solver_get_grid(solver);
    const double *V = pde_solver_get_solution(solver);
    const size_t n = solver->grid.n_points;
    const double K = data->strike;

    fprintf(fp, "# S (spot) | V(S,0) (option value) | Intrinsic Value\n");
    for (size_t i = 0; i < n; i++) {
        double S = K * exp(x[i]);
        double intrinsic;

        if (data->option_type == OPTION_CALL) {
            intrinsic = fmax(S - K, 0.0);
        } else {
            intrinsic = fmax(K - S, 0.0);
        }

        fprintf(fp, "%.6f %.10e %.10e\n", S, V[i], intrinsic);
    }

    fclose(fp);
    printf("Option values written to %s\n", filename);
}

int main(void) {
    printf("=== American Option Pricing with PDE Solver ===\n\n");

    // Example 1: American Call Option (no dividends)
    printf("Example 1: American Call Option (no dividends)\n");
    printf("-----------------------------------------------\n");

    OptionData call_data = {
        .strike = 100.0,
        .volatility = 0.2,      // 20% volatility
        .risk_free_rate = 0.05,  // 5% risk-free rate
        .time_to_maturity = 1.0, // 1 year to maturity
        .option_type = OPTION_CALL,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    // Create spatial grid in log-price coordinates: x = ln(S/K)
    // Cover spot prices from S/K ∈ [0.5, 2.0] → x ∈ [ln(0.5), ln(2.0)] ≈ [-0.69, 0.69]
    SpatialGrid grid = pde_create_grid(-0.7, 0.7, 141); // Fine grid for accuracy

    // Time domain: solve forward in time-to-maturity
    TimeDomain time = {
        .t_start = 0.0,
        .t_end = call_data.time_to_maturity,
        .dt = 0.001,  // Smaller time step for stability
        .n_steps = 1000
    };

    // Setup callbacks
    PDECallbacks callbacks = {
        .initial_condition = option_terminal_condition,
        .left_boundary = option_left_boundary,
        .right_boundary = option_right_boundary,
        .spatial_operator = option_spatial_operator,
        .jump_condition = nullptr, // No dividends in this example
        .obstacle = option_obstacle, // American constraint
        .user_data = &call_data
    };

    // Create solver configuration with relaxed tolerance
    BoundaryConfig bc_config = pde_default_boundary_config();
    TRBDF2Config trbdf2_config = pde_default_trbdf2_config();
    trbdf2_config.tolerance = 1e-4;  // Relaxed tolerance for American options
    trbdf2_config.max_iter = 200;    // More iterations allowed

    // Create and run solver
    PDESolver *solver = pde_solver_create(&grid, &time, &bc_config,
                                          &trbdf2_config, &callbacks);
    pde_solver_initialize(solver);
    int status = pde_solver_solve(solver);

    if (status == 0) {
        print_option_values(solver, &call_data, "american_call_values.dat");

        // Calculate option value at spot = strike (ATM)
        double x_atm = 0.0; // ln(S/K) = 0 when S = K
        double V_atm = pde_solver_interpolate(solver, x_atm);
        printf("American Call value at S=K=%.2f: %.6f\n", call_data.strike, V_atm);
    } else {
        printf("Solver failed to converge for American Call\n");
    }

    pde_solver_destroy(solver);

    // Example 2: American Put Option (no dividends)
    printf("\nExample 2: American Put Option (no dividends)\n");
    printf("----------------------------------------------\n");

    // Create new grid (ownership was transferred to previous solver)
    grid = pde_create_grid(-0.7, 0.7, 141);

    OptionData put_data = {
        .strike = 100.0,
        .volatility = 0.2,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    callbacks.user_data = &put_data;

    solver = pde_solver_create(&grid, &time, &bc_config, &trbdf2_config, &callbacks);
    pde_solver_initialize(solver);
    status = pde_solver_solve(solver);

    if (status == 0) {
        print_option_values(solver, &put_data, "american_put_values.dat");

        double x_atm = 0.0;
        double V_atm = pde_solver_interpolate(solver, x_atm);
        printf("American Put value at S=K=%.2f: %.6f\n", put_data.strike, V_atm);
    } else {
        printf("Solver failed to converge for American Put\n");
    }

    pde_solver_destroy(solver);
    // Note: grid ownership was transferred to last solver, no need to free

    printf("\nAll examples completed!\n");
    printf("\nNote: For American options:\n");
    printf("  - Early exercise premium = American value - European value\n");
    printf("  - Call options have minimal early exercise value (no dividends)\n");
    printf("  - Put options have significant early exercise value\n");

    return 0;
}
