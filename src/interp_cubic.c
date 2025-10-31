#include "interp_cubic.h"
#include "iv_surface.h"
#include "price_table.h"
#include "cubic_spline.h"
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

/**
 * Cubic interpolation context
 *
 * For tensor-product cubic splines, we pre-compute spline coefficients
 * for slices along each dimension to avoid solving tridiagonal systems
 * at query time.
 */
typedef struct {
    size_t dimensions;        // 2, 4, or 5
    size_t *grid_sizes;       // Size of each dimension
    void *coefficients;       // Strategy-specific storage
} CubicContext;

/**
 * Pre-computed coefficients for 2D interpolation
 * Stores one spline per maturity slice (varying moneyness)
 */
typedef struct {
    size_t n_maturity;
    const double *moneyness_grid;  // Not owned, points to surface grid
    const double *maturity_grid;   // Not owned, points to surface grid
    CubicSpline **moneyness_splines;  // Array of n_maturity splines
} Cubic2DCoeffs;

/**
 * Pre-computed coefficients for 4D interpolation
 * Stores one spline per (tau, sigma, r) combination (varying moneyness)
 * Uses flat array with index = j_tau * (n_sigma * n_r) + j_sigma * n_r + j_r
 */
typedef struct {
    size_t n_maturity;
    size_t n_volatility;
    size_t n_rate;
    const double *moneyness_grid;   // Not owned, points to table grid
    CubicSpline **moneyness_splines;  // Flat array of n_tau * n_sigma * n_r spline pointers
} Cubic4DCoeffs;

/**
 * Pre-computed coefficients for 5D interpolation
 * Stores one spline per (tau, sigma, r, q) combination (varying moneyness)
 * Uses flat array with index = j_tau * (n_sigma * n_r * n_q) + j_sigma * (n_r * n_q) + j_r * n_q + j_q
 */
typedef struct {
    size_t n_maturity;
    size_t n_volatility;
    size_t n_rate;
    size_t n_dividend;
    const double *moneyness_grid;   // Not owned, points to table grid
    CubicSpline **moneyness_splines;  // Flat array of n_tau * n_sigma * n_r * n_q spline pointers
} Cubic5DCoeffs;

// Forward declarations
static double cubic_interpolate_2d(const IVSurface *surface,
                                   double moneyness, double maturity,
                                   InterpContext context);

static double cubic_interpolate_4d(const OptionPriceTable *table,
                                   double moneyness, double maturity,
                                   double volatility, double rate,
                                   InterpContext context);

static double cubic_interpolate_5d(const OptionPriceTable *table,
                                   double moneyness, double maturity,
                                   double volatility, double rate,
                                   double dividend,
                                   InterpContext context);

static InterpContext cubic_create_context(size_t dimensions,
                                          const size_t *grid_sizes);

static void cubic_destroy_context(InterpContext context);

static int cubic_precompute(const void *grid_data, InterpContext context);

// ---------- Strategy Definition ----------

const InterpolationStrategy INTERP_CUBIC = {
    .name = "cubic",
    .description = "Tensor-product cubic splines (C2 continuous, smooth derivatives)",
    .interpolate_2d = cubic_interpolate_2d,
    .interpolate_4d = cubic_interpolate_4d,
    .interpolate_5d = cubic_interpolate_5d,
    .create_context = cubic_create_context,
    .destroy_context = cubic_destroy_context,
    .precompute = cubic_precompute
};

// ---------- Helper Functions ----------

// Check if query is within grid bounds (with small tolerance for rounding)
static bool is_within_bounds(double query, const double *grid, size_t n) {
    const double tolerance = 1e-10;  // Small tolerance for floating point
    return (query >= grid[0] - tolerance && query <= grid[n-1] + tolerance);
}

// Evaluate cubic polynomial: a + b·dx + c·dx² + d·dx³
static inline double eval_cubic(double a, double b, double c, double d, double dx) {
    return a + dx * (b + dx * (c + dx * d));
}

// ---------- 2D Interpolation (IV Surface) ----------

// Workspace-based 2D cubic interpolation (zero malloc version)
double cubic_interpolate_2d_workspace(const IVSurface *surface,
                                       double moneyness, double maturity,
                                       CubicInterpWorkspace workspace) {
    if (surface == NULL) {
        return NAN;
    }

    const size_t n_m = surface->n_moneyness;
    const size_t n_tau = surface->n_maturity;

    // Use workspace slices
    double *intermediate_values = workspace.intermediate_arrays;  // n_tau doubles
    double *moneyness_slice = workspace.slice_buffers;           // n_m doubles

    // Stage 1: Interpolate along moneyness for each maturity point
    for (size_t j_tau = 0; j_tau < n_tau; j_tau++) {
        // Extract moneyness slice at this maturity
        for (size_t i_m = 0; i_m < n_m; i_m++) {
            moneyness_slice[i_m] = surface->iv_surface[j_tau * n_m + i_m];
        }

        // Create spline using workspace (zero malloc)
        CubicSpline m_spline;
        int ret = pde_spline_init(&m_spline, surface->moneyness_grid, moneyness_slice,
                                  n_m, workspace.spline_coeff_workspace,
                                  workspace.spline_temp_workspace);
        if (ret != 0) {
            return NAN;
        }

        // Evaluate at query moneyness
        intermediate_values[j_tau] = pde_spline_eval(&m_spline, moneyness);
    }

    // Stage 2: Interpolate along maturity using intermediate values
    CubicSpline tau_spline;
    int ret = pde_spline_init(&tau_spline, surface->maturity_grid, intermediate_values,
                              n_tau, workspace.spline_coeff_workspace,
                              workspace.spline_temp_workspace);
    if (ret != 0) {
        return NAN;
    }

    return pde_spline_eval(&tau_spline, maturity);
}

// Workspace-based 4D cubic interpolation (zero malloc version)
double cubic_interpolate_4d_workspace(const OptionPriceTable *table,
                                       double moneyness, double maturity,
                                       double volatility, double rate,
                                       CubicInterpWorkspace workspace) {
    if (table == NULL || table->n_dividend > 0) {
        return NAN;
    }

    const size_t n_m = table->n_moneyness;
    const size_t n_tau = table->n_maturity;
    const size_t n_sigma = table->n_volatility;
    const size_t n_r = table->n_rate;

    // Slice workspace into intermediate arrays
    const size_t n1 = n_tau * n_sigma * n_r;
    const size_t n2 = n_sigma * n_r;

    double *intermediate1 = workspace.intermediate_arrays;
    double *intermediate2 = intermediate1 + n1;
    double *intermediate3 = intermediate2 + n2;
    double *slice = workspace.slice_buffers;

    // Stage 1: Interpolate along moneyness (n_tau × n_sigma × n_r splines)
    for (size_t j_tau = 0; j_tau < n_tau; j_tau++) {
        for (size_t k_sigma = 0; k_sigma < n_sigma; k_sigma++) {
            for (size_t l_r = 0; l_r < n_r; l_r++) {
                // Extract moneyness slice
                for (size_t i_m = 0; i_m < n_m; i_m++) {
                    size_t idx = i_m * table->stride_m + j_tau * table->stride_tau +
                                 k_sigma * table->stride_sigma + l_r * table->stride_r;
                    slice[i_m] = table->prices[idx];
                }

                // Create spline and evaluate
                CubicSpline m_spline;
                int ret = pde_spline_init(&m_spline, table->moneyness_grid, slice, n_m,
                                          workspace.spline_coeff_workspace,
                                          workspace.spline_temp_workspace);
                if (ret != 0) return NAN;

                size_t idx1 = j_tau * n_sigma * n_r + k_sigma * n_r + l_r;
                intermediate1[idx1] = pde_spline_eval(&m_spline, moneyness);
            }
        }
    }

    // Stage 2: Interpolate along maturity (n_sigma × n_r splines)
    for (size_t k_sigma = 0; k_sigma < n_sigma; k_sigma++) {
        for (size_t l_r = 0; l_r < n_r; l_r++) {
            // Extract maturity slice from intermediate1
            for (size_t j_tau = 0; j_tau < n_tau; j_tau++) {
                slice[j_tau] = intermediate1[j_tau * n_sigma * n_r + k_sigma * n_r + l_r];
            }

            CubicSpline tau_spline;
            int ret = pde_spline_init(&tau_spline, table->maturity_grid, slice, n_tau,
                                      workspace.spline_coeff_workspace,
                                      workspace.spline_temp_workspace);
            if (ret != 0) return NAN;

            size_t idx2 = k_sigma * n_r + l_r;
            intermediate2[idx2] = pde_spline_eval(&tau_spline, maturity);
        }
    }

    // Stage 3: Interpolate along volatility (n_r splines)
    for (size_t l_r = 0; l_r < n_r; l_r++) {
        // Extract volatility slice from intermediate2
        for (size_t k_sigma = 0; k_sigma < n_sigma; k_sigma++) {
            slice[k_sigma] = intermediate2[k_sigma * n_r + l_r];
        }

        CubicSpline sigma_spline;
        int ret = pde_spline_init(&sigma_spline, table->volatility_grid, slice, n_sigma,
                                  workspace.spline_coeff_workspace,
                                  workspace.spline_temp_workspace);
        if (ret != 0) return NAN;

        intermediate3[l_r] = pde_spline_eval(&sigma_spline, volatility);
    }

    // Stage 4: Final interpolation along rate (1 spline)
    CubicSpline r_spline;
    int ret = pde_spline_init(&r_spline, table->rate_grid, intermediate3, n_r,
                              workspace.spline_coeff_workspace,
                              workspace.spline_temp_workspace);
    if (ret != 0) return NAN;

    return pde_spline_eval(&r_spline, rate);
}

// Workspace-based 5D cubic interpolation (zero malloc version)
double cubic_interpolate_5d_workspace(const OptionPriceTable *table,
                                       double moneyness, double maturity,
                                       double volatility, double rate,
                                       double dividend,
                                       CubicInterpWorkspace workspace) {
    if (table == NULL || table->n_dividend == 0) {
        return NAN;
    }

    const size_t n_m = table->n_moneyness;
    const size_t n_tau = table->n_maturity;
    const size_t n_sigma = table->n_volatility;
    const size_t n_r = table->n_rate;
    const size_t n_q = table->n_dividend;

    // Slice workspace into intermediate arrays
    const size_t n1 = n_tau * n_sigma * n_r * n_q;
    const size_t n2 = n_sigma * n_r * n_q;
    const size_t n3 = n_r * n_q;

    double *intermediate1 = workspace.intermediate_arrays;
    double *intermediate2 = intermediate1 + n1;
    double *intermediate3 = intermediate2 + n2;
    double *intermediate4 = intermediate3 + n3;
    double *slice = workspace.slice_buffers;

    // Stage 1: Moneyness (n_tau × n_sigma × n_r × n_q splines)
    for (size_t j = 0; j < n_tau; j++) {
        for (size_t k = 0; k < n_sigma; k++) {
            for (size_t l = 0; l < n_r; l++) {
                for (size_t m = 0; m < n_q; m++) {
                    // Extract moneyness slice
                    for (size_t i = 0; i < n_m; i++) {
                        size_t idx = i * table->stride_m + j * table->stride_tau +
                                     k * table->stride_sigma + l * table->stride_r +
                                     m * table->stride_q;
                        slice[i] = table->prices[idx];
                    }

                    CubicSpline spline;
                    int ret = pde_spline_init(&spline, table->moneyness_grid, slice, n_m,
                                              workspace.spline_coeff_workspace,
                                              workspace.spline_temp_workspace);
                    if (ret != 0) return NAN;

                    size_t idx1 = j * n_sigma * n_r * n_q + k * n_r * n_q + l * n_q + m;
                    intermediate1[idx1] = pde_spline_eval(&spline, moneyness);
                }
            }
        }
    }

    // Stage 2: Maturity (n_sigma × n_r × n_q splines)
    for (size_t k = 0; k < n_sigma; k++) {
        for (size_t l = 0; l < n_r; l++) {
            for (size_t m = 0; m < n_q; m++) {
                for (size_t j = 0; j < n_tau; j++) {
                    slice[j] = intermediate1[j * n_sigma * n_r * n_q + k * n_r * n_q + l * n_q + m];
                }

                CubicSpline spline;
                int ret = pde_spline_init(&spline, table->maturity_grid, slice, n_tau,
                                          workspace.spline_coeff_workspace,
                                          workspace.spline_temp_workspace);
                if (ret != 0) return NAN;

                size_t idx2 = k * n_r * n_q + l * n_q + m;
                intermediate2[idx2] = pde_spline_eval(&spline, maturity);
            }
        }
    }

    // Stage 3: Volatility (n_r × n_q splines)
    for (size_t l = 0; l < n_r; l++) {
        for (size_t m = 0; m < n_q; m++) {
            for (size_t k = 0; k < n_sigma; k++) {
                slice[k] = intermediate2[k * n_r * n_q + l * n_q + m];
            }

            CubicSpline spline;
            int ret = pde_spline_init(&spline, table->volatility_grid, slice, n_sigma,
                                      workspace.spline_coeff_workspace,
                                      workspace.spline_temp_workspace);
            if (ret != 0) return NAN;

            size_t idx3 = l * n_q + m;
            intermediate3[idx3] = pde_spline_eval(&spline, volatility);
        }
    }

    // Stage 4: Rate (n_q splines)
    for (size_t m = 0; m < n_q; m++) {
        for (size_t l = 0; l < n_r; l++) {
            slice[l] = intermediate3[l * n_q + m];
        }

        CubicSpline spline;
        int ret = pde_spline_init(&spline, table->rate_grid, slice, n_r,
                                  workspace.spline_coeff_workspace,
                                  workspace.spline_temp_workspace);
        if (ret != 0) return NAN;

        intermediate4[m] = pde_spline_eval(&spline, rate);
    }

    // Stage 5: Dividend (final)
    CubicSpline q_spline;
    int ret = pde_spline_init(&q_spline, table->dividend_grid, intermediate4, n_q,
                              workspace.spline_coeff_workspace,
                              workspace.spline_temp_workspace);
    if (ret != 0) return NAN;

    return pde_spline_eval(&q_spline, dividend);
}

/**
 * Proper tensor-product cubic spline interpolation
 *
 * Algorithm:
 * 1. For EACH maturity grid point, build cubic spline along moneyness and evaluate at query_m
 *    This gives intermediate[j] = spline_m(data[:, j], query_m) for all j
 * 2. Build cubic spline along maturity using intermediate values and evaluate at query_τ
 *    This gives result = spline_τ(intermediate, query_τ)
 *
 * This is the standard separable tensor-product approach that guarantees:
 * - On-grid points are exact (splines interpolate data exactly)
 * - C² continuous everywhere
 * - Uses full grid information in each dimension
 */
static double cubic_interpolate_2d(const IVSurface *surface,
                                   double moneyness, double maturity,
                                   InterpContext context) {
    // Need at least 2 points in each dimension for cubic spline (natural boundary conditions)
    if (surface->n_maturity < 2 || surface->n_moneyness < 2) {
        return NAN;
    }

    // Validate query point is within grid bounds
    if (!is_within_bounds(moneyness, surface->moneyness_grid, surface->n_moneyness) ||
        !is_within_bounds(maturity, surface->maturity_grid, surface->n_maturity)) {
        return NAN;  // Query point outside grid coverage
    }

    // Check if we have pre-computed coefficients
    CubicContext *ctx = (CubicContext*)context;
    Cubic2DCoeffs *coeffs = (ctx && ctx->coefficients) ? (Cubic2DCoeffs*)ctx->coefficients : NULL;

    // Allocate workspaces for zero-malloc spline creation (slow path only)
    size_t max_grid_size = surface->n_moneyness > surface->n_maturity ?
                          surface->n_moneyness : surface->n_maturity;
    double *spline_coeff_workspace = malloc(4 * max_grid_size * sizeof(double));
    double *spline_temp_workspace = malloc(6 * max_grid_size * sizeof(double));
    if (!spline_coeff_workspace || !spline_temp_workspace) {
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    // Step 1: For each maturity grid point, interpolate along moneyness at query point
    // This gives us intermediate_values[j] for all j in [0, n_maturity)
    double *intermediate_values = malloc(surface->n_maturity * sizeof(double));
    if (!intermediate_values) {
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    if (coeffs) {
        // Fast path: Use pre-computed spline coefficients
        #pragma omp simd
        for (size_t j_tau = 0; j_tau < surface->n_maturity; j_tau++) {
            intermediate_values[j_tau] = pde_spline_eval(coeffs->moneyness_splines[j_tau], moneyness);
        }
    } else {
        // Slow path: Compute splines on-the-fly using workspace-based API
        double *moneyness_slice = malloc(surface->n_moneyness * sizeof(double));
        if (!moneyness_slice) {
            free(intermediate_values);
            free(spline_coeff_workspace);
            free(spline_temp_workspace);
            return NAN;
        }

        CubicSpline m_spline;  // Stack-allocated

        for (size_t j_tau = 0; j_tau < surface->n_maturity; j_tau++) {
            // Extract moneyness slice at this maturity level
            // Data layout: iv_surface[j_tau * n_moneyness + i_m] (maturity varies slowest)
            #pragma omp simd
            for (size_t i_m = 0; i_m < surface->n_moneyness; i_m++) {
                size_t idx = j_tau * surface->n_moneyness + i_m;
                moneyness_slice[i_m] = surface->iv_surface[idx];
            }

            // Initialize cubic spline using workspace (zero-malloc)
            int result = pde_spline_init(&m_spline, surface->moneyness_grid,
                                         moneyness_slice, surface->n_moneyness,
                                         spline_coeff_workspace, spline_temp_workspace);
            if (result != 0) {
                free(intermediate_values);
                free(moneyness_slice);
                free(spline_coeff_workspace);
                free(spline_temp_workspace);
                return NAN;
            }

            // Evaluate at query moneyness
            intermediate_values[j_tau] = pde_spline_eval(&m_spline, moneyness);
            // No destroy needed - workspace is reused
        }

        free(moneyness_slice);
    }

    // Step 2: Build cubic spline along maturity using intermediate values
    CubicSpline tau_spline;  // Stack-allocated
    int result_init = pde_spline_init(&tau_spline, surface->maturity_grid,
                                      intermediate_values, surface->n_maturity,
                                      spline_coeff_workspace, spline_temp_workspace);
    if (result_init != 0) {
        free(intermediate_values);
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    // Evaluate at query maturity
    double result = pde_spline_eval(&tau_spline, maturity);
    // No destroy needed - workspace managed by caller

    free(intermediate_values);
    free(spline_coeff_workspace);
    free(spline_temp_workspace);

    return result;
}

// ---------- 4D Interpolation (Price Table) ----------

/**
 * 4D tensor-product cubic spline interpolation
 *
 * Algorithm (separable tensor-product):
 * 1. For each (maturity, volatility, rate) combo: interpolate along moneyness
 *    → Gives n_tau × n_sigma × n_r intermediate values
 * 2. For each (volatility, rate) combo: interpolate along maturity
 *    → Gives n_sigma × n_r intermediate values
 * 3. For each rate: interpolate along volatility
 *    → Gives n_r intermediate values
 * 4. Interpolate along rate
 *    → Gives final result
 *
 * This is C² continuous and allows accurate computation of second derivatives (gamma, volga, etc.)
 */
static double cubic_interpolate_4d(const OptionPriceTable *table,
                                   double moneyness, double maturity,
                                   double volatility, double rate,
                                   InterpContext context) {
    // Need at least 2 points in each dimension for cubic splines
    if (table->n_moneyness < 2 || table->n_maturity < 2 ||
        table->n_volatility < 2 || table->n_rate < 2) {
        return NAN;
    }

    // Validate query point is within grid bounds
    if (!is_within_bounds(moneyness, table->moneyness_grid, table->n_moneyness) ||
        !is_within_bounds(maturity, table->maturity_grid, table->n_maturity) ||
        !is_within_bounds(volatility, table->volatility_grid, table->n_volatility) ||
        !is_within_bounds(rate, table->rate_grid, table->n_rate)) {
        return NAN;  // Query point outside grid coverage
    }

    // Check if we have pre-computed coefficients
    CubicContext *ctx = (CubicContext*)context;
    Cubic4DCoeffs *coeffs = (ctx && ctx->coefficients) ? (Cubic4DCoeffs*)ctx->coefficients : NULL;

    // Allocate workspaces for zero-malloc spline creation (slow path only)
    // Find maximum grid size across all dimensions
    size_t max_grid_size = table->n_moneyness;
    if (table->n_maturity > max_grid_size) max_grid_size = table->n_maturity;
    if (table->n_volatility > max_grid_size) max_grid_size = table->n_volatility;
    if (table->n_rate > max_grid_size) max_grid_size = table->n_rate;

    // Allocate workspaces once for the entire interpolation
    double *spline_coeff_workspace = malloc(4 * max_grid_size * sizeof(double));
    double *spline_temp_workspace = malloc(6 * max_grid_size * sizeof(double));
    if (!spline_coeff_workspace || !spline_temp_workspace) {
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    // Stage 1: Interpolate along moneyness for each (tau, sigma, r) combination
    // Result: intermediate1[j_tau][j_sigma][j_r]
    size_t n1 = table->n_maturity * table->n_volatility * table->n_rate;
    double *intermediate1 = malloc(n1 * sizeof(double));
    if (!intermediate1) {
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    if (coeffs) {
        // Fast path: Use pre-computed spline coefficients
        for (size_t j_tau = 0; j_tau < table->n_maturity; j_tau++) {
            for (size_t j_sigma = 0; j_sigma < table->n_volatility; j_sigma++) {
                for (size_t j_r = 0; j_r < table->n_rate; j_r++) {
                    size_t idx1 = j_tau * (table->n_volatility * table->n_rate)
                                + j_sigma * table->n_rate
                                + j_r;
                    size_t spline_idx = j_tau * (coeffs->n_volatility * coeffs->n_rate)
                                      + j_sigma * coeffs->n_rate
                                      + j_r;
                    intermediate1[idx1] = pde_spline_eval(coeffs->moneyness_splines[spline_idx], moneyness);
                }
            }
        }
    } else {
        // Slow path: Compute splines on-the-fly using workspace-based API
        double *moneyness_slice = malloc(table->n_moneyness * sizeof(double));
        if (!moneyness_slice) {
            free(intermediate1);
            free(spline_coeff_workspace);
            free(spline_temp_workspace);
            return NAN;
        }

        CubicSpline m_spline;  // Stack-allocated spline structure

        for (size_t j_tau = 0; j_tau < table->n_maturity; j_tau++) {
            for (size_t j_sigma = 0; j_sigma < table->n_volatility; j_sigma++) {
                for (size_t j_r = 0; j_r < table->n_rate; j_r++) {
                    // Extract moneyness slice: fix (tau, sigma, r), vary moneyness
                    for (size_t i_m = 0; i_m < table->n_moneyness; i_m++) {
                        size_t idx = i_m * table->stride_m
                                   + j_tau * table->stride_tau
                                   + j_sigma * table->stride_sigma
                                   + j_r * table->stride_r;
                        moneyness_slice[i_m] = table->prices[idx];
                    }

                    // Initialize cubic spline using workspace (zero-malloc)
                    int result = pde_spline_init(&m_spline, table->moneyness_grid,
                                                  moneyness_slice, table->n_moneyness,
                                                  spline_coeff_workspace, spline_temp_workspace);
                    if (result != 0) {
                        free(intermediate1);
                        free(moneyness_slice);
                        free(spline_coeff_workspace);
                        free(spline_temp_workspace);
                        return NAN;
                    }

                    size_t idx1 = j_tau * (table->n_volatility * table->n_rate)
                                + j_sigma * table->n_rate
                                + j_r;
                    intermediate1[idx1] = pde_spline_eval(&m_spline, moneyness);
                    // No destroy needed - workspace is reused
                }
            }
        }
        free(moneyness_slice);
    }

    // Stage 2: Interpolate along maturity for each (sigma, r) combination
    // Result: intermediate2[j_sigma][j_r]
    size_t n2 = table->n_volatility * table->n_rate;
    double *intermediate2 = malloc(n2 * sizeof(double));
    if (!intermediate2) {
        free(intermediate1);
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    double *maturity_slice = malloc(table->n_maturity * sizeof(double));
    if (!maturity_slice) {
        free(intermediate1);
        free(intermediate2);
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    CubicSpline tau_spline;  // Stack-allocated

    for (size_t j_sigma = 0; j_sigma < table->n_volatility; j_sigma++) {
        for (size_t j_r = 0; j_r < table->n_rate; j_r++) {
            // Extract maturity slice from intermediate1: fix (sigma, r), vary tau
            for (size_t j_tau = 0; j_tau < table->n_maturity; j_tau++) {
                size_t idx1 = j_tau * (table->n_volatility * table->n_rate)
                            + j_sigma * table->n_rate
                            + j_r;
                maturity_slice[j_tau] = intermediate1[idx1];
            }

            // Initialize cubic spline using workspace (zero-malloc)
            int result = pde_spline_init(&tau_spline, table->maturity_grid,
                                         maturity_slice, table->n_maturity,
                                         spline_coeff_workspace, spline_temp_workspace);
            if (result != 0) {
                free(intermediate1);
                free(intermediate2);
                free(maturity_slice);
                free(spline_coeff_workspace);
                free(spline_temp_workspace);
                return NAN;
            }

            size_t idx2 = j_sigma * table->n_rate + j_r;
            intermediate2[idx2] = pde_spline_eval(&tau_spline, maturity);
            // No destroy needed - workspace is reused
        }
    }
    free(intermediate1);
    free(maturity_slice);

    // Stage 3: Interpolate along volatility for each rate
    // Result: intermediate3[j_r]
    double *intermediate3 = malloc(table->n_rate * sizeof(double));
    if (!intermediate3) {
        free(intermediate2);
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    double *volatility_slice = malloc(table->n_volatility * sizeof(double));
    if (!volatility_slice) {
        free(intermediate2);
        free(intermediate3);
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    CubicSpline sigma_spline;  // Stack-allocated

    for (size_t j_r = 0; j_r < table->n_rate; j_r++) {
        // Extract volatility slice from intermediate2: fix r, vary sigma
        for (size_t j_sigma = 0; j_sigma < table->n_volatility; j_sigma++) {
            size_t idx2 = j_sigma * table->n_rate + j_r;
            volatility_slice[j_sigma] = intermediate2[idx2];
        }

        // Initialize cubic spline using workspace (zero-malloc)
        int result_init = pde_spline_init(&sigma_spline, table->volatility_grid,
                                          volatility_slice, table->n_volatility,
                                          spline_coeff_workspace, spline_temp_workspace);
        if (result_init != 0) {
            free(intermediate2);
            free(intermediate3);
            free(volatility_slice);
            free(spline_coeff_workspace);
            free(spline_temp_workspace);
            return NAN;
        }

        intermediate3[j_r] = pde_spline_eval(&sigma_spline, volatility);
        // No destroy needed - workspace is reused
    }
    free(intermediate2);
    free(volatility_slice);

    // Stage 4: Final interpolation along rate
    CubicSpline r_spline;  // Stack-allocated
    int result_init = pde_spline_init(&r_spline, table->rate_grid,
                                      intermediate3, table->n_rate,
                                      spline_coeff_workspace, spline_temp_workspace);
    if (result_init != 0) {
        free(intermediate3);
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    double result = pde_spline_eval(&r_spline, rate);
    // No destroy needed - workspace managed by caller

    free(intermediate3);
    free(spline_coeff_workspace);
    free(spline_temp_workspace);

    return result;
}

// ---------- 5D Interpolation (Price Table with Dividend) ----------

/**
 * 5D tensor-product cubic spline interpolation
 *
 * Algorithm (separable tensor-product):
 * 1. For each (maturity, volatility, rate, dividend) combo: interpolate along moneyness
 *    → Gives n_tau × n_sigma × n_r × n_q intermediate values
 * 2. For each (volatility, rate, dividend) combo: interpolate along maturity
 *    → Gives n_sigma × n_r × n_q intermediate values
 * 3. For each (rate, dividend) combo: interpolate along volatility
 *    → Gives n_r × n_q intermediate values
 * 4. For each dividend: interpolate along rate
 *    → Gives n_q intermediate values
 * 5. Interpolate along dividend
 *    → Gives final result
 */
static double cubic_interpolate_5d(const OptionPriceTable *table,
                                   double moneyness, double maturity,
                                   double volatility, double rate,
                                   double dividend,
                                   InterpContext context) {
    // Need at least 2 points in each dimension for cubic splines
    if (table->n_moneyness < 2 || table->n_maturity < 2 ||
        table->n_volatility < 2 || table->n_rate < 2 || table->n_dividend < 2) {
        return NAN;
    }

    // Validate query point is within grid bounds
    if (!is_within_bounds(moneyness, table->moneyness_grid, table->n_moneyness) ||
        !is_within_bounds(maturity, table->maturity_grid, table->n_maturity) ||
        !is_within_bounds(volatility, table->volatility_grid, table->n_volatility) ||
        !is_within_bounds(rate, table->rate_grid, table->n_rate) ||
        !is_within_bounds(dividend, table->dividend_grid, table->n_dividend)) {
        return NAN;  // Query point outside grid coverage
    }

    // Check if we have pre-computed coefficients
    CubicContext *ctx = (CubicContext*)context;
    Cubic5DCoeffs *coeffs = (ctx && ctx->coefficients) ? (Cubic5DCoeffs*)ctx->coefficients : NULL;

    // Allocate workspaces for zero-malloc spline creation (slow path only)
    // Find maximum grid size across all dimensions
    size_t max_grid_size = table->n_moneyness;
    if (table->n_maturity > max_grid_size) max_grid_size = table->n_maturity;
    if (table->n_volatility > max_grid_size) max_grid_size = table->n_volatility;
    if (table->n_rate > max_grid_size) max_grid_size = table->n_rate;
    if (table->n_dividend > max_grid_size) max_grid_size = table->n_dividend;

    // Allocate workspaces once for the entire interpolation
    double *spline_coeff_workspace = malloc(4 * max_grid_size * sizeof(double));
    double *spline_temp_workspace = malloc(6 * max_grid_size * sizeof(double));
    if (!spline_coeff_workspace || !spline_temp_workspace) {
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    // Stage 1: Interpolate along moneyness for each (tau, sigma, r, q) combination
    // Result: intermediate1[j_tau][j_sigma][j_r][j_q]
    size_t n1 = table->n_maturity * table->n_volatility * table->n_rate * table->n_dividend;
    double *intermediate1 = malloc(n1 * sizeof(double));
    if (!intermediate1) {
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    if (coeffs) {
        // Fast path: Use pre-computed spline coefficients
        for (size_t j_tau = 0; j_tau < table->n_maturity; j_tau++) {
            for (size_t j_sigma = 0; j_sigma < table->n_volatility; j_sigma++) {
                for (size_t j_r = 0; j_r < table->n_rate; j_r++) {
                    for (size_t j_q = 0; j_q < table->n_dividend; j_q++) {
                        size_t idx1 = j_tau * (table->n_volatility * table->n_rate * table->n_dividend)
                                    + j_sigma * (table->n_rate * table->n_dividend)
                                    + j_r * table->n_dividend
                                    + j_q;
                        size_t spline_idx = j_tau * (coeffs->n_volatility * coeffs->n_rate * coeffs->n_dividend)
                                          + j_sigma * (coeffs->n_rate * coeffs->n_dividend)
                                          + j_r * coeffs->n_dividend
                                          + j_q;
                        intermediate1[idx1] = pde_spline_eval(coeffs->moneyness_splines[spline_idx], moneyness);
                    }
                }
            }
        }
    } else {
        // Slow path: Compute splines on-the-fly using workspace-based API
        double *moneyness_slice = malloc(table->n_moneyness * sizeof(double));
        if (!moneyness_slice) {
            free(intermediate1);
            free(spline_coeff_workspace);
            free(spline_temp_workspace);
            return NAN;
        }

        CubicSpline m_spline;  // Stack-allocated spline structure

        for (size_t j_tau = 0; j_tau < table->n_maturity; j_tau++) {
            for (size_t j_sigma = 0; j_sigma < table->n_volatility; j_sigma++) {
                for (size_t j_r = 0; j_r < table->n_rate; j_r++) {
                    for (size_t j_q = 0; j_q < table->n_dividend; j_q++) {
                        // Extract moneyness slice: fix (tau, sigma, r, q), vary moneyness
                        for (size_t i_m = 0; i_m < table->n_moneyness; i_m++) {
                            size_t idx = i_m * table->stride_m
                                       + j_tau * table->stride_tau
                                       + j_sigma * table->stride_sigma
                                       + j_r * table->stride_r
                                       + j_q * table->stride_q;
                            moneyness_slice[i_m] = table->prices[idx];
                        }

                        // Initialize cubic spline using workspace (zero-malloc)
                        int result = pde_spline_init(&m_spline, table->moneyness_grid,
                                                      moneyness_slice, table->n_moneyness,
                                                      spline_coeff_workspace, spline_temp_workspace);
                        if (result != 0) {
                            free(intermediate1);
                            free(moneyness_slice);
                            free(spline_coeff_workspace);
                            free(spline_temp_workspace);
                            return NAN;
                        }

                        size_t idx1 = j_tau * (table->n_volatility * table->n_rate * table->n_dividend)
                                    + j_sigma * (table->n_rate * table->n_dividend)
                                    + j_r * table->n_dividend
                                    + j_q;
                        intermediate1[idx1] = pde_spline_eval(&m_spline, moneyness);
                        // No destroy needed - workspace is reused
                    }
                }
            }
        }
        free(moneyness_slice);
    }

    // Stage 2: Interpolate along maturity for each (sigma, r, q) combination
    // Result: intermediate2[j_sigma][j_r][j_q]
    size_t n2 = table->n_volatility * table->n_rate * table->n_dividend;
    double *intermediate2 = malloc(n2 * sizeof(double));
    if (!intermediate2) {
        free(intermediate1);
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    double *maturity_slice = malloc(table->n_maturity * sizeof(double));
    if (!maturity_slice) {
        free(intermediate1);
        free(intermediate2);
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    CubicSpline tau_spline;  // Stack-allocated

    for (size_t j_sigma = 0; j_sigma < table->n_volatility; j_sigma++) {
        for (size_t j_r = 0; j_r < table->n_rate; j_r++) {
            for (size_t j_q = 0; j_q < table->n_dividend; j_q++) {
                // Extract maturity slice from intermediate1: fix (sigma, r, q), vary tau
                for (size_t j_tau = 0; j_tau < table->n_maturity; j_tau++) {
                    size_t idx1 = j_tau * (table->n_volatility * table->n_rate * table->n_dividend)
                                + j_sigma * (table->n_rate * table->n_dividend)
                                + j_r * table->n_dividend
                                + j_q;
                    maturity_slice[j_tau] = intermediate1[idx1];
                }

                // Initialize cubic spline using workspace (zero-malloc)
                int result = pde_spline_init(&tau_spline, table->maturity_grid,
                                             maturity_slice, table->n_maturity,
                                             spline_coeff_workspace, spline_temp_workspace);
                if (result != 0) {
                    free(intermediate1);
                    free(intermediate2);
                    free(maturity_slice);
                    free(spline_coeff_workspace);
                    free(spline_temp_workspace);
                    return NAN;
                }

                size_t idx2 = j_sigma * (table->n_rate * table->n_dividend)
                            + j_r * table->n_dividend
                            + j_q;
                intermediate2[idx2] = pde_spline_eval(&tau_spline, maturity);
                // No destroy needed - workspace is reused
            }
        }
    }
    free(intermediate1);
    free(maturity_slice);

    // Stage 3: Interpolate along volatility for each (r, q) combination
    // Result: intermediate3[j_r][j_q]
    size_t n3 = table->n_rate * table->n_dividend;
    double *intermediate3 = malloc(n3 * sizeof(double));
    if (!intermediate3) {
        free(intermediate2);
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    double *volatility_slice = malloc(table->n_volatility * sizeof(double));
    if (!volatility_slice) {
        free(intermediate2);
        free(intermediate3);
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    CubicSpline sigma_spline;  // Stack-allocated

    for (size_t j_r = 0; j_r < table->n_rate; j_r++) {
        for (size_t j_q = 0; j_q < table->n_dividend; j_q++) {
            // Extract volatility slice from intermediate2: fix (r, q), vary sigma
            for (size_t j_sigma = 0; j_sigma < table->n_volatility; j_sigma++) {
                size_t idx2 = j_sigma * (table->n_rate * table->n_dividend)
                            + j_r * table->n_dividend
                            + j_q;
                volatility_slice[j_sigma] = intermediate2[idx2];
            }

            // Initialize cubic spline using workspace (zero-malloc)
            int result_init = pde_spline_init(&sigma_spline, table->volatility_grid,
                                              volatility_slice, table->n_volatility,
                                              spline_coeff_workspace, spline_temp_workspace);
            if (result_init != 0) {
                free(intermediate2);
                free(intermediate3);
                free(volatility_slice);
                free(spline_coeff_workspace);
                free(spline_temp_workspace);
                return NAN;
            }

            size_t idx3 = j_r * table->n_dividend + j_q;
            intermediate3[idx3] = pde_spline_eval(&sigma_spline, volatility);
            // No destroy needed - workspace is reused
        }
    }
    free(intermediate2);
    free(volatility_slice);

    // Stage 4: Interpolate along rate for each dividend
    // Result: intermediate4[j_q]
    double *intermediate4 = malloc(table->n_dividend * sizeof(double));
    if (!intermediate4) {
        free(intermediate3);
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    double *rate_slice = malloc(table->n_rate * sizeof(double));
    if (!rate_slice) {
        free(intermediate3);
        free(intermediate4);
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    CubicSpline r_spline;  // Stack-allocated

    for (size_t j_q = 0; j_q < table->n_dividend; j_q++) {
        // Extract rate slice from intermediate3: fix q, vary r
        for (size_t j_r = 0; j_r < table->n_rate; j_r++) {
            size_t idx3 = j_r * table->n_dividend + j_q;
            rate_slice[j_r] = intermediate3[idx3];
        }

        // Initialize cubic spline using workspace (zero-malloc)
        int result_init = pde_spline_init(&r_spline, table->rate_grid,
                                          rate_slice, table->n_rate,
                                          spline_coeff_workspace, spline_temp_workspace);
        if (result_init != 0) {
            free(intermediate3);
            free(intermediate4);
            free(rate_slice);
            free(spline_coeff_workspace);
            free(spline_temp_workspace);
            return NAN;
        }

        intermediate4[j_q] = pde_spline_eval(&r_spline, rate);
        // No destroy needed - workspace is reused
    }
    free(intermediate3);
    free(rate_slice);

    // Stage 5: Final interpolation along dividend
    CubicSpline q_spline;  // Stack-allocated
    int result_init = pde_spline_init(&q_spline, table->dividend_grid,
                                      intermediate4, table->n_dividend,
                                      spline_coeff_workspace, spline_temp_workspace);
    if (result_init != 0) {
        free(intermediate4);
        free(spline_coeff_workspace);
        free(spline_temp_workspace);
        return NAN;
    }

    double result = pde_spline_eval(&q_spline, dividend);
    // No destroy needed - workspace managed by caller

    free(intermediate4);
    free(spline_coeff_workspace);
    free(spline_temp_workspace);

    return result;
}

// ---------- Context Management ----------

static InterpContext cubic_create_context(size_t dimensions,
                                          const size_t *grid_sizes) {
    CubicContext *ctx = malloc(sizeof(CubicContext));
    if (!ctx) return NULL;

    ctx->dimensions = dimensions;
    ctx->grid_sizes = malloc(dimensions * sizeof(size_t));
    if (!ctx->grid_sizes) {
        free(ctx);
        return NULL;
    }

    #pragma omp simd
    for (size_t i = 0; i < dimensions; i++) {
        ctx->grid_sizes[i] = grid_sizes[i];
    }

    ctx->coefficients = NULL;  // Will be allocated in precompute if needed

    return (InterpContext)ctx;
}

static void cubic_destroy_context(InterpContext context) {
    if (!context) return;

    CubicContext *ctx = (CubicContext*)context;

    // Free pre-computed coefficients if they exist
    if (ctx->coefficients) {
        if (ctx->dimensions == 2) {
            Cubic2DCoeffs *coeffs = (Cubic2DCoeffs*)ctx->coefficients;
            if (coeffs->moneyness_splines) {
                for (size_t j = 0; j < coeffs->n_maturity; j++) {
                    pde_spline_destroy(coeffs->moneyness_splines[j]);
                }
                free(coeffs->moneyness_splines);
            }
            free(coeffs);
        }
        else if (ctx->dimensions == 4) {
            Cubic4DCoeffs *coeffs = (Cubic4DCoeffs*)ctx->coefficients;
            if (coeffs->moneyness_splines) {
                size_t n_splines = coeffs->n_maturity * coeffs->n_volatility * coeffs->n_rate;
                for (size_t i = 0; i < n_splines; i++) {
                    pde_spline_destroy(coeffs->moneyness_splines[i]);
                }
                free(coeffs->moneyness_splines);
            }
            free(coeffs);
        }
        else if (ctx->dimensions == 5) {
            Cubic5DCoeffs *coeffs = (Cubic5DCoeffs*)ctx->coefficients;
            if (coeffs->moneyness_splines) {
                size_t n_splines = coeffs->n_maturity * coeffs->n_volatility * coeffs->n_rate * coeffs->n_dividend;
                for (size_t i = 0; i < n_splines; i++) {
                    pde_spline_destroy(coeffs->moneyness_splines[i]);
                }
                free(coeffs->moneyness_splines);
            }
            free(coeffs);
        }
    }

    free(ctx->grid_sizes);
    free(ctx);
}

static int cubic_precompute(const void *grid_data, InterpContext context) {
    if (!grid_data || !context) return -1;

    CubicContext *ctx = (CubicContext*)context;

    if (ctx->dimensions == 2) {
        // Pre-compute splines for 2D IV surface
        const IVSurface *surface = (const IVSurface*)grid_data;

        if (surface->n_maturity < 2 || surface->n_moneyness < 2) {
            return -1;
        }

        // Allocate 2D coefficients structure
        Cubic2DCoeffs *coeffs = malloc(sizeof(Cubic2DCoeffs));
        if (!coeffs) return -1;

        coeffs->n_maturity = surface->n_maturity;
        coeffs->moneyness_grid = surface->moneyness_grid;
        coeffs->maturity_grid = surface->maturity_grid;
        coeffs->moneyness_splines = malloc(surface->n_maturity * sizeof(CubicSpline*));
        if (!coeffs->moneyness_splines) {
            free(coeffs);
            return -1;
        }

        // Pre-compute a spline for each maturity slice
        double *moneyness_slice = malloc(surface->n_moneyness * sizeof(double));
        if (!moneyness_slice) {
            free(coeffs->moneyness_splines);
            free(coeffs);
            return -1;
        }

        for (size_t j_tau = 0; j_tau < surface->n_maturity; j_tau++) {
            // Extract moneyness slice at this maturity level
            #pragma omp simd
            for (size_t i_m = 0; i_m < surface->n_moneyness; i_m++) {
                size_t idx = j_tau * surface->n_moneyness + i_m;
                moneyness_slice[i_m] = surface->iv_surface[idx];
            }

            // Create and store spline for this slice
            coeffs->moneyness_splines[j_tau] = pde_spline_create(surface->moneyness_grid,
                                                                   moneyness_slice,
                                                                   surface->n_moneyness);
            if (!coeffs->moneyness_splines[j_tau]) {
                // Cleanup on error
                for (size_t k = 0; k < j_tau; k++) {
                    pde_spline_destroy(coeffs->moneyness_splines[k]);
                }
                free(moneyness_slice);
                free(coeffs->moneyness_splines);
                free(coeffs);
                return -1;
            }
        }

        free(moneyness_slice);
        ctx->coefficients = coeffs;
        return 0;
    }

    else if (ctx->dimensions == 4) {
        // Pre-compute splines for 4D price table
        const OptionPriceTable *table = (const OptionPriceTable*)grid_data;

        if (table->n_moneyness < 2 || table->n_maturity < 2 ||
            table->n_volatility < 2 || table->n_rate < 2) {
            return -1;
        }

        // Allocate 4D coefficients structure
        Cubic4DCoeffs *coeffs = malloc(sizeof(Cubic4DCoeffs));
        if (!coeffs) return -1;

        coeffs->n_maturity = table->n_maturity;
        coeffs->n_volatility = table->n_volatility;
        coeffs->n_rate = table->n_rate;
        coeffs->moneyness_grid = table->moneyness_grid;

        // Allocate flat array of spline pointers
        size_t n_splines = table->n_maturity * table->n_volatility * table->n_rate;
        coeffs->moneyness_splines = malloc(n_splines * sizeof(CubicSpline*));
        if (!coeffs->moneyness_splines) {
            free(coeffs);
            return -1;
        }

        // Pre-compute a spline for each (tau, sigma, r) combination
        double *moneyness_slice = malloc(table->n_moneyness * sizeof(double));
        if (!moneyness_slice) {
            free(coeffs->moneyness_splines);
            free(coeffs);
            return -1;
        }

        for (size_t j_tau = 0; j_tau < table->n_maturity; j_tau++) {
            for (size_t j_sigma = 0; j_sigma < table->n_volatility; j_sigma++) {
                for (size_t j_r = 0; j_r < table->n_rate; j_r++) {
                    // Extract moneyness slice: fix (tau, sigma, r), vary moneyness
                    for (size_t i_m = 0; i_m < table->n_moneyness; i_m++) {
                        size_t idx = i_m * table->stride_m
                                   + j_tau * table->stride_tau
                                   + j_sigma * table->stride_sigma
                                   + j_r * table->stride_r;
                        moneyness_slice[i_m] = table->prices[idx];
                    }

                    // Calculate flat array index
                    size_t spline_idx = j_tau * (table->n_volatility * table->n_rate)
                                      + j_sigma * table->n_rate
                                      + j_r;

                    // Create and store spline for this slice
                    coeffs->moneyness_splines[spline_idx] =
                        pde_spline_create(table->moneyness_grid,
                                         moneyness_slice,
                                         table->n_moneyness);

                    if (!coeffs->moneyness_splines[spline_idx]) {
                        // Cleanup on error - destroy all previously created splines
                        for (size_t k = 0; k < spline_idx; k++) {
                            pde_spline_destroy(coeffs->moneyness_splines[k]);
                        }
                        free(moneyness_slice);
                        free(coeffs->moneyness_splines);
                        free(coeffs);
                        return -1;
                    }
                }
            }
        }

        free(moneyness_slice);
        ctx->coefficients = coeffs;
        return 0;
    }
    else if (ctx->dimensions == 5) {
        // Pre-compute splines for 5D price table
        const OptionPriceTable *table = (const OptionPriceTable*)grid_data;

        if (table->n_moneyness < 2 || table->n_maturity < 2 ||
            table->n_volatility < 2 || table->n_rate < 2 || table->n_dividend < 2) {
            return -1;
        }

        // Allocate 5D coefficients structure
        Cubic5DCoeffs *coeffs = malloc(sizeof(Cubic5DCoeffs));
        if (!coeffs) return -1;

        coeffs->n_maturity = table->n_maturity;
        coeffs->n_volatility = table->n_volatility;
        coeffs->n_rate = table->n_rate;
        coeffs->n_dividend = table->n_dividend;
        coeffs->moneyness_grid = table->moneyness_grid;

        // Allocate flat array of spline pointers
        size_t n_splines = table->n_maturity * table->n_volatility * table->n_rate * table->n_dividend;
        coeffs->moneyness_splines = malloc(n_splines * sizeof(CubicSpline*));
        if (!coeffs->moneyness_splines) {
            free(coeffs);
            return -1;
        }

        // Pre-compute a spline for each (tau, sigma, r, q) combination
        double *moneyness_slice = malloc(table->n_moneyness * sizeof(double));
        if (!moneyness_slice) {
            free(coeffs->moneyness_splines);
            free(coeffs);
            return -1;
        }

        for (size_t j_tau = 0; j_tau < table->n_maturity; j_tau++) {
            for (size_t j_sigma = 0; j_sigma < table->n_volatility; j_sigma++) {
                for (size_t j_r = 0; j_r < table->n_rate; j_r++) {
                    for (size_t j_q = 0; j_q < table->n_dividend; j_q++) {
                        // Extract moneyness slice: fix (tau, sigma, r, q), vary moneyness
                        for (size_t i_m = 0; i_m < table->n_moneyness; i_m++) {
                            size_t idx = i_m * table->stride_m
                                       + j_tau * table->stride_tau
                                       + j_sigma * table->stride_sigma
                                       + j_r * table->stride_r
                                       + j_q * table->stride_q;
                            moneyness_slice[i_m] = table->prices[idx];
                        }

                        // Calculate flat array index
                        size_t spline_idx = j_tau * (table->n_volatility * table->n_rate * table->n_dividend)
                                          + j_sigma * (table->n_rate * table->n_dividend)
                                          + j_r * table->n_dividend
                                          + j_q;

                        // Create and store spline for this slice
                        coeffs->moneyness_splines[spline_idx] =
                            pde_spline_create(table->moneyness_grid,
                                             moneyness_slice,
                                             table->n_moneyness);

                        if (!coeffs->moneyness_splines[spline_idx]) {
                            // Cleanup on error - destroy all previously created splines
                            for (size_t k = 0; k < spline_idx; k++) {
                                pde_spline_destroy(coeffs->moneyness_splines[k]);
                            }
                            free(moneyness_slice);
                            free(coeffs->moneyness_splines);
                            free(coeffs);
                            return -1;
                        }
                    }
                }
            }
        }

        free(moneyness_slice);
        ctx->coefficients = coeffs;
        return 0;
    }

    // Unknown dimension
    return -1;
}
