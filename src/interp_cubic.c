#include "interp_cubic.h"
#include "iv_surface.h"
#include "price_table.h"
#include "cubic_spline.h"
#include <stdlib.h>
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

// Evaluate cubic polynomial: a + b·dx + c·dx² + d·dx³
static inline double eval_cubic(double a, double b, double c, double d, double dx) {
    return a + dx * (b + dx * (c + dx * d));
}

// ---------- 2D Interpolation (IV Surface) ----------

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

    // Check if we have pre-computed coefficients
    CubicContext *ctx = (CubicContext*)context;
    Cubic2DCoeffs *coeffs = (ctx && ctx->coefficients) ? (Cubic2DCoeffs*)ctx->coefficients : NULL;

    // Step 1: For each maturity grid point, interpolate along moneyness at query point
    // This gives us intermediate_values[j] for all j in [0, n_maturity)
    double *intermediate_values = malloc(surface->n_maturity * sizeof(double));
    if (!intermediate_values) return NAN;

    if (coeffs) {
        // Fast path: Use pre-computed spline coefficients
        #pragma omp simd
        for (size_t j_tau = 0; j_tau < surface->n_maturity; j_tau++) {
            intermediate_values[j_tau] = pde_spline_eval(coeffs->moneyness_splines[j_tau], moneyness);
        }
    } else {
        // Slow path: Compute splines on-the-fly
        double *moneyness_slice = malloc(surface->n_moneyness * sizeof(double));
        if (!moneyness_slice) {
            free(intermediate_values);
            return NAN;
        }

        for (size_t j_tau = 0; j_tau < surface->n_maturity; j_tau++) {
            // Extract moneyness slice at this maturity level
            // Data layout: iv_surface[j_tau * n_moneyness + i_m] (maturity varies slowest)
            #pragma omp simd
            for (size_t i_m = 0; i_m < surface->n_moneyness; i_m++) {
                size_t idx = j_tau * surface->n_moneyness + i_m;
                moneyness_slice[i_m] = surface->iv_surface[idx];
            }

            // Build cubic spline along moneyness using this slice
            CubicSpline *m_spline = pde_spline_create(surface->moneyness_grid,
                                                        moneyness_slice,
                                                        surface->n_moneyness);
            if (!m_spline) {
                free(intermediate_values);
                free(moneyness_slice);
                return NAN;
            }

            // Evaluate at query moneyness
            intermediate_values[j_tau] = pde_spline_eval(m_spline, moneyness);
            pde_spline_destroy(m_spline);
        }

        free(moneyness_slice);
    }

    // Step 2: Build cubic spline along maturity using intermediate values
    CubicSpline *tau_spline = pde_spline_create(surface->maturity_grid,
                                                  intermediate_values,
                                                  surface->n_maturity);
    if (!tau_spline) {
        free(intermediate_values);
        return NAN;
    }

    // Evaluate at query maturity
    double result = pde_spline_eval(tau_spline, maturity);

    pde_spline_destroy(tau_spline);
    free(intermediate_values);

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
                                   [[maybe_unused]] InterpContext context) {
    // Need at least 2 points in each dimension for cubic splines
    if (table->n_moneyness < 2 || table->n_maturity < 2 ||
        table->n_volatility < 2 || table->n_rate < 2) {
        return NAN;
    }

    // Stage 1: Interpolate along moneyness for each (tau, sigma, r) combination
    // Result: intermediate1[j_tau][j_sigma][j_r]
    size_t n1 = table->n_maturity * table->n_volatility * table->n_rate;
    double *intermediate1 = malloc(n1 * sizeof(double));
    if (!intermediate1) return NAN;

    double *moneyness_slice = malloc(table->n_moneyness * sizeof(double));
    if (!moneyness_slice) {
        free(intermediate1);
        return NAN;
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

                // Build cubic spline along moneyness and evaluate at query point
                CubicSpline *m_spline = pde_spline_create(table->moneyness_grid,
                                                           moneyness_slice,
                                                           table->n_moneyness);
                if (!m_spline) {
                    free(intermediate1);
                    free(moneyness_slice);
                    return NAN;
                }

                size_t idx1 = j_tau * (table->n_volatility * table->n_rate)
                            + j_sigma * table->n_rate
                            + j_r;
                intermediate1[idx1] = pde_spline_eval(m_spline, moneyness);
                pde_spline_destroy(m_spline);
            }
        }
    }
    free(moneyness_slice);

    // Stage 2: Interpolate along maturity for each (sigma, r) combination
    // Result: intermediate2[j_sigma][j_r]
    size_t n2 = table->n_volatility * table->n_rate;
    double *intermediate2 = malloc(n2 * sizeof(double));
    if (!intermediate2) {
        free(intermediate1);
        return NAN;
    }

    double *maturity_slice = malloc(table->n_maturity * sizeof(double));
    if (!maturity_slice) {
        free(intermediate1);
        free(intermediate2);
        return NAN;
    }

    for (size_t j_sigma = 0; j_sigma < table->n_volatility; j_sigma++) {
        for (size_t j_r = 0; j_r < table->n_rate; j_r++) {
            // Extract maturity slice from intermediate1: fix (sigma, r), vary tau
            for (size_t j_tau = 0; j_tau < table->n_maturity; j_tau++) {
                size_t idx1 = j_tau * (table->n_volatility * table->n_rate)
                            + j_sigma * table->n_rate
                            + j_r;
                maturity_slice[j_tau] = intermediate1[idx1];
            }

            // Build cubic spline along maturity and evaluate
            CubicSpline *tau_spline = pde_spline_create(table->maturity_grid,
                                                         maturity_slice,
                                                         table->n_maturity);
            if (!tau_spline) {
                free(intermediate1);
                free(intermediate2);
                free(maturity_slice);
                return NAN;
            }

            size_t idx2 = j_sigma * table->n_rate + j_r;
            intermediate2[idx2] = pde_spline_eval(tau_spline, maturity);
            pde_spline_destroy(tau_spline);
        }
    }
    free(intermediate1);
    free(maturity_slice);

    // Stage 3: Interpolate along volatility for each rate
    // Result: intermediate3[j_r]
    double *intermediate3 = malloc(table->n_rate * sizeof(double));
    if (!intermediate3) {
        free(intermediate2);
        return NAN;
    }

    double *volatility_slice = malloc(table->n_volatility * sizeof(double));
    if (!volatility_slice) {
        free(intermediate2);
        free(intermediate3);
        return NAN;
    }

    for (size_t j_r = 0; j_r < table->n_rate; j_r++) {
        // Extract volatility slice from intermediate2: fix r, vary sigma
        for (size_t j_sigma = 0; j_sigma < table->n_volatility; j_sigma++) {
            size_t idx2 = j_sigma * table->n_rate + j_r;
            volatility_slice[j_sigma] = intermediate2[idx2];
        }

        // Build cubic spline along volatility and evaluate
        CubicSpline *sigma_spline = pde_spline_create(table->volatility_grid,
                                                       volatility_slice,
                                                       table->n_volatility);
        if (!sigma_spline) {
            free(intermediate2);
            free(intermediate3);
            free(volatility_slice);
            return NAN;
        }

        intermediate3[j_r] = pde_spline_eval(sigma_spline, volatility);
        pde_spline_destroy(sigma_spline);
    }
    free(intermediate2);
    free(volatility_slice);

    // Stage 4: Final interpolation along rate
    CubicSpline *r_spline = pde_spline_create(table->rate_grid,
                                               intermediate3,
                                               table->n_rate);
    if (!r_spline) {
        free(intermediate3);
        return NAN;
    }

    double result = pde_spline_eval(r_spline, rate);
    pde_spline_destroy(r_spline);
    free(intermediate3);

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
                                   [[maybe_unused]] InterpContext context) {
    // Need at least 2 points in each dimension for cubic splines
    if (table->n_moneyness < 2 || table->n_maturity < 2 ||
        table->n_volatility < 2 || table->n_rate < 2 || table->n_dividend < 2) {
        return NAN;
    }

    // Stage 1: Interpolate along moneyness for each (tau, sigma, r, q) combination
    // Result: intermediate1[j_tau][j_sigma][j_r][j_q]
    size_t n1 = table->n_maturity * table->n_volatility * table->n_rate * table->n_dividend;
    double *intermediate1 = malloc(n1 * sizeof(double));
    if (!intermediate1) return NAN;

    double *moneyness_slice = malloc(table->n_moneyness * sizeof(double));
    if (!moneyness_slice) {
        free(intermediate1);
        return NAN;
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

                    // Build cubic spline along moneyness and evaluate
                    CubicSpline *m_spline = pde_spline_create(table->moneyness_grid,
                                                               moneyness_slice,
                                                               table->n_moneyness);
                    if (!m_spline) {
                        free(intermediate1);
                        free(moneyness_slice);
                        return NAN;
                    }

                    size_t idx1 = j_tau * (table->n_volatility * table->n_rate * table->n_dividend)
                                + j_sigma * (table->n_rate * table->n_dividend)
                                + j_r * table->n_dividend
                                + j_q;
                    intermediate1[idx1] = pde_spline_eval(m_spline, moneyness);
                    pde_spline_destroy(m_spline);
                }
            }
        }
    }
    free(moneyness_slice);

    // Stage 2: Interpolate along maturity for each (sigma, r, q) combination
    // Result: intermediate2[j_sigma][j_r][j_q]
    size_t n2 = table->n_volatility * table->n_rate * table->n_dividend;
    double *intermediate2 = malloc(n2 * sizeof(double));
    if (!intermediate2) {
        free(intermediate1);
        return NAN;
    }

    double *maturity_slice = malloc(table->n_maturity * sizeof(double));
    if (!maturity_slice) {
        free(intermediate1);
        free(intermediate2);
        return NAN;
    }

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

                // Build cubic spline along maturity and evaluate
                CubicSpline *tau_spline = pde_spline_create(table->maturity_grid,
                                                             maturity_slice,
                                                             table->n_maturity);
                if (!tau_spline) {
                    free(intermediate1);
                    free(intermediate2);
                    free(maturity_slice);
                    return NAN;
                }

                size_t idx2 = j_sigma * (table->n_rate * table->n_dividend)
                            + j_r * table->n_dividend
                            + j_q;
                intermediate2[idx2] = pde_spline_eval(tau_spline, maturity);
                pde_spline_destroy(tau_spline);
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
        return NAN;
    }

    double *volatility_slice = malloc(table->n_volatility * sizeof(double));
    if (!volatility_slice) {
        free(intermediate2);
        free(intermediate3);
        return NAN;
    }

    for (size_t j_r = 0; j_r < table->n_rate; j_r++) {
        for (size_t j_q = 0; j_q < table->n_dividend; j_q++) {
            // Extract volatility slice from intermediate2: fix (r, q), vary sigma
            for (size_t j_sigma = 0; j_sigma < table->n_volatility; j_sigma++) {
                size_t idx2 = j_sigma * (table->n_rate * table->n_dividend)
                            + j_r * table->n_dividend
                            + j_q;
                volatility_slice[j_sigma] = intermediate2[idx2];
            }

            // Build cubic spline along volatility and evaluate
            CubicSpline *sigma_spline = pde_spline_create(table->volatility_grid,
                                                           volatility_slice,
                                                           table->n_volatility);
            if (!sigma_spline) {
                free(intermediate2);
                free(intermediate3);
                free(volatility_slice);
                return NAN;
            }

            size_t idx3 = j_r * table->n_dividend + j_q;
            intermediate3[idx3] = pde_spline_eval(sigma_spline, volatility);
            pde_spline_destroy(sigma_spline);
        }
    }
    free(intermediate2);
    free(volatility_slice);

    // Stage 4: Interpolate along rate for each dividend
    // Result: intermediate4[j_q]
    double *intermediate4 = malloc(table->n_dividend * sizeof(double));
    if (!intermediate4) {
        free(intermediate3);
        return NAN;
    }

    double *rate_slice = malloc(table->n_rate * sizeof(double));
    if (!rate_slice) {
        free(intermediate3);
        free(intermediate4);
        return NAN;
    }

    for (size_t j_q = 0; j_q < table->n_dividend; j_q++) {
        // Extract rate slice from intermediate3: fix q, vary r
        for (size_t j_r = 0; j_r < table->n_rate; j_r++) {
            size_t idx3 = j_r * table->n_dividend + j_q;
            rate_slice[j_r] = intermediate3[idx3];
        }

        // Build cubic spline along rate and evaluate
        CubicSpline *r_spline = pde_spline_create(table->rate_grid,
                                                   rate_slice,
                                                   table->n_rate);
        if (!r_spline) {
            free(intermediate3);
            free(intermediate4);
            free(rate_slice);
            return NAN;
        }

        intermediate4[j_q] = pde_spline_eval(r_spline, rate);
        pde_spline_destroy(r_spline);
    }
    free(intermediate3);
    free(rate_slice);

    // Stage 5: Final interpolation along dividend
    CubicSpline *q_spline = pde_spline_create(table->dividend_grid,
                                               intermediate4,
                                               table->n_dividend);
    if (!q_spline) {
        free(intermediate4);
        return NAN;
    }

    double result = pde_spline_eval(q_spline, dividend);
    pde_spline_destroy(q_spline);
    free(intermediate4);

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
        // TODO: Add cleanup for 4D and 5D when implemented
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

    // 4D and 5D not yet implemented
    return -1;
}
