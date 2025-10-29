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

// Evaluate cubic polynomial: a + b*dx + c*dx^2 + d*dx^3
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
 * 2. Build cubic spline along maturity using intermediate values and evaluate at query_tau
 *    This gives result = spline_tau(intermediate, query_tau)
 *
 * This is the standard separable tensor-product approach that guarantees:
 * - On-grid points are exact (splines interpolate data exactly)
 * - C2 continuous everywhere
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

static double cubic_interpolate_4d(const OptionPriceTable *table,
                                   double moneyness, double maturity,
                                   double volatility, double rate,
                                   InterpContext context) {
    (void)context;
    (void)table;
    (void)moneyness;
    (void)maturity;
    (void)volatility;
    (void)rate;

    // TODO: Implement 4D tensor-product cubic spline
    // For now, return NAN to indicate not implemented
    return NAN;
}

// ---------- 5D Interpolation (Price Table with Dividend) ----------

static double cubic_interpolate_5d(const OptionPriceTable *table,
                                   double moneyness, double maturity,
                                   double volatility, double rate,
                                   double dividend,
                                   InterpContext context) {
    (void)context;
    (void)table;
    (void)moneyness;
    (void)maturity;
    (void)volatility;
    (void)rate;
    (void)dividend;

    // TODO: Implement 5D tensor-product cubic spline
    return NAN;
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
