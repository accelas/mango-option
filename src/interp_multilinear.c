#include "interp_multilinear.h"
#include "iv_surface.h"
#include "price_table.h"
#include <math.h>
#include <stddef.h>

// Forward declarations
static double multilinear_interpolate_2d(const IVSurface *surface,
                                          double moneyness, double maturity,
                                          InterpContext context);

static double multilinear_interpolate_4d(const OptionPriceTable *table,
                                          double moneyness, double maturity,
                                          double volatility, double rate,
                                          InterpContext context);

static double multilinear_interpolate_5d(const OptionPriceTable *table,
                                          double moneyness, double maturity,
                                          double volatility, double rate,
                                          double dividend,
                                          InterpContext context);

static InterpContext multilinear_create_context(size_t dimensions,
                                                  const size_t *grid_sizes);

static void multilinear_destroy_context(InterpContext context);

// ---------- Strategy Definition ----------

const InterpolationStrategy INTERP_MULTILINEAR = {
    .name = "multilinear",
    .description = "Separable multi-linear interpolation (fast, C0 continuous)",
    .interpolate_2d = multilinear_interpolate_2d,
    .interpolate_4d = multilinear_interpolate_4d,
    .interpolate_5d = multilinear_interpolate_5d,
    .create_context = multilinear_create_context,
    .destroy_context = multilinear_destroy_context,
    .precompute = NULL  // No pre-computation needed for linear
};

// ---------- Helper Functions ----------

size_t find_bracket(const double *grid, size_t n, double query) {
    // Handle boundary cases
    if (query <= grid[0]) return 0;
    if (query >= grid[n-1]) return n - 2;

    // Binary search for bracketing interval
    size_t left = 0;
    size_t right = n - 1;

    while (right - left > 1) {
        size_t mid = left + (right - left) / 2;
        if (query < grid[mid]) {
            right = mid;
        } else {
            left = mid;
        }
    }

    return left;
}

double lerp(double x0, double x1, double y0, double y1, double x) {
    // Linear interpolation: y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    if (x1 == x0) return y0;  // Avoid division by zero
    double t = (x - x0) / (x1 - x0);
    return y0 + (y1 - y0) * t;
}

// ---------- 2D Interpolation (IV Surface) ----------

static double multilinear_interpolate_2d(const IVSurface *surface,
                                          double moneyness, double maturity,
                                          InterpContext context) {
    (void)context;  // Unused for stateless strategy

    // Find bracketing indices
    size_t i_m = find_bracket(surface->moneyness_grid, surface->n_moneyness, moneyness);
    size_t i_tau = find_bracket(surface->maturity_grid, surface->n_maturity, maturity);

    // Get grid values
    double m0 = surface->moneyness_grid[i_m];
    double m1 = surface->moneyness_grid[i_m + 1];
    double tau0 = surface->maturity_grid[i_tau];
    double tau1 = surface->maturity_grid[i_tau + 1];

    // Get 4 corner values (row-major: moneyness varies fastest)
    double v00 = surface->iv_surface[i_tau * surface->n_moneyness + i_m];
    double v10 = surface->iv_surface[i_tau * surface->n_moneyness + (i_m + 1)];
    double v01 = surface->iv_surface[(i_tau + 1) * surface->n_moneyness + i_m];
    double v11 = surface->iv_surface[(i_tau + 1) * surface->n_moneyness + (i_m + 1)];

    // Bilinear interpolation:
    // 1. Interpolate along moneyness at tau0 and tau1
    double v_tau0 = lerp(m0, m1, v00, v10, moneyness);
    double v_tau1 = lerp(m0, m1, v01, v11, moneyness);

    // 2. Interpolate along maturity
    double result = lerp(tau0, tau1, v_tau0, v_tau1, maturity);

    return result;
}

// ---------- 4D Interpolation (Price Table) ----------

static double multilinear_interpolate_4d(const OptionPriceTable *table,
                                          double moneyness, double maturity,
                                          double volatility, double rate,
                                          InterpContext context) {
    (void)context;  // Unused

    // Find bracketing indices for each dimension
    size_t i_m = find_bracket(table->moneyness_grid, table->n_moneyness, moneyness);
    size_t i_tau = find_bracket(table->maturity_grid, table->n_maturity, maturity);
    size_t i_sigma = find_bracket(table->volatility_grid, table->n_volatility, volatility);
    size_t i_r = find_bracket(table->rate_grid, table->n_rate, rate);

    // Get grid values
    double m0 = table->moneyness_grid[i_m];
    double m1 = table->moneyness_grid[i_m + 1];
    double tau0 = table->maturity_grid[i_tau];
    double tau1 = table->maturity_grid[i_tau + 1];
    double sigma0 = table->volatility_grid[i_sigma];
    double sigma1 = table->volatility_grid[i_sigma + 1];
    double r0 = table->rate_grid[i_r];
    double r1 = table->rate_grid[i_r + 1];

    // Get 16 hypercube corner values (2^4 = 16)
    // Index: i_m * stride_m + i_tau * stride_tau + i_sigma * stride_sigma + i_r * stride_r
    double values[16];
    for (int dm = 0; dm < 2; dm++) {
        for (int dtau = 0; dtau < 2; dtau++) {
            for (int dsigma = 0; dsigma < 2; dsigma++) {
                for (int dr = 0; dr < 2; dr++) {
                    size_t idx = (i_m + dm) * table->stride_m
                               + (i_tau + dtau) * table->stride_tau
                               + (i_sigma + dsigma) * table->stride_sigma
                               + (i_r + dr) * table->stride_r;
                    values[dm*8 + dtau*4 + dsigma*2 + dr] = table->prices[idx];
                }
            }
        }
    }

    // 4D multilinear interpolation: 15 lerps
    // Stage 1: Interpolate along moneyness (8 → 4)
    double v_m[8];
    for (int i = 0; i < 8; i++) {
        v_m[i] = lerp(m0, m1, values[i*2], values[i*2 + 1], moneyness);
    }

    // Stage 2: Interpolate along maturity (4 → 2)
    double v_tau[4];
    for (int i = 0; i < 4; i++) {
        v_tau[i] = lerp(tau0, tau1, v_m[i*2], v_m[i*2 + 1], maturity);
    }

    // Stage 3: Interpolate along volatility (2 → 1)
    double v_sigma[2];
    v_sigma[0] = lerp(sigma0, sigma1, v_tau[0], v_tau[1], volatility);
    v_sigma[1] = lerp(sigma0, sigma1, v_tau[2], v_tau[3], volatility);

    // Stage 4: Interpolate along rate (1 → final)
    double result = lerp(r0, r1, v_sigma[0], v_sigma[1], rate);

    return result;
}

// ---------- 5D Interpolation (Price Table with Dividend) ----------

static double multilinear_interpolate_5d(const OptionPriceTable *table,
                                          double moneyness, double maturity,
                                          double volatility, double rate,
                                          double dividend,
                                          InterpContext context) {
    (void)context;  // Unused

    // Find bracketing indices for each dimension
    size_t i_m = find_bracket(table->moneyness_grid, table->n_moneyness, moneyness);
    size_t i_tau = find_bracket(table->maturity_grid, table->n_maturity, maturity);
    size_t i_sigma = find_bracket(table->volatility_grid, table->n_volatility, volatility);
    size_t i_r = find_bracket(table->rate_grid, table->n_rate, rate);
    size_t i_q = find_bracket(table->dividend_grid, table->n_dividend, dividend);

    // Get grid values
    double m0 = table->moneyness_grid[i_m];
    double m1 = table->moneyness_grid[i_m + 1];
    double tau0 = table->maturity_grid[i_tau];
    double tau1 = table->maturity_grid[i_tau + 1];
    double sigma0 = table->volatility_grid[i_sigma];
    double sigma1 = table->volatility_grid[i_sigma + 1];
    double r0 = table->rate_grid[i_r];
    double r1 = table->rate_grid[i_r + 1];
    double q0 = table->dividend_grid[i_q];
    double q1 = table->dividend_grid[i_q + 1];

    // Get 32 hypercube corner values (2^5 = 32)
    double values[32];
    for (int dm = 0; dm < 2; dm++) {
        for (int dtau = 0; dtau < 2; dtau++) {
            for (int dsigma = 0; dsigma < 2; dsigma++) {
                for (int dr = 0; dr < 2; dr++) {
                    for (int dq = 0; dq < 2; dq++) {
                        size_t idx = (i_m + dm) * table->stride_m
                                   + (i_tau + dtau) * table->stride_tau
                                   + (i_sigma + dsigma) * table->stride_sigma
                                   + (i_r + dr) * table->stride_r
                                   + (i_q + dq) * table->stride_q;
                        values[dm*16 + dtau*8 + dsigma*4 + dr*2 + dq] = table->prices[idx];
                    }
                }
            }
        }
    }

    // 5D multilinear interpolation: 31 lerps
    // Stage 1: Interpolate along moneyness (16 → 8)
    double v_m[16];
    for (int i = 0; i < 16; i++) {
        v_m[i] = lerp(m0, m1, values[i*2], values[i*2 + 1], moneyness);
    }

    // Stage 2: Interpolate along maturity (8 → 4)
    double v_tau[8];
    for (int i = 0; i < 8; i++) {
        v_tau[i] = lerp(tau0, tau1, v_m[i*2], v_m[i*2 + 1], maturity);
    }

    // Stage 3: Interpolate along volatility (4 → 2)
    double v_sigma[4];
    for (int i = 0; i < 4; i++) {
        v_sigma[i] = lerp(sigma0, sigma1, v_tau[i*2], v_tau[i*2 + 1], volatility);
    }

    // Stage 4: Interpolate along rate (2 → 1)
    double v_r[2];
    v_r[0] = lerp(r0, r1, v_sigma[0], v_sigma[1], rate);
    v_r[1] = lerp(r0, r1, v_sigma[2], v_sigma[3], rate);

    // Stage 5: Interpolate along dividend (1 → final)
    double result = lerp(q0, q1, v_r[0], v_r[1], dividend);

    return result;
}

// ---------- Context Management ----------

static InterpContext multilinear_create_context(size_t dimensions,
                                                  const size_t *grid_sizes) {
    (void)dimensions;
    (void)grid_sizes;
    return NULL;  // Multilinear is stateless, no context needed
}

static void multilinear_destroy_context(InterpContext context) {
    (void)context;  // No-op
}
