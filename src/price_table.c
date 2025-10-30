#include "price_table.h"
#include "american_option.h"
#include "interp_multilinear.h"
#include "ivcalc_trace.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

// File format constants
#define PRICE_TABLE_MAGIC 0x50545442  // "PTTB"
#define PRICE_TABLE_VERSION 1

// File header structure
typedef struct {
    uint32_t magic;
    uint32_t version;
    size_t n_moneyness;
    size_t n_maturity;
    size_t n_volatility;
    size_t n_rate;
    size_t n_dividend;
    OptionType type;
    ExerciseType exercise;
    char underlying[32];
    time_t generation_time;
    uint8_t padding[128];  // Reserved for future use
} PriceTableHeader;

// ---------- Helper Functions ----------

/**
 * Convert flat index to multi-dimensional grid indices.
 * Maps a linear array index to (moneyness, maturity, volatility, rate, dividend)
 * indices based on the table's stride configuration.
 */
static void unflatten_index(size_t idx, const OptionPriceTable *table,
                           size_t *i_m, size_t *i_tau, size_t *i_sigma,
                           size_t *i_r, size_t *i_q) {
    size_t remaining = idx;

    *i_m = remaining / table->stride_m;
    remaining %= table->stride_m;

    *i_tau = remaining / table->stride_tau;
    remaining %= table->stride_tau;

    *i_sigma = remaining / table->stride_sigma;
    remaining %= table->stride_sigma;

    *i_r = remaining / table->stride_r;
    remaining %= table->stride_r;

    *i_q = remaining;
}

/**
 * Convert grid point indices to OptionData structure.
 * Extracts grid values at the specified indices and constructs an option
 * with fixed reference strike K_ref = 100.0.
 *
 * Moneyness scaling approach:
 * - All precomputed prices use K_ref = 100.0 as the strike
 * - Moneyness m = S/K is stored in the grid
 * - Actual spot price S will be computed when needed: S = m * K_ref
 * - This allows one table to serve all strikes via moneyness interpolation
 */
static OptionData grid_point_to_option(const OptionPriceTable *table,
                                       size_t i_m, size_t i_tau,
                                       size_t i_sigma, size_t i_r,
                                       size_t i_q) {
    const double K_ref = 100.0;  // Reference strike for moneyness scaling

    double m = table->moneyness_grid[i_m];
    double tau = table->maturity_grid[i_tau];
    double sigma = table->volatility_grid[i_sigma];
    double r = table->rate_grid[i_r];
    double q = (table->n_dividend > 0) ? table->dividend_grid[i_q] : 0.0;

    // Note: Moneyness m is extracted but not yet used
    // Spot price calculation S = m * K_ref will be performed in Task 2
    (void)m;
    (void)q;

    OptionData option = {
        .strike = K_ref,
        .volatility = sigma,
        .risk_free_rate = r,
        .time_to_maturity = tau,
        .option_type = table->type,
        .n_dividends = 0,
        .dividend_times = NULL,
        .dividend_amounts = NULL
    };

    return option;
}

/**
 * Get batch size for parallel computation from environment variable.
 * Defaults to 100 if IVCALC_PRECOMPUTE_BATCH_SIZE is not set or invalid.
 * Valid range: 1 to 100000.
 */
static size_t get_batch_size(void) {
    size_t batch_size = 100;  // Default

    char *env_batch = getenv("IVCALC_PRECOMPUTE_BATCH_SIZE");
    if (env_batch) {
        long val = atol(env_batch);
        if (val >= 1 && val <= 100000) {
            batch_size = (size_t)val;
        }
    }

    return batch_size;
}

// ---------- Creation and Destruction ----------

OptionPriceTable* price_table_create_with_strategy(
    const double *moneyness, size_t n_m,
    const double *maturity, size_t n_tau,
    const double *volatility, size_t n_sigma,
    const double *rate, size_t n_r,
    const double *dividend, size_t n_q,
    OptionType type, ExerciseType exercise,
    const InterpolationStrategy *strategy) {

    if (!moneyness || !maturity || !volatility || !rate) return NULL;
    if (n_m == 0 || n_tau == 0 || n_sigma == 0 || n_r == 0) return NULL;
    if (n_q > 0 && !dividend) return NULL;

    // Default to multilinear if no strategy specified
    if (!strategy) {
        strategy = &INTERP_MULTILINEAR;
    }

    OptionPriceTable *table = malloc(sizeof(OptionPriceTable));
    if (!table) return NULL;

    // Copy dimensions
    table->n_moneyness = n_m;
    table->n_maturity = n_tau;
    table->n_volatility = n_sigma;
    table->n_rate = n_r;
    table->n_dividend = n_q;

    // Allocate and copy grids
    table->moneyness_grid = malloc(n_m * sizeof(double));
    table->maturity_grid = malloc(n_tau * sizeof(double));
    table->volatility_grid = malloc(n_sigma * sizeof(double));
    table->rate_grid = malloc(n_r * sizeof(double));
    table->dividend_grid = (n_q > 0) ? malloc(n_q * sizeof(double)) : NULL;

    if (!table->moneyness_grid || !table->maturity_grid ||
        !table->volatility_grid || !table->rate_grid ||
        (n_q > 0 && !table->dividend_grid)) {
        free(table->moneyness_grid);
        free(table->maturity_grid);
        free(table->volatility_grid);
        free(table->rate_grid);
        free(table->dividend_grid);
        free(table);
        return NULL;
    }

    memcpy(table->moneyness_grid, moneyness, n_m * sizeof(double));
    memcpy(table->maturity_grid, maturity, n_tau * sizeof(double));
    memcpy(table->volatility_grid, volatility, n_sigma * sizeof(double));
    memcpy(table->rate_grid, rate, n_r * sizeof(double));
    if (n_q > 0) {
        memcpy(table->dividend_grid, dividend, n_q * sizeof(double));
    }

    // Allocate price array
    size_t n_points = n_m * n_tau * n_sigma * n_r * (n_q > 0 ? n_q : 1);
    table->prices = malloc(n_points * sizeof(double));
    if (!table->prices) {
        free(table->moneyness_grid);
        free(table->maturity_grid);
        free(table->volatility_grid);
        free(table->rate_grid);
        free(table->dividend_grid);
        free(table);
        return NULL;
    }

    // Initialize prices to NaN
    #pragma omp simd
    for (size_t i = 0; i < n_points; i++) {
        table->prices[i] = NAN;
    }

    // Set metadata
    table->type = type;
    table->exercise = exercise;
    memset(table->underlying, 0, sizeof(table->underlying));
    table->generation_time = time(NULL);

    // Compute strides for fast indexing
    if (n_q > 0) {
        // 5D mode
        table->stride_q = 1;
        table->stride_r = n_q;
        table->stride_sigma = n_r * n_q;
        table->stride_tau = n_sigma * n_r * n_q;
        table->stride_m = n_tau * n_sigma * n_r * n_q;
    } else {
        // 4D mode
        table->stride_q = 0;
        table->stride_r = 1;
        table->stride_sigma = n_r;
        table->stride_tau = n_sigma * n_r;
        table->stride_m = n_tau * n_sigma * n_r;
    }

    // Set strategy
    table->strategy = strategy;
    size_t dimensions = (n_q > 0) ? 5 : 4;
    size_t grid_sizes[5] = {n_m, n_tau, n_sigma, n_r, n_q};
    table->interp_context = NULL;
    if (strategy->create_context) {
        table->interp_context = strategy->create_context(dimensions, grid_sizes);
    }

    return table;
}

OptionPriceTable* price_table_create(
    const double *moneyness, size_t n_m,
    const double *maturity, size_t n_tau,
    const double *volatility, size_t n_sigma,
    const double *rate, size_t n_r,
    const double *dividend, size_t n_q,
    OptionType type, ExerciseType exercise) {
    return price_table_create_with_strategy(
        moneyness, n_m, maturity, n_tau, volatility, n_sigma,
        rate, n_r, dividend, n_q, type, exercise, &INTERP_MULTILINEAR);
}

void price_table_destroy(OptionPriceTable *table) {
    if (!table) return;

    // Destroy interpolation context
    if (table->strategy && table->strategy->destroy_context) {
        table->strategy->destroy_context(table->interp_context);
    }

    // Free arrays
    free(table->moneyness_grid);
    free(table->maturity_grid);
    free(table->volatility_grid);
    free(table->rate_grid);
    free(table->dividend_grid);
    free(table->prices);
    free(table);
}

// ---------- Pre-computation ----------

int price_table_precompute([[maybe_unused]] OptionPriceTable *table,
                            [[maybe_unused]] const void *pde_solver_template) {
    // Note: This is a placeholder for Phase 2
    // In Phase 2, we'll implement the actual FDM-based pre-computation
    // with OpenMP parallelization
    return -1;  // Not yet implemented
}

// ---------- Data Access ----------

double price_table_get(const OptionPriceTable *table,
                       size_t i_m, size_t i_tau, size_t i_sigma,
                       size_t i_r, size_t i_q) {
    if (!table) return NAN;
    if (i_m >= table->n_moneyness || i_tau >= table->n_maturity ||
        i_sigma >= table->n_volatility || i_r >= table->n_rate) {
        return NAN;
    }
    if (table->n_dividend > 0 && i_q >= table->n_dividend) {
        return NAN;
    }

    size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
               + i_sigma * table->stride_sigma + i_r * table->stride_r
               + i_q * table->stride_q;

    return table->prices[idx];
}

int price_table_set(OptionPriceTable *table,
                    size_t i_m, size_t i_tau, size_t i_sigma,
                    size_t i_r, size_t i_q, double price) {
    if (!table) return -1;
    if (i_m >= table->n_moneyness || i_tau >= table->n_maturity ||
        i_sigma >= table->n_volatility || i_r >= table->n_rate) {
        return -1;
    }
    if (table->n_dividend > 0 && i_q >= table->n_dividend) {
        return -1;
    }

    size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
               + i_sigma * table->stride_sigma + i_r * table->stride_r
               + i_q * table->stride_q;

    table->prices[idx] = price;
    return 0;
}

// ---------- Interpolation ----------

double price_table_interpolate_4d(const OptionPriceTable *table,
                                   double moneyness, double maturity,
                                   double volatility, double rate) {
    if (!table || !table->strategy || !table->strategy->interpolate_4d) {
        return NAN;
    }

    if (table->n_dividend > 0) {
        // Table is 5D, can't use 4D interpolation
        return NAN;
    }

    return table->strategy->interpolate_4d(table, moneyness, maturity,
                                            volatility, rate,
                                            table->interp_context);
}

double price_table_interpolate_5d(const OptionPriceTable *table,
                                   double moneyness, double maturity,
                                   double volatility, double rate,
                                   double dividend) {
    if (!table || !table->strategy || !table->strategy->interpolate_5d) {
        return NAN;
    }

    if (table->n_dividend == 0) {
        // Table is 4D, can't use 5D interpolation
        return NAN;
    }

    return table->strategy->interpolate_5d(table, moneyness, maturity,
                                            volatility, rate, dividend,
                                            table->interp_context);
}

OptionGreeks price_table_greeks_4d(const OptionPriceTable *table,
                                    double moneyness, double maturity,
                                    double volatility, double rate) {
    OptionGreeks greeks = {0};

    if (!table) return greeks;

    // Finite difference steps
    const double h_m = 0.001;      // 0.1% for moneyness
    const double h_tau = 1.0/365;  // 1 day for maturity
    const double h_sigma = 0.001;  // 0.1% for volatility
    const double h_r = 0.0001;     // 1 basis point for rate

    // Base price
    double V = price_table_interpolate_4d(table, moneyness, maturity, volatility, rate);

    // Delta: ∂V/∂m (moneyness sensitivity)
    double V_up = price_table_interpolate_4d(table, moneyness + h_m, maturity, volatility, rate);
    double V_down = price_table_interpolate_4d(table, moneyness - h_m, maturity, volatility, rate);
    greeks.delta = (V_up - V_down) / (2 * h_m);

    // Gamma: ∂²V/∂m²
    greeks.gamma = (V_up - 2*V + V_down) / (h_m * h_m);

    // Vega: ∂V/∂σ
    double V_vega_up = price_table_interpolate_4d(table, moneyness, maturity, volatility + h_sigma, rate);
    double V_vega_down = price_table_interpolate_4d(table, moneyness, maturity, volatility - h_sigma, rate);
    greeks.vega = (V_vega_up - V_vega_down) / (2 * h_sigma);

    // Theta: -∂V/∂τ (note negative sign)
    double V_theta_up = price_table_interpolate_4d(table, moneyness, maturity + h_tau, volatility, rate);
    double V_theta_down = price_table_interpolate_4d(table, moneyness, maturity - h_tau, volatility, rate);
    greeks.theta = -(V_theta_up - V_theta_down) / (2 * h_tau);

    // Rho: ∂V/∂r
    double V_rho_up = price_table_interpolate_4d(table, moneyness, maturity, volatility, rate + h_r);
    double V_rho_down = price_table_interpolate_4d(table, moneyness, maturity, volatility, rate - h_r);
    greeks.rho = (V_rho_up - V_rho_down) / (2 * h_r);

    return greeks;
}

OptionGreeks price_table_greeks_5d(const OptionPriceTable *table,
                                    double moneyness, double maturity,
                                    double volatility, double rate,
                                    double dividend) {
    OptionGreeks greeks = {0};

    if (!table) return greeks;

    // Finite difference steps
    const double h_m = 0.001;
    const double h_tau = 1.0/365;
    const double h_sigma = 0.001;
    const double h_r = 0.0001;

    // Base price
    double V = price_table_interpolate_5d(table, moneyness, maturity, volatility, rate, dividend);

    // Delta
    double V_up = price_table_interpolate_5d(table, moneyness + h_m, maturity, volatility, rate, dividend);
    double V_down = price_table_interpolate_5d(table, moneyness - h_m, maturity, volatility, rate, dividend);
    greeks.delta = (V_up - V_down) / (2 * h_m);

    // Gamma
    greeks.gamma = (V_up - 2*V + V_down) / (h_m * h_m);

    // Vega
    double V_vega_up = price_table_interpolate_5d(table, moneyness, maturity, volatility + h_sigma, rate, dividend);
    double V_vega_down = price_table_interpolate_5d(table, moneyness, maturity, volatility - h_sigma, rate, dividend);
    greeks.vega = (V_vega_up - V_vega_down) / (2 * h_sigma);

    // Theta
    double V_theta_up = price_table_interpolate_5d(table, moneyness, maturity + h_tau, volatility, rate, dividend);
    double V_theta_down = price_table_interpolate_5d(table, moneyness, maturity - h_tau, volatility, rate, dividend);
    greeks.theta = -(V_theta_up - V_theta_down) / (2 * h_tau);

    // Rho
    double V_rho_up = price_table_interpolate_5d(table, moneyness, maturity, volatility, rate + h_r, dividend);
    double V_rho_down = price_table_interpolate_5d(table, moneyness, maturity, volatility, rate - h_r, dividend);
    greeks.rho = (V_rho_up - V_rho_down) / (2 * h_r);

    return greeks;
}

int price_table_set_strategy(OptionPriceTable *table,
                              const InterpolationStrategy *strategy) {
    if (!table || !strategy) return -1;

    // Destroy old context
    if (table->strategy && table->strategy->destroy_context) {
        table->strategy->destroy_context(table->interp_context);
    }

    // Set new strategy
    table->strategy = strategy;

    // Create new context
    size_t dimensions = (table->n_dividend > 0) ? 5 : 4;
    size_t grid_sizes[5] = {
        table->n_moneyness, table->n_maturity, table->n_volatility,
        table->n_rate, table->n_dividend
    };
    table->interp_context = NULL;
    if (strategy->create_context) {
        table->interp_context = strategy->create_context(dimensions, grid_sizes);
    }

    // Pre-compute if supported
    if (strategy->precompute) {
        strategy->precompute(table, table->interp_context);
    }

    return 0;
}

// ---------- Metadata ----------

void price_table_set_underlying(OptionPriceTable *table, const char *underlying) {
    if (!table || !underlying) return;
    strncpy(table->underlying, underlying, sizeof(table->underlying) - 1);
    table->underlying[sizeof(table->underlying) - 1] = '\0';
}

const char* price_table_get_underlying(const OptionPriceTable *table) {
    return table ? table->underlying : NULL;
}

// ---------- I/O ----------

int price_table_save(const OptionPriceTable *table, const char *filename) {
    if (!table || !filename) return -1;

    FILE *fp = fopen(filename, "wb");
    if (!fp) return -1;

    // Write header
    PriceTableHeader header = {
        .magic = PRICE_TABLE_MAGIC,
        .version = PRICE_TABLE_VERSION,
        .n_moneyness = table->n_moneyness,
        .n_maturity = table->n_maturity,
        .n_volatility = table->n_volatility,
        .n_rate = table->n_rate,
        .n_dividend = table->n_dividend,
        .type = table->type,
        .exercise = table->exercise,
        .generation_time = table->generation_time
    };
    memcpy(header.underlying, table->underlying, sizeof(header.underlying));

    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    // Write grid arrays
    if (fwrite(table->moneyness_grid, sizeof(double), table->n_moneyness, fp) != table->n_moneyness ||
        fwrite(table->maturity_grid, sizeof(double), table->n_maturity, fp) != table->n_maturity ||
        fwrite(table->volatility_grid, sizeof(double), table->n_volatility, fp) != table->n_volatility ||
        fwrite(table->rate_grid, sizeof(double), table->n_rate, fp) != table->n_rate) {
        fclose(fp);
        return -1;
    }

    if (table->n_dividend > 0) {
        if (fwrite(table->dividend_grid, sizeof(double), table->n_dividend, fp) != table->n_dividend) {
            fclose(fp);
            return -1;
        }
    }

    // Write price data
    size_t n_points = table->n_moneyness * table->n_maturity * table->n_volatility
                    * table->n_rate * (table->n_dividend > 0 ? table->n_dividend : 1);
    if (fwrite(table->prices, sizeof(double), n_points, fp) != n_points) {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

OptionPriceTable* price_table_load(const char *filename) {
    if (!filename) return NULL;

    FILE *fp = fopen(filename, "rb");
    if (!fp) return NULL;

    // Read header
    PriceTableHeader header;
    if (fread(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    // Validate magic and version
    if (header.magic != PRICE_TABLE_MAGIC || header.version != PRICE_TABLE_VERSION) {
        fclose(fp);
        return NULL;
    }

    // Allocate grid arrays
    double *moneyness = malloc(header.n_moneyness * sizeof(double));
    double *maturity = malloc(header.n_maturity * sizeof(double));
    double *volatility = malloc(header.n_volatility * sizeof(double));
    double *rate = malloc(header.n_rate * sizeof(double));
    double *dividend = (header.n_dividend > 0) ? malloc(header.n_dividend * sizeof(double)) : NULL;

    if (!moneyness || !maturity || !volatility || !rate ||
        (header.n_dividend > 0 && !dividend)) {
        free(moneyness);
        free(maturity);
        free(volatility);
        free(rate);
        free(dividend);
        fclose(fp);
        return NULL;
    }

    // Read grids
    if (fread(moneyness, sizeof(double), header.n_moneyness, fp) != header.n_moneyness ||
        fread(maturity, sizeof(double), header.n_maturity, fp) != header.n_maturity ||
        fread(volatility, sizeof(double), header.n_volatility, fp) != header.n_volatility ||
        fread(rate, sizeof(double), header.n_rate, fp) != header.n_rate) {
        free(moneyness);
        free(maturity);
        free(volatility);
        free(rate);
        free(dividend);
        fclose(fp);
        return NULL;
    }

    if (header.n_dividend > 0) {
        if (fread(dividend, sizeof(double), header.n_dividend, fp) != header.n_dividend) {
            free(moneyness);
            free(maturity);
            free(volatility);
            free(rate);
            free(dividend);
            fclose(fp);
            return NULL;
        }
    }

    // Create table
    OptionPriceTable *table = price_table_create(
        moneyness, header.n_moneyness,
        maturity, header.n_maturity,
        volatility, header.n_volatility,
        rate, header.n_rate,
        dividend, header.n_dividend,
        header.type, header.exercise);

    free(moneyness);
    free(maturity);
    free(volatility);
    free(rate);
    free(dividend);

    if (!table) {
        fclose(fp);
        return NULL;
    }

    // Read price data
    size_t n_points = header.n_moneyness * header.n_maturity * header.n_volatility
                    * header.n_rate * (header.n_dividend > 0 ? header.n_dividend : 1);
    if (fread(table->prices, sizeof(double), n_points, fp) != n_points) {
        price_table_destroy(table);
        fclose(fp);
        return NULL;
    }

    // Set metadata
    memcpy(table->underlying, header.underlying, sizeof(table->underlying));
    table->generation_time = header.generation_time;

    fclose(fp);
    return table;
}
